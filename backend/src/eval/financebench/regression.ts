/**
 * Garde-fous de régression, port de regression.py. À lancer avant de merger un changement de
 * prompt, de chunking, de retrieval ou de modèle.
 *
 *   pnpm eval:gate retrieval                    # rang de la page de preuve vs baseline versionnée
 *   pnpm eval:gate retrieval --update-baseline  # (re)crée la baseline de la config courante
 *   pnpm eval:gate sentinels                    # ~10 questions réussies aujourd'hui, jugées de bout en bout
 *
 * Code de sortie : 0 si le garde-fou passe, 1 en cas de régression, 2 si le run est invalide
 * (un service a lâché en silence, ex. rerank en échec : le résultat ne mesure pas le code).
 *
 * La baseline de retrieval dépend des embeddings et du reranker : un fichier par configuration.
 */
import { startTelemetry, shutdownTelemetry } from "../../observability/instrumentation";
startTelemetry();

import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs } from "node:util";
import { env, REPO_ROOT } from "../../config/settings";
import { AgentWorkflow } from "../../agent/workflow";
import { judgeModel, modelSlug } from "../../llm/providers";
import { withBackoff } from "../../llm/resilience";
import { FB_DIR, loadDataset, loadIndex } from "./corpus";
import { FinanceBenchJudge } from "./judge";
import { retrievalMetrics } from "./metrics";
import { evaluateExample, type Row } from "./run";

const TOP_K = 10; // = RESEARCH_TOP_K : ce que le modèle voit sans outils
const K_VALUES = [5, 10, 20];
const GATE_DIR = path.join(REPO_ROOT, "eval-outputs", "regression");
const log = (msg: string) => process.stdout.write(`[regression] ${msg}\n`);

type QuestionMetrics = Record<string, number | null>;

const inTop = (rank: number | null | undefined, k: number) => rank !== null && rank !== undefined && rank <= k;

/**
 * Compare deux runs question par question. Une moyenne ne suffit pas : perdre la preuve de
 * 2 questions et la gagner sur 2 autres laisse page_hit@10 inchangé.
 */
export function compareRetrieval(
  baseline: Record<string, QuestionMetrics>,
  current: Record<string, QuestionMetrics>,
  k = TOP_K,
  maxLost = 0,
  maxMeanDrop = 0.05,
) {
  const lost: unknown[] = [];
  const gained: unknown[] = [];
  const moved: unknown[] = [];
  for (const [id, base] of Object.entries(baseline)) {
    const cur = current[id];
    if (!cur) continue;
    const b = base.gold_rank;
    const c = cur.gold_rank;
    if (inTop(b, k) && !inTop(c, k)) lost.push({ id, before: b, after: c });
    else if (inTop(c, k) && !inTop(b, k)) gained.push({ id, before: b, after: c });
    else if (b !== c) moved.push({ id, before: b, after: c });
  }
  const common = Object.keys(baseline).filter((id) => id in current);
  const missing = Object.keys(baseline).filter((id) => !(id in current));
  const avg = (rows: Record<string, QuestionMetrics>, key: string) => {
    const vals = common.map((id) => rows[id][key]).filter((v): v is number => typeof v === "number");
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  const metrics: Record<string, { baseline: number | null; current: number | null; delta: number | null }> = {};
  for (const key of [`page_hit@${k}`, `page_recall@${k}`, `mrr@${k}`]) {
    const b = avg(baseline, key);
    const c = avg(current, key);
    metrics[key] = { baseline: b, current: c, delta: b === null || c === null ? null : c - b };
  }
  const hitDelta = metrics[`page_hit@${k}`].delta ?? 0;
  const reasons: string[] = [];
  if (lost.length > maxLost) reasons.push(`${lost.length} question(s) perdent la page de preuve du top-${k} (tolérance ${maxLost})`);
  if (hitDelta < -maxMeanDrop) reasons.push(`page_hit@${k} baisse de ${(-hitDelta * 100).toFixed(1)} pts (tolérance ${maxMeanDrop * 100})`);
  if (missing.length) reasons.push(`${missing.length} question(s) de la baseline absentes du run`);
  return { ok: reasons.length === 0, reasons, lost, gained, moved, missing, metrics };
}

/** Raisons d'échec d'une sentinelle (vide = elle passe). */
export function checkSentinel(row: Row, expect: { verdict?: string; evidence_seen?: boolean; min_tool_calls?: number }): string[] {
  if (row.failed) return [`échec technique: ${String(row.error ?? "").slice(0, 120)}`];
  const failures: string[] = [];
  const want = expect.verdict ?? "CORRECT";
  if (row.verdict !== want) failures.push(`verdict ${row.verdict} au lieu de ${want}`);
  if (expect.evidence_seen && !row.evidence_seen) failures.push("la page de preuve n'a jamais été montrée au modèle");
  if ((row.tool_calls ?? 0) < (expect.min_tool_calls ?? 0)) failures.push(`${row.tool_calls ?? 0} appel(s) d'outils, ${expect.min_tool_calls} attendu(s) au moins`);
  return failures;
}

function baselinePath() {
  const rerank = env.RERANK_ENABLED ? modelSlug(env.RERANK_PROVIDER, env.RERANK_MODEL_ID) : "norerank";
  return path.join(GATE_DIR, `retrieval_baseline.${modelSlug(env.EMBEDDING_PROVIDER, env.EMBEDDING_MODEL_ID)}.${rerank}.${env.VECTOR_STORE}.json`);
}

async function cmdRetrieval(updateBaseline: boolean, maxLost: number, maxMeanDrop: number, tolerance: number): Promise<number> {
  const dataset = loadDataset();
  const docs = [...new Set(dataset.map((ex) => ex.doc_name))].sort();
  const { retriever } = await loadIndex(docs);
  log(`Retrieval sur ${dataset.length} questions`);
  const current: Record<string, QuestionMetrics> = {};
  for (const ex of dataset) {
    const found = await withBackoff(() => retriever.invoke(ex.question), `retrieval ${ex.id}`);
    const m = retrievalMetrics(found, ex, K_VALUES, tolerance);
    current[ex.id] = Object.fromEntries(
      ["gold_rank", ...K_VALUES.flatMap((k) => ["page_hit", "page_recall", "mrr"].map((n) => `${n}@${k}`))].map((key) => [key, m[key] as number | null]),
    );
  }
  if (retriever.reranker.failures.length) {
    log(`RUN INVALIDE — rerank en échec sur ${retriever.reranker.failures.length}/${dataset.length} requêtes (quota, accès modèle ou région).`);
    return 2;
  }
  fs.mkdirSync(GATE_DIR, { recursive: true });
  const file = baselinePath();
  if (updateBaseline) {
    fs.writeFileSync(file, JSON.stringify({ created: new Date().toISOString().slice(0, 10), top_k: TOP_K, questions: current }, null, 2) + "\n");
    const hit = Object.values(current).filter((q) => inTop(q.gold_rank, TOP_K)).length / dataset.length;
    log(`Baseline écrite : ${file} (page_hit@${TOP_K} = ${(hit * 100).toFixed(1)} %)`);
    return 0;
  }
  if (!fs.existsSync(file)) throw new Error(`Pas de baseline (${file}). Lancer d'abord avec --update-baseline.`);
  const baseline = JSON.parse(fs.readFileSync(file, "utf-8")).questions;
  const report = compareRetrieval(baseline, current, TOP_K, maxLost, maxMeanDrop);
  fs.writeFileSync(path.join(GATE_DIR, "last_retrieval.json"), JSON.stringify({ report, questions: current }, null, 2));
  for (const [key, v] of Object.entries(report.metrics)) {
    log(`${key}: ${v.baseline?.toFixed(3)} -> ${v.current?.toFixed(3)} (${v.delta === null ? "—" : (v.delta >= 0 ? "+" : "") + v.delta.toFixed(3)})`);
  }
  for (const l of report.lost as { id: string; before: number; after: number | null }[]) log(`  PERDUE ${l.id}: rang ${l.before} -> ${l.after ?? "absente"}`);
  for (const g of report.gained as { id: string; before: number | null; after: number }[]) log(`  gagnée ${g.id}: rang ${g.before ?? "absente"} -> ${g.after}`);
  log(report.ok ? "OK — pas de régression du retrieval." : `RÉGRESSION — ${report.reasons.join(" ; ")}`);
  return report.ok ? 0 : 1;
}

async function cmdSentinels(tolerance: number): Promise<number> {
  const dataset = loadDataset();
  const spec = JSON.parse(fs.readFileSync(path.join(FB_DIR, "sentinels.json"), "utf-8")) as {
    sentinels: { id: string; verdict?: string; evidence_seen?: boolean; min_tool_calls?: number; why?: string }[];
  };
  const byId = new Map(dataset.map((ex) => [ex.id, ex]));
  const unknown = spec.sentinels.filter((s) => !byId.has(s.id));
  if (unknown.length) throw new Error(`Sentinelles absentes du dataset : ${unknown.map((s) => s.id).join(", ")}`);
  const docs = [...new Set(dataset.map((ex) => ex.doc_name))].sort(); // même index que l'éval complète
  const { retriever, pageStore } = await loadIndex(docs);
  const workflow = new AgentWorkflow();
  const judge = new FinanceBenchJudge(judgeModel());
  log(`${spec.sentinels.length} sentinelles, mode agentic + juge`);
  let failed = 0;
  for (const s of spec.sentinels) {
    const [row] = await evaluateExample({ ex: byId.get(s.id)!, modes: ["agentic"], retriever, pageStore, workflow, judge, kValues: K_VALUES, tolerance });
    const reasons = checkSentinel(row, s);
    if (reasons.length) failed++;
    log(`${reasons.length ? "ÉCHEC" : "ok   "} ${s.id} — ${reasons.join(" ; ") || s.why}`);
  }
  if (retriever.reranker.failures.length) {
    log(`RUN INVALIDE — rerank en échec sur ${retriever.reranker.failures.length} requêtes.`);
    return 2;
  }
  log(failed ? `RÉGRESSION — ${failed}/${spec.sentinels.length} sentinelle(s) en échec.` : "OK — toutes les sentinelles passent.");
  return failed ? 1 : 0;
}

async function main(): Promise<number> {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      "update-baseline": { type: "boolean", default: false },
      "max-lost": { type: "string", default: "0" },
      "max-mean-drop": { type: "string", default: "0.05" },
      "page-tolerance": { type: "string", default: "1" },
    },
  });
  const tolerance = Number(values["page-tolerance"]);
  switch (positionals[0]) {
    case "retrieval":
      return cmdRetrieval(values["update-baseline"]!, Number(values["max-lost"]), Number(values["max-mean-drop"]), tolerance);
    case "sentinels":
      return cmdSentinels(tolerance);
    default:
      throw new Error("Usage : pnpm eval:gate <retrieval|sentinels> [--update-baseline]");
  }
}

if (require.main === module) {
  main()
    .then((code) => (process.exitCode = code))
    .catch((error) => {
      process.stderr.write(`${(error as Error).stack ?? error}\n`);
      process.exitCode = 2;
    })
    .finally(() => shutdownTelemetry());
}

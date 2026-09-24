/**
 * Évaluation FinanceBench du RAG agentique, port de run_financebench_eval.py.
 *
 * Protocole identique : 26 questions, 4 10-K, index combiné, un retrieval par question
 * partagé par les deux modes, juge LLM, erreurs techniques hors dénominateur, IC95.
 *
 *   pnpm eval                                   # les deux modes (ablation outils / sans outils)
 *   pnpm eval --mode agentic --label bedrock-mistral-large
 *   pnpm eval --max-items 3 --no-judge --out-dir /tmp/fb-essai
 *   pnpm eval --no-langfuse                     # pas d'experiment Langfuse (traces seules)
 *
 * Chaque run écrit summary.json (avec la configuration des modèles : un run se relit seul),
 * results.json et errors.json dans eval-outputs/<label>/. Avec les clés Langfuse, un run complet
 * tourne comme une experiment par mode sur le dataset `financebench` (langfuse-experiment.ts).
 */
import { getActiveTraceId } from "@langfuse/tracing";
import { startTelemetry, shutdownTelemetry, traced } from "../../observability/instrumentation";
const tracing = startTelemetry();

import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs } from "node:util";
import { env, REPO_ROOT, settings } from "../../config/settings";
import type { Passage } from "../../core/passage";
import { AgentWorkflow, type Mode } from "../../agent/workflow";
import { judgeModel } from "../../llm/providers";
import { isRateLimit, rootCause, withBackoff } from "../../llm/resilience";
import type { PageStore } from "../../retrieval/page-store";
import type { Retriever } from "../../retrieval/retriever";
import { loadDataset, loadIndex, type Example } from "./corpus";
import { computeCost, formatCost } from "./cost";
import { runLangfuseExperiments } from "./langfuse-experiment";
import { aggregateVerdicts, FinanceBenchJudge, type Verdict } from "./judge";
import { contextHit, f1Score, pageFlags, retrievalMetrics } from "./metrics";

export type Row = Record<string, any> & { id: string; mode: Mode };

const log = (msg: string) => process.stdout.write(`[financebench] ${msg}\n`);
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const round = (x: number | null, d = 4) => (x === null ? null : Math.round(x * 10 ** d) / 10 ** d);

export function modelConfig() {
  return {
    llm: `${env.LLM_PROVIDER}:${env.MODEL_ID}`,
    llm_small: `${env.LLM_PROVIDER}:${env.MODEL_SMALL_ID}`,
    embeddings: `${env.EMBEDDING_PROVIDER}:${env.EMBEDDING_MODEL_ID}`,
    rerank: env.RERANK_ENABLED ? `${env.RERANK_PROVIDER}:${env.RERANK_MODEL_ID}` : "off",
    judge: `${env.JUDGE_PROVIDER}:${env.JUDGE_MODEL_ID}`,
    vector_store: env.VECTOR_STORE,
    region: env.AWS_REGION,
  };
}

export type EvalContext = {
  retriever: Retriever;
  pageStore: PageStore;
  workflow: AgentWorkflow;
  judge: FinanceBenchJudge | null;
  kValues: number[];
  tolerance: number;
};

export type Retrieval = Awaited<ReturnType<typeof retrieve>>;

/** Le retrieval d'une question, partagé par les deux modes : seule la génération diffère. */
export async function retrieve(ex: Example, ctx: EvalContext) {
  const question = ex.question.trim();
  const t0 = Date.now();
  const scope = await withBackoff(() => ctx.retriever.route(question), `routage ${ex.id}`);
  const docs = await withBackoff(() => ctx.retriever.invokeWithScope(question, scope), `retrieval ${ex.id}`);
  return { scope, docs, metrics: { ...retrievalMetrics(docs, ex, ctx.kValues, ctx.tolerance), retrieval_sec: (Date.now() - t0) / 1000 } };
}

/** Réponse et verdict d'un mode. Un échec de génération ou du juge devient une ligne en échec, jamais une exception. */
export async function evaluateMode(ex: Example, mode: Mode, retrieval: Retrieval, ctx: EvalContext): Promise<Row> {
  const { retriever, pageStore, workflow, judge } = ctx;
  const question = ex.question.trim();
  const { scope, docs } = retrieval;
  const row: Row = {
    id: ex.id,
    doc_name: ex.doc_name,
    question,
    expected_answer: ex.expected_answer,
    question_type: ex.question_type,
    question_reasoning: ex.question_reasoning,
    mode,
    ...retrieval.metrics,
  };
  const t1 = Date.now();
  let run;
  try {
    run = await withBackoff(
      () =>
        traced("financebench-question", { id: ex.id, question, mode }, () => {
          row.trace_id = getActiveTraceId();
          return workflow.run({ question, passages: docs, retriever, pageStore, scope, mode });
        },
          { asType: "chain", attrs: { tags: ["eval", "financebench", mode], metadata: { id: ex.id } }, output: (r) => r.draft_answer },
        ),
      `génération ${mode} ${ex.id}`,
    );
  } catch (error) {
    Object.assign(row, { failed: true, error: rootCause(error), rate_limited: isRateLimit(error), answer: "", generation_sec: (Date.now() - t1) / 1000 });
    if (judge) row.verdict = "ERROR";
    return row;
  }
  const llmDocs: Passage[] = run.passages;
  const flags = pageFlags(llmDocs, ex.gold_pages, ctx.tolerance);
  Object.assign(row, {
    answer: run.draft_answer,
    generation_sec: (Date.now() - t1) / 1000,
    evidence_seen: ex.gold_pages.length ? flags.some(Boolean) : null,
    context_hit: contextHit(llmDocs, ex.expected_answer, ex.answer_keywords),
    answer_f1: ex.expected_answer ? f1Score(run.draft_answer, ex.expected_answer) : 0,
  });
  if (mode === "agentic") {
    Object.assign(row, {
      relevance: run.relevance,
      tool_calls: run.toolCalls,
      corrective_rounds: run.toolCalls > 0 ? 1 : 0,
      corrective_queries: run.calls,
      pages_read: [...new Set(llmDocs.filter((d) => d.metadata.origin === "read_page").map((d) => `${d.metadata.docName} p.${d.metadata.page}`))].sort(),
    });
  }
  if (judge) {
    try {
      const v = await withBackoff(
        () =>
          judge.evaluate({
            question,
            expected: ex.expected_answer,
            answer: run.draft_answer,
            context: llmDocs.map((d) => d.content).join("\n\n"),
            justification: ex.justification,
          }),
        `juge ${ex.id}`,
      );
      Object.assign(row, { verdict: v.verdict, judge_faithfulness: v.faithfulness, judge_reason: v.reason });
    } catch (error) {
      Object.assign(row, { verdict: "ERROR", judge_reason: `juge indisponible: ${rootCause(error)}` });
    }
  }
  return row;
}

/** Un retrieval, puis une réponse par mode (chaque mode isolé : l'échec de l'un ne perd pas l'autre). */
export async function evaluateExample(opts: EvalContext & { ex: Example; modes: Mode[] }): Promise<Row[]> {
  const retrieval = await retrieve(opts.ex, opts);
  const rows: Row[] = [];
  for (const mode of opts.modes) rows.push(await evaluateMode(opts.ex, mode, retrieval, opts));
  return rows;
}

export function aggregate(rows: Row[], kValues: number[]) {
  const failed = rows.filter((r) => r.failed);
  const ok = rows.filter((r) => !r.failed);
  if (!ok.length) return { count: 0, failed: failed.length, attempted: rows.length };
  const m = (key: string) => round(mean(ok.map((r) => r[key]).filter((v) => typeof v === "number")));
  const out: Record<string, any> = {
    count: ok.length,
    failed: failed.length,
    rate_limited: failed.filter((r) => r.rate_limited).length,
    attempted: rows.length,
    mean_f1: m("answer_f1"),
    context_hit_rate: round(ok.filter((r) => r.context_hit).length / ok.length),
    mean_generation_sec: round(mean(ok.map((r) => r.generation_sec)), 2),
    mean_retrieval_sec: round(mean(ok.map((r) => r.retrieval_sec)), 2),
  };
  const seen = ok.filter((r) => r.evidence_seen !== null && r.evidence_seen !== undefined);
  if (seen.length) out.evidence_seen_rate = round(seen.filter((r) => r.evidence_seen).length / seen.length);
  const corrected = ok.filter((r) => r.corrective_rounds);
  if (ok.some((r) => r.corrective_rounds !== undefined)) {
    out.corrective_rate = round(corrected.length / ok.length);
    if (corrected.length) {
      out.mean_tool_calls_when_corrected = round(mean(corrected.map((r) => r.tool_calls)), 2);
      out.pages_read_rate_when_corrected = round(corrected.filter((r) => r.pages_read?.length).length / corrected.length);
    }
  }
  const retrieval: Record<string, number | null> = {};
  for (const k of kValues) {
    for (const metric of ["recall", "precision", "mrr", "ndcg", "page_hit", "page_recall", "page_precision"]) {
      const key = `${metric}@${k}`;
      const v = m(key);
      if (v !== null) retrieval[key] = v;
    }
  }
  out.retrieval = retrieval;
  const verdicts = ok.filter((r) => r.verdict).map((r) => ({ verdict: r.verdict, faithfulness: r.judge_faithfulness ?? 0, reason: "" }) as Verdict);
  // Les échecs techniques comptent aussi comme ERROR côté juge (hors dénominateur).
  if (verdicts.length) out.financebench = aggregateVerdicts([...verdicts, ...failed.map(() => ({ verdict: "ERROR", faithfulness: 0, reason: "" }) as Verdict)]);
  return out;
}

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);

export function printReport(summary: Record<string, any>, modes: Mode[], kValues: number[], nProtocol: number) {
  const lines = ["", "=".repeat(78), "RÉSULTATS — FinanceBench (protocole Patronus AI)", "=".repeat(78)];
  const fb = (b: any) => b?.financebench ?? {};
  const hasVerdicts = modes.some((m) => fb(summary[m]).count);
  if (!hasVerdicts) lines.push("", "⚠️  AUCUN VERDICT — le juge n'a pas tourné (--no-judge) : ce run ne mesure que le retrieval et la latence.");
  else if (summary.n_questions < nProtocol)
    lines.push("", `⚠️  RUN PARTIEL — ${summary.n_questions}/${nProtocol} questions : pas comparable aux chiffres publiés.`);
  lines.push(`${"Métrique".padEnd(36)}${modes.map((m) => m.padStart(14)).join("")}`, "-".repeat(78));
  const row = (label: string, get: (b: any) => string) =>
    lines.push(`${label.padEnd(36)}${modes.map((m) => get(summary[m] ?? {}).padStart(14)).join("")}`);
  const count = (b: any, key: string) => (fb(b).counts ? `${fb(b).counts[key]}/${fb(b).count}` : "—");
  const ci = (b: any, key: string) => {
    const c = fb(b)[`${key}_ci95`];
    return c ? `[${Math.round(c[0] * 100)}-${Math.round(c[1] * 100)}%]` : "—";
  };
  if (hasVerdicts) {
    row("Accuracy (CORRECT)", (b) => pct(fb(b).accuracy));
    row("  questions", (b) => count(b, "correct"));
    row("  IC95 %", (b) => ci(b, "accuracy"));
    row("Hallucinations (INCORRECT)", (b) => pct(fb(b).hallucination_rate));
    row("  questions", (b) => count(b, "hallucination"));
    row("Refus (REFUSAL)", (b) => pct(fb(b).refusal_rate));
    row("  questions", (b) => count(b, "refusal"));
    row("Faithfulness moyenne /5", (b) => String(fb(b).mean_faithfulness ?? "—"));
    lines.push("-".repeat(78));
  }
  const kS = Math.min(...kValues);
  const kL = Math.max(...kValues);
  row("Preuve transmise au LLM", (b) => pct(b.evidence_seen_rate));
  row("Outils appelés (part des questions)", (b) => pct(b.corrective_rate));
  row("  appels d'outils (moy., si appelés)", (b) => String(b.mean_tool_calls_when_corrected ?? "—"));
  row("  page entière lue (si appelés)", (b) => pct(b.pages_read_rate_when_corrected));
  lines.push("-".repeat(78));
  row(`page_hit@${kS} (retrieval exact)`, (b) => pct(b.retrieval?.[`page_hit@${kS}`]));
  row(`page_hit@${kL}`, (b) => pct(b.retrieval?.[`page_hit@${kL}`]));
  row(`page_recall@${kL}`, (b) => pct(b.retrieval?.[`page_recall@${kL}`]));
  row("context_hit_rate", (b) => pct(b.context_hit_rate));
  row("mean_f1", (b) => pct(b.mean_f1));
  lines.push("-".repeat(78));
  row("Latence retrieval (s/question)", (b) => String(b.mean_retrieval_sec ?? "—"));
  row("Latence génération (s/question)", (b) => String(b.mean_generation_sec ?? "—"));
  row("Questions évaluées", (b) => `${b.count ?? 0}/${b.attempted ?? 0}`);
  row("Questions en échec", (b) => String(b.failed ?? 0));
  lines.push("=".repeat(78), `Durée totale: ${summary.elapsed_sec}s`, formatCost(summary.cost));
  lines.push("", `Modèles : ${Object.entries(summary.config).map(([k, v]) => `${k}=${v}`).join(" | ")}`);
  if (modes.length === 2 && fb(summary.baseline).counts && fb(summary.agentic).counts) {
    const gap = fb(summary.agentic).counts.correct - fb(summary.baseline).counts.correct;
    lines.push("", `ℹ️  Écart agentic − baseline : ${gap >= 0 ? "+" : ""}${gap} question(s) correcte(s) sur ${fb(summary.agentic).count}.`);
    if (Math.abs(gap) <= 2) lines.push("    Les IC95 se recouvrent largement : ce n'est pas une amélioration démontrable.");
  }
  lines.push("", "Référence Python (4 sept. 2026) : agentic 83,3 % (20/24), baseline 65,4 % (17/26).", "");
  process.stdout.write(lines.join("\n") + "\n");
}

async function pool<T, R>(items: T[], workers: number, fn: (item: T, i: number) => Promise<R>, onDone: (r: R, item: T, i: number) => void) {
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.max(1, workers) }, async () => {
      while (next < items.length) {
        const i = next++;
        onDone(await fn(items[i], i), items[i], i);
      }
    }),
  );
}

async function main() {
  const { values: args } = parseArgs({
    options: {
      mode: { type: "string", default: "both" },
      label: { type: "string" },
      "out-dir": { type: "string" },
      "max-items": { type: "string", default: "0" },
      docs: { type: "string", default: "" },
      "k-values": { type: "string", default: "5,10,20" },
      workers: { type: "string", default: "2" },
      "no-judge": { type: "boolean", default: false },
      "page-tolerance": { type: "string", default: "1" },
      "force-overwrite": { type: "boolean", default: false },
      "no-langfuse": { type: "boolean", default: false },
    },
  });
  const modes: Mode[] = args.mode === "both" ? ["baseline", "agentic"] : [args.mode as Mode];
  const label = args.label ?? `${env.LLM_PROVIDER}-${env.MODEL_ID}`.replace(/[^a-zA-Z0-9._-]+/g, "_");
  const defaultDir = path.join(REPO_ROOT, "eval-outputs", label);
  const outDir = args["out-dir"] ?? defaultDir;
  const kValues = args["k-values"]!.split(",").map(Number);
  const tolerance = Number(args["page-tolerance"]);

  const partial = [
    Number(args["max-items"]) ? `--max-items ${args["max-items"]}` : "",
    args["no-judge"] ? "--no-judge" : "",
    args.docs ? `--docs ${args.docs}` : "",
  ].filter(Boolean);
  // Un run partiel n'écrase jamais les sorties d'un run complet.
  if (partial.length && outDir === defaultDir && !args["force-overwrite"] && fs.existsSync(path.join(outDir, "summary.json"))) {
    throw new Error(`Refus d'écrire un run partiel (${partial.join(", ")}) dans ${outDir}. Utiliser --out-dir ou --force-overwrite.`);
  }

  let dataset = loadDataset();
  const nProtocol = dataset.length;
  if (args.docs) {
    const wanted = new Set(args.docs.split(",").map((d) => d.trim()));
    dataset = dataset.filter((ex) => wanted.has(ex.doc_name));
  }
  if (Number(args["max-items"])) dataset = dataset.slice(0, Number(args["max-items"]));
  const docs = [...new Set(loadDataset().map((ex) => ex.doc_name))].sort(); // index complet : mêmes distracteurs

  log(`${dataset.length} questions, index sur ${docs.join(", ")} | modes: ${modes.join(", ")} | juge: ${!args["no-judge"]}`);
  log(`Modèles : ${JSON.stringify(modelConfig())}`);
  const start = Date.now();
  const { retriever, pageStore, chunkCount } = await loadIndex(docs);
  log(`Index prêt : ${chunkCount} chunks, ${pageStore.documents().length} documents (${((Date.now() - start) / 1000).toFixed(0)}s)`);

  const workflow = new AgentWorkflow();
  const judge = args["no-judge"] ? null : new FinanceBenchJudge(judgeModel());
  const ctx: EvalContext = { retriever, pageStore, workflow, judge, kValues, tolerance };
  const settingsUsed = { research_top_k: settings.RESEARCH_TOP_K, max_tool_calls: settings.GENERATOR_MAX_TOOL_CALLS, tools: env.GENERATOR_TOOLS_ENABLED };
  const rows: Row[] = [];
  const errors: unknown[] = [];
  const recordFailures = (result: Row[]) => {
    for (const r of result) if (r.failed) errors.push({ id: r.id, mode: r.mode, error: r.error, rate_limited: r.rate_limited });
  };
  const verdictsOf = (result: Row[]) => result.map((r) => (r.failed ? "ÉCHEC" : (r.verdict ?? "ok"))).join("/") || "ÉCHEC";

  // Un run partiel ou sans juge fausserait la comparaison des experiments du dataset.
  const experiment = tracing && !args["no-langfuse"] && !partial.length;
  if (tracing && !args["no-langfuse"] && partial.length) log(`Langfuse : run partiel (${partial.join(", ")}), pas d'experiment (traces seules).`);

  if (experiment) {
    // Retrieval mémorisé : les experiments tournent mode par mode, la question n'est cherchée qu'une fois.
    const retrievals = new Map<string, Promise<Retrieval>>();
    const done = new Map<Mode, number>();
    const evaluate = async (ex: Example, mode: Mode): Promise<Row> => {
      if (!retrievals.has(ex.id)) retrievals.set(ex.id, retrieve(ex, ctx));
      const row = await retrievals
        .get(ex.id)!
        .then((r) => evaluateMode(ex, mode, r, ctx))
        .catch((error: unknown): Row => ({ id: ex.id, mode, doc_name: ex.doc_name, question: ex.question, failed: true, error: rootCause(error), rate_limited: isRateLimit(error), answer: "", ...(judge ? { verdict: "ERROR" } : {}) }));
      done.set(mode, (done.get(mode) ?? 0) + 1);
      log(`[${mode} ${done.get(mode)}/${dataset.length}] ${ex.id} (${ex.doc_name}) -> ${verdictsOf([row])}`);
      return row;
    };
    const result = await runLangfuseExperiments({
      examples: dataset,
      modes,
      label,
      metadata: { config: modelConfig(), settings: settingsUsed, n_questions: dataset.length },
      workers: Number(args.workers),
      kValues,
      evaluate,
      summarize: (modeRows) => aggregate(modeRows, kValues),
      log,
    });
    rows.push(...result);
    recordFailures(result);
  } else {
    let done = 0;
    await pool(
      dataset,
      Number(args.workers),
      (ex) =>
        evaluateExample({ ex, modes, ...ctx }).catch((error) => {
          errors.push({ id: ex.id, mode: "*", error: rootCause(error), rate_limited: isRateLimit(error) });
          return [] as Row[];
        }),
      (result, ex) => {
        done++;
        rows.push(...result);
        recordFailures(result);
        log(`[${done}/${dataset.length}] ${ex.id} (${ex.doc_name}) -> ${verdictsOf(result)}`);
      },
    );
  }

  const summary: Record<string, any> = {
    label,
    date: new Date().toISOString(),
    config: modelConfig(),
    settings: settingsUsed,
    documents: docs,
    n_questions: dataset.length,
    elapsed_sec: Math.round((Date.now() - start) / 100) / 10,
    errors: errors.length,
    rerank_failures: retriever.reranker.failures.length,
    cost: computeCost(),
  };
  for (const mode of modes) summary[mode] = aggregate(rows.filter((r) => r.mode === mode), kValues);

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "summary.json"), JSON.stringify(summary, null, 2));
  fs.writeFileSync(path.join(outDir, "results.json"), JSON.stringify(rows, null, 2));
  fs.writeFileSync(path.join(outDir, "errors.json"), JSON.stringify(errors, null, 2));
  printReport(summary, modes, kValues, nProtocol);
  if (summary.rerank_failures) log(`⚠️  Rerank en échec sur ${summary.rerank_failures} requêtes : run dégradé.`);
  log(`Résultats écrits dans ${outDir}/`);

}

if (require.main === module) {
  main()
    .catch((error) => {
      process.stderr.write(`${(error as Error).stack ?? error}\n`);
      process.exitCode = 1;
    })
    .finally(() => shutdownTelemetry());
}

/**
 * Fait tourner l'éval FinanceBench comme une experiment Langfuse (Datasets > financebench, onglet Experiments).
 *
 * L'onglet Experiments lit des attributs posés sur les spans pendant l'exécution : seul
 * experiment.run du SDK les pose, un lien créé après coup n'y apparaît pas. Chaque mode
 * devient donc une experiment sur le dataset, et la génération + le juge tournent dans sa tâche.
 *
 * - Le dataset `financebench` contient les questions (upsert par id : relancer ne duplique rien).
 * - Les scores par question (correct, hallucination, faithfulness, page_hit@k...) sont posés sur
 *   la trace : Langfuse affiche leur moyenne par run, donc correct = accuracy.
 * - Les agrégats sans équivalent par question (IC95) sont des scores du run.
 */
import { LangfuseClient } from "@langfuse/client";
import type { Mode } from "../../agent/workflow";
import type { Example } from "./corpus";
import type { Row } from "./run";

const DATASET = "financebench";
const itemId = (id: string) => `${DATASET}-${id}`;

type Score = Omit<Parameters<LangfuseClient["score"]["create"]>[0], "traceId" | "datasetRunId">;
const numeric = (name: string, value: unknown): Score[] =>
  typeof value === "number" && Number.isFinite(value) ? [{ name, value, dataType: "NUMERIC" }] : [];
const bool = (name: string, value: unknown): Score[] =>
  typeof value === "boolean" ? [{ name, value: value ? 1 : 0, dataType: "BOOLEAN" }] : [];

/** Scores par question. Les échecs (génération ou juge) restent hors dénominateur, comme dans run.ts. */
function itemScores(row: Row, kValues: number[]): Score[] {
  const scores: Score[] = [];
  if (row.verdict && row.verdict !== "ERROR") {
    scores.push({ name: "verdict", value: row.verdict, dataType: "CATEGORICAL", comment: row.judge_reason });
    scores.push(...bool("correct", row.verdict === "CORRECT"), ...bool("hallucination", row.verdict === "INCORRECT"), ...bool("refusal", row.verdict === "REFUSAL"));
    if (row.judge_faithfulness > 0) scores.push(...numeric("faithfulness", row.judge_faithfulness));
  }
  if (row.failed) return scores;
  const kS = Math.min(...kValues);
  const kL = Math.max(...kValues);
  scores.push(
    ...bool("evidence_seen", row.evidence_seen),
    ...bool("context_hit", row.context_hit),
    ...numeric("answer_f1", row.answer_f1),
    ...numeric(`page_hit@${kS}`, row[`page_hit@${kS}`]),
    ...numeric(`page_hit@${kL}`, row[`page_hit@${kL}`]),
    ...numeric(`page_recall@${kL}`, row[`page_recall@${kL}`]),
    ...numeric("retrieval_sec", row.retrieval_sec),
    ...numeric("generation_sec", row.generation_sec),
  );
  if (row.mode === "agentic") scores.push(...bool("tools_called", row.tool_calls > 0), ...numeric("tool_calls", row.tool_calls));
  return scores;
}

function runScores(agg: Record<string, any>): Score[] {
  const fb = agg.financebench ?? {};
  const ci = (key: string): Score[] => (fb[`${key}_ci95`] ? [...numeric(`${key}_ci95_low`, fb[`${key}_ci95`][0]), ...numeric(`${key}_ci95_high`, fb[`${key}_ci95`][1])] : []);
  return [
    ...numeric("accuracy", fb.accuracy),
    ...ci("accuracy"),
    ...numeric("hallucination_rate", fb.hallucination_rate),
    ...ci("hallucination_rate"),
    ...numeric("refusal_rate", fb.refusal_rate),
    ...numeric("mean_faithfulness", fb.mean_faithfulness),
    ...numeric("questions_failed", agg.failed),
  ];
}

async function syncDataset(langfuse: LangfuseClient, examples: Example[]) {
  await langfuse.api.datasets.create({
    name: DATASET,
    description: "FinanceBench (Patronus AI) : 26 questions sur 4 10-K, protocole du projet Python.",
  });
  for (const ex of examples) {
    await langfuse.api.datasetItems.create({
      datasetName: DATASET,
      id: itemId(ex.id),
      input: { question: ex.question, doc_name: ex.doc_name },
      expectedOutput: ex.expected_answer,
      metadata: { id: ex.id, company: ex.company, question_type: ex.question_type, question_reasoning: ex.question_reasoning, gold_pages: ex.gold_pages },
    });
  }
}

/** Une experiment par mode, l'une après l'autre (le retrieval mémorisé par run.ts sert aux deux). */
export async function runLangfuseExperiments(opts: {
  examples: Example[];
  modes: Mode[];
  label: string;
  metadata: Record<string, unknown>;
  workers: number;
  kValues: number[];
  evaluate: (ex: Example, mode: Mode) => Promise<Row>;
  summarize: (rows: Row[]) => Record<string, any>;
  log: (msg: string) => void;
}): Promise<Row[]> {
  const { examples, modes, kValues, log } = opts;
  const langfuse = new LangfuseClient();
  await syncDataset(langfuse, examples);
  const byItem = new Map(examples.map((ex) => [itemId(ex.id), ex]));
  const dataset = await langfuse.dataset.get(DATASET);
  const items = dataset.items.filter((item) => byItem.has(item.id));

  const stamp = new Date().toLocaleString("sv-SE").slice(0, 16); // heure locale, « 2026-09-24 13:40 »
  const all: Row[] = [];
  for (const mode of modes) {
    const rows = new Map<string, Row>();
    const result = await langfuse.experiment.run({
      name: `FinanceBench ${mode}`,
      runName: `${opts.label} · ${mode} · ${stamp}`,
      description: `FinanceBench ${mode} — ${String((opts.metadata.config as Record<string, string>)?.llm ?? "")}`,
      metadata: { mode, ...opts.metadata },
      data: items,
      maxConcurrency: opts.workers,
      task: async (item) => {
        const ex = byItem.get((item as { id: string }).id)!;
        const row = await opts.evaluate(ex, mode);
        rows.set(ex.id, row);
        return row.answer;
      },
      evaluators: [async ({ metadata }) => itemScores(rows.get(metadata!.id)!, kValues)],
      runEvaluators: [async () => runScores(opts.summarize([...rows.values()]))],
    });
    all.push(...rows.values());
    log(`Langfuse : experiment « ${result.runName} » (${rows.size}/${items.length} questions)${result.datasetRunUrl ? ` → ${result.datasetRunUrl}` : ""}`);
  }
  return all;
}

/**
 * Juge LLM au protocole FinanceBench : verdict CORRECT / INCORRECT / REFUSAL + faithfulness.
 * Prompt, détection des messages figés et parsing repris de evaluation/llm_judge.py.
 * Le juge doit rester le même d'un run à l'autre (JUDGE_PROVIDER / JUDGE_MODEL_ID).
 */
import { generateText, type LanguageModel } from "ai";
import { settings } from "../../config/settings";
import { isRateLimit } from "../../llm/resilience";
import { modelIdOf, usageMeter } from "../../llm/usage";
import { wilson } from "./metrics";

export type VerdictLabel = "CORRECT" | "INCORRECT" | "REFUSAL" | "ERROR";
export type Verdict = { verdict: VerdictLabel; faithfulness: number; reason: string };

// Messages figés du pipeline : ils doivent constituer TOUTE la réponse pour valoir refus
// (le premier apparaît légitimement au milieu d'une réponse complète).
const REFUSAL_MARKERS = [
  "cette information n'est pas disponible dans le document",
  "je ne peux pas répondre à cette question basée sur les documents fournis",
  "cette question n'est pas liée",
];
const ERROR_MARKERS = [
  "une erreur est survenue lors du traitement de votre question",
  "une erreur est survenue lors de la génération de la réponse",
];
const CANNED_MAX_EXTRA_CHARS = 160;

const JUDGE_PROMPT = (q: string, expected: string, justification: string, answer: string, context: string) => `Tu es un évaluateur expert en analyse financière. Tu compares la réponse d'un système RAG à la réponse de référence annotée par des experts sur le benchmark FinanceBench.

**Question:** ${q}

**Réponse de référence (ground truth):** ${expected}

**Justification de la référence:** ${justification}

**Réponse générée par le système:** ${answer}

**Extraits du document fournis au système:**
${context}

Rends DEUX jugements.

1) VERDICT — classe la réponse générée:
- "CORRECT": elle donne la même information que la référence. Tolère les écarts de formulation, d'unité, d'arrondi et de mise en forme ($1,577 = 1577.00 = 1 577 millions). Une réponse plus détaillée que la référence reste CORRECT si elle ne la contredit pas.
- "INCORRECT": elle contredit la référence, donne un chiffre faux, ou répond à côté.
- "REFUSAL": elle déclare ne pas pouvoir répondre ou que l'information n'est pas dans le document.

RÈGLE DE DÉPARTAGE (à appliquer AVANT de choisir REFUSAL) : si la réponse de référence dit elle-même que la métrique n'est pas applicable, pas publiée ou pas utilisée pour cette entreprise (ex. "Performance is not measured through gross margin", "There are none"), alors une réponse générée qui constate que le document ne fournit pas cette métrique, qu'elle n'y figure pas ou qu'il n'y en a pas est CORRECT — ce n'est pas un refus, c'est la bonne réponse. REFUSAL est réservé au cas où la référence contient une vraie information que la réponse générée déclare introuvable.

2) FAITHFULNESS (1-5) — tout ce qu'affirme la réponse est-il appuyé par les extraits fournis ?
- 5: intégralement appuyé | 4: inférences mineures raisonnables | 3: quelques affirmations non appuyées
- 2: plusieurs affirmations inventées | 1: hallucinations majeures
Un refus honnête vaut 5.

Réponds EXACTEMENT dans ce format:
VERDICT: [CORRECT|INCORRECT|REFUSAL]
FAITHFULNESS: [1-5]
RAISON: [1-2 phrases]`;

const normalize = (a: string) => (a || "").toLowerCase().replace(/[*_#`>]/g, " ").split(/\s+/).filter(Boolean).join(" ");

export function detectCanned(answer: string): "REFUSAL" | "ERROR" | null {
  const low = normalize(answer);
  if (!low) return "ERROR";
  if (ERROR_MARKERS.some((m) => low.startsWith(m))) return "ERROR";
  for (const m of REFUSAL_MARKERS) {
    if (low.startsWith(m) && low.length <= m.length + CANNED_MAX_EXTRA_CHARS) return "REFUSAL";
  }
  return null;
}

// « **VERDICT:** CORRECT » ou « VERDICT**: CORRECT » : un juge qui met ses champs en gras
// voyait sinon tous ses CORRECT comptés INCORRECT.
const field = (name: string) => `${name}\\**\\s*:[\\s*\\[]*`;

export function parseVerdict(text: string): Verdict {
  const v = new RegExp(field("VERDICT") + "(CORRECT|INCORRECT|REFUSAL)", "i").exec(text);
  const f = new RegExp(field("FAITHFULNESS") + "(\\d(?:\\.\\d)?)", "i").exec(text);
  const r = new RegExp(field("RAISON") + "([\\s\\S]+?)(?:\\n\\s*\\n|$)", "i").exec(text);
  return {
    verdict: (v ? v[1].toUpperCase() : "INCORRECT") as VerdictLabel,
    faithfulness: f ? Math.max(1, Math.min(5, parseFloat(f[1]))) : 3,
    reason: r ? r[1].split(/\s+/).join(" ").trim() : "Impossible de parser la réponse du juge",
  };
}

export class FinanceBenchJudge {
  constructor(private readonly model: LanguageModel) {}

  async evaluate(opts: { question: string; expected: string; answer: string; context: string; justification?: string }): Promise<Verdict> {
    const canned = detectCanned(opts.answer);
    if (canned === "ERROR") return { verdict: "ERROR", faithfulness: 0, reason: "Erreur technique du pipeline" };
    if (canned === "REFUSAL") return { verdict: "REFUSAL", faithfulness: 5, reason: "Refus explicite du pipeline (message figé)" };
    const context = opts.context.length > 6000 ? opts.context.slice(0, 6000) + "..." : opts.context;
    try {
      const result = await generateText({
        model: this.model,
        prompt: JUDGE_PROMPT(opts.question, opts.expected || "(non fournie)", opts.justification || "(non fournie)", opts.answer, context || "(aucun extrait)"),
        temperature: 0,
        maxOutputTokens: 250,
        maxRetries: settings.LLM_MAX_RETRIES,
        timeout: settings.LLM_TIMEOUT_MS,
        telemetry: { functionId: "financebench-judge" },
      });
      usageMeter.record(modelIdOf(this.model), result.usage);
      return parseVerdict(result.text);
    } catch (error) {
      if (isRateLimit(error)) throw error;
      return { verdict: "ERROR", faithfulness: 0, reason: `Erreur du juge: ${(error as Error).message}` };
    }
  }
}

/** Agrégat au protocole FinanceBench : erreurs techniques hors dénominateur, IC95 de Wilson. */
export function aggregateVerdicts(verdicts: Verdict[]) {
  const scored = verdicts.filter((v) => v.verdict !== "ERROR");
  const errors = verdicts.length - scored.length;
  const n = scored.length;
  if (!n) return { count: 0, errors };
  const counts = {
    correct: scored.filter((v) => v.verdict === "CORRECT").length,
    refusal: scored.filter((v) => v.verdict === "REFUSAL").length,
    hallucination: scored.filter((v) => v.verdict === "INCORRECT").length,
  };
  const faith = scored.map((v) => v.faithfulness).filter((f) => f > 0);
  const r = (x: number) => Math.round(x * 10000) / 10000;
  return {
    count: n,
    errors,
    counts,
    accuracy: r(counts.correct / n),
    refusal_rate: r(counts.refusal / n),
    hallucination_rate: r(counts.hallucination / n),
    accuracy_ci95: wilson(counts.correct, n),
    hallucination_rate_ci95: wilson(counts.hallucination, n),
    refusal_rate_ci95: wilson(counts.refusal, n),
    mean_faithfulness: faith.length ? Math.round((faith.reduce((a, b) => a + b, 0) / faith.length) * 100) / 100 : null,
  };
}

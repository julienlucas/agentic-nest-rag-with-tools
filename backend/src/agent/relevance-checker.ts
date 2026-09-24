/**
 * Vérificateur de pertinence (modèle léger) : CAN_ANSWER / PARTIAL / NO_MATCH.
 * Son verdict ne bloque plus la génération : il devient un indice dans le prompt du modèle
 * à outils (« le contexte a été jugé partiel, cherchez ce qui manque »).
 */
import { generateText, type LanguageModel } from "ai";
import { settings } from "../config/settings";
import type { Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import { isRateLimit } from "../llm/resilience";
import { modelIdOf, usageMeter } from "../llm/usage";
import { relevancePrompt, type Relevance } from "./prompts";

const logger = getLogger("RelevanceChecker");

const LABEL_RE = /(?<![A-Z_])(CAN[ _-]ANSWER|PARTIAL|NO[ _-]MATCH)(?![A-Z_])/;

/**
 * Premier label cité, quelle que soit sa mise en forme (« CAN_ANSWER. », « **PARTIAL** »),
 * ou null. Une égalité stricte classait en NO_MATCH des réponses pourtant claires.
 */
export function parseRelevanceLabel(text: string): Relevance | null {
  const m = LABEL_RE.exec((text || "").toUpperCase());
  return m ? (m[1].replace(/[ -]/g, "_") as Relevance) : null;
}

export class RelevanceChecker {
  constructor(private readonly model: LanguageModel) {}

  async check(question: string, passages: Passage[], k = 3): Promise<Relevance> {
    if (!passages.length) return "NO_MATCH";
    const content = passages.slice(0, k).map((p) => p.content).join("\n\n");
    try {
      const result = await generateText({
        model: this.model,
        prompt: relevancePrompt(question, content),
        temperature: 0,
        maxOutputTokens: 10,
        maxRetries: settings.LLM_MAX_RETRIES,
        timeout: settings.LLM_TIMEOUT_MS,
        telemetry: { functionId: "relevance-checker" },
      });
      usageMeter.record(modelIdOf(this.model), result.usage);
      const label = parseRelevanceLabel(result.text);
      if (!label) {
        logger.warn(`Label illisible (${JSON.stringify(result.text.slice(0, 80))}), NO_MATCH forcé`);
        return "NO_MATCH";
      }
      return label;
    } catch (error) {
      if (isRateLimit(error)) throw error; // à rejouer par l'appelant, pas un vrai NO_MATCH
      logger.error(`Erreur du vérificateur : ${(error as Error).message}`);
      return "NO_MATCH";
    }
  }
}

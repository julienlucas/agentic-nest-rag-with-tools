/**
 * Reranking (Cohere Rerank via Bedrock, ou l'API Cohere directe). Exposé séparément du
 * pipeline pour pouvoir reclasser un ensemble contre la question d'origine.
 *
 * En cas d'échec, les candidats sont renvoyés non reclassés (comme en Python) : le garde-fou
 * de régression repère ces runs dégradés au lieu de les prendre pour une régression du code.
 */
import { rerank } from "ai";
import { env, settings } from "../config/settings";
import type { Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import { rerankingModel } from "../llm/providers";
import { isRateLimit } from "../llm/resilience";
import { usageMeter } from "../llm/usage";

const logger = getLogger("Reranker");

export class Reranker {
  private readonly model = env.RERANK_ENABLED ? rerankingModel() : null;
  /** Requêtes où le rerank a échoué (run dégradé). */
  readonly failures: string[] = [];

  get enabled() {
    return this.model !== null;
  }

  async rerank(query: string, docs: Passage[], topN: number = settings.RERANK_TOP_K): Promise<Passage[]> {
    if (!this.model || !docs.length) return docs;
    const candidates = docs.slice(0, settings.MAX_RERANK_CANDIDATES);
    usageMeter.recordRerank();
    try {
      const result = await rerank({
        model: this.model,
        query,
        documents: candidates.map((d) => d.content),
        topN: Math.min(topN, candidates.length),
        maxRetries: settings.LLM_MAX_RETRIES,
      });
      const out = result.ranking.map(({ originalIndex, score }) => {
        const doc = candidates[originalIndex];
        return { content: doc.content, metadata: { ...doc.metadata, rerankScore: score } };
      });
      if (out.length) {
        const avg = out.reduce((s, d) => s + (d.metadata.rerankScore ?? 0), 0) / out.length;
        logger.debug(`${candidates.length} docs -> top ${out.length}, top=${out[0].metadata.rerankScore?.toFixed(3)}, avg=${avg.toFixed(3)}`);
      }
      return out;
    } catch (error) {
      if (isRateLimit(error)) throw error;
      logger.warn(`Erreur de rerank : ${(error as Error).message}. Candidats renvoyés non reclassés.`);
      this.failures.push(query.slice(0, 80));
      return candidates;
    }
  }
}

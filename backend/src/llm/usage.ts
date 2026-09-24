/**
 * Compteur de consommation (tokens par modèle, appels de rerank), cumulé sur le process.
 * L'éval le lit pour chiffrer le coût d'un run, comme `cost.py` côté Python.
 */
type ModelUsage = { inputTokens: number; outputTokens: number; calls: number };

class UsageMeter {
  private models = new Map<string, ModelUsage>();
  rerankCalls = 0;
  embeddingTokens = new Map<string, number>();

  record(modelId: string, usage: { inputTokens?: number; outputTokens?: number } | undefined) {
    const row = this.models.get(modelId) ?? { inputTokens: 0, outputTokens: 0, calls: 0 };
    row.inputTokens += usage?.inputTokens ?? 0;
    row.outputTokens += usage?.outputTokens ?? 0;
    row.calls += 1;
    this.models.set(modelId, row);
  }

  recordEmbedding(modelId: string, tokens: number | undefined) {
    this.embeddingTokens.set(modelId, (this.embeddingTokens.get(modelId) ?? 0) + (tokens ?? 0));
  }

  recordRerank() {
    this.rerankCalls += 1;
  }

  snapshot() {
    return {
      models: Object.fromEntries(this.models),
      embeddings: Object.fromEntries(this.embeddingTokens),
      rerankCalls: this.rerankCalls,
    };
  }

  reset() {
    this.models.clear();
    this.embeddingTokens.clear();
    this.rerankCalls = 0;
  }
}

export const usageMeter = new UsageMeter();

export function modelIdOf(model: unknown): string {
  const m = model as { modelId?: string; provider?: string };
  return m?.modelId ?? String(model);
}

/**
 * Coût d'un run à partir des tokens réellement consommés (compteur usageMeter).
 *
 * Grille La Plateforme reprise de cost.py (relevée le 5 septembre 2026). Les tarifs Bedrock
 * ne sont PAS renseignés : les copier depuis https://aws.amazon.com/bedrock/pricing/ pour la
 * région utilisée. Un modèle sans tarif apparaît avec ses tokens et « — ».
 */
import { usageMeter } from "../../llm/usage";

type Price = { inputPerM: number; outputPerM: number };

export const PRICES_USD: Record<string, Price> = {
  // préfixe du modèle -> $ par million de tokens
  "mistral-large": { inputPerM: 0.5, outputPerM: 1.5 },
  "mistral-small": { inputPerM: 0.15, outputPerM: 0.6 },
  // À compléter (grille Bedrock de ta région), par exemple :
  // "mistral.mistral-large": { inputPerM: ?, outputPerM: ? },
  // "eu.anthropic.claude-sonnet": { inputPerM: ?, outputPerM: ? },
};
/** $ par requête de rerank, à compléter (Cohere API : 0,0025 $ la recherche). */
export const RERANK_PRICE_USD: number | null = null;
export const USD_TO_EUR = 0.86;

function priceFor(model: string): Price | undefined {
  const name = model.toLowerCase();
  return Object.entries(PRICES_USD).find(([prefix]) => name.startsWith(prefix))?.[1];
}

export function computeCost() {
  const snap = usageMeter.snapshot();
  let total = 0;
  const unpriced: string[] = [];
  const byModel: Record<string, unknown> = {};
  for (const [model, u] of Object.entries(snap.models)) {
    const price = priceFor(model);
    const usd = price ? (u.inputTokens * price.inputPerM + u.outputTokens * price.outputPerM) / 1e6 : null;
    if (usd === null) unpriced.push(model);
    else total += usd;
    byModel[model] = { calls: u.calls, input_tokens: u.inputTokens, output_tokens: u.outputTokens, usd: usd === null ? null : +usd.toFixed(4) };
  }
  const rerankUsd = RERANK_PRICE_USD === null ? null : snap.rerankCalls * RERANK_PRICE_USD;
  if (rerankUsd !== null) total += rerankUsd;
  byModel.rerank = { calls: snap.rerankCalls, usd: rerankUsd };
  return {
    total_usd: +total.toFixed(2),
    total_eur: +(total * USD_TO_EUR).toFixed(2),
    complete: unpriced.length === 0 && rerankUsd !== null,
    unpriced_models: unpriced,
    by_model: byModel,
    embedding_tokens: snap.embeddings,
  };
}

export function formatCost(cost: ReturnType<typeof computeCost>): string {
  const lines = Object.entries(cost.by_model).map(([model, u]) => {
    const row = u as { calls: number; input_tokens?: number; output_tokens?: number; usd: number | null };
    const usd = row.usd === null ? "—" : `${row.usd.toFixed(2)} $`;
    return row.input_tokens === undefined
      ? `${model}: ${row.calls} appels, ${usd}`
      : `${model}: ${row.calls} appels, ${row.input_tokens.toLocaleString("fr-FR")} in / ${row.output_tokens!.toLocaleString("fr-FR")} out, ${usd}`;
  });
  const head = `Coût du run: ${cost.total_usd.toFixed(2)} $ ≈ ${cost.total_eur.toFixed(2)} €` +
    (cost.complete ? "" : " (INCOMPLET : tarifs manquants dans eval/financebench/cost.ts)");
  return `${head}\n  ${lines.join("\n  ")}`;
}

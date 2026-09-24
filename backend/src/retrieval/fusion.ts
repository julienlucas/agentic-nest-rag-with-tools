/**
 * Fusion RRF pondérée, identique à l'EnsembleRetriever LangChain :
 * score(d) = Σ poids_i / (rang_i(d) + c), c = 60, dédoublonnage sur le contenu.
 */
import type { Passage } from "../core/passage";

export function weightedRrf(lists: Passage[][], weights: readonly number[], c = 60): Passage[] {
  const scores = new Map<string, number>();
  const first = new Map<string, Passage>();
  lists.forEach((list, li) => {
    list.forEach((doc, rank) => {
      const key = doc.content;
      scores.set(key, (scores.get(key) ?? 0) + (weights[li] ?? 1) / (rank + 1 + c));
      if (!first.has(key)) first.set(key, doc);
    });
  });
  // Tri stable : à score égal, l'ordre d'apparition (BM25 puis vecteurs) est conservé.
  return [...first.keys()]
    .map((key, order) => ({ key, order, score: scores.get(key)! }))
    .sort((a, b) => b.score - a.score || a.order - b.order)
    .map(({ key }) => first.get(key)!);
}

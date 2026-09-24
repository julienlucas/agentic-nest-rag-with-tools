/**
 * Pré-calcule les embeddings des ~12 400 chunks FinanceBench (et l'index OpenSearch si
 * VECTOR_STORE=opensearch). Une seule fois par modèle d'embeddings ; l'éval réutilise le cache.
 *
 *   pnpm index:financebench
 */
import { loadDataset, loadIndex } from "../eval/financebench/corpus";

async function main() {
  const docs = [...new Set(loadDataset().map((ex) => ex.doc_name))].sort();
  const t0 = Date.now();
  const { chunkCount, pageStore } = await loadIndex(docs);
  console.log(`Index prêt : ${chunkCount} chunks, ${pageStore.documents().length} documents, en ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

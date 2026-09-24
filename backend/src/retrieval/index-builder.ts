/**
 * Construit le Retriever d'un corpus : embeddings (cache disque), index lexical et vectoriel
 * (mémoire ou OpenSearch selon VECTOR_STORE), reranker et routeur.
 */
import { env } from "../config/settings";
import type { Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import { smallModel } from "../llm/providers";
import { DocumentRouter } from "./document-router";
import { Embedder } from "./embedder";
import { MemoryBm25Index, MemoryVectorIndex, type LexicalIndex, type VectorIndex } from "./indexes";
import { OpenSearchIndex } from "./opensearch-index";
import { Reranker } from "./reranker";
import { Retriever } from "./retriever";

const logger = getLogger("IndexBuilder");

export type CorpusPart = { key: string; chunks: Passage[] };

let sharedEmbedder: Embedder | undefined;
const embedder = () => (sharedEmbedder ??= new Embedder());

/**
 * `parts` : un morceau par document, chacun avec sa clé de cache d'embeddings (le nom du
 * 10-K, ou le hash du fichier uploadé). L'index est construit sur l'union.
 */
export async function buildRetriever(parts: CorpusPart[], indexName: string): Promise<Retriever> {
  const passages: Passage[] = [];
  const vectors: Float32Array[] = [];
  for (const part of parts) {
    const v = await embedder().embedCorpus(part.key, part.chunks.map((c) => c.content));
    passages.push(...part.chunks);
    vectors.push(...v);
  }

  let lexical: LexicalIndex;
  let vector: VectorIndex;
  if (env.VECTOR_STORE === "opensearch") {
    const index = await OpenSearchIndex.build(`${indexName}-${embedder().slug}`, passages, vectors);
    lexical = index;
    vector = index;
  } else {
    lexical = new MemoryBm25Index(passages);
    vector = new MemoryVectorIndex(passages, vectors);
  }

  const sources = [...new Set(passages.map((p) => p.metadata.source).filter(Boolean))];
  const router = env.DOCUMENT_ROUTING_ENABLED && sources.length > 1 ? new DocumentRouter(sources, smallModel()) : null;
  logger.log(`Retriever prêt : ${passages.length} chunks, ${sources.length} source(s), index ${env.VECTOR_STORE}`);
  return new Retriever(lexical, vector, embedder(), new Reranker(), router);
}

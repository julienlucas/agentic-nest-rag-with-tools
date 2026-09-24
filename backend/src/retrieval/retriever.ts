/**
 * La chaîne de retrieval, dans le même ordre que le projet Python :
 *
 *   routage par document -> hybride (BM25 + vecteurs, RRF) dans ce périmètre
 *   -> parent-child -> rerank (top 30)
 *
 * (Le multi-query Python est configuré à MULTI_QUERY_COUNT = 1, soit désactivé : il n'est
 * pas porté.)
 */
import { settings } from "../config/settings";
import type { Passage } from "../core/passage";
import { DocumentRouter } from "./document-router";
import { Embedder } from "./embedder";
import { weightedRrf } from "./fusion";
import type { LexicalIndex, Scope, VectorIndex } from "./indexes";
import { toParents } from "./parent-child";
import { Reranker } from "./reranker";

export class Retriever {
  constructor(
    private readonly lexical: LexicalIndex,
    private readonly vector: VectorIndex,
    private readonly embedder: Embedder,
    readonly reranker: Reranker,
    readonly router: DocumentRouter | null,
  ) {}

  /** Périmètre documentaire de la question (null = tous les documents). */
  async route(question: string): Promise<Scope> {
    return this.router ? this.router.route(question) : null;
  }

  /** Candidats hybrides fusionnés, avant parent-child et rerank. */
  async hybrid(query: string, scope: Scope): Promise<Passage[]> {
    const [lexical, vector] = await Promise.all([
      this.lexical.searchLexical(query, settings.BM25_K, scope),
      this.embedder.embedQuery(query).then((v) => this.vector.searchVector(v, settings.VECTOR_SEARCH_K, scope)),
    ]);
    return weightedRrf([lexical, vector], settings.HYBRID_WEIGHTS, settings.RRF_C);
  }

  async invokeWithScope(query: string, scope: Scope): Promise<Passage[]> {
    const parents = toParents(await this.hybrid(query, scope));
    return this.reranker.rerank(query, parents);
  }

  async invoke(query: string): Promise<Passage[]> {
    return this.invokeWithScope(query, await this.route(query));
  }
}

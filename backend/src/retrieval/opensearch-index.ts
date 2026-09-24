/**
 * Adaptateur OpenSearch des ports LexicalIndex / VectorIndex : un seul index porte le texte
 * (BM25 natif, analyseur standard) et le vecteur (knn_vector HNSW, cosinus, moteur Lucene
 * pour le filtrage par source pendant la recherche k-NN).
 *
 * La fusion RRF reste côté application (fusion.ts), pour que les adaptateurs mémoire et
 * OpenSearch ne diffèrent que par l'index : c'est ce qu'on veut mesurer à l'éval.
 */
import { Client } from "@opensearch-project/opensearch";
import { env } from "../config/settings";
import type { Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import type { LexicalIndex, Scope, VectorIndex } from "./indexes";

const logger = getLogger("OpenSearch");

type Source = {
  content: string;
  source: string;
  doc_name: string | null;
  page: number | null;
  parent_id: string | null;
  parent_content: string | null;
};

let clientInstance: Client | undefined;
export function openSearchClient(): Client {
  if (!clientInstance) {
    clientInstance = new Client({
      node: env.OPENSEARCH_URL,
      ...(env.OPENSEARCH_USERNAME
        ? { auth: { username: env.OPENSEARCH_USERNAME, password: env.OPENSEARCH_PASSWORD ?? "" } }
        : {}),
    });
  }
  return clientInstance;
}

function toPassage(hit: { _source?: unknown }): Passage {
  const s = hit._source as Source;
  return {
    content: s.content,
    metadata: {
      source: s.source,
      docName: s.doc_name,
      page: s.page,
      parentId: s.parent_id,
      parentContent: s.parent_content,
    },
  };
}

function scopeFilter(scope: Scope) {
  return scope && scope.length ? [{ terms: { source: scope } }] : [];
}

export class OpenSearchIndex implements LexicalIndex, VectorIndex {
  private constructor(
    private readonly client: Client,
    readonly index: string,
  ) {}

  /** Crée (ou réutilise, s'il est complet) l'index du corpus et y charge passages + vecteurs. */
  static async build(indexName: string, passages: Passage[], vectors: Float32Array[]): Promise<OpenSearchIndex> {
    const client = openSearchClient();
    const index = indexName.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").slice(0, 200);
    const exists = (await client.indices.exists({ index })).body;
    if (exists) {
      const count = (await client.count({ index })).body.count;
      if (count === passages.length) {
        logger.log(`Index ${index} réutilisé (${count} documents)`);
        return new OpenSearchIndex(client, index);
      }
      logger.warn(`Index ${index} incomplet (${count}/${passages.length}), reconstruction`);
      await client.indices.delete({ index });
    }

    const dimension = vectors[0]?.length ?? 1024;
    await client.indices.create({
      index,
      body: {
        settings: { index: { knn: true, number_of_shards: 1, number_of_replicas: 0 } },
        mappings: {
          properties: {
            content: { type: "text" },
            source: { type: "keyword" },
            doc_name: { type: "keyword" },
            page: { type: "integer" },
            parent_id: { type: "keyword" },
            parent_content: { type: "text", index: false },
            embedding: {
              type: "knn_vector",
              dimension,
              method: { name: "hnsw", space_type: "cosinesimil", engine: "lucene" },
            },
          },
        },
      },
    });

    const BATCH = 500;
    for (let i = 0; i < passages.length; i += BATCH) {
      const body: unknown[] = [];
      passages.slice(i, i + BATCH).forEach((p, j) => {
        body.push({ index: { _index: index, _id: String(i + j) } });
        body.push({
          content: p.content,
          source: p.metadata.source,
          doc_name: p.metadata.docName ?? null,
          page: p.metadata.page ?? null,
          parent_id: p.metadata.parentId ?? null,
          parent_content: p.metadata.parentContent ?? null,
          embedding: Array.from(vectors[i + j]),
        });
      });
      const res = await client.bulk({ body: body as never, refresh: false });
      if (res.body.errors) throw new Error(`Indexation OpenSearch en erreur (lot ${i / BATCH})`);
    }
    await client.indices.refresh({ index });
    logger.log(`Index ${index} créé (${passages.length} documents, dimension ${dimension})`);
    return new OpenSearchIndex(client, index);
  }

  async searchLexical(query: string, k: number, scope: Scope): Promise<Passage[]> {
    const res = await this.client.search({
      index: this.index,
      body: {
        size: k,
        _source: { excludes: ["embedding"] },
        query: { bool: { must: [{ match: { content: query } }], filter: scopeFilter(scope) } },
      },
    });
    return res.body.hits.hits.map(toPassage);
  }

  async searchVector(vector: number[], k: number, scope: Scope): Promise<Passage[]> {
    const filter = scopeFilter(scope);
    const res = await this.client.search({
      index: this.index,
      body: {
        size: k,
        _source: { excludes: ["embedding"] },
        query: {
          knn: { embedding: { vector, k, ...(filter.length ? { filter: { bool: { filter } } } : {}) } },
        },
      },
    });
    return res.body.hits.hits.map(toPassage);
  }
}

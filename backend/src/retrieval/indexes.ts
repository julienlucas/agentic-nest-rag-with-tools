/**
 * Ports du retrieval : un index lexical et un index vectoriel, filtrables par source.
 * Deux adaptateurs : en mémoire (ce fichier) et OpenSearch (opensearch-index.ts).
 */
import type { Passage } from "../core/passage";
import { Bm25 } from "./bm25";

export type Scope = string[] | null;

export interface LexicalIndex {
  searchLexical(query: string, k: number, scope: Scope): Promise<Passage[]>;
}

export interface VectorIndex {
  searchVector(vector: number[], k: number, scope: Scope): Promise<Passage[]>;
}

/** BM25 en mémoire, reconstruit par périmètre (mis en cache : c'est très rapide). */
export class MemoryBm25Index implements LexicalIndex {
  private readonly cache = new Map<string, { bm25: Bm25; docs: Passage[] }>();

  constructor(private readonly passages: Passage[]) {}

  private forScope(scope: Scope) {
    const key = scope && scope.length ? [...scope].sort().join("|") : "*";
    let entry = this.cache.get(key);
    if (!entry) {
      const wanted = scope && scope.length ? new Set(scope) : null;
      let docs = wanted ? this.passages.filter((p) => wanted.has(p.metadata.source)) : this.passages;
      if (!docs.length) docs = this.passages;
      entry = { bm25: new Bm25(docs.map((d) => d.content)), docs };
      this.cache.set(key, entry);
    }
    return entry;
  }

  async searchLexical(query: string, k: number, scope: Scope): Promise<Passage[]> {
    const { bm25, docs } = this.forScope(scope);
    return bm25.topN(query, k).map((i) => docs[i]);
  }
}

/** Recherche vectorielle exacte (cosinus) : ~12 000 vecteurs, pas besoin d'ANN. */
export class MemoryVectorIndex implements VectorIndex {
  private readonly norms: Float32Array;

  constructor(
    private readonly passages: Passage[],
    private readonly vectors: Float32Array[],
  ) {
    if (passages.length !== vectors.length) {
      throw new Error(`Index vectoriel incohérent : ${passages.length} passages, ${vectors.length} vecteurs`);
    }
    this.norms = new Float32Array(vectors.map((v) => Math.sqrt(dot(v, v)) || 1));
  }

  async searchVector(vector: number[], k: number, scope: Scope): Promise<Passage[]> {
    const q = Float32Array.from(vector);
    const qNorm = Math.sqrt(dot(q, q)) || 1;
    const wanted = scope && scope.length ? new Set(scope) : null;
    const scored: { i: number; s: number }[] = [];
    for (let i = 0; i < this.vectors.length; i++) {
      if (wanted && !wanted.has(this.passages[i].metadata.source)) continue;
      scored.push({ i, s: dot(q, this.vectors[i]) / (qNorm * this.norms[i]) });
    }
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, k).map(({ i }) => this.passages[i]);
  }
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * BM25 Okapi, port fidèle de `rank_bm25.BM25Okapi` (k1 = 1,5, b = 0,75, epsilon = 0,25),
 * que le BM25Retriever LangChain utilise côté Python.
 *
 * Même tokenisation que LangChain par défaut : découpage sur les espaces, sans mise en
 * minuscules. C'est volontaire : on veut mesurer le changement de stack, pas celui du
 * tokenizer. L'adaptateur OpenSearch, lui, utilise l'analyseur standard d'OpenSearch.
 */
export type Tokenizer = (text: string) => string[];

export const whitespaceTokenizer: Tokenizer = (text) => text.split(/\s+/).filter(Boolean);

export class Bm25 {
  private readonly docFreqs: Map<string, number>[] = [];
  private readonly docLens: number[] = [];
  private readonly idf = new Map<string, number>();
  private readonly avgdl: number;

  constructor(
    corpus: string[],
    private readonly tokenize: Tokenizer = whitespaceTokenizer,
    private readonly k1 = 1.5,
    private readonly b = 0.75,
    epsilon = 0.25,
  ) {
    const nd = new Map<string, number>();
    let totalLen = 0;
    for (const text of corpus) {
      const tokens = this.tokenize(text);
      this.docLens.push(tokens.length);
      totalLen += tokens.length;
      const freqs = new Map<string, number>();
      for (const t of tokens) freqs.set(t, (freqs.get(t) ?? 0) + 1);
      this.docFreqs.push(freqs);
      for (const t of freqs.keys()) nd.set(t, (nd.get(t) ?? 0) + 1);
    }
    const n = corpus.length;
    this.avgdl = n ? totalLen / n : 0;

    // idf = log(N - n + 0.5) - log(n + 0.5) ; les idf négatifs (termes présents dans plus
    // de la moitié des documents) sont remplacés par epsilon * idf moyen, comme rank_bm25.
    let idfSum = 0;
    const negatives: string[] = [];
    for (const [word, freq] of nd) {
      const idf = Math.log(n - freq + 0.5) - Math.log(freq + 0.5);
      this.idf.set(word, idf);
      idfSum += idf;
      if (idf < 0) negatives.push(word);
    }
    const eps = epsilon * (nd.size ? idfSum / nd.size : 0);
    for (const word of negatives) this.idf.set(word, eps);
  }

  scores(query: string): Float64Array {
    const out = new Float64Array(this.docFreqs.length);
    const terms = this.tokenize(query);
    for (let i = 0; i < this.docFreqs.length; i++) {
      const freqs = this.docFreqs[i];
      const norm = this.k1 * (1 - this.b + (this.b * this.docLens[i]) / (this.avgdl || 1));
      let s = 0;
      for (const t of terms) {
        const tf = freqs.get(t);
        if (!tf) continue;
        s += (this.idf.get(t) ?? 0) * ((tf * (this.k1 + 1)) / (tf + norm));
      }
      out[i] = s;
    }
    return out;
  }

  /** Indices des `n` meilleurs documents, score décroissant. */
  topN(query: string, n: number): number[] {
    const s = this.scores(query);
    return Array.from(s.keys())
      .sort((a, b) => s[b] - s[a])
      .slice(0, n);
  }
}

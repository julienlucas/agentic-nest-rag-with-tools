/**
 * Contexte numéroté : chaque passage transmis au modèle reçoit un [n] qu'il doit citer, et le
 * frontend affiche la source derrière chaque marqueur. Les passages initiaux gardent leurs
 * numéros ; les outils ne font qu'AJOUTER après eux.
 */
import { contentKey, locator, passageLabel, squash, type Passage } from "../core/passage";

export type Citation = {
  n: number;
  source: string;
  page: number | null;
  locator: string;
  excerpt: string;
};

export function buildNumberedContext(passages: Passage[]): { context: string; citations: Citation[] } {
  const parts: string[] = [];
  const citations: Citation[] = [];
  passages.forEach((p, i) => {
    const n = i + 1;
    const loc = locator(p);
    parts.push(`[${n}] (${loc})\n${p.content}`);
    citations.push({
      n,
      source: passageLabel(p),
      page: p.metadata.page !== null && p.metadata.page !== undefined ? p.metadata.page + 1 : null,
      locator: loc,
      excerpt: squash(p.content).slice(0, 280),
    });
  });
  return { context: parts.join("\n\n"), citations };
}

/** Contexte qui grandit pendant la boucle d'outils, sans jamais renuméroter. */
export class GrowingContext {
  readonly passages: Passage[];
  private readonly index = new Map<string, number>();

  constructor(initial: Passage[]) {
    this.passages = [...initial];
    this.passages.forEach((p, i) => this.index.set(contentKey(p.content), i + 1));
  }

  /** Ajoute les passages inconnus ; renvoie le numéro [n] de chacun. */
  add(docs: Passage[]): number[] {
    return docs.map((d) => {
      const key = contentKey(d.content);
      let n = this.index.get(key);
      if (n === undefined) {
        this.passages.push(d);
        n = this.passages.length;
        this.index.set(key, n);
      }
      return n;
    });
  }
}

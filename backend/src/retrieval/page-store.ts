/**
 * Accès page par page aux documents OCRisés : les outils grep / read_page de l'agent.
 *
 * Le retrieval par chunks ne montre jamais une page entière au modèle : un tableau de 1 200
 * caractères y est coupé, un signe négatif se perd. Le PageStore garde le markdown de chaque
 * page tel que l'OCR l'a produit. Port de page_store.py.
 *
 * Les pages sont indexées à partir de 0 en interne ; les outils affichent et reçoivent des
 * numéros à partir de 1.
 */
import { docLabel, type Passage } from "../core/passage";

export type GrepHit = { doc: string; page: number; line: string };
export type GrepResult = { hits: GrepHit[]; total: number; error?: string; documents?: string[] };

function compilePattern(pattern: string): RegExp {
  try {
    return new RegExp(pattern, "i");
  } catch {
    // Regex invalide : repli sur une recherche littérale.
    return new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
  }
}

export class PageStore {
  private readonly pages = new Map<string, string[]>();
  private readonly sources = new Map<string, string>();

  constructor(pagesBySource: Record<string, string[]>) {
    for (const [source, pages] of Object.entries(pagesBySource)) {
      const label = docLabel(source);
      this.pages.set(label, [...(pages ?? [])]);
      this.sources.set(label, source);
    }
  }

  documents(): string[] {
    return [...this.pages.keys()];
  }

  pageCount(doc: string): number {
    return this.pages.get(doc)?.length ?? 0;
  }

  sourceOf(doc: string): string | undefined {
    return this.sources.get(doc);
  }

  /** Accepte un libellé exact, un chemin, ou une sous-chaîne non ambiguë (insensible à la casse). */
  resolve(doc: string | null | undefined): string | null {
    if (!doc) return null;
    if (this.pages.has(doc)) return doc;
    const label = docLabel(doc);
    if (this.pages.has(label)) return label;
    const low = label.toLowerCase();
    const matches = this.documents().filter((d) => d.toLowerCase().includes(low) || low.includes(d.toLowerCase()));
    return matches.length === 1 ? matches[0] : null;
  }

  grep(pattern: string, doc?: string | null, maxHits = 20, contextChars = 160): GrepResult {
    const rx = compilePattern(pattern);
    const target = doc ? this.resolve(doc) : null;
    if (doc && !target) return { hits: [], total: 0, error: `document inconnu: ${doc}`, documents: this.documents() };
    const docs = target ? [target] : this.documents();

    const hits: GrepHit[] = [];
    let total = 0;
    for (const d of docs) {
      this.pages.get(d)!.forEach((text, pageIdx) => {
        for (const line of text.split("\n")) {
          if (!rx.test(line)) continue;
          total++;
          if (hits.length >= maxHits) continue;
          let snippet = line.split(/\s+/).filter(Boolean).join(" ");
          if (snippet.length > contextChars) {
            // Centre l'extrait sur la première occurrence.
            const m = rx.exec(snippet);
            const start = Math.max(0, (m ? m.index : 0) - Math.floor(contextChars / 3));
            snippet = (start ? "…" : "") + snippet.slice(start, start + contextChars) + "…";
          }
          hits.push({ doc: d, page: pageIdx, line: snippet });
        }
      });
    }
    return { hits, total };
  }

  /** Le markdown d'une page (index 0), tronqué au-delà de maxChars. */
  readPage(doc: string, page: number, maxChars = 8000): { doc: string; page: number; text: string } | { error: string } {
    const target = this.resolve(doc);
    if (!target) return { error: `document inconnu: ${doc}. Documents : ${this.documents().join(", ")}` };
    const pages = this.pages.get(target)!;
    if (!Number.isInteger(page)) return { error: `page invalide: ${page}` };
    if (page < 0 || page >= pages.length) return { error: `page ${page + 1} hors limites (1-${pages.length})` };
    const text = pages[page];
    return { doc: target, page, text: text.length > maxChars ? text.slice(0, maxChars) + "\n… [page tronquée]" : text };
  }

  /**
   * Plusieurs pages consécutives (index 0, bornes incluses) en passages, pour les tableaux
   * à cheval sur deux pages. Plafonné à maxPages, chaque page tronquée à 8 000 caractères
   * (comme page_documents côté Python).
   */
  pagePassages(doc: string, start: number, end: number, maxPages = 3, maxCharsPerPage = 8000): Passage[] {
    const target = this.resolve(doc);
    if (!target) return [];
    const n = this.pages.get(target)!.length;
    if (end < start) [start, end] = [end, start];
    start = Math.max(0, start);
    end = Math.min(n - 1, end, start + maxPages - 1);
    if (start >= n) return [];
    const out: Passage[] = [];
    for (let p = start; p <= end; p++) {
      const r = this.readPage(target, p, maxCharsPerPage);
      if ("error" in r) continue;
      out.push({
        content: r.text,
        metadata: { source: this.sources.get(target)!, docName: target, page: p, origin: "read_page" },
      });
    }
    return out;
  }
}

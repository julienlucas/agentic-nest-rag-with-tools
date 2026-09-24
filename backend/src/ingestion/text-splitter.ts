/**
 * Découpage récursif par séparateurs, sur le modèle du RecursiveCharacterTextSplitter de
 * LangChain : on coupe sur le séparateur le plus « gros » présent, on regroupe les morceaux
 * jusqu'à la taille cible avec chevauchement, et on redescend d'un séparateur si un morceau
 * reste trop grand.
 */
const DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""];

export class RecursiveTextSplitter {
  constructor(
    private readonly chunkSize: number,
    private readonly overlap: number,
    private readonly separators: string[] = DEFAULT_SEPARATORS,
  ) {
    if (overlap >= chunkSize) throw new Error("Le chevauchement doit être inférieur à la taille des chunks");
  }

  split(text: string): string[] {
    return this.splitWith(text, this.separators).map((c) => c.trim()).filter(Boolean);
  }

  private splitWith(text: string, separators: string[]): string[] {
    let sep = separators[separators.length - 1];
    let rest: string[] = [];
    for (let i = 0; i < separators.length; i++) {
      if (separators[i] === "" || text.includes(separators[i])) {
        sep = separators[i];
        rest = separators.slice(i + 1);
        break;
      }
    }
    // Le séparateur est conservé en tête du morceau suivant (keep_separator de LangChain).
    const pieces = sep === "" ? [...text] : splitKeep(text, sep);

    const out: string[] = [];
    let good: string[] = [];
    for (const piece of pieces) {
      if (piece.length < this.chunkSize) {
        good.push(piece);
        continue;
      }
      if (good.length) {
        out.push(...this.merge(good));
        good = [];
      }
      if (!rest.length) out.push(piece);
      else out.push(...this.splitWith(piece, rest));
    }
    if (good.length) out.push(...this.merge(good));
    return out;
  }

  private merge(pieces: string[]): string[] {
    const docs: string[] = [];
    let current: string[] = [];
    let total = 0;
    for (const piece of pieces) {
      if (total + piece.length > this.chunkSize && current.length) {
        docs.push(current.join(""));
        // Chevauchement : on garde la fin du chunk précédent.
        while (total > this.overlap || (total + piece.length > this.chunkSize && total > 0)) {
          total -= current[0].length;
          current = current.slice(1);
        }
      }
      current.push(piece);
      total += piece.length;
    }
    if (current.length) docs.push(current.join(""));
    return docs;
  }
}

function splitKeep(text: string, sep: string): string[] {
  const parts = text.split(sep);
  return parts.map((p, i) => (i === 0 ? p : sep + p)).filter((p) => p.length);
}

/**
 * Un passage : un chunk (enfant ou parent) ou une page entière lue par un outil.
 * Équivalent du `Document` LangChain du projet Python.
 */
import { createHash } from "node:crypto";
import * as path from "node:path";

export type PassageMetadata = {
  /** Fichier d'origine ou identifiant du document (sert au périmètre du routage). */
  source: string;
  /** Libellé lisible du document (FinanceBench : "AMD_2022_10K"). */
  docName?: string | null;
  /** Page d'origine, indexée à partir de 0. */
  page?: number | null;
  parentId?: string | null;
  parentContent?: string | null;
  matchedChildren?: number;
  rerankScore?: number;
  /** "read_page" quand le passage est une page lue par l'outil. */
  origin?: "read_page";
};

export type Passage = {
  content: string;
  metadata: PassageMetadata;
};

export function contentKey(content: string): string {
  return createHash("md5").update(content).digest("hex");
}

export function sha256(data: string | Buffer): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Nom lisible d'une source : basename sans extension. */
export function docLabel(source: string): string {
  const base = path.basename(String(source));
  const ext = path.extname(base);
  return ext ? base.slice(0, -ext.length) : base;
}

export function passageLabel(p: Passage): string {
  return docLabel(p.metadata.docName || p.metadata.source || "document");
}

/** "AMD_2022_10K, p. 12" : la page affichée est 1-indexée. */
export function locator(p: Passage): string {
  const name = passageLabel(p);
  const page = p.metadata.page;
  return page !== null && page !== undefined ? `${name}, p. ${page + 1}` : name;
}

export function squash(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(" ");
}

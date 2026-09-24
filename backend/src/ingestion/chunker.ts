/**
 * Chunking parent-child des documents uploadés : parents de 1 200 caractères (ce que le modèle
 * lit), enfants de 400 (ce qui est indexé et matché). Les chunks sont ensuite rattachés à leur
 * page d'origine, pour les citations et les outils.
 *
 * Écart assumé avec le Python : les parents y sont découpés sémantiquement (embeddings des
 * paragraphes). Ici le découpage est récursif. FinanceBench n'est pas concerné : l'éval
 * réutilise les chunks exportés du projet Python.
 */
import { randomUUID } from "node:crypto";
import { settings } from "../config/settings";
import type { Passage } from "../core/passage";
import { RecursiveTextSplitter } from "./text-splitter";

const SEPARATOR = "\n\n";

export function chunkPages(pages: string[], source: string): Passage[] {
  const markdown = pages.join(SEPARATOR);
  const spans: { start: number; page: number }[] = [];
  let cursor = 0;
  pages.forEach((p, i) => {
    spans.push({ start: cursor, page: i });
    cursor += p.length + SEPARATOR.length;
  });

  const parentSplitter = new RecursiveTextSplitter(settings.PARENT_CHUNK_SIZE, 100);
  const childSplitter = new RecursiveTextSplitter(settings.CHILD_CHUNK_SIZE, settings.CHILD_OVERLAP);

  const children: Passage[] = [];
  for (const parent of parentSplitter.split(markdown)) {
    const parentId = randomUUID();
    for (const child of childSplitter.split(parent)) {
      children.push({ content: child, metadata: { source, parentId, parentContent: parent } });
    }
  }
  attachPages(children, markdown, spans);
  return children;
}

/**
 * Rattache chaque chunk à sa page (index 0) en retrouvant ses premiers caractères dans le
 * markdown assemblé ; le curseur avance de façon monotone. Port de _attach_pages.
 */
export function attachPages(chunks: Passage[], markdown: string, spans: { start: number; page: number }[]) {
  if (!spans.length) return;
  let cursor = 0;
  for (const chunk of chunks) {
    const probe = chunk.content.slice(0, 120).trim();
    if (!probe) continue;
    let pos = markdown.indexOf(probe, cursor);
    if (pos === -1) pos = markdown.indexOf(probe);
    if (pos === -1) continue;
    cursor = pos;
    let page = 0;
    for (const s of spans) {
      if (s.start <= pos) page = s.page;
      else break;
    }
    chunk.metadata.page = page;
  }
}

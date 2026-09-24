/**
 * Corpus FinanceBench exporté du projet Python (scripts/export_from_python.py) : mêmes pages
 * OCR, mêmes chunks. Seule la stack change.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { env } from "../../config/settings";
import type { Passage } from "../../core/passage";
import { buildRetriever, type CorpusPart } from "../../retrieval/index-builder";
import { PageStore } from "../../retrieval/page-store";

export const FB_DIR = path.join(env.DATA_DIR, "financebench");

export type Example = {
  id: string;
  doc_name: string;
  company?: string;
  question: string;
  expected_answer: string;
  justification?: string;
  answer_keywords: string[];
  gold_passages: string[];
  gold_pages: [string, number][];
  question_type?: string;
  question_reasoning?: string;
};

function requireFile(file: string): string {
  if (!fs.existsSync(file)) {
    throw new Error(`${file} introuvable. Lancer d'abord : ../agentic-rag/.venv/bin/python scripts/export_from_python.py`);
  }
  return file;
}

export function loadDataset(file = path.join(FB_DIR, "dataset.jsonl")): Example[] {
  return fs
    .readFileSync(requireFile(file), "utf-8")
    .split("\n")
    .filter((l) => l.trim() && !l.startsWith("#"))
    .map((l) => JSON.parse(l) as Example);
}

export function loadChunks(doc: string): Passage[] {
  return fs
    .readFileSync(requireFile(path.join(FB_DIR, `${doc}.chunks.jsonl`)), "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Passage);
}

export function loadPages(doc: string): string[] {
  return JSON.parse(fs.readFileSync(requireFile(path.join(FB_DIR, `${doc}.pages.json`)), "utf-8")) as string[];
}

/** Index combiné des documents (comme l'éval Python) + PageStore pour grep / read_page. */
export async function loadIndex(docs: string[]) {
  const parts: CorpusPart[] = docs.map((d) => ({ key: `financebench-${d}`, chunks: loadChunks(d) }));
  const retriever = await buildRetriever(parts, `financebench-${docs.length}docs`);
  const pageStore = new PageStore(Object.fromEntries(docs.map((d) => [d, loadPages(d)])));
  return { retriever, pageStore, chunkCount: parts.reduce((s, p) => s + p.chunks.length, 0) };
}

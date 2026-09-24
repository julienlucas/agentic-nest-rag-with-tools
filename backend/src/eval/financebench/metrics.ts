/**
 * Métriques de retrieval et de réponse, même calcul que evaluation/metrics.py et
 * run_financebench_eval.py (page_hit@k sur les pages annotées, tolérance de pagination 1).
 */
import type { Passage } from "../../core/passage";
import type { Example } from "./corpus";

export function normalize(text: string): string {
  return (text || "")
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const tokens = (t: string) => normalize(t).split(" ").filter(Boolean);

export function f1Score(pred: string, ref: string): number {
  const p = tokens(pred);
  const r = tokens(ref);
  if (!p.length || !r.length) return 0;
  const counts = new Map<string, number>();
  for (const t of p) counts.set(t, (counts.get(t) ?? 0) + 1);
  let overlap = 0;
  for (const t of r) {
    const c = counts.get(t) ?? 0;
    if (c > 0) {
      overlap++;
      counts.set(t, c - 1);
    }
  }
  const precision = overlap / p.length;
  const recall = overlap / r.length;
  return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
}

export function contextHit(docs: Passage[], expected: string, keywords: string[]): boolean {
  const ctx = normalize(docs.map((d) => d.content).join("\n\n"));
  if (expected && ctx.includes(normalize(expected))) return true;
  return keywords.some((k) => k && ctx.includes(normalize(k)));
}

function tokenOverlap(gold: string, doc: string): number {
  const g = tokens(gold);
  if (!g.length) return 0;
  const set = new Set(tokens(doc));
  return g.filter((t) => set.has(t)).length / g.length;
}

export function textFlags(docs: Passage[], goldPassages: string[], threshold = 0.6): number[] {
  if (!goldPassages.length) return [];
  const gold = goldPassages.filter(Boolean).map(normalize);
  return docs.map((d) => {
    const content = normalize(d.content);
    if (gold.some((g) => content.includes(g))) return 1;
    return gold.some((g) => tokenOverlap(g, content) >= threshold) ? 1 : 0;
  });
}

export function pageFlags(docs: Passage[], goldPages: [string, number][], tolerance = 1): number[] {
  if (!goldPages.length) return [];
  return docs.map((d) => {
    const name = String(d.metadata.docName || d.metadata.source || "");
    const page = d.metadata.page;
    if (page === null || page === undefined) return 0;
    return goldPages.some(([gd, gp]) => name === gd && Math.abs(page - gp) <= tolerance) ? 1 : 0;
  });
}

export const recallAtK = (f: number[], k: number) => {
  if (!f.length) return null;
  const total = f.reduce((a, b) => a + b, 0);
  return total === 0 ? 0 : Math.min(f.slice(0, k).reduce((a, b) => a + b, 0) / total, 1);
};
export const precisionAtK = (f: number[], k: number) => (!f.length || k <= 0 ? null : f.slice(0, k).reduce((a, b) => a + b, 0) / k);
export const mrrAtK = (f: number[], k: number) => {
  if (!f.length) return null;
  const i = f.slice(0, k).indexOf(1);
  return i === -1 ? 0 : 1 / (i + 1);
};
export const ndcgAtK = (f: number[], k: number) => {
  if (!f.length) return null;
  const dcg = (xs: number[]) => xs.slice(0, k).reduce((s, rel, i) => s + (rel ? 1 / Math.log2(i + 2) : 0), 0);
  const ideal = dcg([...f].sort((a, b) => b - a));
  return ideal === 0 ? 0 : dcg(f) / ideal;
};
export const pageHitAtK = (f: number[], k: number) => (!f.length ? null : f.slice(0, k).some(Boolean) ? 1 : 0);

export function pageRecallAtK(docs: Passage[], goldPages: [string, number][], k: number, tolerance = 1): number | null {
  if (!goldPages.length) return null;
  const covered = new Set<string>();
  for (const d of docs.slice(0, k)) {
    const name = String(d.metadata.docName || d.metadata.source || "");
    const page = d.metadata.page;
    if (page === null || page === undefined) continue;
    for (const [gd, gp] of goldPages) if (name === gd && Math.abs(page - gp) <= tolerance) covered.add(`${gd}:${gp}`);
  }
  const uniqueGold = new Set(goldPages.map(([d, p]) => `${d}:${p}`));
  return covered.size / uniqueGold.size;
}

export function retrievalMetrics(docs: Passage[], ex: Example, kValues: number[], tolerance: number) {
  const tf = textFlags(docs, ex.gold_passages);
  const pf = pageFlags(docs, ex.gold_pages, tolerance);
  const m: Record<string, unknown> = { n_docs: docs.length };
  for (const k of kValues) {
    m[`recall@${k}`] = recallAtK(tf, k);
    m[`precision@${k}`] = precisionAtK(tf, k);
    m[`mrr@${k}`] = mrrAtK(tf, k);
    m[`ndcg@${k}`] = ndcgAtK(tf, k);
    m[`page_hit@${k}`] = pageHitAtK(pf, k);
    m[`page_recall@${k}`] = pageRecallAtK(docs, ex.gold_pages, k, tolerance);
    m[`page_precision@${k}`] = precisionAtK(pf, k);
  }
  m.retrieved = docs.slice(0, 20).map((d) => [String(d.metadata.docName || d.metadata.source), d.metadata.page]);
  const rank = pf.indexOf(1);
  m.gold_rank = rank === -1 ? null : rank + 1;
  return m;
}

/** Intervalle de Wilson à 95 % : sur 26 questions, ~30 points de large. */
export function wilson(successes: number, n: number, z = 1.96): [number, number] | null {
  if (n <= 0) return null;
  const p = successes / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  const r = (x: number) => Math.round(x * 10000) / 10000;
  return [r(Math.max(0, center - half)), r(Math.min(1, center + half))];
}

import { describe, expect, it } from "vitest";
import { aggregateVerdicts, detectCanned, parseVerdict } from "../src/eval/financebench/judge";
import { f1Score, pageFlags, retrievalMetrics, wilson } from "../src/eval/financebench/metrics";
import { checkSentinel, compareRetrieval } from "../src/eval/financebench/regression";
import { passage } from "./helpers";

describe("Juge FinanceBench", () => {
  it("parse un verdict en gras", () => {
    expect(parseVerdict("**VERDICT:** CORRECT\n**FAITHFULNESS:** 4\n**RAISON:** ok.")).toMatchObject({ verdict: "CORRECT", faithfulness: 4 });
  });

  it("un refus figé doit être TOUTE la réponse", () => {
    expect(detectCanned("Cette information n'est pas disponible dans le document.")).toBe("REFUSAL");
    const long = "Le chiffre est 12 [1]. " + "x".repeat(300) + " Cette information n'est pas disponible dans le document.";
    expect(detectCanned(long)).toBeNull();
    expect(detectCanned("")).toBe("ERROR");
  });

  it("agrège hors erreurs techniques, avec IC95", () => {
    const agg = aggregateVerdicts([
      { verdict: "CORRECT", faithfulness: 5, reason: "" },
      { verdict: "INCORRECT", faithfulness: 2, reason: "" },
      { verdict: "ERROR", faithfulness: 0, reason: "" },
    ]);
    expect(agg).toMatchObject({ count: 2, errors: 1, accuracy: 0.5 });
  });
});

describe("Métriques", () => {
  it("Wilson : mêmes bornes que llm_judge.wilson_interval (20/24)", () => {
    const [lo, hi] = wilson(20, 24)!;
    expect(lo).toBe(0.6415);
    expect(hi).toBe(0.9332);
  });

  it("F1 normalisé : mêmes valeurs que metrics._f1_score", () => {
    expect(f1Score("$1,577.00 million", "1577.00")).toBeCloseTo(1 / 3, 10);
    expect(f1Score("the answer is 42", "42")).toBeCloseTo(0.4, 5);
  });

  it("page de preuve avec tolérance de pagination", () => {
    const docs = [passage("a", "AMD_2022_10K", 10), passage("b", "AMD_2022_10K", 30)];
    expect(pageFlags(docs, [["AMD_2022_10K", 31]])).toEqual([0, 1]);
    const m = retrievalMetrics(docs, { gold_pages: [["AMD_2022_10K", 31]], gold_passages: [] } as never, [1, 5], 1);
    expect(m.gold_rank).toBe(2);
    expect(m["page_hit@1"]).toBe(0);
  });
});

describe("Garde-fou de régression", () => {
  it("détecte une preuve sortie du top-10, même si la moyenne ne bouge pas", () => {
    const base = { q1: { gold_rank: 2, "page_hit@10": 1 }, q2: { gold_rank: 15, "page_hit@10": 0 } };
    const cur = { q1: { gold_rank: 14, "page_hit@10": 0 }, q2: { gold_rank: 3, "page_hit@10": 1 } };
    const r = compareRetrieval(base, cur);
    expect(r.ok).toBe(false);
    expect(r.lost).toHaveLength(1);
    expect(r.gained).toHaveLength(1);
  });

  it("sentinelle : verdict et preuve vue", () => {
    expect(checkSentinel({ id: "x", mode: "agentic", verdict: "CORRECT", evidence_seen: true }, { evidence_seen: true })).toEqual([]);
    expect(checkSentinel({ id: "x", mode: "agentic", verdict: "INCORRECT", evidence_seen: false }, { evidence_seen: true })).toHaveLength(2);
  });
});

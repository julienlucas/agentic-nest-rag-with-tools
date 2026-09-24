import { describe, expect, it } from "vitest";
import { Bm25 } from "../src/retrieval/bm25";
import { weightedRrf } from "../src/retrieval/fusion";
import { MemoryBm25Index, MemoryVectorIndex } from "../src/retrieval/indexes";
import { toParents } from "../src/retrieval/parent-child";
import { DocumentRouter, identityTokens } from "../src/retrieval/document-router";
import { passage } from "./helpers";

describe("BM25 (port de rank_bm25.BM25Okapi)", () => {
  // Valeurs de référence calculées avec rank_bm25 dans le venv du projet Python.
  const corpus = ["the quick brown fox", "the lazy dog sleeps", "quick quick fox jumps over the dog", "Revenue grew 12% in FY2022", "the the the"];
  const bm = new Bm25(corpus);
  const round = (xs: Float64Array) => Array.from(xs, (x) => Math.round(x * 1e6) / 1e6);

  it("donne les mêmes scores que Python", () => {
    expect(round(bm.scores("quick fox"))).toEqual([0.714906, 0, 0.684137, 0, 0]);
    expect(round(bm.scores("the dog"))).toEqual([0.206722, 0.564175, 0.430084, 0, 0.3552]);
    expect(round(bm.scores("Revenue FY2022"))).toEqual([0, 0, 0, 2.114484, 0]);
    expect(round(bm.scores("absent"))).toEqual([0, 0, 0, 0, 0]);
  });

  it("classe par score décroissant", () => {
    expect(bm.topN("Revenue FY2022", 1)).toEqual([3]);
  });
});

describe("Fusion RRF pondérée (EnsembleRetriever)", () => {
  it("additionne poids / (rang + 60) et dédoublonne sur le contenu", () => {
    const a = passage("A");
    const b = passage("B");
    const c = passage("C");
    const fused = weightedRrf([[a, b], [b, c]], [0.5, 0.5]);
    expect(fused.map((p) => p.content)).toEqual(["B", "A", "C"]);
  });

  it("garde l'ordre d'apparition à score égal", () => {
    const fused = weightedRrf([[passage("X")], [passage("Y")]], [0.5, 0.5]);
    expect(fused.map((p) => p.content)).toEqual(["X", "Y"]);
  });
});

describe("Parent-child", () => {
  it("remplace les enfants par leur parent, triés par nombre d'enfants retrouvés", () => {
    const kids = [
      passage("c1", "D", 0, { parentId: "p1", parentContent: "PARENT 1" }),
      passage("c2", "D", 0, { parentId: "p2", parentContent: "PARENT 2" }),
      passage("c3", "D", 0, { parentId: "p2", parentContent: "PARENT 2" }),
      passage("orphelin", "D", 1),
    ];
    const out = toParents(kids);
    expect(out.map((p) => p.content)).toEqual(["PARENT 2", "PARENT 1", "orphelin"]);
    expect(out[0].metadata.matchedChildren).toBe(2);
  });
});

describe("Index en mémoire", () => {
  const docs = [passage("alpha beta", "A"), passage("beta gamma", "B"), passage("gamma delta", "B")];

  it("restreint BM25 au périmètre", async () => {
    const idx = new MemoryBm25Index(docs);
    const res = await idx.searchLexical("beta", 5, ["A"]);
    expect(res.every((p) => p.metadata.source === "A")).toBe(true);
  });

  it("classe par cosinus et filtre par source", async () => {
    const vecs = [new Float32Array([1, 0]), new Float32Array([0.7, 0.7]), new Float32Array([0, 1])];
    const idx = new MemoryVectorIndex(docs, vecs);
    expect((await idx.searchVector([0, 1], 3, null)).map((p) => p.content)[0]).toBe("gamma delta");
    expect((await idx.searchVector([0, 1], 3, ["A"])).map((p) => p.content)).toEqual(["alpha beta"]);
  });
});

describe("Routage par document", () => {
  it("extrait les tokens d'identité du nom de fichier", () => {
    expect([...identityTokens("AMERICANEXPRESS_2022_10K.pdf")]).toEqual(["americanexpress"]);
  });

  it("route sur le nom de l'entreprise sans appel LLM", async () => {
    const router = new DocumentRouter(["AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K"], null);
    expect(await router.route("What is American Express's largest liability?")).toEqual(["AMERICANEXPRESS_2022_10K"]);
    expect(await router.route("Quelle entreprise a le plus de dette ?")).toBeNull();
  });
});

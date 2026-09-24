import { describe, expect, it } from "vitest";
import { ResearchAgent } from "../src/agent/research-agent";
import { GrowingContext, buildNumberedContext } from "../src/agent/numbered-context";
import { parseRelevanceLabel } from "../src/agent/relevance-checker";
import { buildVerificationReport } from "../src/agent/verification-report";
import { PageStore } from "../src/retrieval/page-store";
import type { Retriever } from "../src/retrieval/retriever";
import { passage, scriptedModel, text, toolCall } from "./helpers";

const pageStore = new PageStore({ AMD_2022_10K: ["cover", "Total current liabilities 6,369\nCash and cash equivalents 4,835"] });
const fakeRetriever = {
  invokeWithScope: async (query: string) => [passage(`résultat pour ${query}`, "AMD_2022_10K", 1)],
} as unknown as Retriever;

describe("Contexte numéroté", () => {
  it("numérote à partir de 1 et affiche les pages 1-indexées", () => {
    const { context, citations } = buildNumberedContext([passage("texte", "AMD_2022_10K", 4)]);
    expect(context).toContain("[1] (AMD_2022_10K, p. 5)");
    expect(citations[0]).toMatchObject({ n: 1, page: 5, source: "AMD_2022_10K" });
  });

  it("n'ajoute qu'après les passages initiaux et ne renumérote jamais", () => {
    const ctx = new GrowingContext([passage("a"), passage("b")]);
    expect(ctx.add([passage("b"), passage("c")])).toEqual([2, 3]);
    expect(ctx.passages.map((p) => p.content)).toEqual(["a", "b", "c"]);
  });
});

describe("Label de pertinence", () => {
  it("tolère la mise en forme", () => {
    expect(parseRelevanceLabel("**PARTIAL**")).toBe("PARTIAL");
    expect(parseRelevanceLabel("Label: CAN ANSWER.")).toBe("CAN_ANSWER");
    expect(parseRelevanceLabel("je ne sais pas")).toBeNull();
  });
});

describe("Agent à outils (modèle factice)", () => {
  it("lit la page, la numérote après les initiaux, et répond en la citant", async () => {
    const model = scriptedModel([
      toolCall("read_page", { doc: "AMD_2022_10K", page: 2 }),
      text("Les passifs courants s'élèvent à 6 369 M$ [2]."),
    ]);
    const agent = new ResearchAgent(model);
    const r = await agent.generateWithTools({
      question: "Passifs courants AMD ?",
      passages: [passage("extrait initial", "AMD_2022_10K", 0)],
      retriever: fakeRetriever,
      pageStore,
      scope: null,
    });
    expect(r.toolCalls).toBe(1);
    expect(r.calls).toEqual(["read_page: AMD_2022_10K p. 2"]);
    expect(r.passages).toHaveLength(2);
    expect(r.passages[1].metadata.origin).toBe("read_page");
    expect(r.citations[1]).toMatchObject({ n: 2, page: 2 });
    expect(r.answer).toContain("[2]");
  });

  it("n'exécute pas les appels au-delà du budget ; le modèle conclut dans la conversation", async () => {
    const calls = Array.from({ length: 3 }, (_, i) => toolCall("grep", { pattern: `terme${i}` }, `c${i}`));
    const model = scriptedModel([...calls, text("Conclusion avec ce que j'ai.")]);
    const r = await new ResearchAgent(model).generateWithTools({
      question: "q",
      passages: [passage("init")],
      retriever: fakeRetriever,
      pageStore,
      scope: null,
      budget: 2,
    });
    expect(r.toolCalls).toBe(2); // le 3e appel a reçu « budget épuisé » sans s'exécuter
    expect(r.answer).toBe("Conclusion avec ce que j'ai.");
  });

  it("si le modèle insiste au-delà de la limite d'étapes, réponse forcée sans outils", async () => {
    // budget 2 -> 4 étapes au plus, toutes des appels d'outils ; puis l'appel forcé.
    const calls = Array.from({ length: 4 }, (_, i) => toolCall("grep", { pattern: `terme${i}` }, `c${i}`));
    const model = scriptedModel([...calls, text("Réponse finale forcée.")]);
    const r = await new ResearchAgent(model).generateWithTools({
      question: "q",
      passages: [passage("init")],
      retriever: fakeRetriever,
      pageStore,
      scope: null,
      budget: 2,
    });
    expect(r.toolCalls).toBe(2);
    expect(r.answer).toBe("Réponse finale forcée.");
    // L'appel forcé reçoit le journal des outils déjà appelés, sans outils déclarés.
    const last = model.doGenerateCalls[model.doGenerateCalls.length - 1];
    expect(last.tools ?? []).toHaveLength(0);
    expect(JSON.stringify(last.prompt)).toContain("Journal des outils");
  });

  it("rapport de vérification : outils, confiance du reranker et pages", () => {
    const report = buildVerificationReport({
      relevance: "PARTIAL",
      passages: [passage("x", "AMD_2022_10K", 3, { rerankScore: 0.82 })],
      toolCalls: 1,
      calls: ['grep: "goodwill"'],
    });
    expect(report).toContain("**Confiance retrieval (reranker):** 0.82 — élevée");
    expect(report).toContain("déclenchée (1 appel d'outils)");
    expect(report).toContain("AMD_2022_10K (p. 4)");
  });
});

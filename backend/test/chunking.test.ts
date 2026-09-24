import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { chunkPages } from "../src/ingestion/chunker";
import { RecursiveTextSplitter } from "../src/ingestion/text-splitter";

describe("Découpage récursif (propriétés)", () => {
  const splitter = new RecursiveTextSplitter(100, 20);

  it("aucun chunk ne dépasse la taille cible", () => {
    fc.assert(
      fc.property(fc.array(fc.stringMatching(/^[a-z]{1,12}$/), { maxLength: 300 }), (words) => {
        const chunks = splitter.split(words.join(" "));
        return chunks.every((c) => c.length <= 100);
      }),
    );
  });

  it("chaque mot du texte se retrouve dans au moins un chunk", () => {
    fc.assert(
      fc.property(fc.array(fc.stringMatching(/^[a-z]{3,10}$/), { minLength: 1, maxLength: 200 }), (words) => {
        const joined = splitter.split(words.join(" ")).join(" ");
        return words.every((w) => joined.includes(w));
      }),
    );
  });
});

describe("Chunking parent-child", () => {
  it("rattache chaque enfant à son parent et à sa page", () => {
    const pages = ["Page un. ".repeat(100), "Deuxième page, tableau des passifs. ".repeat(60)];
    const chunks = chunkPages(pages, "doc.pdf");
    expect(chunks.length).toBeGreaterThan(4);
    expect(chunks.every((c) => c.metadata.parentId && c.metadata.parentContent?.includes(c.content.slice(0, 30)))).toBe(true);
    expect(chunks[0].metadata.page).toBe(0);
    expect(chunks[chunks.length - 1].metadata.page).toBe(1);
  });
});

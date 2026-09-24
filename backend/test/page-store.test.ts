import { describe, expect, it } from "vitest";
import { PageStore } from "../src/retrieval/page-store";

const store = new PageStore({
  "AMD_2022_10K": ["# Cover\nAdvanced Micro Devices", "Total current liabilities 6,369\nCash 4,835", "Legal Proceedings none"],
  "/tmp/BOEING_2022_10K.pdf": ["Boeing page one"],
});

describe("PageStore", () => {
  it("résout libellés, chemins et sous-chaînes", () => {
    expect(store.resolve("AMD_2022_10K")).toBe("AMD_2022_10K");
    expect(store.resolve("boeing")).toBe("BOEING_2022_10K");
    expect(store.resolve("inconnu")).toBeNull();
  });

  it("grep est exhaustif et insensible à la casse, pages indexées à 0", () => {
    const r = store.grep("total current liabilities", "AMD");
    expect(r.total).toBe(1);
    expect(r.hits[0]).toMatchObject({ doc: "AMD_2022_10K", page: 1 });
    expect(store.grep("goodwill", "AMD").total).toBe(0);
  });

  it("une regex invalide retombe sur une recherche littérale", () => {
    expect(() => store.grep("(unclosed", "AMD")).not.toThrow();
  });

  it("lit plusieurs pages en passages, 3 au plus", () => {
    const pages = store.pagePassages("AMD_2022_10K", 0, 10);
    expect(pages.map((p) => p.metadata.page)).toEqual([0, 1, 2]);
    expect(pages[0].metadata.origin).toBe("read_page");
    expect(store.readPage("AMD_2022_10K", 99)).toHaveProperty("error");
  });
});

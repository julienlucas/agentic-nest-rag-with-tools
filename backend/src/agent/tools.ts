/**
 * Les trois outils façon système de fichiers, fermés sur le retriever, le PageStore et le
 * contexte numéroté d'une question. Descriptions reprises de search_agent.py.
 *
 * - search(query, doc)             : le retrieval hybride + rerank, relancé avec une nouvelle requête.
 * - grep(pattern, doc)             : occurrences littérales page par page, exhaustif.
 * - read_page(doc, page, end_page) : la page entière, ou 2-3 pages pour un tableau à cheval.
 */
import { tool } from "ai";
import { z } from "zod";
import { settings } from "../config/settings";
import { locator, squash, type Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import { isRateLimit } from "../llm/resilience";
import type { Scope } from "../retrieval/indexes";
import type { PageStore } from "../retrieval/page-store";
import type { Retriever } from "../retrieval/retriever";
import type { GrowingContext } from "./numbered-context";

const logger = getLogger("Tools");

export const BUDGET_EXHAUSTED = "Budget d'appels épuisé : conclus avec ce que tu as.";

export type ToolTrace = {
  calls: string[];
  /** Sorties des outils (tronquées), pour une éventuelle réponse finale forcée sans outils. */
  outputs: string[];
  searchResults: Passage[][];
  readPages: Passage[];
};

export function buildTools(opts: {
  retriever: Retriever;
  pageStore: PageStore | null;
  scope: Scope;
  context: GrowingContext;
  trace: ToolTrace;
  budget: number;
}) {
  const { retriever, pageStore, scope, context, trace, budget } = opts;

  /** Faux quand le budget est épuisé : l'appel n'est pas exécuté. */
  const spend = () => trace.calls.length < budget;

  /** Garde une trace lisible de la sortie, puis la renvoie au modèle. */
  const out = (text: string) => {
    trace.outputs.push(`${trace.calls[trace.calls.length - 1]}\n${text.slice(0, 1500)}`);
    return text;
  };

  const scopeFor = (doc?: string | null): Scope => {
    if (doc && pageStore) {
      const label = pageStore.resolve(doc);
      if (label) return [pageStore.sourceOf(label)!];
    }
    return doc ? [doc] : scope;
  };

  const search = tool({
    description:
      "Recherche sémantique + lexicale dans les documents indexés. `query` : une phrase " +
      "courte en langage naturel, dans le vocabulaire des rapports financiers (pas " +
      "d'opérateur booléen). `doc` : restreindre à un document (optionnel).",
    inputSchema: z.object({
      query: z.string().describe("Phrase courte en langage naturel"),
      doc: z.string().nullish().describe("Nom du document (optionnel)"),
    }),
    execute: async ({ query, doc }) => {
      if (!spend()) return BUDGET_EXHAUSTED;
      trace.calls.push(`search: "${query}"` + (doc ? ` [${doc}]` : ""));
      let results: Passage[];
      try {
        results = (await retriever.invokeWithScope(query, scopeFor(doc))).slice(0, settings.SEARCH_TOOL_RESULTS);
      } catch (error) {
        if (isRateLimit(error)) throw error;
        logger.warn(`search a échoué : ${(error as Error).message}`);
        return out("Erreur de recherche, réessayez avec une autre formulation.");
      }
      trace.searchResults.push(results);
      if (!results.length) return out("Aucun résultat.");
      const numbers = context.add(results);
      return out(
        results.map((d, i) => `[${numbers[i]}] (${locator(d)}) ${squash(d.content).slice(0, 400)}`).join("\n"),
      );
    },
  });

  const grep = tool({
    description:
      "Occurrences littérales d'un terme (regex insensible à la casse), page par page. " +
      "Exhaustif : 0 résultat signifie que le terme est absent du document. `doc` : " +
      "restreindre à un document (optionnel, recommandé).",
    inputSchema: z.object({
      pattern: z.string().describe("Terme ou regex"),
      doc: z.string().nullish().describe("Nom du document (optionnel, recommandé)"),
    }),
    execute: async ({ pattern, doc }) => {
      if (!spend()) return BUDGET_EXHAUSTED;
      trace.calls.push(`grep: "${pattern}"` + (doc ? ` [${doc}]` : ""));
      if (!pageStore) return out("grep indisponible (pas de pages OCR pour ce corpus).");
      const result = pageStore.grep(pattern, doc, settings.GREP_MAX_HITS);
      if (result.error) return out(`${result.error}. Documents : ${(result.documents ?? []).join(", ")}`);
      if (!result.hits.length) return out(`0 occurrence de « ${pattern} ».`);
      const lines = result.hits.map((h) => `${h.doc} p. ${h.page + 1}: ${h.line}`);
      const more = result.total - result.hits.length;
      if (more > 0) lines.push(`… et ${more} autres occurrences (affinez le motif ou le document).`);
      return out(lines.join("\n"));
    },
  });

  const read_page = tool({
    description:
      "Lit une page entière d'un document (numéro tel qu'affiché par search et grep, à " +
      "partir de 1). Pour un tableau à cheval sur plusieurs pages, donnez `end_page` : " +
      "les pages `page` à `end_page` sont lues ensemble (3 au plus).",
    inputSchema: z.object({
      doc: z.string().describe("Nom du document"),
      page: z.coerce.number().int().describe("Numéro de page, à partir de 1"),
      end_page: z.coerce.number().int().nullish().describe("Dernière page à lire (optionnel)"),
    }),
    execute: async ({ doc, page, end_page }) => {
      if (!spend()) return BUDGET_EXHAUSTED;
      const span = end_page !== null && end_page !== undefined && end_page !== page ? `${page}-${end_page}` : `${page}`;
      trace.calls.push(`read_page: ${doc} p. ${span}`);
      if (!pageStore) return out("read_page indisponible (pas de pages OCR pour ce corpus).");
      const start = page - 1;
      const end = end_page !== null && end_page !== undefined ? end_page - 1 : start;
      const pages = pageStore.pagePassages(doc, start, end);
      if (!pages.length) {
        const r = pageStore.readPage(doc, start);
        return out("error" in r ? r.error : "page introuvable");
      }
      trace.readPages.push(...pages);
      const numbers = context.add(pages);
      return out(pages.map((d, i) => `=== [${numbers[i]}] ${locator(d)} ===\n${d.content}`).join("\n\n"));
    },
  });

  return { search, grep, read_page };
}

/**
 * Vérification en conditions réelles de la boucle d'outils : une question dont la réponse
 * n'est PAS dans le contexte initial (un seul passage hors sujet), sur le rapport DeepSeek.
 * Le modèle doit appeler grep / read_page pour répondre.
 *
 *   pnpm tools:check
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { env } from "../config/settings";
import { ResearchAgent } from "../agent/research-agent";
import { mainModel } from "../llm/providers";
import { DocumentProcessor } from "../ingestion/document-processor";
import { buildRetriever } from "../retrieval/index-builder";
import { PageStore } from "../retrieval/page-store";

async function main() {
  const file = path.join(env.EXAMPLES_DIR, "DeepSeek Technical Report.pdf");
  const doc = await new DocumentProcessor().process("DeepSeek Technical Report.pdf", fs.readFileSync(file));
  const retriever = await buildRetriever([{ key: `doc-${doc.hash}`, chunks: doc.chunks }], "tools-check");
  const pageStore = new PageStore({ [doc.source]: doc.pages });
  const offTopic = doc.chunks.find((c) => c.metadata.page === 0)!;
  const r = await new ResearchAgent(mainModel()).generateWithTools({
    question: "Quel score Codeforces (rating) obtient DeepSeek-R1 ? Cherche dans le document.",
    passages: [offTopic],
    retriever,
    pageStore,
    scope: null,
    relevance: "NO_MATCH",
  });
  console.log("Appels :", r.calls);
  console.log("Passages ajoutés :", r.passages.length - 1);
  console.log("Réponse :", r.answer.slice(0, 600));
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

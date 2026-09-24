/**
 * Premier jalon : chaque modèle configuré répond-il ? (LLM, petit modèle, embeddings, rerank)
 *
 *   pnpm smoke
 */
import { startTelemetry, shutdownTelemetry } from "../observability/instrumentation";
const tracing = startTelemetry();

import { embed, generateText, rerank } from "ai";
import { env } from "../config/settings";
import { embeddingModel, embeddingProviderOptions, mainModel, rerankingModel, smallModel } from "../llm/providers";
import { rootCause } from "../llm/resilience";

async function check(name: string, fn: () => Promise<string>): Promise<boolean> {
  const t0 = Date.now();
  try {
    const detail = await fn();
    console.log(`✅ ${name.padEnd(12)} ${detail} (${Date.now() - t0} ms)`);
    return true;
  } catch (error) {
    console.log(`❌ ${name.padEnd(12)} ${rootCause(error)}`);
    return false;
  }
}

async function main() {
  console.log(`Région AWS ${env.AWS_REGION} | Langfuse ${tracing ? "actif" : "non configuré"}\n`);
  const results = [
    await check("LLM", async () => {
      const r = await generateText({ model: mainModel(), prompt: "Réponds uniquement : pong", maxOutputTokens: 10, telemetry: { functionId: "smoke" } });
      return `${env.LLM_PROVIDER}:${env.MODEL_ID} -> « ${r.text.trim()} »`;
    }),
    await check("LLM small", async () => {
      const r = await generateText({ model: smallModel(), prompt: "Réponds uniquement : pong", maxOutputTokens: 10 });
      return `${env.MODEL_SMALL_ID} -> « ${r.text.trim()} »`;
    }),
    await check("Embeddings", async () => {
      const r = await embed({ model: embeddingModel(), value: "congés payés", providerOptions: embeddingProviderOptions("query") });
      return `${env.EMBEDDING_MODEL_ID} -> dimension ${r.embedding.length}`;
    }),
    await check("Rerank", async () => {
      const model = rerankingModel();
      if (!model) return "désactivé (RERANK_PROVIDER=none)";
      const r = await rerank({ model, query: "jours de congés", documents: ["Le salarié a droit à 2,5 jours ouvrables de congé par mois.", "La météo est clémente."], topN: 2 });
      return `${env.RERANK_MODEL_ID} -> meilleur score ${r.ranking[0].score.toFixed(3)} (doc ${r.ranking[0].originalIndex})`;
    }),
  ];
  process.exitCode = results.every(Boolean) ? 0 : 1;
}

main().finally(() => shutdownTelemetry());

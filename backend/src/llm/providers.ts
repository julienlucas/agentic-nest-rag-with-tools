/**
 * Fabrique de modèles : le seul endroit qui connaît les fournisseurs.
 *
 * Tout le reste du code manipule les interfaces de l'AI SDK (LanguageModel, EmbeddingModel,
 * RerankingModel) : passer de Bedrock à Azure OpenAI ou à Mistral est une ligne de `.env`.
 */
import { createAmazonBedrock } from "@ai-sdk/amazon-bedrock";
import { createAzure } from "@ai-sdk/azure";
import { createCohere } from "@ai-sdk/cohere";
import { createMistral } from "@ai-sdk/mistral";
import { fromNodeProviderChain } from "@aws-sdk/credential-providers";
import type { EmbeddingModel, LanguageModel, RerankingModel } from "ai";
import { env } from "../config/settings";

export type LlmProvider = "bedrock" | "mistral" | "azure";

let bedrockInstance: ReturnType<typeof createAmazonBedrock> | undefined;

function bedrock() {
  if (!bedrockInstance) {
    // Clés explicites dans le .env, sinon la chaîne AWS standard (~/.aws, SSO, rôle IAM).
    const explicit = env.AWS_ACCESS_KEY_ID && env.AWS_SECRET_ACCESS_KEY;
    bedrockInstance = explicit
      ? createAmazonBedrock({
          region: env.AWS_REGION,
          accessKeyId: env.AWS_ACCESS_KEY_ID,
          secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
          sessionToken: env.AWS_SESSION_TOKEN,
        })
      : createAmazonBedrock({
          region: env.AWS_REGION,
          credentialProvider: fromNodeProviderChain(),
        });
  }
  return bedrockInstance;
}

function requireKey(value: string | undefined, name: string): string {
  if (!value) throw new Error(`${name} manquante dans le .env`);
  return value;
}

const mistral = () => createMistral({ apiKey: requireKey(env.MISTRALAI_API_KEY, "MISTRALAI_API_KEY") });
const cohere = () => createCohere({ apiKey: requireKey(env.COHERE_API_KEY, "COHERE_API_KEY") });
const azure = () =>
  createAzure({
    resourceName: requireKey(env.AZURE_RESOURCE_NAME, "AZURE_RESOURCE_NAME"),
    apiKey: requireKey(env.AZURE_API_KEY, "AZURE_API_KEY"),
  });

/** Modèle de langage. Pour Azure, `modelId` est le nom du déploiement. */
export function languageModel(provider: LlmProvider, modelId: string): LanguageModel {
  switch (provider) {
    case "bedrock":
      return bedrock()(modelId);
    case "mistral":
      return mistral()(modelId);
    case "azure":
      return azure()(modelId);
  }
}

export const mainModel = () => languageModel(env.LLM_PROVIDER, env.MODEL_ID);
export const smallModel = () => languageModel(env.LLM_PROVIDER, env.MODEL_SMALL_ID);
export const judgeModel = () => languageModel(env.JUDGE_PROVIDER, env.JUDGE_MODEL_ID);

export function embeddingModel(): EmbeddingModel {
  return env.EMBEDDING_PROVIDER === "bedrock"
    ? bedrock().embedding(env.EMBEDDING_MODEL_ID)
    : mistral().embedding(env.EMBEDDING_MODEL_ID);
}

/** Options fournisseur d'un appel d'embedding, selon qu'on indexe ou qu'on interroge. */
export function embeddingProviderOptions(purpose: "document" | "query") {
  if (env.EMBEDDING_PROVIDER !== "bedrock") return undefined;
  const id = env.EMBEDDING_MODEL_ID.toLowerCase();
  if (id.includes("cohere")) {
    // Cohere sur Bedrock : input_type obligatoire, et différent à l'indexation.
    return { bedrock: { inputType: purpose === "document" ? "search_document" : "search_query" } };
  }
  if (id.includes("titan")) {
    // Vecteurs unitaires : le cosinus reste exact (même raison que VECTOR_SPACE côté Python).
    return { bedrock: { normalize: true } };
  }
  return undefined;
}

export function rerankingModel(): RerankingModel | null {
  switch (env.RERANK_PROVIDER) {
    case "bedrock":
      return bedrock().reranking(env.RERANK_MODEL_ID);
    case "cohere":
      return cohere().reranking(env.RERANK_MODEL_ID);
    case "none":
      return null;
  }
}

/** Identifiant stable d'un modèle, pour les caches et le coût. */
export function modelSlug(provider: string, modelId: string): string {
  return `${provider}-${modelId}`.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

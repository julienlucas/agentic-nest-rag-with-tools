/**
 * Configuration : variables d'environnement validées par Zod, et réglages du pipeline repris
 * à l'identique de `agentic-rag/backend/config/settings.py` (mêmes valeurs, mêmes commentaires
 * quand ils expliquent un choix mesuré).
 */
import * as path from "node:path";
import * as dotenv from "dotenv";
import { z } from "zod";

// src/config (ou dist/config) -> backend -> racine du repo
export const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
dotenv.config({ path: path.join(REPO_ROOT, ".env"), quiet: true });
dotenv.config({ path: path.join(REPO_ROOT, "backend", ".env"), quiet: true });

const provider = z.enum(["bedrock", "mistral", "azure"]);
const bool = z
  .enum(["true", "false", "1", "0"])
  .transform((v) => v === "true" || v === "1");

const envSchema = z.object({
  PORT: z.coerce.number().default(3000),

  // --- Modèles ------------------------------------------------------------
  // Générateur (réponse avec outils) et sous-agents (pertinence, routage).
  LLM_PROVIDER: provider.default("bedrock"),
  MODEL_ID: z.string().default("mistral.mistral-large-2407-v1:0"),
  MODEL_SMALL_ID: z.string().default("mistral.mistral-small-2402-v1:0"),

  EMBEDDING_PROVIDER: z.enum(["bedrock", "mistral"]).default("bedrock"),
  EMBEDDING_MODEL_ID: z.string().default("cohere.embed-multilingual-v3"),

  RERANK_PROVIDER: z.enum(["bedrock", "cohere", "none"]).default("bedrock"),
  RERANK_MODEL_ID: z.string().default("cohere.rerank-v3-5:0"),

  // Juge de l'éval : doit rester CONSTANT d'un run à l'autre pour que les runs se comparent.
  // Par défaut, le même juge que le projet Python (Mistral Large via La Plateforme).
  JUDGE_PROVIDER: provider.default("mistral"),
  JUDGE_MODEL_ID: z.string().default("mistral-large-latest"),

  // --- Accès fournisseurs ----------------------------------------------------
  AWS_REGION: z.string().default("eu-west-3"),
  AWS_ACCESS_KEY_ID: z.string().optional(),
  AWS_SECRET_ACCESS_KEY: z.string().optional(),
  AWS_SESSION_TOKEN: z.string().optional(),
  MISTRALAI_API_KEY: z.string().optional(),
  COHERE_API_KEY: z.string().optional(),
  AZURE_RESOURCE_NAME: z.string().optional(),
  AZURE_API_KEY: z.string().optional(),

  // --- Stockage vectoriel ------------------------------------------------------
  VECTOR_STORE: z.enum(["memory", "opensearch"]).default("memory"),
  OPENSEARCH_URL: z.string().default("http://localhost:9200"),
  OPENSEARCH_USERNAME: z.string().optional(),
  OPENSEARCH_PASSWORD: z.string().optional(),

  // --- Observabilité -------------------------------------------------------------
  LANGFUSE_PUBLIC_KEY: z.string().optional(),
  LANGFUSE_SECRET_KEY: z.string().optional(),
  LANGFUSE_BASE_URL: z.string().default("https://cloud.langfuse.com"),

  // --- Pipeline (surchargeables pour les ablations) ------------------------------
  GENERATOR_TOOLS_ENABLED: bool.default(true),
  RERANK_ENABLED: bool.default(true),
  DOCUMENT_ROUTING_ENABLED: bool.default(true),
  RELEVANCE_CHECK_ENABLED: bool.default(true),

  DATA_DIR: z.string().default(path.join(REPO_ROOT, "data")),
  EXAMPLES_DIR: z.string().default(path.join(REPO_ROOT, "backend", "static")),
  FRONTEND_DIST: z.string().default(path.join(REPO_ROOT, "frontend", "dist")),
  CORS_ORIGINS: z.string().default("http://localhost:8080"),
});

export type Env = z.infer<typeof envSchema>;

function loadEnv(): Env {
  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `  ${i.path.join(".")}: ${i.message}`).join("\n");
    throw new Error(`Configuration invalide (.env) :\n${issues}`);
  }
  return parsed.data;
}

export const env = loadEnv();

/** Réglages du pipeline, mêmes valeurs que le projet Python. */
export const settings = {
  LLM_TIMEOUT_MS: 30_000, // par appel
  LLM_MAX_RETRIES: 2,

  // Retrieval hybride
  VECTOR_SEARCH_K: 20,
  BM25_K: 20,
  HYBRID_WEIGHTS: [0.5, 0.5] as const, // équilibré — BM25 crucial pour les termes exacts
  RRF_C: 60, // constante de l'EnsembleRetriever LangChain
  RERANK_TOP_K: 30, // top N après reranking de tous les candidats
  MAX_RERANK_CANDIDATES: 40, // au-delà, latence sans gain de qualité notable

  // Passages transmis au modèle de réponse
  RESEARCH_TOP_K: 10,
  GENERATOR_MAX_TOOL_CALLS: 5,
  GENERATOR_MAX_TOKENS: 700, // la réponse peut suivre plusieurs tours d'outils et poser un calcul
  BASELINE_MAX_TOKENS: 500,

  // Chunking parent-child (documents uploadés ; FinanceBench réutilise les chunks Python)
  PARENT_CHUNK_SIZE: 1200,
  CHILD_CHUNK_SIZE: 400,
  CHILD_OVERLAP: 50,

  // Outils
  SEARCH_TOOL_RESULTS: 8,
  GREP_MAX_HITS: 20,
  READ_PAGE_MAX_CHARS: 8000,

  MAX_FILE_BYTES: 50 * 1024 * 1024,
  ALLOWED_TYPES: [".pdf", ".txt", ".md"],
} as const;

export function isLangfuseConfigured(): boolean {
  return Boolean(env.LANGFUSE_PUBLIC_KEY && env.LANGFUSE_SECRET_KEY);
}

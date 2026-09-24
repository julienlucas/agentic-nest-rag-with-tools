/**
 * OpenTelemetry + Langfuse. À importer AVANT tout le reste (main.ts, scripts d'éval).
 *
 * - LangfuseSpanProcessor exporte les spans vers Langfuse (EU par défaut).
 * - LangfuseVercelAiSdkIntegration trace chaque appel AI SDK v7 (generateText, embed, rerank),
 *   tokens et latence compris.
 * - traced() ouvre un span pour les étapes qui ne sont pas des appels LLM (retrieval, pipeline).
 *
 * Sans clés Langfuse, tout est inactif et traced() appelle simplement la fonction.
 */
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { propagateAttributes, startActiveObservation } from "@langfuse/tracing";
import { LangfuseVercelAiSdkIntegration } from "@langfuse/vercel-ai-sdk";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { registerTelemetry } from "ai";
import { isLangfuseConfigured } from "../config/settings";

let spanProcessor: LangfuseSpanProcessor | undefined;
let sdk: NodeSDK | undefined;

export function startTelemetry(): boolean {
  if (sdk || !isLangfuseConfigured()) return Boolean(sdk);
  spanProcessor = new LangfuseSpanProcessor();
  sdk = new NodeSDK({ spanProcessors: [spanProcessor] });
  sdk.start();
  registerTelemetry(new LangfuseVercelAiSdkIntegration());
  return true;
}

/** À appeler avant la fin d'un script : les spans encore en lot seraient perdus. */
export async function flushTelemetry(): Promise<void> {
  await spanProcessor?.forceFlush();
}

export async function shutdownTelemetry(): Promise<void> {
  await flushTelemetry();
  await sdk?.shutdown();
}

type Attrs = { sessionId?: string; tags?: string[]; metadata?: Record<string, string> };

/** Span Langfuse autour de `fn` (entrée et sortie enregistrées), ou appel direct sans Langfuse. */
export async function traced<T>(
  name: string,
  input: unknown,
  fn: () => Promise<T>,
  opts: { asType?: "span" | "chain" | "retriever" | "agent"; attrs?: Attrs; output?: (r: T) => unknown } = {},
): Promise<T> {
  if (!sdk) return fn();
  const run = () =>
    startActiveObservation(
      name,
      async (span) => {
        span.update({ input });
        const result = await fn();
        span.update({ output: opts.output ? opts.output(result) : result });
        return result;
      },
      { asType: (opts.asType ?? "span") as "span" },
    );
  return opts.attrs ? propagateAttributes(opts.attrs, run) : run();
}

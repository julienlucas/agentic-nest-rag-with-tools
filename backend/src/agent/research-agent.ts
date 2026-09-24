/**
 * Le modèle de réponse. Deux modes :
 *
 * - generate()          : une seule génération sur les passages initiaux (baseline de l'ablation).
 * - generateWithTools() : le modèle dispose de search / grep / read_page ; le modèle qui cherche
 *                         est celui qui répond, dans la même conversation (Agentic Search).
 */
import { generateText, isStepCount, type LanguageModel } from "ai";
import { settings } from "../config/settings";
import type { Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import { modelIdOf, usageMeter } from "../llm/usage";
import type { Scope } from "../retrieval/indexes";
import type { PageStore } from "../retrieval/page-store";
import type { Retriever } from "../retrieval/retriever";
import { buildNumberedContext, GrowingContext, type Citation } from "./numbered-context";
import { baselinePrompt, CANNOT_ANSWER, toolsSystem, toolsUser, type Relevance } from "./prompts";
import { buildTools, type ToolTrace } from "./tools";

const logger = getLogger("ResearchAgent");

export type AnswerResult = {
  answer: string;
  citations: Citation[];
  /** Tout ce que le modèle a vu : initiaux + passages ramenés par les outils. */
  passages: Passage[];
  toolCalls: number;
  calls: string[];
};

export class ResearchAgent {
  constructor(private readonly model: LanguageModel) {}

  async generate(question: string, passages: Passage[]): Promise<AnswerResult> {
    const { context, citations } = buildNumberedContext(passages);
    const result = await generateText({
      model: this.model,
      prompt: baselinePrompt(question, context),
      temperature: 0,
      maxOutputTokens: settings.BASELINE_MAX_TOKENS,
      maxRetries: settings.LLM_MAX_RETRIES,
      timeout: settings.LLM_TIMEOUT_MS,
      telemetry: { functionId: "answer-baseline" },
    });
    usageMeter.record(modelIdOf(this.model), result.usage);
    return { answer: result.text.trim() || CANNOT_ANSWER, citations, passages, toolCalls: 0, calls: [] };
  }

  async generateWithTools(opts: {
    question: string;
    passages: Passage[];
    retriever: Retriever;
    pageStore: PageStore | null;
    scope: Scope;
    relevance?: Relevance | null;
    budget?: number;
  }): Promise<AnswerResult> {
    const budget = opts.budget ?? settings.GENERATOR_MAX_TOOL_CALLS;
    const context = new GrowingContext(opts.passages);
    const trace: ToolTrace = { calls: [], outputs: [], searchResults: [], readPages: [] };
    const tools = buildTools({
      retriever: opts.retriever,
      pageStore: opts.pageStore,
      scope: opts.scope,
      context,
      trace,
      budget,
    });

    const system = toolsSystem(budget, opts.relevance);
    const { context: initialContext } = buildNumberedContext(opts.passages);
    const result = await generateText({
      model: this.model,
      system,
      prompt: toolsUser(opts.question, initialContext),
      tools,
      // Les appels au-delà du budget renvoient « budget épuisé » sans s'exécuter ; la limite
      // d'étapes n'est qu'un filet si le modèle insiste.
      stopWhen: isStepCount(budget + 2),
      temperature: 0,
      maxOutputTokens: settings.GENERATOR_MAX_TOKENS,
      maxRetries: settings.LLM_MAX_RETRIES,
      timeout: settings.LLM_TIMEOUT_MS * (budget + 2),
      telemetry: { functionId: "answer-with-tools" },
    });
    usageMeter.record(modelIdOf(this.model), result.usage);

    let answer = result.text.trim();
    if (!answer) {
      // La boucle s'est arrêtée sur un appel d'outil : réponse forcée SANS outils. On ne peut
      // pas simplement retirer les outils de la conversation (Bedrock efface alors tout
      // l'historique d'outils) : on repart d'un prompt qui contient tout ce qui a été trouvé.
      answer = await this.forceFinalAnswer(opts.question, context.passages, trace, opts.relevance);
    }

    const { citations } = buildNumberedContext(context.passages);
    logger.log(
      `${trace.calls.length} appel(s) d'outils, ${context.passages.length - opts.passages.length} passage(s) ajouté(s)`,
    );
    return {
      answer: answer || CANNOT_ANSWER,
      citations,
      passages: context.passages,
      toolCalls: trace.calls.length,
      calls: trace.calls,
    };
  }

  private async forceFinalAnswer(
    question: string,
    passages: Passage[],
    trace: ToolTrace,
    relevance?: Relevance | null,
  ): Promise<string> {
    const { context } = buildNumberedContext(passages);
    const log = trace.outputs.length ? `\n\n**Journal des outils déjà appelés:**\n${trace.outputs.join("\n\n")}` : "";
    const result = await generateText({
      model: this.model,
      system: toolsSystem(0, relevance),
      prompt: toolsUser(question, context + log),
      temperature: 0,
      maxOutputTokens: settings.GENERATOR_MAX_TOKENS,
      maxRetries: settings.LLM_MAX_RETRIES,
      timeout: settings.LLM_TIMEOUT_MS,
      telemetry: { functionId: "answer-forced-final" },
    });
    usageMeter.record(modelIdOf(this.model), result.usage);
    return result.text.trim();
  }
}

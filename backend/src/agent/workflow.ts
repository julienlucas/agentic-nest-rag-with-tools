/**
 * Le pipeline complet, équivalent du graphe LangGraph Python — sans LangGraph : avec le
 * générateur à outils, il ne reste qu'une séquence.
 *
 *   retrieval (routage + hybride + rerank)
 *   -> vérificateur de pertinence (indice, ne bloque plus)
 *   -> réponse : générateur à outils (défaut) ou génération unique (ablation)
 *   -> rapport de vérification
 */
import { env, settings } from "../config/settings";
import type { Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import { mainModel, smallModel } from "../llm/providers";
import { isRateLimit } from "../llm/resilience";
import { traced } from "../observability/instrumentation";
import type { Scope } from "../retrieval/indexes";
import type { PageStore } from "../retrieval/page-store";
import type { Retriever } from "../retrieval/retriever";
import type { Citation } from "./numbered-context";
import { GENERATION_ERROR, OFF_TOPIC, PIPELINE_ERROR, type Relevance } from "./prompts";
import { RelevanceChecker } from "./relevance-checker";
import { ResearchAgent } from "./research-agent";
import { buildVerificationReport } from "./verification-report";

const logger = getLogger("AgentWorkflow");

export type PipelineResult = {
  draft_answer: string;
  verification_report: string;
  citations: Citation[];
};

export type RunResult = PipelineResult & {
  relevance: Relevance | null;
  /** Passages réellement vus par le modèle de réponse. */
  passages: Passage[];
  toolCalls: number;
  calls: string[];
};

export type Mode = "agentic" | "baseline";

export class AgentWorkflow {
  private readonly researcher = new ResearchAgent(mainModel());
  private readonly checker = new RelevanceChecker(smallModel());

  /** Point d'entrée de l'API : retrieval compris, et aucune exception ne remonte. */
  async fullPipeline(question: string, retriever: Retriever, pageStore: PageStore | null, sessionId?: string) {
    try {
      return await traced(
        "rag-pipeline",
        { question },
        async () => {
          const scope = await retriever.route(question);
          const passages = await traced("retrieval", { question, scope }, () => retriever.invokeWithScope(question, scope), {
            asType: "retriever",
            output: (docs) => docs.slice(0, settings.RESEARCH_TOP_K).map((d) => ({ locator: d.metadata, score: d.metadata.rerankScore })),
          });
          const r = await this.run({ question, passages, retriever, pageStore, scope, mode: "agentic" });
          return { draft_answer: r.draft_answer, verification_report: r.verification_report, citations: r.citations };
        },
        { asType: "chain", attrs: { sessionId, tags: ["api"] }, output: (r) => ({ answer: r.draft_answer }) },
      );
    } catch (error) {
      logger.error(`L'exécution du workflow a échoué : ${(error as Error).message}`);
      return {
        draft_answer: PIPELINE_ERROR,
        verification_report: `Erreur: ${(error as Error).name}`,
        citations: [],
      } satisfies PipelineResult;
    }
  }

  /**
   * À partir de passages déjà récupérés (l'éval récupère une fois pour les deux modes).
   * Les rate limits remontent pour être rejoués ; les autres erreurs de génération donnent
   * le message figé, que le juge classe ERROR (hors dénominateur).
   */
  async run(opts: {
    question: string;
    passages: Passage[];
    retriever: Retriever;
    pageStore: PageStore | null;
    scope: Scope;
    mode: Mode;
  }): Promise<RunResult> {
    const { question, passages, mode } = opts;
    const top = passages.slice(0, settings.RESEARCH_TOP_K);

    if (mode === "baseline") {
      const r = await this.researcher.generate(question, top);
      return this.result(r.answer, null, top, r.citations, 0, []);
    }

    const relevance = env.RELEVANCE_CHECK_ENABLED ? await this.checker.check(question, passages, 3) : null;
    const toolsEnabled = env.GENERATOR_TOOLS_ENABLED;
    if (!passages.length || (!toolsEnabled && relevance === "NO_MATCH")) {
      return this.result(OFF_TOPIC, relevance ?? "NO_MATCH", top, [], 0, []);
    }

    try {
      const r = toolsEnabled
        ? await this.researcher.generateWithTools({
            question,
            passages: top,
            retriever: opts.retriever,
            pageStore: opts.pageStore,
            scope: opts.scope,
            relevance,
          })
        : await this.researcher.generate(question, top);
      return this.result(r.answer, relevance, r.passages, r.citations, r.toolCalls, r.calls);
    } catch (error) {
      if (isRateLimit(error)) throw error;
      logger.error(`Le modèle de réponse a échoué : ${(error as Error).message}`);
      return this.result(GENERATION_ERROR, relevance, top, [], 0, []);
    }
  }

  private result(
    answer: string,
    relevance: Relevance | null,
    passages: Passage[],
    citations: Citation[],
    toolCalls: number,
    calls: string[],
  ): RunResult {
    return {
      draft_answer: answer,
      verification_report: buildVerificationReport({ relevance, passages, toolCalls, calls }),
      citations,
      relevance,
      passages,
      toolCalls,
      calls,
    };
  }
}

/**
 * Routage par document : avant de chercher, décider DANS QUEL(S) document(s) chercher.
 * D'abord un matching déterministe sur le nom des fichiers (gratuit), puis un LLM léger
 * seulement si nécessaire. Port de document_router.py.
 */
import { generateText, type LanguageModel } from "ai";
import { settings } from "../config/settings";
import { docLabel } from "../core/passage";
import { getLogger } from "../core/logger";
import { isRateLimit } from "../llm/resilience";
import { modelIdOf, usageMeter } from "../llm/usage";

const logger = getLogger("DocumentRouter");

// Mots de fichiers trop génériques pour identifier un document.
const GENERIC_TOKENS = new Set([
  "pdf", "docx", "txt", "md", "10k", "10q", "8k", "annual", "report", "rapport", "annuel",
  "document", "doc", "file", "fichier", "final", "copy", "copie", "version", "draft",
  "earnings", "q1", "q2", "q3", "q4", "fy", "the", "and", "inc", "corp", "co", "ltd",
]);

/** "AMERICANEXPRESS_2022_10K.pdf" -> {"americanexpress"} */
export function identityTokens(source: string): Set<string> {
  const parts = docLabel(source).toLowerCase().split(/[^a-z]+/);
  return new Set(parts.filter((p) => p.length >= 3 && !GENERIC_TOKENS.has(p)));
}

const ROUTER_PROMPT = (sources: string, question: string) => `Tu dois choisir dans quel(s) document(s) chercher la réponse à une question.

Documents disponibles (un nom par ligne) :
${sources}

Question : ${question}

Règles :
- Si la question désigne clairement un ou plusieurs documents (entreprise, sujet, période), réponds avec leurs noms EXACTS, un par ligne.
- Si la question ne permet pas de choisir, ou concerne tous les documents, réponds ALL.
- N'ajoute aucune explication.`;

export class DocumentRouter {
  readonly sources: string[];
  private readonly tokens: Map<string, Set<string>>;

  constructor(sources: string[], private readonly llm: LanguageModel | null) {
    this.sources = [...new Set(sources)];
    this.tokens = new Map(this.sources.map((s) => [s, identityTokens(s)]));
  }

  matchByName(question: string): string[] {
    const q = question.toLowerCase().replace(/[^a-z]/g, ""); // "American Express" -> "americanexpress"
    return this.sources.filter((s) => {
      const t = this.tokens.get(s)!;
      return t.size > 0 && [...t].some((tok) => q.includes(tok));
    });
  }

  private async matchByLlm(question: string): Promise<string[] | null> {
    if (!this.llm) return null;
    const labels = new Map(this.sources.map((s) => [docLabel(s), s]));
    try {
      const result = await generateText({
        model: this.llm,
        prompt: ROUTER_PROMPT([...labels.keys()].join("\n"), question),
        temperature: 0,
        maxOutputTokens: 200,
        maxRetries: settings.LLM_MAX_RETRIES,
        timeout: settings.LLM_TIMEOUT_MS,
        telemetry: { functionId: "document-router" },
      });
      usageMeter.record(modelIdOf(this.llm), result.usage);
      const content = result.text.trim();
      if (content.toUpperCase().split(/\s+/).includes("ALL")) return null;
      const chosen: string[] = [];
      for (const raw of content.split("\n")) {
        const line = raw.trim().replace(/^[-•*\s]+|[-•*\s]+$/g, "");
        for (const [label, source] of labels) {
          if (line && (line === label || line.toLowerCase().includes(label.toLowerCase()))) chosen.push(source);
        }
      }
      return chosen.length ? [...new Set(chosen)] : null;
    } catch (error) {
      if (isRateLimit(error)) throw error;
      logger.warn(`LLM indisponible (${(error as Error).message}), recherche sur tous les documents`);
      return null;
    }
  }

  /** Sources à interroger, ou null pour toutes. Un seul document indexé : pas de routage. */
  async route(question: string): Promise<string[] | null> {
    if (this.sources.length <= 1) return null;
    const matched = this.matchByName(question);
    if (matched.length) return matched;
    return this.matchByLlm(question);
  }
}

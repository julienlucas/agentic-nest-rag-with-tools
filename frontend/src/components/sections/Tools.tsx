import { Section } from "@/components/site/primitives";

/*
 * Les outils, et pourquoi ils comptent. Chiffres Mistral : https://mistral.ai/news/agentic-search/
 * (20 août 2026), FinanceBench 150 questions, boucle complète (search + open / navigate / read /
 * grep) comparée à une boucle search seule. Chiffres de ce RAG : financebench_summary.json,
 * ablation à retrieval identique (baseline vs agentic).
 */
const tools = [
  {
    verb: "Trouver",
    signature: "search(query, doc)",
    mistral: "search",
    text: "Relance le retrieval hybride du pipeline (BM25 + vecteurs, routage, rerank Cohere) avec une requête reformulée, sur tout le corpus ou un seul document. 8 extraits, chacun avec son document et sa page.",
  },
  {
    verb: "Localiser",
    signature: "grep(pattern, doc)",
    mistral: "grep",
    text: "Toutes les occurrences littérales d'un terme, page par page, sur les 260 pages. Exhaustif : zéro résultat permet d'affirmer qu'une donnée est absente du rapport, au lieu de le supposer.",
  },
  {
    verb: "Lire",
    signature: "read_page(doc, page, end_page)",
    mistral: "open + navigate + read",
    text: "La page entière telle que l'OCR l'a produite, tableau compris, sur 1 à 3 pages quand il est à cheval. Ce qu'un chunk de 1 200 caractères ne montre jamais.",
  },
];

const toolGains = [
  {
    value: "−24 à −34 %",
    label: "de tokens consommés",
  },
  {
    value: "+7 à +9 pts",
    label: "de réponses correctes",
  },
  {
    value: "−40 %",
    label: "de latence",
    detail: "au p90 : 255 s → 154 s · en moyenne : 108 s → 71 s",
  },
];

const toolsHere = [
  {
    value: "80,8 → 92,3 %",
    label: "de réponses correctes",
    detail: "même index, même retrieval, mêmes 10 passages initiaux : la seule différence, ce sont les outils (Claude Sonnet 4.6 dans les deux cas)",
  },
  {
    value: "15,4 → 7,7 %",
    label: "d'hallucinations",
    detail: "4 réponses fausses sur 26 sans outils, 2 sur 26 avec",
  },
  {
    value: "15 sur 26",
    label: "questions ont appelé un outil",
    detail: "2,5 appels en moyenne, une page entière lue dans 14 cas sur 15 ; les 11 autres répondent en une passe, sans un token de plus",
  },
];

export function Tools() {
  return (
    <Section
      id="outils"
      index="03"
      eyebrow="Le levier déterminant"
      title={
        <>
          Ce RAG agentique ne se contente pas des passages qu&apos;on lui donne
          : il <span className="accent-italic">va chercher la preuve</span>,
          grâce à des outils.
        </>
      }
      intro="Un RAG classique choisit dix passages avant que le modèle ne lise la question, puis lui demande de répondre en une seule passe. Ici l'approche est différente, l'agent peut lui-même venir lire les documents, analyser un tableau, le contexte autour, partir, revenir, ect."
    >
      {/* d'où ça vient */}

      {/* les trois outils */}
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        {tools.map((t, i) => (
          <div
            key={t.signature}
            className="card-paper border-hairline flex h-full flex-col p-6"
          >
            <div className="flex items-center justify-between">
              <span className="eyebrow">{t.verb}</span>
              <span className="mono-xs text-ink-faint">0{i + 1}</span>
            </div>
            <code className="mono-xs mt-3 w-fit rounded-sm bg-brand-surface px-2 py-1 text-brand-deep">
              {t.signature}
            </code>
            <p className="mb-4 mt-3 text-sm leading-relaxed text-ink-muted">
              {t.text}
            </p>
          </div>
        ))}
      </div>

      {/* moins de tokens, plus de précision */}
      <div className="card-paper border-hairline mt-10 p-8">
        <p className="display-sm mt-8">Voici les gains avec des outils</p>
        <p className="mono-xs mt-1 text-muted-foreground">
          Mistral Agentic Search, modèle Mistral Medium 3.5 · sur FinanceBench, 150 questions, 368 documents, soit 53900
          pages indéxées ensemble
        </p>
        <div className="mt-6 grid gap-6 sm:grid-cols-3">
          {toolGains.map((g) => (
            <div key={g.label} className="border-l-2 border-brand pl-4">
              <div className="font-display text-4xl font-normal tracking-tight tabular-nums">
                {g.value}
              </div>
              <div className="mt-1 text-sm font-medium">{g.label}</div>
              <div className="mono-xs mt-1 text-muted-foreground">
                {g.detail}
              </div>
            </div>
          ))}
        </div>
        <blockquote className="pt-6">
          <p className="display-sm text-ink">
            « Les outils de retrieval ne sont pas un surcoût : ils remplacent
            des recherches relancées pour rien par une{" "}
            <span className="accent-italic">navigation précise</span>. »
          </p>
          <cite className="mono-xs mt-2 block not-italic text-muted-foreground">
            Mistral AI, Introducing Agentic Search
          </cite>
        </blockquote>
      </div>
    </Section>
  );
}

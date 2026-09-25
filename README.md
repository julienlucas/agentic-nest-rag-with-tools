# RAG agentique — TypeScript, NestJS, Amazon Bedrock

Port TypeScript de [agentic-rag](https://github.com/julienlucas/agentic-rag-with-tools) (Python, Django, LangGraph), sur une stack
de production AWS : **NestJS, Vercel AI SDK v7, Amazon Bedrock, OpenSearch, Langfuse et
OpenTelemetry**. Le frontend React est repris tel quel, avec le même contrat d'API.

## Modèles utilisés

| Rôle | Modèle | Accès |
|---|---|---|
| Générateur à outils (recherche + réponse) | **Claude Sonnet 4.6** (`eu.anthropic.claude-sonnet-4-6`) | Amazon Bedrock, Paris |
| Vérificateur de pertinence, routeur | **Claude Haiku 4.5** (`eu.anthropic.claude-haiku-4-5-20251001-v1:0`) | Amazon Bedrock, Paris |
| Embeddings | **Cohere Embed Multilingual v3** (`cohere.embed-multilingual-v3`) | Amazon Bedrock, Paris |
| Reranking | **Cohere Rerank v4 Pro** (`rerank-v4.0-pro`) | API Cohere directe |
| OCR des PDF uploadés | Mistral OCR | API Mistral |
| Juge de l'éval | Mistral Large (`mistral-large-latest`) | API Mistral (le même juge que le projet Python) |

Pourquoi le rerank ne passe pas par Bedrock : à Paris, Bedrock n'a pas de reranker (Cohere
Rerank 3.5 n'existe qu'en `eu-central-1`), et Rerank v4 Pro est le reranker du projet Python.
Tout se change dans le `.env` : `RERANK_PROVIDER=bedrock` avec `AWS_REGION=eu-central-1` donne
une stack 100 % AWS.

Le projet Python est mesuré sur [FinanceBench](https://github.com/patronus-ai/financebench) :
**83,3 % de réponses correctes avec les outils, contre 65,4 % sans**, à retrieval identique.
Ce port réutilise les mêmes pages OCR, les mêmes chunks, les mêmes prompts et le même juge. Le
runner d'éval est porté lui aussi : chaque changement de fournisseur se mesure au lieu de se
supposer.

## Ce qui est porté

| Python (agentic-rag) | TypeScript (ce repo) |
|---|---|
| Django `views.py` | `api/rag.controller.ts` (Nest), même contrat HTTP |
| `pydantic-settings` | `config/settings.ts` (Zod), mêmes réglages |
| LangGraph `StateGraph` | `agent/workflow.ts` : une séquence, la boucle est dans l'AI SDK |
| `@tool` LangChain + `run_tool_loop` | `agent/tools.ts` : `tool()` + Zod, budget de 5 appels |
| `ChatMistralAI` (Mistral Large / Small) | `llm/providers.ts` : Claude Sonnet 4.6 / Haiku 4.5 via Bedrock ; Mistral ou Azure OpenAI en une ligne de `.env` |
| Mistral Embed + Chroma | `retrieval/embedder.ts` : Cohere Embed via Bedrock (cache disque) + index mémoire ou **OpenSearch** |
| `rank_bm25` + `EnsembleRetriever` | `retrieval/bm25.ts` (mêmes scores, testés contre Python) + `fusion.ts` (RRF) |
| Cohere Rerank v4 (API) | `retrieval/reranker.ts` : Rerank v4 Pro par l'API Cohere (ou Rerank 3.5 via Bedrock) |
| `page_store.py` | `retrieval/page-store.ts` (grep, read_page) |
| `document_router.py` | `retrieval/document-router.ts` |
| `relevance_checker.py` | `agent/relevance-checker.ts` |
| LangSmith | **Langfuse** via OpenTelemetry (`observability/instrumentation.ts`) |
| `run_financebench_eval.py`, `llm_judge.py` | `eval/financebench/run.ts`, `judge.ts` |
| `regression.py` (garde-fous) | `eval/financebench/regression.ts` |
| pytest + Hypothesis | Vitest + fast-check (31 tests hors ligne) |

## Démarrage

Prérequis : Node 22+, pnpm, le projet Python à côté (`../agentic-rag`) pour les données.

```bash
# 1. Données : pages OCR, chunks et dataset exportés du projet Python (aucun appel d'API)
../agentic-rag/.venv/bin/python scripts/export_from_python.py

# 2. Configuration
cp .env.example .env        # puis renseigner région et modèles Bedrock, clés Langfuse

# 3. Backend
cd backend && pnpm install
pnpm smoke                  # chaque modèle configuré répond-il ? (LLM, embeddings, rerank)
pnpm dev                    # API sur http://localhost:3000

# 4. Frontend (autre terminal)
cd frontend && pnpm install && pnpm dev    # http://localhost:8080, proxy /api -> :3000
```

En production, `pnpm build` du frontend puis du backend : Nest sert le frontend buildé.
Le `Dockerfile` construit l'image complète.

## Évaluation FinanceBench

```bash
cd backend
pnpm index:financebench                     # embeddings des ~12 400 chunks, une fois par modèle
pnpm eval                                   # 26 questions, avec et sans outils, juge LLM
pnpm eval --mode agentic --label bedrock-sonnet-4-6
pnpm eval --max-items 3 --no-judge --out-dir /tmp/essai   # essai rapide
```

Chaque run écrit `eval-outputs/<label>/summary.json` avec la configuration des modèles, les
comptages bruts, les IC95, la latence et le coût.

**Isoler les variables, une par run.** Pour savoir d'où vient un écart avec le 83,3 % Python :

| Run | LLM | Embeddings | Rerank | Ce qu'il mesure |
|---|---|---|---|---|
| 0 | Mistral Large (API Mistral) | mistral-embed | Cohere v4 Pro | le port TS seul (mêmes modèles que Python) |
| 1 | **Claude Sonnet 4.6** (Bedrock) | mistral-embed | Cohere v4 Pro | le changement de LLM |
| 2 | Claude Sonnet 4.6 (Bedrock) | **Cohere Embed v3** (Bedrock) | Cohere v4 Pro | les embeddings — **configuration par défaut** |
| 3 | Claude Sonnet 4.6 (Bedrock, `eu-central-1`) | Cohere Embed v3 (Bedrock) | **Rerank 3.5** (Bedrock) | le reranker (stack 100 % AWS) |

Mistral Large n'est pas proposé dans une version récente sur Bedrock en Europe
(`mistral-large-2402` seulement, à Paris) : le run 1 change donc le fournisseur et le modèle à
la fois.

Le juge reste le même (Mistral Large, La Plateforme) sur tous les runs.

**Garde-fous de régression** (appellent les vraies API, à lancer avant de merger) :

```bash
pnpm eval:gate retrieval --update-baseline   # une fois par configuration d'embeddings / rerank
pnpm eval:gate retrieval                     # la page de preuve sort-elle du top-10 ?
pnpm eval:gate sentinels                     # 10 questions réussies, jugées de bout en bout
```

Code de sortie 0 (ok), 1 (régression) ou 2 (run invalide : un service a lâché en silence).

## OpenSearch

```bash
docker compose up -d          # OpenSearch 2.19 + Dashboards (http://localhost:5601)
VECTOR_STORE=opensearch pnpm index:financebench
```

Un seul index par corpus porte le texte (BM25 natif) et le vecteur (`knn_vector` HNSW, cosinus,
moteur Lucene pour filtrer par document pendant la recherche k-NN). La fusion RRF reste dans
l'application : les adaptateurs mémoire et OpenSearch ne diffèrent que par l'index, et l'éval
mesure exactement cette différence.

## Observabilité

Avec `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, chaque question produit une trace Langfuse :
le span du pipeline, le retrieval, chaque appel LLM (vérificateur, boucle d'outils, juge) avec
ses tokens et sa latence, et les appels d'outils. Les runs d'éval sont tagués `eval`,
`financebench` et le mode.

Un `pnpm eval` complet et jugé tourne comme une experiment Langfuse par mode sur le dataset
`financebench` (onglet Experiments) : chaque question est une trace, avec les scores `correct`,
`hallucination`, `refusal`, `faithfulness`, `page_hit@k`, `evidence_seen`, `tool_calls`,
latences… La moyenne de `correct` par run est l'accuracy ; les IC95 sont des scores du run.
Les deux modes partagent toujours le même retrieval par question. `--no-langfuse` revient au
run local seul ; les runs partiels (`--max-items`, `--docs`, `--no-judge`) ne créent pas
d'experiment.

## Écarts connus avec le Python

- **Chunking des documents uploadés** : parents découpés récursivement, pas sémantiquement.
  FinanceBench n'est pas concerné (chunks exportés du Python).
- **Tokenisation BM25** identique au Python en mémoire (découpage sur les espaces) ; OpenSearch
  utilise son analyseur standard (minuscules), ce qui peut déplacer quelques rangs.
- **Réponse forcée après épuisement du budget d'outils** : Bedrock efface l'historique d'outils
  quand on retire les outils d'un appel. La réponse forcée repart donc d'un prompt qui contient
  tous les passages trouvés et le journal des outils, au lieu de la conversation brute.
- **Multi-query** non porté : il était configuré à 1 requête, donc inactif, côté Python.
- **Answer relevancy** (RAGAS) non portée dans l'éval.

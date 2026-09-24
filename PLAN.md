# agentic-rag-nest — plan de portage

Port TypeScript / NestJS de `../agentic-rag` (Python, Django, LangGraph, Mistral), sur la stack
de la mission PayFit : **Vercel AI SDK, Amazon Bedrock, Langfuse / OpenTelemetry, OpenSearch**,
Azure OpenAI en option.

Objectif mesurable : relancer l'éval FinanceBench (26 questions, 4 10-K) et comparer au
**83,3 % (20/24)** du projet Python. Le score n'est pas acquis : modèle, embeddings et reranker
changent. On isole les variables une par une (voir phase 6).

## État d'avancement (24 septembre 2026)

| Phase | État |
|---|---|
| 0. Comptes | **À faire de ton côté** (voir le tableau ci-dessous) |
| 1. Squelette Nest + Bedrock | Fait (`pnpm smoke` ; non testé sur Bedrock faute de clés AWS valides) |
| 2. Données exportées du Python | Fait (`scripts/export_from_python.py`) |
| 3. Retrieval | Fait (BM25 testé contre rank_bm25, RRF, parent-child, routage, rerank) |
| 4. Agent à outils | Fait, vérifié en réel avec Mistral (boucle d'outils, refus hors document) |
| 5. Langfuse / OpenTelemetry | Fait (intégration AI SDK v7) ; non testé sans clés Langfuse |
| 6. Éval + garde-fous | Fait ; runner validé sur 3 questions (stack Mistral, sans rerank) |
| 7. OpenSearch | Adaptateur + docker-compose écrits ; non lancé |
| 8. Vérificateur de pertinence | Fait (dans le pipeline par défaut, comme en Python) |
| 9. Azure OpenAI | Adaptateur prêt (`LLM_PROVIDER=azure`) ; non testé |
| 10. Ingestion serverless S3/SQS/Lambda | **Non fait** (sujet d'entretien, pas nécessaire pour la démo) |
| 11. Tests + CI | 31 tests Vitest, workflow GitHub Actions |
| Frontend | Repris du Python, textes de stack mis à jour, build OK |

---

## 0. Comptes et accès à créer (toi)

| # | Service | À faire | Variables produites |
|---|---|---|---|
| 1 | **AWS** | Tes clés locales (`~/.aws/credentials`, région `eu-west-3`) sont **invalides** (`InvalidClientTokenId`). Dans la console IAM : créer un utilisateur `agentic-rag-dev` avec la politique `AmazonBedrockFullAccess`, générer une clé d'accès, puis `aws configure`. Vérifier : `aws sts get-caller-identity`. | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| 2 | **AWS Budgets** | Budget mensuel de 20 $ avec alerte e-mail à 80 %. Indispensable avant de toucher à OpenSearch. | — |
| 3 | **Bedrock, accès modèles** | Console Bedrock → *Model access*, dans la région choisie (voir ci-dessous). Activer : un **Mistral Large**, un **Claude Sonnet** (formulaire de cas d'usage Anthropic), **Cohere Embed Multilingual** et/ou **Titan Text Embeddings V2**, **Cohere Rerank 3.5**. Faire ça **en premier** : l'activation peut prendre du temps. | IDs de modèles → `.env` |
| 4 | **Langfuse Cloud** | Région EU (`cloud.langfuse.com`). Tu as déjà un compte (closechat) : créer un nouveau projet `agentic-rag-nest`, générer les clés API. | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` |
| 5 | **GitHub** | Nouveau repo privé `agentic-rag-nest`. On ne pousse que sur ta demande. | — |
| 6 | **Docker Desktop** | Déjà installé (28.3). Sert à OpenSearch en local (phase 7). | — |
| 7 | **Vercel** | Compte existant. Utile seulement en phase 10 (frontend). | — |
| 8 | **Azure** *(optionnel, phase 9)* | Abonnement Azure → ressource Azure OpenAI (AI Foundry) → un déploiement de modèle. | `AZURE_RESOURCE_NAME`, `AZURE_API_KEY`, nom du déploiement |
| 9 | **Mistral** | Clé existante. Pas nécessaire au départ : l'OCR des 4 10-K est déjà en cache JSON. | `MISTRALAI_API_KEY` |

**Choix de région Bedrock.** Toutes les régions n'ont pas tous les modèles (Cohere Rerank en
particulier). Une fois les clés valides, lancer :

```bash
aws bedrock list-foundation-models --region eu-central-1 --query "modelSummaries[].modelId" --output text | tr '\t' '\n' | grep -Ei 'mistral|cohere|titan-embed|anthropic'
```

Même chose pour `eu-west-3` et `us-west-2`. Préférer une région EU (argument RGPD pour de la
paie), sinon `us-west-2` qui a le catalogue le plus large. Les modèles Claude récents passent par
des *inference profiles* (`eu.anthropic...`).

**Coûts attendus.** Embedding des ~12 400 chunks une fois : moins de 1 $. Un run d'éval complet :
de l'ordre de 0,5 $ (le run Python coûtait 0,35 $). OpenSearch : **en local via Docker**, pas de
domaine AWS managé tant que ce n'est pas utile (facturé à l'heure).

---

## 1. Squelette Nest et premier appel Bedrock

- `pnpm dlx @nestjs/cli new . --package-manager pnpm --strict`
- Dépendances : `ai`, `@ai-sdk/amazon-bedrock`, `zod`, `@nestjs/config`.
- `ConfigModule` avec validation Zod du `.env` (équivalent de `backend/config/settings.py`).
- Un script `pnpm smoke` : `generateText` sur Bedrock qui répond « pong ». **Premier jalon.**

## 2. Données : réutiliser ce que le Python a déjà produit

- Pages OCR : `../agentic-rag/evaluation/financebench/cache/*.ocr.json` sont **déjà en JSON**,
  lus tels quels par le TS. Pas de ré-OCR (économie de 4,4 $ et 20 min).
- Chunks : `*.chunks.pkl` sont des pickles LangChain. Petit script Python one-shot dans
  `agentic-rag` qui les exporte en `*.chunks.jsonl` (`page_content` + `metadata`).
  **Garder exactement le même chunking** : c'est ce qui rend la comparaison au 83,3 % honnête.
- Dataset : `dataset.jsonl` (26 questions) copié dans `eval/`.

## 3. Retrieval (module `retrieval`)

Architecture en ports/adaptateurs (ta Clean Archi) :

- Port `Retriever` : `search(query, scope?) → Passage[]`.
- Port `Embedder` → adaptateur Bedrock (`embedMany`), vecteurs mis en cache sur disque
  (`data/embeddings/*.json`) pour ne pas ré-embedder à chaque run.
- Adaptateur **in-memory** d'abord : BM25 (implémentation maison ~60 lignes, ou lib) +
  cosinus, fusion **RRF 0,5/0,5**, k=20 de chaque côté (mêmes réglages que `settings.py`).
- Reranker : port `Reranker` → Cohere Rerank via Bedrock, top 30.
- Parent-child : les chunks enfants matchent, les parents (1 200 car.) remontent.
- Routage par document : port de la partie déterministe de `document_router.py`
  (nom d'entreprise / fichier → périmètre).
- Multi-query (1 reformulation, modèle small) : à porter en fin de phase, flag de config.
- `PageStore` : `grep`, `readPage`, `resolve` — port direct de `page_store.py`.

## 4. Agent (module `agent`)

- Outils AI SDK : `tool({ description, inputSchema: z.object(...), execute })` pour
  **`search`**, **`grep`**, **`read_page`**. Descriptions reprises **mot pour mot** du Python.
- Contexte numéroté : les 10 passages initiaux sont intouchables, les outils ajoutent
  après eux, chaque résultat affiche son `[n]`.
- Boucle : `generateText({ tools, stopWhen: stepCountIs(6) })` (5 appels d'outils max,
  comme `GENERATOR_MAX_TOOL_CALLS`). LangGraph n'est plus nécessaire.
- Prompts `RULES` et `TOOLS_GUIDE` de `research_agent.py` repris à l'identique
  (citations, calcul des ratios, « non disponible » seulement après un `grep` vide).
- Rapport de vérification (port de `build_verification_report`), sans appel LLM.
- Endpoint `POST /ask` (contrôleur Nest) ; streaming SSE en option.
- Résilience : timeouts, retries sur throttling Bedrock (équivalent de `resilience.py`).

**→ Jalon démo entretien : phases 0 à 5 + 3 ou 4 questions qui tournent de bout en bout.**

## 5. Observabilité

- `@opentelemetry/sdk-node` + span processor Langfuse, initialisé avant le bootstrap Nest.
- `experimental_telemetry: { isEnabled: true, functionId, metadata }` sur chaque appel AI SDK.
- Une trace par question : retrieval, rerank, chaque appel d'outil, génération, tokens, coût.

## 6. Évaluation (le cœur du sujet)

- Port de `run_financebench_eval.py` et `llm_judge.py` (même prompt de juge, même
  détection de refus/erreurs sans LLM, même protocole : erreurs techniques hors dénominateur).
- Modes `--tools` / `--no-tools` pour reproduire l'**ablation**.
- Comptage des tokens et du coût par run (port de `cost.py`).
- Garde-fou de régression (port de `regression.py`) : `pnpm eval:gate` sort en erreur si le
  score baisse. Branché en CI.
- **Isoler les variables**, une par run :
  1. Mistral Large sur Bedrock + même juge → effet du fournisseur seul.
  2. Changement d'embeddings (mistral-embed n'existe pas sur Bedrock) → effet des embeddings.
  3. Cohere Rerank 3.5 (Bedrock) au lieu de v4 Pro → effet du reranker.
  4. Claude Sonnet en générateur → effet du modèle.
- Option : pousser les résultats en *dataset / experiment* Langfuse.

## 7. OpenSearch (adaptateur `Retriever` n°2)

- `docker compose` : OpenSearch 2.x + Dashboards, sécurité désactivée en local.
- Index `chunks` : champ `knn_vector` (HNSW, cosinus) + champ texte BM25 + `doc_name`, `page`.
- Recherche hybride native : *search pipeline* avec `normalization-processor`.
- Index `pages` pour `grep` / `read_page`.
- Relancer l'éval : l'adaptateur ne doit pas faire perdre de points.

## 8. Vérificateur de pertinence et mode correctif *(optionnel)*

- Port de `relevance_checker.py` en `generateObject` avec un enum Zod
  `CAN_ANSWER | PARTIAL | NO_MATCH`, verdict transmis comme indice au générateur.
- Mode `SearchAgent` séparé : seulement si l'éval montre un gain.

## 9. Azure OpenAI *(optionnel)*

- Port `LlmProvider` → adaptateur `@ai-sdk/azure`, sélection par config.
- Un run d'éval comparatif Bedrock vs Azure : le changement de fournisseur est une ligne de config.

## 10. Ingestion serverless et déploiement

- Dépôt d'un PDF sur **S3** → événement → **SQS** → **Lambda** (OCR Mistral, chunking,
  embeddings, indexation OpenSearch). Retries et dead-letter queue.
- Variante **Step Functions** si le pipeline grossit (OCR par lots, reprise).
- API : Lambda (cohérent avec la stack PayFit) ou Railway en Docker (plus simple).
- Frontend : réutiliser `../agentic-rag/frontend` sur Vercel, pointé sur la nouvelle API.

## 11. Qualité

- Tests Vitest : port des tests hors-ligne (`test_page_store`, `test_retrieval_chain`,
  `test_llm_output_parsing`, `test_regression_gate`...) + fast-check pour les propriétés.
- GitHub Actions : lint, tests, garde-fou d'éval.
- README avec la table de comparaison Python vs TS (score, hallucinations, coût, latence).

---

## Correspondance Python → TS

| Python | TS / AWS |
|---|---|
| Django `views.py` | Contrôleur Nest |
| `pydantic-settings` | `@nestjs/config` + Zod |
| LangGraph `StateGraph` | Code async + `generateText` / `stopWhen` |
| `@tool` LangChain | `tool()` AI SDK + Zod |
| `ChatMistralAI` | `@ai-sdk/amazon-bedrock` (`@ai-sdk/azure` en option) |
| Mistral Embed | Cohere Embed / Titan V2 via Bedrock |
| Cohere Rerank v4 | Cohere Rerank 3.5 via Bedrock |
| Chroma + `rank_bm25` + RRF | In-memory, puis OpenSearch hybride |
| `PageStore` | `PageStore` TS, puis index OpenSearch `pages` |
| LangSmith | Langfuse via OpenTelemetry |
| pytest + Hypothesis | Vitest + fast-check |

## `.env` cible

```bash
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
BEDROCK_MODEL_ID=            # générateur (Mistral Large ou Claude)
BEDROCK_SMALL_MODEL_ID=      # sous-agents (routage, multi-query)
BEDROCK_EMBEDDING_MODEL_ID=
BEDROCK_RERANK_MODEL_ID=
JUDGE_MODEL_ID=              # constant d'un run à l'autre
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
OPENSEARCH_URL=http://localhost:9200
# Optionnel
AZURE_RESOURCE_NAME=
AZURE_API_KEY=
MISTRALAI_API_KEY=
```

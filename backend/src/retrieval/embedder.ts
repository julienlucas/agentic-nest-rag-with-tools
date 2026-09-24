/**
 * Embeddings avec cache disque : on n'embedde jamais deux fois le même corpus avec le même
 * modèle. Format : data/embeddings/<modèle>/<corpus>.bin (Float32 à la suite) + .json
 * (dimension et empreinte de chaque texte, pour détecter un corpus qui a changé).
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { embed, embedMany } from "ai";
import { env } from "../config/settings";
import { contentKey } from "../core/passage";
import { getLogger } from "../core/logger";
import { embeddingModel, embeddingProviderOptions, modelSlug } from "../llm/providers";
import { withBackoff } from "../llm/resilience";
import { usageMeter } from "../llm/usage";

const logger = getLogger("Embedder");
const SLICE = 512; // textes par sauvegarde intermédiaire

type CacheMeta = { model: string; dim: number; hashes: string[] };

export class Embedder {
  private readonly queryCache = new Map<string, number[]>();
  readonly slug = modelSlug(env.EMBEDDING_PROVIDER, env.EMBEDDING_MODEL_ID);

  constructor(private readonly cacheDir = path.join(env.DATA_DIR, "embeddings")) {}

  private paths(corpusKey: string) {
    const dir = path.join(this.cacheDir, this.slug);
    const safe = corpusKey.replace(/[^a-zA-Z0-9._-]+/g, "_");
    return { dir, bin: path.join(dir, `${safe}.bin`), meta: path.join(dir, `${safe}.json`) };
  }

  private load(corpusKey: string): Map<string, Float32Array> {
    const p = this.paths(corpusKey);
    const out = new Map<string, Float32Array>();
    if (!fs.existsSync(p.meta) || !fs.existsSync(p.bin)) return out;
    const meta = JSON.parse(fs.readFileSync(p.meta, "utf-8")) as CacheMeta;
    const buf = fs.readFileSync(p.bin);
    const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    meta.hashes.forEach((h, i) => out.set(h, all.slice(i * meta.dim, (i + 1) * meta.dim)));
    return out;
  }

  private save(corpusKey: string, vectors: Map<string, Float32Array>) {
    const p = this.paths(corpusKey);
    fs.mkdirSync(p.dir, { recursive: true });
    const hashes = [...vectors.keys()];
    const dim = hashes.length ? vectors.get(hashes[0])!.length : 0;
    const all = new Float32Array(hashes.length * dim);
    hashes.forEach((h, i) => all.set(vectors.get(h)!, i * dim));
    fs.writeFileSync(p.bin, Buffer.from(all.buffer));
    fs.writeFileSync(p.meta, JSON.stringify({ model: this.slug, dim, hashes } satisfies CacheMeta));
  }

  /** Vecteurs des `texts`, dans le même ordre ; seuls les textes absents du cache sont embeddés. */
  async embedCorpus(corpusKey: string, texts: string[]): Promise<Float32Array[]> {
    const cached = this.load(corpusKey);
    const hashes = texts.map(contentKey);
    const missing = [...new Set(hashes.filter((h) => !cached.has(h)))];
    if (missing.length) {
      const byHash = new Map(texts.map((t, i) => [hashes[i], t]));
      logger.log(`${corpusKey}: ${missing.length}/${texts.length} textes à embedder (${this.slug})`);
      const model = embeddingModel();
      for (let i = 0; i < missing.length; i += SLICE) {
        const slice = missing.slice(i, i + SLICE);
        const result = await withBackoff(
          () =>
            embedMany({
              model,
              values: slice.map((h) => byHash.get(h)!),
              maxParallelCalls: 8,
              providerOptions: embeddingProviderOptions("document"),
            }),
          `embeddings ${corpusKey}`,
        );
        usageMeter.recordEmbedding(this.slug, result.usage?.tokens);
        slice.forEach((h, j) => cached.set(h, Float32Array.from(result.embeddings[j])));
        this.save(corpusKey, cached);
        logger.log(`${corpusKey}: ${Math.min(i + SLICE, missing.length)}/${missing.length}`);
      }
    }
    return hashes.map((h) => cached.get(h)!);
  }

  async embedQuery(text: string): Promise<number[]> {
    const hit = this.queryCache.get(text);
    if (hit) return hit;
    const result = await withBackoff(
      () =>
        embed({
          model: embeddingModel(),
          value: text,
          providerOptions: embeddingProviderOptions("query"),
        }),
      "embedding requête",
    );
    usageMeter.recordEmbedding(this.slug, result.usage?.tokens);
    if (this.queryCache.size > 2000) this.queryCache.clear();
    this.queryCache.set(text, result.embedding);
    return result.embedding;
  }
}

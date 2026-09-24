/**
 * Fichier -> pages + chunks, avec cache par contenu (sha256) dans data/document_cache/.
 * Les documents d'exemple déjà OCRisés par le projet Python y sont exportés
 * (scripts/export_from_python.py) : ils se chargent sans OCR.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { env, settings } from "../config/settings";
import { sha256, type Passage } from "../core/passage";
import { getLogger } from "../core/logger";
import { chunkPages } from "./chunker";
import { extractPages } from "./ocr";

const logger = getLogger("DocumentProcessor");

export type ProcessedDocument = { hash: string; source: string; pages: string[]; chunks: Passage[] };

export class DocumentProcessor {
  constructor(private readonly cacheDir = path.join(env.DATA_DIR, "document_cache")) {}

  async process(fileName: string, content: Buffer): Promise<ProcessedDocument> {
    if (content.length > settings.MAX_FILE_BYTES) {
      throw new Error(`Fichier trop volumineux (max ${settings.MAX_FILE_BYTES / 1024 / 1024} Mo)`);
    }
    const source = path.basename(fileName);
    const hash = sha256(content);
    const cachePath = path.join(this.cacheDir, `${hash}.json`);
    if (fs.existsSync(cachePath)) {
      const cached = JSON.parse(fs.readFileSync(cachePath, "utf-8")) as { pages: string[]; chunks: Passage[] };
      logger.log(`${source} : chargé depuis le cache (${cached.chunks.length} chunks)`);
      // La source suit le nom sous lequel le fichier est chargé (routage et citations).
      const chunks = cached.chunks.map((c) => ({ ...c, metadata: { ...c.metadata, source } }));
      return { hash, source, pages: cached.pages, chunks };
    }

    const pages = await extractPages(source, content);
    const chunks = chunkPages(pages, source);
    if (!chunks.length) throw new Error(`Aucun texte extrait de ${source}`);
    fs.mkdirSync(this.cacheDir, { recursive: true });
    fs.writeFileSync(cachePath, JSON.stringify({ source, pages, chunks }));
    logger.log(`${source} : ${pages.length} pages, ${chunks.length} chunks`);
    return { hash, source, pages, chunks };
  }
}

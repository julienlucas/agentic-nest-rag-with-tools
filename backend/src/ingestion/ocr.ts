/**
 * Extraction du texte page par page.
 *
 * - PDF avec MISTRALAI_API_KEY : Mistral OCR (markdown, tableaux compris), comme le Python.
 * - PDF sans clé : extraction du texte natif (unpdf). Suffisant pour un PDF texte, pas pour
 *   un scan, et les tableaux perdent leur structure.
 * - .txt / .md : le fichier est une seule « page ».
 */
import * as path from "node:path";
import { extractText, getDocumentProxy } from "unpdf";
import { env } from "../config/settings";
import { getLogger } from "../core/logger";

const logger = getLogger("Ocr");

export async function extractPages(fileName: string, content: Buffer): Promise<string[]> {
  const ext = path.extname(fileName).toLowerCase();
  if (ext === ".txt" || ext === ".md") return [content.toString("utf-8")];
  if (ext !== ".pdf") throw new Error(`Type de fichier non supporté: ${fileName}`);
  if (env.MISTRALAI_API_KEY) {
    try {
      return await mistralOcr(content);
    } catch (error) {
      logger.warn(`Mistral OCR indisponible (${(error as Error).message}), repli sur le texte natif du PDF`);
    }
  }
  return pdfText(content);
}

async function mistralOcr(content: Buffer): Promise<string[]> {
  const response = await fetch("https://api.mistral.ai/v1/ocr", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.MISTRALAI_API_KEY}` },
    body: JSON.stringify({
      model: "mistral-ocr-latest",
      document: { type: "document_url", document_url: `data:application/pdf;base64,${content.toString("base64")}` },
    }),
    signal: AbortSignal.timeout(300_000),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status} ${(await response.text()).slice(0, 200)}`);
  const data = (await response.json()) as { pages: { index: number; markdown: string }[] };
  logger.log(`Mistral OCR : ${data.pages.length} pages`);
  return [...data.pages].sort((a, b) => a.index - b.index).map((p) => p.markdown);
}

async function pdfText(content: Buffer): Promise<string[]> {
  const pdf = await getDocumentProxy(new Uint8Array(content));
  const { text } = await extractText(pdf, { mergePages: false });
  logger.log(`Texte natif du PDF : ${text.length} pages`);
  return text;
}

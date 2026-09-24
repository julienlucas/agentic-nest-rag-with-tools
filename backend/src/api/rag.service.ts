import { BadRequestException, Injectable } from "@nestjs/common";
import * as fs from "node:fs";
import * as path from "node:path";
import { env, settings } from "../config/settings";
import { AgentWorkflow, type PipelineResult } from "../agent/workflow";
import { DocumentProcessor } from "../ingestion/document-processor";
import { buildRetriever } from "../retrieval/index-builder";
import { PageStore } from "../retrieval/page-store";
import { SessionStore } from "./session.store";

@Injectable()
export class RagService {
  private readonly processor = new DocumentProcessor();
  private workflowInstance?: AgentWorkflow;

  constructor(private readonly sessions: SessionStore) {}

  // Instancié au premier usage : l'API démarre même si les clés du fournisseur manquent.
  private get workflow() {
    return (this.workflowInstance ??= new AgentWorkflow());
  }

  /** Charge un document d'exemple du dossier static/ (le nom vient du client : on le borne). */
  async loadExample(fileName: string, sessionId: string) {
    const dir = path.resolve(env.EXAMPLES_DIR);
    const filePath = path.resolve(dir, fileName || "");
    if (!fileName || filePath === dir || path.relative(dir, filePath).startsWith("..") || !fs.existsSync(filePath)) {
      throw new BadRequestException(`Fichier d'exemple inconnu: ${fileName}`);
    }
    const result = await this.index(path.basename(filePath), fs.readFileSync(filePath), sessionId);
    return { message: "Fichier chargé avec succès", chunks_count: result.chunks, filename: fileName };
  }

  async upload(fileName: string, content: Buffer, sessionId: string) {
    const ext = path.extname(fileName).toLowerCase();
    if (!(settings.ALLOWED_TYPES as readonly string[]).includes(ext)) {
      throw new BadRequestException(`Type de fichier non supporté: ${fileName}`);
    }
    const result = await this.index(fileName, content, sessionId);
    return { message: "Fichier traité avec succès", chunks_count: result.chunks };
  }

  private async index(fileName: string, content: Buffer, sessionId: string) {
    const doc = await this.processor.process(fileName, content);
    const retriever = await buildRetriever([{ key: `doc-${doc.hash}`, chunks: doc.chunks }], `doc-${doc.hash.slice(0, 16)}`);
    const pageStore = new PageStore({ [doc.source]: doc.pages });
    this.sessions.set(sessionId, { retriever, pageStore, files: [doc.source] });
    return { chunks: doc.chunks.length };
  }

  async ask(question: string, sessionId: string): Promise<PipelineResult> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new BadRequestException("Aucun document chargé. Veuillez d'abord charger un document.");
    if (!question?.trim()) throw new BadRequestException("Question vide.");
    return this.workflow.fullPipeline(question.trim(), session.retriever, session.pageStore, sessionId);
  }
}

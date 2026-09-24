/**
 * Même contrat HTTP que l'API Django : le frontend React est repris tel quel.
 *   POST /api/load-file         { file_name, session_id }
 *   POST /api/upload-file       multipart : file, session_id
 *   POST /api/process-question  { question, session_id }
 */
import {
  BadRequestException,
  Body,
  Controller,
  Get,
  HttpCode,
  Post,
  UploadedFile,
  UseInterceptors,
} from "@nestjs/common";
import { FileInterceptor } from "@nestjs/platform-express";
import { z } from "zod";
import { env, settings } from "../config/settings";
import { RagService } from "./rag.service";

const loadFileBody = z.object({ file_name: z.string().min(1), session_id: z.string().default("default") });
const questionBody = z.object({ question: z.string().min(1).max(4000), session_id: z.string().default("default") });
const uploadBody = z.object({ session_id: z.string().default("default") });

function parse<T>(schema: z.ZodType<T>, body: unknown): T {
  const r = schema.safeParse(body ?? {});
  if (!r.success) throw new BadRequestException(r.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; "));
  return r.data;
}

@Controller("api")
export class RagController {
  constructor(private readonly rag: RagService) {}

  @Get("health")
  health() {
    return {
      status: "ok",
      llm: `${env.LLM_PROVIDER}:${env.MODEL_ID}`,
      embeddings: `${env.EMBEDDING_PROVIDER}:${env.EMBEDDING_MODEL_ID}`,
      rerank: env.RERANK_ENABLED ? `${env.RERANK_PROVIDER}:${env.RERANK_MODEL_ID}` : "off",
      vectorStore: env.VECTOR_STORE,
    };
  }

  @Post("load-file")
  @HttpCode(200)
  loadFile(@Body() body: unknown) {
    const { file_name, session_id } = parse(loadFileBody, body);
    return this.rag.loadExample(file_name, session_id);
  }

  @Post("upload-file")
  @HttpCode(200)
  @UseInterceptors(FileInterceptor("file", { limits: { fileSize: settings.MAX_FILE_BYTES } }))
  uploadFile(@UploadedFile() file: Express.Multer.File | undefined, @Body() body: unknown) {
    if (!file) throw new BadRequestException("Aucun fichier reçu (champ 'file').");
    const { session_id } = parse(uploadBody, body);
    return this.rag.upload(file.originalname, file.buffer, session_id);
  }

  @Post("process-question")
  @HttpCode(200)
  processQuestion(@Body() body: unknown) {
    const { question, session_id } = parse(questionBody, body);
    return this.rag.ask(question, session_id);
  }
}

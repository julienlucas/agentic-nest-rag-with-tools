// L'instrumentation doit être chargée avant tout module qui appelle l'AI SDK.
import { startTelemetry, shutdownTelemetry } from "./observability/instrumentation";
const tracing = startTelemetry();

import "reflect-metadata";
import { Logger } from "@nestjs/common";
import { NestFactory } from "@nestjs/core";
import { AppModule } from "./app.module";
import { ErrorFilter } from "./api/error.filter";
import { env } from "./config/settings";

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.useGlobalFilters(new ErrorFilter());
  app.enableCors({ origin: env.CORS_ORIGINS.split(",").map((o) => o.trim()) });
  app.enableShutdownHooks();
  process.on("SIGTERM", () => void shutdownTelemetry());
  await app.listen(env.PORT);
  const logger = new Logger("Bootstrap");
  logger.log(`API sur http://localhost:${env.PORT} — LLM ${env.LLM_PROVIDER}:${env.MODEL_ID}, index ${env.VECTOR_STORE}`);
  logger.log(tracing ? "Traces Langfuse actives" : "Langfuse non configuré (traces désactivées)");
}

void bootstrap();

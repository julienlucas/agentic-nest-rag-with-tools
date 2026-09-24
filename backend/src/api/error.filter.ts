import { ArgumentsHost, Catch, ExceptionFilter, HttpException, Logger } from "@nestjs/common";
import type { Response } from "express";

/** Erreurs au format { error } attendu par le frontend (comme l'API Django). */
@Catch()
export class ErrorFilter implements ExceptionFilter {
  private readonly logger = new Logger("Api");

  catch(exception: unknown, host: ArgumentsHost) {
    const res = host.switchToHttp().getResponse<Response>();
    if (exception instanceof HttpException) {
      const body = exception.getResponse();
      const message = typeof body === "string" ? body : ((body as { message?: string | string[] }).message ?? exception.message);
      res.status(exception.getStatus()).json({ error: Array.isArray(message) ? message.join("; ") : message });
      return;
    }
    this.logger.error((exception as Error)?.stack ?? String(exception));
    res.status(500).json({ error: (exception as Error)?.message ?? "Erreur interne" });
  }
}

import { Logger } from "@nestjs/common";

/** Logger Nest utilisable aussi hors du conteneur (scripts d'éval). */
export function getLogger(context: string): Logger {
  return new Logger(context);
}

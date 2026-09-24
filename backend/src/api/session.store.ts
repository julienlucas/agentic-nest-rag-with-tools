import { Injectable } from "@nestjs/common";
import type { PageStore } from "../retrieval/page-store";
import type { Retriever } from "../retrieval/retriever";

export type Session = { retriever: Retriever; pageStore: PageStore; files: string[]; touchedAt: number };

/**
 * Sessions en mémoire (comme le Python). En production : Redis pour l'état, et l'index dans
 * OpenSearch plutôt qu'en mémoire du process.
 */
@Injectable()
export class SessionStore {
  private readonly sessions = new Map<string, Session>();
  private static readonly MAX_SESSIONS = 50;

  get(id: string): Session | undefined {
    const s = this.sessions.get(id);
    if (s) s.touchedAt = Date.now();
    return s;
  }

  set(id: string, session: Omit<Session, "touchedAt">) {
    this.sessions.set(id, { ...session, touchedAt: Date.now() });
    if (this.sessions.size > SessionStore.MAX_SESSIONS) {
      const oldest = [...this.sessions].sort((a, b) => a[1].touchedAt - b[1].touchedAt)[0];
      this.sessions.delete(oldest[0]);
    }
  }
}

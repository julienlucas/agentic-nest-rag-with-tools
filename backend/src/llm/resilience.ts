/**
 * Rate limits : ils doivent remonter à l'appelant pour être rejoués avec backoff,
 * jamais être avalés en réponse figée (une question perdue fausse l'éval).
 */
import { getLogger } from "../core/logger";

const logger = getLogger("Resilience");

export function isRateLimit(error: unknown): boolean {
  const e = error as { statusCode?: number; status?: number; name?: string; message?: string; cause?: unknown };
  if (!e) return false;
  if (e.statusCode === 429 || e.status === 429) return true;
  const text = `${e.name ?? ""} ${e.message ?? ""}`.toLowerCase();
  if (/throttl|rate.?limit|too many requests|429/.test(text)) return true;
  // AI SDK : RetryError enveloppe la dernière erreur
  const last = (e as { lastError?: unknown }).lastError;
  if (last && last !== error) return isRateLimit(last);
  if (e.cause && e.cause !== error) return isRateLimit(e.cause);
  return false;
}

export function rootCause(error: unknown): string {
  let e: any = error;
  const seen = new Set<unknown>();
  while (e && !seen.has(e)) {
    seen.add(e);
    const next = e.lastError ?? e.cause;
    if (!next) break;
    e = next;
  }
  return `${e?.name ?? "Error"}: ${e?.message ?? String(e)}`.slice(0, 500);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Rejoue `fn` sur rate limit, avec backoff exponentiel (2, 4, 8, 16, 32 s). */
export async function withBackoff<T>(fn: () => Promise<T>, what: string, attempts = 5): Promise<T> {
  for (let i = 0; ; i++) {
    try {
      return await fn();
    } catch (error) {
      if (!isRateLimit(error) || i >= attempts - 1) throw error;
      const wait = 2000 * 2 ** i;
      logger.warn(`Rate limit sur ${what}, nouvel essai dans ${wait / 1000}s`);
      await sleep(wait);
    }
  }
}

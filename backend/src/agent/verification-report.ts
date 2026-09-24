/**
 * Rapport de vérification construit à partir des signaux réels du pipeline, sans appel LLM.
 * Le frontend convertit **gras** et *italique*. Port de build_verification_report.
 */
import { passageLabel, type Passage } from "../core/passage";
import type { Relevance } from "./prompts";

const REL_LABELS: Record<Relevance, string> = {
  CAN_ANSWER: "les passages récupérés permettent de répondre à la question",
  PARTIAL: "couverture partielle — la réponse peut être incomplète",
  NO_MATCH: "aucun passage pertinent trouvé dans les documents",
};

export function buildVerificationReport(state: {
  relevance: Relevance | null;
  passages: Passage[];
  toolCalls: number;
  calls: string[];
}): string {
  const lines: string[] = [];
  const { relevance } = state;
  lines.push(`**Pertinent:** ${relevance === "NO_MATCH" ? "Non" : "Oui"}`);
  if (relevance) lines.push(`**Pertinence des passages:** ${relevance} — ${REL_LABELS[relevance]}`);

  const scores = state.passages
    .map((p) => p.metadata.rerankScore)
    .filter((s): s is number => typeof s === "number");
  if (scores.length) {
    const top = Math.max(...scores);
    const level = top >= 0.7 ? "élevée" : top >= 0.4 ? "moyenne" : "faible";
    lines.push(`**Confiance retrieval (reranker):** ${top.toFixed(2)} — ${level}`);
  }

  if (state.toolCalls) {
    const n = state.toolCalls;
    lines.push(`**Recherche corrective:** déclenchée (${n} appel${n > 1 ? "s" : ""} d'outils)`);
    for (const q of state.calls.slice(0, 5)) lines.push(`  • *${q}*`);
  } else {
    lines.push("**Recherche corrective:** non nécessaire");
  }

  const bySource = new Map<string, Set<number>>();
  for (const p of state.passages) {
    const name = passageLabel(p);
    if (!bySource.has(name)) bySource.set(name, new Set());
    if (p.metadata.page !== null && p.metadata.page !== undefined) bySource.get(name)!.add(p.metadata.page);
  }
  if (bySource.size) {
    const parts = [...bySource].map(([name, pages]) => {
      if (!pages.size) return name;
      const sorted = [...pages].sort((a, b) => a - b);
      const shown = sorted.slice(0, 6).map((p) => p + 1).join(", ");
      return `${name} (p. ${shown}${sorted.length > 6 ? "…" : ""})`;
    });
    lines.push(`**Sources utilisées:** ${parts.join(" · ")} — ${state.passages.length} passages transmis au modèle`);
  }
  return lines.join("\n");
}

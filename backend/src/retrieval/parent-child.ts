/**
 * Petits chunks pour matcher, gros chunks pour répondre : les enfants trouvés sont remplacés
 * par leur parent, trié par nombre d'enfants retrouvés (port de parent_child_retriever.py).
 */
import type { Passage } from "../core/passage";

export function toParents(children: Passage[]): Passage[] {
  const parents = new Map<string, Passage>();
  const orphans: Passage[] = [];
  for (const child of children) {
    const { parentId, parentContent, ...rest } = child.metadata;
    if (parentId && parentContent) {
      const existing = parents.get(parentId);
      if (existing) {
        existing.metadata.matchedChildren = (existing.metadata.matchedChildren ?? 1) + 1;
      } else {
        parents.set(parentId, {
          content: parentContent,
          metadata: { ...rest, parentId, matchedChildren: 1 },
        });
      }
    } else {
      orphans.push(child);
    }
  }
  // Tri stable : à égalité, l'ordre de la fusion RRF est conservé.
  const sorted = [...parents.values()]
    .map((p, order) => ({ p, order }))
    .sort((a, b) => (b.p.metadata.matchedChildren ?? 0) - (a.p.metadata.matchedChildren ?? 0) || a.order - b.order)
    .map(({ p }) => p);
  return [...sorted, ...orphans];
}

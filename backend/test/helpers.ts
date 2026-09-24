import { MockLanguageModelV4 } from "ai/test";
import type { Passage } from "../src/core/passage";

export const usage = (input = 10, output = 5) => ({
  inputTokens: { total: input, noCache: input, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: output, text: output, reasoning: 0 },
});

export const text = (t: string) => ({
  content: [{ type: "text" as const, text: t }],
  finishReason: { unified: "stop" as const, raw: "stop" },
  usage: usage(),
  warnings: [],
});

export const toolCall = (toolName: string, input: object, id = `call-${toolName}`) => ({
  content: [{ type: "tool-call" as const, toolCallId: id, toolName, input: JSON.stringify(input) }],
  finishReason: { unified: "tool-calls" as const, raw: "tool_use" },
  usage: usage(),
  warnings: [],
});

/** Modèle factice qui rejoue une séquence de réponses. */
export function scriptedModel(responses: ReturnType<typeof text | typeof toolCall>[]) {
  let i = 0;
  return new MockLanguageModelV4({
    modelId: "mock",
    doGenerate: async () => responses[Math.min(i++, responses.length - 1)] as never,
  });
}

export const passage = (content: string, source = "DOC_A", page: number | null = 0, extra: Partial<Passage["metadata"]> = {}): Passage => ({
  content,
  metadata: { source, docName: source, page, ...extra },
});

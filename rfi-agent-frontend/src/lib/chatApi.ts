type SendChatInput = {
  message: string;
  sessionId: string;
};

type SendBatchInput = {
  questions: string[];
  sessionId: string;
  prefix?: string;
};

type SendChatResult = {
  reply: string;
  sessionId: string;
  transport: "direct-webhook";
};

export type BatchReply = {
  index: number;
  question: string;
  answer: string;
};

type SendBatchResult = {
  replies: BatchReply[];
  sessionId: string;
  transport: "direct-webhook-batch";
};

const DEFAULT_DIRECT_WEBHOOK =
  "https://coe-n8n.coe-untrust-eu-de.prod.tddrift.net/webhook/6b55959d-96d7-46c5-be2e-11fa3736e1c3";

function resolveConfig() {
  return {
    directWebhookUrl: String(
      import.meta.env.VITE_RFI_DIRECT_WEBHOOK_URL || DEFAULT_DIRECT_WEBHOOK,
    ).trim(),
  };
}

async function parseResponsePayload(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  const text = await response.text();
  if (!text.trim()) return "";
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function postJson(url: string, body: Record<string, unknown>) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const payload = await parseResponsePayload(response);
  if (!response.ok) {
    throw new Error(
      `HTTP ${response.status} from ${url}: ${extractAssistantText(payload) || "No details"}`,
    );
  }

  return payload;
}

function extractFromObject(record: Record<string, unknown>): string | null {
  const directCandidates = ["reply", "response", "output", "answer", "text", "message"];
  for (const key of directCandidates) {
    const value = record[key];
    const extracted = extractAssistantText(value);
    if (extracted) return extracted;
  }

  const nestedCandidates = ["data", "result", "body", "content", "messages"];
  for (const key of nestedCandidates) {
    const value = record[key];
    const extracted = extractAssistantText(value);
    if (extracted) return extracted;
  }

  return null;
}

function extractAssistantText(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (value == null) return "";

  if (Array.isArray(value)) {
    const parts = value.map((entry) => extractAssistantText(entry)).filter(Boolean);
    return parts.join("\n").trim();
  }

  if (typeof value === "object") {
    const direct = extractFromObject(value as Record<string, unknown>);
    if (direct) return direct.trim();
    return JSON.stringify(value);
  }

  return String(value);
}

async function sendDirect(input: SendChatInput): Promise<SendChatResult> {
  const config = resolveConfig();

  try {
    const payload = await postJson(config.directWebhookUrl, {
      chatInput: input.message,
      message: input.message,
      text: input.message,
      sessionId: input.sessionId,
    });

    return {
      reply: extractAssistantText(payload),
      sessionId: input.sessionId,
      transport: "direct-webhook",
    };
  } catch (error) {
    throw new Error(error instanceof Error ? error.message : "direct-webhook failed");
  }
}

export async function sendMessageToRfiAgent(input: SendChatInput): Promise<SendChatResult> {
  return sendDirect(input);
}

function normalizeQuestions(questions: string[]) {
  return questions
    .map((question) => question.trim())
    .filter(Boolean)
    .map((question) => question.replace(/^\d+[\)\.\:\-]\s*/, "").trim());
}

export async function sendBatchToRfiAgent(input: SendBatchInput): Promise<SendBatchResult> {
  const normalizedQuestions = normalizeQuestions(input.questions);
  if (normalizedQuestions.length === 0) {
    throw new Error("No questions supplied for batch submission.");
  }

  const maxConcurrency = Math.min(3, normalizedQuestions.length);
  const replies: BatchReply[] = new Array(normalizedQuestions.length);
  let nextIndex = 0;

  const workers = Array.from({ length: maxConcurrency }, async () => {
    while (true) {
      const current = nextIndex;
      nextIndex += 1;
      if (current >= normalizedQuestions.length) break;

      const question = normalizedQuestions[current];
      const prompt = input.prefix
        ? `${input.prefix.trim()}\n\nQuestion:\n${question}`
        : question;

      try {
        const result = await sendDirect({
          sessionId: input.sessionId,
          message: prompt,
        });
        replies[current] = {
          index: current + 1,
          question,
          answer: result.reply || "No response text returned.",
        };
      } catch (error) {
        replies[current] = {
          index: current + 1,
          question,
          answer:
            error instanceof Error
              ? `Request failed: ${error.message}`
              : "Request failed: unknown error",
        };
      }
    }
  });

  await Promise.all(workers);

  return {
    replies,
    sessionId: input.sessionId,
    transport: "direct-webhook-batch",
  };
}

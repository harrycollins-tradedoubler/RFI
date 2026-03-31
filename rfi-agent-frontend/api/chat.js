const DEFAULT_DIRECT_WEBHOOK =
  "https://coe-n8n.coe-untrust-eu-de.prod.tddrift.net/webhook/6b55959d-96d7-46c5-be2e-11fa3736e1c3";

function extractFromObject(record) {
  const directCandidates = ["reply", "response", "output", "answer", "text", "message"];
  for (const key of directCandidates) {
    const value = extractAssistantText(record[key]);
    if (value) return value;
  }

  const nestedCandidates = ["data", "result", "body", "content", "messages"];
  for (const key of nestedCandidates) {
    const value = extractAssistantText(record[key]);
    if (value) return value;
  }

  return null;
}

function extractAssistantText(value) {
  if (typeof value === "string") return value.trim();
  if (value == null) return "";

  if (Array.isArray(value)) {
    return value.map((entry) => extractAssistantText(entry)).filter(Boolean).join("\n").trim();
  }

  if (typeof value === "object") {
    const direct = extractFromObject(value);
    if (direct) return direct;
    try {
      return JSON.stringify(value);
    } catch {
      return "";
    }
  }

  return String(value);
}

async function parseResponsePayload(response) {
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

async function postJson(url, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify(body),
    });
  } finally {
    clearTimeout(timeout);
  }

  const payload = await parseResponsePayload(response);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${extractAssistantText(payload) || "No details"}`);
  }
  return payload;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
  const message = String(body.message || "").trim();
  const sessionId = String(body.sessionId || `rfi-ui-${Date.now()}`);

  if (!message) {
    res.status(400).json({ error: "Missing message" });
    return;
  }

  const directWebhookUrl = String(
    process.env.RFI_DIRECT_WEBHOOK_URL || DEFAULT_DIRECT_WEBHOOK,
  ).trim();

  try {
    const payload = await postJson(directWebhookUrl, {
      chatInput: message,
      message,
      text: message,
      sessionId,
    });

    res.status(200).json({
      reply: extractAssistantText(payload),
      sessionId,
      transport: "direct-webhook",
    });
    return;
  } catch (error) {
    const detail = error instanceof Error ? error.message : "direct-webhook failed";
    console.error("[api/chat] direct-webhook request failed", detail);
    res.status(502).json({
      error: "Failed to reach RFI webhook",
      detail,
    });
    return;
  }
}

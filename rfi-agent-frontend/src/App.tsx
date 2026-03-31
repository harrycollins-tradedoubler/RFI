import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bot, FileUp, MessageSquare, Send, Sparkles } from "lucide-react";
import { BatchReply, sendBatchToRfiAgent, sendMessageToRfiAgent } from "./lib/chatApi";

type MessageRole = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
};

type InputMode = "chat" | "file";

const QUICK_PROMPTS = [
  "Do you enforce MFA for all privileged/admin accounts?",
  "Is customer data encrypted in transit and at rest?",
  "What is your incident response SLA for notifying clients after a confirmed breach?",
  "Do you restrict personal data storage/processing to EU or approved countries?",
  "How often are user access rights reviewed and recertified?",
];

const SESSION_KEY = "rfi_agent_session_id";
const MAX_BATCH_QUESTIONS = 25;
const SUPPORTED_FILE_EXTENSIONS = [".txt", ".md", ".csv", ".json", ".docx", ".xlsx", ".xls"];

function createId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function getInitialSessionId() {
  try {
    const fromStorage = localStorage.getItem(SESSION_KEY);
    if (fromStorage) return fromStorage;
    const generated = `rfi-ui-${createId()}`;
    localStorage.setItem(SESSION_KEY, generated);
    return generated;
  } catch {
    return `rfi-ui-${createId()}`;
  }
}

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function normalizeQuestion(line: string) {
  return line.replace(/^\s*(?:\d+[\)\.\:\-]\s*|[-*]\s+)/, "").trim();
}

function dedupeQuestions(questions: string[]) {
  const seen = new Set<string>();
  const output: string[] = [];
  for (const question of questions) {
    const clean = normalizeQuestion(question);
    if (!clean) continue;
    const key = clean.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    output.push(clean);
    if (output.length >= MAX_BATCH_QUESTIONS) break;
  }
  return output;
}

function extractQuestionsFromText(rawInput: string) {
  const input = rawInput.trim();
  if (!input) return [];

  const lines = input
    .split(/\r?\n/)
    .map((line) => normalizeQuestion(line))
    .filter(Boolean);

  if (lines.length > 1) {
    return dedupeQuestions(lines);
  }

  const splitByQuestionMark = input
    .split(/(?<=\?)\s+/)
    .map((entry) => normalizeQuestion(entry))
    .filter(Boolean);

  if (splitByQuestionMark.length > 1) {
    return dedupeQuestions(splitByQuestionMark);
  }

  return dedupeQuestions([input]);
}

function getFileExtension(fileName: string) {
  const lastDot = fileName.lastIndexOf(".");
  if (lastDot === -1) return "";
  return fileName.slice(lastDot).toLowerCase();
}

function stripCsvCell(cell: string) {
  return cell.trim().replace(/^"(.*)"$/, "$1").trim();
}

function parseQuestionsFromCsv(text: string) {
  const rows = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (rows.length === 0) return [];

  const firstRowLower = rows[0].toLowerCase();
  const hasHeader =
    firstRowLower.includes("question") || firstRowLower.includes("rfi") || firstRowLower.includes("query");

  const contentRows = hasHeader ? rows.slice(1) : rows;
  const candidates = contentRows.map((row) => {
    const cells = row.split(",").map(stripCsvCell).filter(Boolean);
    if (cells.length === 0) return "";
    const withQuestionMark = cells.find((cell) => cell.includes("?"));
    return withQuestionMark || cells[0];
  });
  return dedupeQuestions(candidates);
}

function parseQuestionsFromRows(rows: string[][]) {
  if (rows.length === 0) return [];

  const normalizedRows = rows
    .map((row) => row.map((cell) => String(cell ?? "").trim()))
    .filter((row) => row.some(Boolean));
  if (normalizedRows.length === 0) return [];

  const header = normalizedRows[0].map((cell) => cell.toLowerCase());
  const questionColumn = header.findIndex(
    (cell) =>
      cell.includes("question") ||
      cell.includes("rfi") ||
      cell.includes("query") ||
      cell.includes("prompt"),
  );

  const candidates = normalizedRows.map((row, rowIndex) => {
    if (questionColumn >= 0) {
      if (rowIndex === 0) return "";
      return row[questionColumn] || "";
    }

    const cells = row.filter(Boolean);
    if (cells.length === 0) return "";
    const withQuestionMark = cells.find((cell) => cell.includes("?"));
    return withQuestionMark || cells[0];
  });

  return dedupeQuestions(candidates);
}

function parseQuestionsFromJson(text: string) {
  const candidates: string[] = [];
  const parsed = JSON.parse(text);

  if (Array.isArray(parsed)) {
    for (const entry of parsed) {
      if (typeof entry === "string") {
        candidates.push(entry);
        continue;
      }
      if (entry && typeof entry === "object") {
        const record = entry as Record<string, unknown>;
        const value =
          record.question ||
          record.questions ||
          record.query ||
          record.prompt ||
          record.text;
        if (typeof value === "string") {
          candidates.push(value);
        } else if (Array.isArray(value)) {
          for (const nested of value) {
            if (typeof nested === "string") candidates.push(nested);
          }
        }
      }
    }
  } else if (parsed && typeof parsed === "object") {
    const record = parsed as Record<string, unknown>;
    const value = record.questions || record.question || record.items || record.data;
    if (typeof value === "string") {
      candidates.push(value);
    } else if (Array.isArray(value)) {
      for (const entry of value) {
        if (typeof entry === "string") {
          candidates.push(entry);
          continue;
        }
        if (entry && typeof entry === "object") {
          const item = entry as Record<string, unknown>;
          if (typeof item.question === "string") candidates.push(item.question);
          else if (typeof item.text === "string") candidates.push(item.text);
        }
      }
    }
  }

  return dedupeQuestions(candidates);
}

async function parseQuestionsFromDocx(file: File) {
  const mammoth = await import("mammoth/mammoth.browser");
  const arrayBuffer = await file.arrayBuffer();
  const extracted = await mammoth.extractRawText({ arrayBuffer });
  return extractQuestionsFromText(extracted.value || "");
}

async function parseQuestionsFromExcel(file: File) {
  const XLSX = await import("xlsx");
  const arrayBuffer = await file.arrayBuffer();
  const workbook = XLSX.read(arrayBuffer, { type: "array" });

  const candidates: string[] = [];
  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];
    if (!sheet) continue;
    const rows = XLSX.utils.sheet_to_json<(string | number | boolean | null)[]>(sheet, {
      header: 1,
      raw: false,
      defval: "",
    });
    const parsed = parseQuestionsFromRows(
      rows.map((row) => row.map((cell) => String(cell ?? ""))),
    );
    candidates.push(...parsed);
  }

  return dedupeQuestions(candidates);
}

async function extractQuestionsFromFile(file: File) {
  const extension = getFileExtension(file.name);
  if (!SUPPORTED_FILE_EXTENSIONS.includes(extension)) {
    throw new Error(
      `Unsupported file type (${extension || "none"}). Use ${SUPPORTED_FILE_EXTENSIONS.join(", ")}.`,
    );
  }

  if (extension === ".docx") {
    try {
      return await parseQuestionsFromDocx(file);
    } catch {
      throw new Error("Could not parse questions from DOCX. Ensure the document contains readable text.");
    }
  }

  if (extension === ".xlsx" || extension === ".xls") {
    try {
      return await parseQuestionsFromExcel(file);
    } catch {
      throw new Error("Could not parse questions from Excel. Ensure the file has question text in rows.");
    }
  }

  const text = await file.text();
  if (!text.trim()) {
    throw new Error("The uploaded file is empty.");
  }

  if (extension === ".csv") {
    return parseQuestionsFromCsv(text);
  }

  if (extension === ".json") {
    try {
      return parseQuestionsFromJson(text);
    } catch {
      throw new Error("Could not parse questions from JSON. Expected string array or objects with question/text.");
    }
  }

  return extractQuestionsFromText(text);
}

function formatBatchAssistantReply(replies: BatchReply[]) {
  const completed = replies.filter((reply) => !reply.answer.toLowerCase().startsWith("request failed")).length;
  const lines = replies.map(
    (reply) => `${reply.index}. Question: ${reply.question}\nAnswer: ${reply.answer}`,
  );
  return `Processed ${replies.length} questions (${completed} successful).\n\n${lines.join("\n\n")}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => getInitialSessionId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<InputMode>("chat");
  const [input, setInput] = useState("");
  const [filePrompt, setFilePrompt] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sending, setSending] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const messagesRef = useRef<HTMLElement | null>(null);

  const canSendChat = input.trim().length > 0 && !sending;
  const canSendFile = selectedFile !== null && !sending;

  const messageCount = useMemo(() => messages.length, [messages.length]);

  const scrollToBottom = useCallback(() => {
    const container = messagesRef.current;
    if (!container) return;

    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
      setTimeout(() => {
        if (!messagesRef.current) return;
        messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
      }, 150);
    });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, sending, scrollToBottom]);

  async function submitMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || sending) return;

    const extractedQuestions = extractQuestionsFromText(trimmed);
    if (extractedQuestions.length > 1) {
      await submitBatchQuestions(extractedQuestions, {
        userLabel: `Submitted ${extractedQuestions.length} questions in one message.`,
      });
      setInput("");
      return;
    }

    setMessages((prev) => [
      ...prev,
      {
        id: createId(),
        role: "user",
        content: extractedQuestions[0] || trimmed,
        createdAt: new Date().toISOString(),
      },
    ]);
    setInput("");
    setSending(true);
    setLastError(null);

    try {
      const result = await sendMessageToRfiAgent({
        message: trimmed,
        sessionId,
      });

      if (result.sessionId !== sessionId) {
        setSessionId(result.sessionId);
        try {
          localStorage.setItem(SESSION_KEY, result.sessionId);
        } catch {
          // no-op
        }
      }

      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        content: result.reply || "I did not receive a response body from n8n.",
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const detail =
        error instanceof Error
          ? error.message
          : "Unknown error while calling the RFI workflow.";
      setLastError(detail);
      const assistantError: ChatMessage = {
        id: createId(),
        role: "assistant",
        content:
          "I could not reach the RFI workflow. Please verify the webhook URL and that the workflow is active in n8n.",
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantError]);
    } finally {
      setSending(false);
    }
  }

  async function submitBatchQuestions(
    questions: string[],
    options?: {
      userLabel?: string;
      prefix?: string;
    },
  ) {
    if (questions.length === 0 || sending) return;

    const preview = questions.slice(0, 5).map((question, idx) => `${idx + 1}. ${question}`).join("\n");
    const remainderCount = questions.length - Math.min(questions.length, 5);
    const remainderLine = remainderCount > 0 ? `\n...and ${remainderCount} more.` : "";

    setMessages((prev) => [
      ...prev,
      {
        id: createId(),
        role: "user",
        content:
          options?.userLabel ||
          `Please answer all ${questions.length} questions:\n${preview}${remainderLine}`,
        createdAt: new Date().toISOString(),
      },
    ]);

    setSending(true);
    setLastError(null);

    try {
      const batchResult = await sendBatchToRfiAgent({
        questions,
        sessionId,
        prefix: options?.prefix,
      });

      if (batchResult.sessionId !== sessionId) {
        setSessionId(batchResult.sessionId);
        try {
          localStorage.setItem(SESSION_KEY, batchResult.sessionId);
        } catch {
          // no-op
        }
      }

      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "assistant",
          content: formatBatchAssistantReply(batchResult.replies),
          createdAt: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      const detail =
        error instanceof Error
          ? error.message
          : "Unknown error while processing batch questions.";
      setLastError(detail);
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "assistant",
          content:
            "I could not process the batch request. Please verify the webhook URL and that the workflow is active in n8n.",
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  async function submitFileQuestions(event: FormEvent) {
    event.preventDefault();
    if (!selectedFile || sending) return;

    setSending(true);
    setLastError(null);

    try {
      const questions = await extractQuestionsFromFile(selectedFile);
      if (questions.length === 0) {
        throw new Error("No questions detected in file. Ensure each question is on its own line.");
      }

      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "user",
          content: `Uploaded ${selectedFile.name} and found ${questions.length} question(s).`,
          createdAt: new Date().toISOString(),
        },
      ]);

      const batchResult = await sendBatchToRfiAgent({
        questions,
        sessionId,
        prefix: filePrompt.trim() || undefined,
      });

      if (batchResult.sessionId !== sessionId) {
        setSessionId(batchResult.sessionId);
        try {
          localStorage.setItem(SESSION_KEY, batchResult.sessionId);
        } catch {
          // no-op
        }
      }

      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "assistant",
          content: formatBatchAssistantReply(batchResult.replies),
          createdAt: new Date().toISOString(),
        },
      ]);

      setSelectedFile(null);
      setFilePrompt("");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown file processing error.";
      setLastError(detail);
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "assistant",
          content:
            "I could not process that file. Use a .txt, .md, .csv, .json, .docx, .xlsx, or .xls questionnaire and try again.",
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void submitMessage(input);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-icon">
            <Bot size={20} />
          </div>
          <div>
            <h1>Agent Hub</h1>
            <p>RFI Test Workspace</p>
          </div>
        </div>

        <div className="agent-card">
          <div className="agent-card-icon">
            <MessageSquare size={16} />
          </div>
          <div className="agent-card-copy">
            <strong>RFI Agent</strong>
            <span>Workflow ID: OxplTFYVkEM_iGJMuU5RX</span>
          </div>
          <span className="status-pill">Live</span>
        </div>

        <div className="sidebar-meta">
          <p>Session ID</p>
          <code>{sessionId}</code>
        </div>

        <div className="mode-switch">
          <button
            type="button"
            className={mode === "chat" ? "active" : ""}
            onClick={() => setMode("chat")}
            disabled={sending}
          >
            <MessageSquare size={14} />
            <span>Chat</span>
          </button>
          <button
            type="button"
            className={mode === "file" ? "active" : ""}
            onClick={() => setMode("file")}
            disabled={sending}
          >
            <FileUp size={14} />
            <span>File Upload</span>
          </button>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div>
            <h2>RFI Agent Chat</h2>
            <p>
              {mode === "chat"
                ? "Ask one question or paste multiple questions at once."
                : "Upload a questionnaire file and get responses for all questions."}
            </p>
          </div>
          <div className="message-counter">{messageCount} messages</div>
        </header>

        <section ref={messagesRef} className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">
                <Sparkles size={18} />
              </div>
              <h3>{mode === "chat" ? "Ready for testing" : "Ready for questionnaire upload"}</h3>
              {mode === "chat" ? (
                <>
                  <p>
                    Use a quick prompt, ask one question, or paste multiple questions
                    on separate lines.
                  </p>
                  <div className="quick-prompts">
                    {QUICK_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => void submitMessage(prompt)}
                        disabled={sending}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <p>
                  Upload a questionnaire file (`.txt`, `.md`, `.csv`, `.json`, `.docx`, `.xlsx`, `.xls`) and the
                  agent will answer all parsed questions in one run.
                </p>
              )}
            </div>
          ) : (
            messages.map((message) => (
              <article
                key={message.id}
                className={`message-bubble ${message.role === "user" ? "user" : "assistant"}`}
              >
                <p>{message.content}</p>
                <time>{formatTime(message.createdAt)}</time>
              </article>
            ))
          )}
        </section>

        <footer className="chat-footer">
          {mode === "chat" ? (
            <form onSubmit={handleSubmit} className="input-row">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Type one question, or paste 5+ questions on separate lines..."
                disabled={sending}
                rows={3}
              />
              <button type="submit" disabled={!canSendChat}>
                <Send size={15} />
                <span>{sending ? "Sending..." : "Send"}</span>
              </button>
            </form>
          ) : (
            <form onSubmit={submitFileQuestions} className="file-form">
              <label className="file-picker">
                <span>Questionnaire file</span>
                <input
                  type="file"
                  accept=".txt,.md,.csv,.json,.docx,.xlsx,.xls"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
                  disabled={sending}
                />
                <small>Supported: {SUPPORTED_FILE_EXTENSIONS.join(", ")}</small>
              </label>
              <textarea
                value={filePrompt}
                onChange={(event) => setFilePrompt(event.target.value)}
                placeholder="Optional instruction (for example: prioritize GDPR context)."
                disabled={sending}
                rows={2}
              />
              <button type="submit" disabled={!canSendFile}>
                <FileUp size={15} />
                <span>{sending ? "Processing..." : "Process File"}</span>
              </button>
            </form>
          )}
          {lastError ? <p className="error-line">{lastError}</p> : null}
        </footer>
      </main>
    </div>
  );
}

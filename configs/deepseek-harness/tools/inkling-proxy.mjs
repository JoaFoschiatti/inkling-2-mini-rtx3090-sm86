#!/usr/bin/env node
/**
 * inkling-proxy.mjs — Zero-dependency, byte-preserving HTTP/SSE micro-proxy
 * for Inkling-2-Mini (llama-server :8080) + DeepSeek Harness (dsh -> :8081).
 *
 * Wave 13 upgrades (from 50.7-min live forensic audit):
 *   1. Compaction KV-hit + instant extractive summary:
 *      - Detects compaction via `x-deepseek-harness-compact: 1` OR the exact
 *        `dsh-compaction-basic` prompt marker (needed for `llm-pi-ai` / openai-completions).
 *      - Preserves the session's last `reasoning_effort` onto compaction requests so
 *        `Thinking effort level: 0.99` matches token-for-token (eliminating 100 % KV miss!).
 *      - Clamps `reasoning_budget_tokens = COMPACT_BUDGET` (0) so compaction outputs
 *        the structured Markdown checkpoint directly without 8–12K tokens of thinking.
 *   2. Effort-aware adaptive reasoning budget:
 *      - Removes `write` from `MECHANICAL_TOOLS` (never chop planning after creating a file).
 *      - Scales `LOW_BUDGET` up to `HIGH_EFFORT_LOW_BUDGET` (3072) when `reasoning_effort`
 *        is `high`/`xhigh`/`max`, preventing mid-sentence chops that seed tool loops.
 *   3. Anti-loop rescue (defense-in-depth):
 *      - Detects `dsh`'s repetition warning OR >=2 identical consecutive tool calls, and
 *        injects `HIGH_BUDGET`, `min_p: 0.0`, and `presence_penalty: 0.4` for that turn
 *        to break Transformer induction-head lock-in cleanly.
 *   4. 0 bytes of KV cache drift: NEVER re-serializes `messages`, `tools`, or `system`;
 *      splices only top-level fields directly before the final `}` of the raw UTF-8 JSON body.
 */

import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";

const LISTEN_HOST = process.env.PROXY_HOST || "127.0.0.1";
const LISTEN_PORT = Number(process.env.PROXY_PORT || 8081);
const UPSTREAM_URL = new URL(process.env.UPSTREAM_URL || "http://127.0.0.1:8080");
const LOG_PATH =
  process.env.INKLING_PROXY_LOG ||
  path.join(os.homedir(), ".dsh", "logs", "inkling-proxy.jsonl");

const HIGH_BUDGET = Number(process.env.HIGH_BUDGET ?? 12288);
const LOW_BUDGET = Number(process.env.LOW_BUDGET ?? 1536);
const HIGH_EFFORT_LOW_BUDGET = Number(process.env.HIGH_EFFORT_LOW_BUDGET ?? 3072);
const COMPACT_BUDGET = Number(process.env.COMPACT_BUDGET ?? 0);
const LOOP_PRESENCE_PENALTY = Number(process.env.LOOP_PRESENCE_PENALTY ?? 0.4);

const MSG_HIGH =
  process.env.HIGH_BUDGET_MSG ||
  "\nI have enough context; let me give the final answer or next tool call now.";
const MSG_LOW =
  process.env.LOW_BUDGET_MSG ||
  "\nTool succeeded cleanly; proceed directly with the next action or concise summary.";
const MSG_COMPACT =
  process.env.COMPACT_BUDGET_MSG ||
  "\nWrite the structured checkpoint summary now.";

// Only truly mechanical confirmation tools (NOT `write`, which is followed by planning the next file)
const MECHANICAL_TOOLS = new Set([
  "edit",
  "todo_write",
  "glob",
  "mcp__fast__multi_edit",
]);

const HIGH_EFFORT_LEVELS = new Set(["high", "xhigh", "max"]);

const ERROR_OR_COMPLEX_RE =
  /\b(warn|warning|error|err|fail|failed|failure|traceback|exception|assert|assertion|panic|denied|not found|syntaxerror|typeerror|referenceerror|enoent|segfault|killed|timed out)\b/i;

const BASH_NONZERO_RE = /(?:exit(?:\s+code|\s+status)?\s*[:=]\s*[1-9]|command failed|returned non-zero)/i;
const DIFF_OR_LOG_RE = /(?:^diff --git|^@@ -\d+|^commit [0-9a-f]{7,}|^\+\+\+ b\/)/m;
const DSH_LOOP_STEER_RE =
  /(?:You are repeating the exact same tool call with identical arguments|Repeated tool call detected:)/i;

// Remember last non-compaction reasoning_effort per session (plus global fallback)
const sessionEfforts = new Map();
const sessionHitLowCap = new Map();
let lastGlobalEffort = null;

fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });

function extractTextAndMedia(content) {
  if (typeof content === "string") {
    return { text: content, hasMedia: false, mediaCount: 0 };
  }
  if (!Array.isArray(content)) {
    return { text: "", hasMedia: false, mediaCount: 0 };
  }
  let text = "";
  let mediaCount = 0;
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    if (part.type === "image_url" || part.type === "image" || part.type === "input_audio") {
      mediaCount++;
    } else if (typeof part.text === "string") {
      text += (text ? "\n" : "") + part.text;
    }
  }
  return { text, hasMedia: mediaCount > 0, mediaCount };
}

function countAllImages(messages) {
  let count = 0;
  for (const m of messages) {
    if (!m || !Array.isArray(m.content)) continue;
    for (const part of m.content) {
      if (part && (part.type === "image_url" || part.type === "image")) count++;
    }
  }
  return count;
}

function getExplicitEffort(parsed) {
  if (typeof parsed?.reasoning_effort === "string" && parsed.reasoning_effort.trim()) {
    return parsed.reasoning_effort.trim().toLowerCase();
  }
  const kw = parsed?.chat_template_kwargs?.reasoning_effort;
  if (typeof kw === "string" && kw.trim()) {
    return kw.trim().toLowerCase();
  }
  return null;
}

function isCompactionPrompt(headers, messages) {
  if (String(headers["x-deepseek-harness-compact"] || "").trim() === "1") {
    return "header";
  }
  if (!Array.isArray(messages) || messages.length === 0) return null;
  const last = messages[messages.length - 1];
  if (!last || last.role !== "user") return null;
  const { text } = extractTextAndMedia(last.content);
  if (
    text.includes("Output EXACTLY the Markdown structure below: keep every section, in order.") &&
    text.includes("## Primary Request and Intent")
  ) {
    return "prompt_marker";
  }
  return null;
}

function detectToolLoop(messages) {
  if (!Array.isArray(messages) || messages.length < 2) return false;
  const last = messages[messages.length - 1];
  if (last?.role === "user") {
    const { text } = extractTextAndMedia(last.content);
    if (DSH_LOOP_STEER_RE.test(text)) return true;
  }
  // Look back across the last 8 messages for >=2 consecutive identical single tool calls
  const recentSigs = [];
  for (let i = messages.length - 1; i >= 0 && i >= messages.length - 8 && recentSigs.length < 2; i--) {
    const m = messages[i];
    if (m && m.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls.length === 1) {
      const fn = m.tool_calls[0]?.function;
      if (fn && typeof fn.name === "string") {
        const argsStr = typeof fn.arguments === "string" ? fn.arguments : JSON.stringify(fn.arguments ?? {});
        recentSigs.push(`${fn.name}\0${argsStr}`);
      }
    }
  }
  return recentSigs.length === 2 && recentSigs[0] === recentSigs[1];
}

function resolveLastToolName(messages) {
  if (!Array.isArray(messages) || messages.length === 0) return null;
  const last = messages[messages.length - 1];
  if (!last || last.role !== "tool") return null;
  if (typeof last.name === "string" && last.name) return last.name;
  const tcId = last.tool_call_id;
  if (!tcId) return null;
  for (let i = messages.length - 2; i >= 0 && i >= messages.length - 12; i--) {
    const m = messages[i];
    if (m && m.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        if (tc && tc.id === tcId && tc.function && typeof tc.function.name === "string") {
          return tc.function.name;
        }
      }
    }
  }
  return null;
}

function classifyTurn(headers, parsed) {
  const messages = Array.isArray(parsed?.messages) ? parsed.messages : [];
  const last = messages.length > 0 ? messages[messages.length - 1] : null;
  const lastRole = last?.role || "none";
  const lastTool = resolveLastToolName(messages);
  const sessionId = String(headers["x-deepseek-harness-session-id"] || "").trim() || "_default";
  const explicitEffort = getExplicitEffort(parsed);
  const compactVia = isCompactionPrompt(headers, messages);

  if (compactVia) {
    // dsh sends explicit reasoning_effort:"none" on compactions, which changes
    // "Thinking effort level: 0.99" -> "0.9" / "0" and triggers a 100% KV cache miss.
    // Override absent OR "none" with the session's active effort while clamping budget=0.
    const prevSessionEffort = sessionEfforts.get(sessionId) || lastGlobalEffort || null;
    const shouldOverride =
      Boolean(prevSessionEffort) &&
      (!explicitEffort || explicitEffort === "none") &&
      prevSessionEffort !== explicitEffort;
    const effectiveEffort = shouldOverride ? prevSessionEffort : (explicitEffort || prevSessionEffort || null);
    const injectEffort = shouldOverride ? prevSessionEffort : null;
    return {
      isCompact: true,
      compactVia,
      sessionId,
      effort: effectiveEffort,
      injectEffort,
      overrideExistingEffort: Boolean(injectEffort && parsed?.reasoning_effort !== undefined),
      loopRescue: false,
      lastRole,
      lastTool: lastTool || "compaction",
      budget: COMPACT_BUDGET,
      budgetMessage: MSG_COMPACT,
      reason: `compaction_${compactVia}`,
    };
  }

  // Ignore 1-turn utility calls (e.g. session title generator) with effort:"none" so they
  // never overwrite an active session's xhigh/high effort.
  const isShortUtilityNone = explicitEffort === "none" && messages.length <= 2;
  if (explicitEffort && !isShortUtilityNone) {
    sessionEfforts.set(sessionId, explicitEffort);
    if (sessionEfforts.size > 64) {
      const oldestKey = sessionEfforts.keys().next().value;
      sessionEfforts.delete(oldestKey);
    }
    lastGlobalEffort = explicitEffort;
  }

  const effectiveEffort = explicitEffort || sessionEfforts.get(sessionId) || lastGlobalEffort || "medium";
  const effectiveLowBudget = HIGH_EFFORT_LEVELS.has(effectiveEffort)
    ? Math.max(LOW_BUDGET, HIGH_EFFORT_LOW_BUDGET)
    : LOW_BUDGET;

  if (detectToolLoop(messages)) {
    return {
      isCompact: false,
      compactVia: null,
      sessionId,
      effort: effectiveEffort,
      injectEffort: null,
      overrideExistingEffort: false,
      loopRescue: true,
      lastRole,
      lastTool: lastTool || "-",
      budget: HIGH_BUDGET,
      budgetMessage: MSG_HIGH,
      reason: "loop_rescue",
    };
  }

  // Auto-escalate to HIGH_BUDGET whenever the previous low-budget turn hit its thinking cap,
  // preventing filler-command spirals and double-<|content_thinking|> KV rollbacks.
  if (sessionHitLowCap.get(sessionId)) {
    return {
      isCompact: false,
      compactVia: null,
      sessionId,
      effort: effectiveEffort,
      injectEffort: null,
      overrideExistingEffort: false,
      loopRescue: false,
      lastRole,
      lastTool: lastTool || "-",
      budget: HIGH_BUDGET,
      budgetMessage: MSG_HIGH,
      reason: "low_cap_escalate",
    };
  }

  if (lastRole !== "tool" || !lastTool) {
    return {
      isCompact: false,
      compactVia: null,
      sessionId,
      effort: effectiveEffort,
      injectEffort: null,
      overrideExistingEffort: false,
      loopRescue: false,
      lastRole,
      lastTool: lastTool || "-",
      budget: HIGH_BUDGET,
      budgetMessage: MSG_HIGH,
      reason: `role_${lastRole}`,
    };
  }

  const { text, hasMedia } = extractTextAndMedia(last.content);
  const trimmed = text.trim();

  if (hasMedia) {
    return {
      isCompact: false,
      compactVia: null,
      sessionId,
      effort: effectiveEffort,
      injectEffort: null,
      overrideExistingEffort: false,
      loopRescue: false,
      lastRole,
      lastTool,
      budget: HIGH_BUDGET,
      budgetMessage: MSG_HIGH,
      reason: "tool_has_media",
    };
  }

  if (MECHANICAL_TOOLS.has(lastTool)) {
    const failed =
      ERROR_OR_COMPLEX_RE.test(trimmed) ||
      /syntax:\s*FAILED|verify:\s*FAILED|lint:\s*FAILED|Error in /i.test(trimmed);
    if (!failed) {
      return {
        isCompact: false,
        compactVia: null,
        sessionId,
        effort: effectiveEffort,
        injectEffort: null,
        overrideExistingEffort: false,
        loopRescue: false,
        lastRole,
        lastTool,
        budget: effectiveLowBudget,
        budgetMessage: MSG_LOW,
        reason: "mechanical_tool_ok",
      };
    }
    return {
      isCompact: false,
      compactVia: null,
      sessionId,
      effort: effectiveEffort,
      injectEffort: null,
      overrideExistingEffort: false,
      loopRescue: false,
      lastRole,
      lastTool,
      budget: HIGH_BUDGET,
      budgetMessage: MSG_HIGH,
      reason: "mechanical_tool_error",
    };
  }

  if (lastTool === "bash") {
    // Never chop planning after a quick bash check (ls/rm/pwd) in high/xhigh/max effort:
    // Run 2 audit showed 9/10 cap hits + both full-file SVG writes were truncated right after bash.
    if (HIGH_EFFORT_LEVELS.has(effectiveEffort)) {
      return {
        isCompact: false,
        compactVia: null,
        sessionId,
        effort: effectiveEffort,
        injectEffort: null,
        overrideExistingEffort: false,
        loopRescue: false,
        lastRole,
        lastTool,
        budget: HIGH_BUDGET,
        budgetMessage: MSG_HIGH,
        reason: "bash_high_effort",
      };
    }
    const cleanShort =
      trimmed.length < 300 &&
      !BASH_NONZERO_RE.test(trimmed) &&
      !ERROR_OR_COMPLEX_RE.test(trimmed) &&
      !DIFF_OR_LOG_RE.test(trimmed);
    if (cleanShort) {
      return {
        isCompact: false,
        compactVia: null,
        sessionId,
        effort: effectiveEffort,
        injectEffort: null,
        overrideExistingEffort: false,
        loopRescue: false,
        lastRole,
        lastTool,
        budget: effectiveLowBudget,
        budgetMessage: MSG_LOW,
        reason: "bash_short_clean",
      };
    }
    return {
      isCompact: false,
      compactVia: null,
      sessionId,
      effort: effectiveEffort,
      injectEffort: null,
      overrideExistingEffort: false,
      loopRescue: false,
      lastRole,
      lastTool,
      budget: HIGH_BUDGET,
      budgetMessage: MSG_HIGH,
      reason: "bash_complex_or_error",
    };
  }

  // write, read, grep, read_batch, browser, view_image, read_image, unknown -> full budget
  return {
    isCompact: false,
    compactVia: null,
    sessionId,
    effort: effectiveEffort,
    injectEffort: null,
    overrideExistingEffort: false,
    loopRescue: false,
    lastRole,
    lastTool,
    budget: HIGH_BUDGET,
    budgetMessage: MSG_HIGH,
    reason: "analytical_tool",
  };
}

/**
 * Replace an existing top-level JSON string property (depth === 1 only) without touching
 * a single byte inside `messages`, `tools`, or `system`.
 */
function replaceTopLevelStringField(rawUtf8, targetKey, newValue) {
  let depth = 0;
  let inStr = false;
  let esc = false;
  let strStart = -1;
  const n = rawUtf8.length;
  for (let i = 0; i < n; i++) {
    const ch = rawUtf8[i];
    if (inStr) {
      if (esc) {
        esc = false;
      } else if (ch === "\\") {
        esc = true;
      } else if (ch === '"') {
        inStr = false;
        if (depth === 1 && strStart >= 0) {
          const key = rawUtf8.slice(strStart + 1, i);
          if (key === targetKey) {
            let j = i + 1;
            while (j < n && /\s/.test(rawUtf8[j])) j++;
            if (rawUtf8[j] === ":") {
              j++;
              while (j < n && /\s/.test(rawUtf8[j])) j++;
              if (rawUtf8[j] === '"') {
                let k = j + 1;
                let vEsc = false;
                while (k < n) {
                  const vc = rawUtf8[k];
                  if (vEsc) vEsc = false;
                  else if (vc === "\\") vEsc = true;
                  else if (vc === '"') break;
                  k++;
                }
                if (k < n && rawUtf8[k] === '"') {
                  return rawUtf8.slice(0, j) + JSON.stringify(newValue) + rawUtf8.slice(k + 1);
                }
              }
            }
          }
        }
        strStart = -1;
      }
      continue;
    }
    if (ch === '"') {
      inStr = true;
      strStart = depth === 1 ? i : -1;
    } else if (ch === "{" || ch === "[") {
      depth++;
    } else if (ch === "}" || ch === "]") {
      depth--;
    }
  }
  return null;
}

/**
 * Splice top-level JSON properties right before the final `}` without touching a single
 * byte inside `messages`, `tools`, or `system`.
 */
function spliceTopLevelFields(rawUtf8, fields) {
  const entries = Object.entries(fields).filter(([, v]) => v !== undefined);
  if (entries.length === 0) return Buffer.from(rawUtf8, "utf8");
  const idx = rawUtf8.lastIndexOf("}");
  if (idx === -1) return Buffer.from(rawUtf8, "utf8");
  const prefix = rawUtf8.slice(0, idx);
  const suffix = rawUtf8.slice(idx);
  const hasPriorField = /[^\s{]\s*$/.test(prefix);
  const injected = entries.map(([k, v]) => `${JSON.stringify(k)}:${JSON.stringify(v)}`).join(",");
  return Buffer.from(`${prefix}${hasPriorField ? "," : ""}${injected}${suffix}`, "utf8");
}

function appendTelemetry(record) {
  try {
    fs.appendFileSync(LOG_PATH, JSON.stringify(record) + "\n", "utf8");
  } catch {}
}

function summarizeStats() {
  if (!fs.existsSync(LOG_PATH)) {
    return { count: 0, log_path: LOG_PATH };
  }
  const lines = fs.readFileSync(LOG_PATH, "utf8").trim().split("\n").filter(Boolean);
  const rows = [];
  for (const line of lines.slice(-500)) {
    try {
      rows.push(JSON.parse(line));
    } catch {}
  }
  const pct = (arr, p) => {
    if (!arr.length) return null;
    const s = [...arr].sort((a, b) => a - b);
    return Number(s[Math.min(s.length - 1, Math.floor((p / 100) * (s.length - 1)))].toFixed(2));
  };
  const numField = (k) => rows.map((r) => r[k]).filter((v) => typeof v === "number" && Number.isFinite(v));
  const byReason = {};
  for (const r of rows) {
    const k = r.budget_reason || "unknown";
    byReason[k] = (byReason[k] || 0) + 1;
  }
  return {
    count: rows.length,
    log_path: LOG_PATH,
    config: {
      HIGH_BUDGET,
      LOW_BUDGET,
      HIGH_EFFORT_LOW_BUDGET,
      COMPACT_BUDGET,
      LOOP_PRESENCE_PENALTY,
      upstream: UPSTREAM_URL.origin,
    },
    ttft_ms: { p50: pct(numField("ttft_ms"), 50), p95: pct(numField("ttft_ms"), 95) },
    prompt_ms: { p50: pct(numField("prompt_ms"), 50), p95: pct(numField("prompt_ms"), 95) },
    gen_tps: { p50: pct(numField("gen_tps"), 50), p95: pct(numField("gen_tps"), 95) },
    by_reason: byReason,
    last_5: rows.slice(-5),
  };
}

const upstreamAgent = new http.Agent({ keepAlive: true, maxSockets: 8 });

const server = http.createServer((req, res) => {
  if (req.socket) req.socket.setNoDelay(true);

  if (req.method === "GET" && req.url === "/proxy/stats") {
    const body = JSON.stringify(summarizeStats(), null, 2);
    res.writeHead(200, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
    res.end(body);
    return;
  }

  const isChatCompletion =
    req.method === "POST" && (req.url === "/v1/chat/completions" || req.url?.startsWith("/v1/chat/completions?"));

  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    const tStart = performance.now();
    let outBody = Buffer.concat(chunks);
    let decision = null;
    let isStream = false;
    let nImages = 0;

    if (isChatCompletion && outBody.length > 0) {
      const rawUtf8 = outBody.toString("utf8");
      try {
        const parsed = JSON.parse(rawUtf8);
        isStream = Boolean(parsed.stream);
        nImages = Array.isArray(parsed.messages) ? countAllImages(parsed.messages) : 0;
        decision = classifyTurn(req.headers, parsed);

        const toInject = {};
        const hasExplicitBudget =
          parsed.reasoning_budget_tokens !== undefined || parsed.thinking_budget_tokens !== undefined;
        if (!hasExplicitBudget && decision.budget >= 0) {
          toInject.reasoning_budget_tokens = decision.budget;
        }
        if (parsed.reasoning_budget_message === undefined && decision.budgetMessage) {
          toInject.reasoning_budget_message = decision.budgetMessage;
        }
        // Preserve session reasoning_effort onto compaction requests for KV prefix cache hit
        // (even when dsh explicitly sends reasoning_effort:"none").
        let baseUtf8 = rawUtf8;
        if (decision.injectEffort) {
          if (decision.overrideExistingEffort) {
            const replaced = replaceTopLevelStringField(baseUtf8, "reasoning_effort", decision.injectEffort);
            if (replaced !== null) {
              baseUtf8 = replaced;
            } else {
              toInject.reasoning_effort = decision.injectEffort;
            }
          } else {
            toInject.reasoning_effort = decision.injectEffort;
          }
        }
        // Anti-loop rescue: restore full tail support + mild presence penalty on that turn only
        if (decision.loopRescue) {
          if (parsed.min_p === undefined) toInject.min_p = 0.0;
          if (parsed.presence_penalty === undefined && LOOP_PRESENCE_PENALTY > 0) {
            toInject.presence_penalty = LOOP_PRESENCE_PENALTY;
          }
        }
        if (isStream && parsed.timings_per_token === undefined) {
          toInject.timings_per_token = false;
        }
        if (Object.keys(toInject).length > 0 || baseUtf8 !== rawUtf8) {
          outBody = spliceTopLevelFields(baseUtf8, toInject);
        }
      } catch {
        // Malformed JSON: forward untouched
      }
    }

    const fwdHeaders = { ...req.headers };
    delete fwdHeaders["content-length"];
    delete fwdHeaders["transfer-encoding"];
    fwdHeaders["host"] = UPSTREAM_URL.host;
    if (outBody.length > 0 || req.method === "POST" || req.method === "PUT") {
      fwdHeaders["content-length"] = String(outBody.length);
    }

    let tFirstToken = null;
    let sseBuf = "";
    let lastTimings = null;
    let lastUsage = null;
    let nonStreamBuf = [];

    const upReq = http.request(
      {
        protocol: UPSTREAM_URL.protocol,
        hostname: UPSTREAM_URL.hostname,
        port: UPSTREAM_URL.port || 80,
        method: req.method,
        path: req.url,
        headers: fwdHeaders,
        agent: upstreamAgent,
      },
      (upRes) => {
        if (upRes.socket) upRes.socket.setNoDelay(true);
        const respHeaders = { ...upRes.headers };
        res.writeHead(upRes.statusCode || 502, respHeaders);
        if (typeof res.flushHeaders === "function") res.flushHeaders();

        upRes.on("data", (chunk) => {
          res.write(chunk);

          if (!isChatCompletion) return;

          if (isStream) {
            const text = chunk.toString("utf8");
            sseBuf += text;
            let nl;
            while ((nl = sseBuf.indexOf("\n")) !== -1) {
              const line = sseBuf.slice(0, nl).trim();
              sseBuf = sseBuf.slice(nl + 1);
              if (!line.startsWith("data:")) continue;
              const payload = line.slice(5).trim();
              if (!payload || payload === "[DONE]") continue;
              try {
                const evt = JSON.parse(payload);
                if (tFirstToken === null && evt.choices?.[0]?.delta) {
                  const d = evt.choices[0].delta;
                  if (d.content || d.reasoning_content || d.tool_calls) {
                    tFirstToken = performance.now();
                  }
                }
                if (evt.timings) lastTimings = evt.timings;
                if (evt.usage) lastUsage = evt.usage;
              } catch {}
            }
          } else {
            nonStreamBuf.push(chunk);
          }
        });

        upRes.on("end", () => {
          if (isChatCompletion) {
            const totalMs = performance.now() - tStart;
            if (!isStream && nonStreamBuf.length > 0) {
              try {
                const full = JSON.parse(Buffer.concat(nonStreamBuf).toString("utf8"));
                if (full.timings) lastTimings = full.timings;
                if (full.usage) lastUsage = full.usage;
              } catch {}
            }

            const t = lastTimings || {};
            const u = lastUsage || {};
            const promptN = t.prompt_n ?? u.prompt_tokens ?? null;
            const cacheN = t.cache_n ?? null;
            const promptMs = typeof t.prompt_ms === "number" ? Number(t.prompt_ms.toFixed(2)) : null;
            const predN = t.predicted_n ?? u.completion_tokens ?? null;
            const predMs = typeof t.predicted_ms === "number" ? Number(t.predicted_ms.toFixed(2)) : null;
            const promptTps =
              typeof t.prompt_per_second === "number" ? Number(t.prompt_per_second.toFixed(2)) : null;
            const genTps =
              typeof t.predicted_per_second === "number" ? Number(t.predicted_per_second.toFixed(2)) : null;
            const ttftMs =
              tFirstToken !== null
                ? Number((tFirstToken - tStart).toFixed(2))
                : promptMs;

            if (decision && !decision.isCompact && decision.sessionId) {
              const hitLowCap =
                typeof decision.budget === "number" &&
                decision.budget > 0 &&
                decision.budget < HIGH_BUDGET &&
                typeof predN === "number" &&
                predN >= decision.budget;
              sessionHitLowCap.set(decision.sessionId, hitLowCap);
              if (sessionHitLowCap.size > 64) {
                const oldestKey = sessionHitLowCap.keys().next().value;
                sessionHitLowCap.delete(oldestKey);
              }
            }

            appendTelemetry({
              ts: new Date().toISOString(),
              status: upRes.statusCode,
              stream: isStream,
              is_compact: decision?.isCompact ?? false,
              effort: decision?.effort ?? null,
              effort_injected: decision?.injectEffort ?? null,
              loop_rescue: decision?.loopRescue ?? false,
              n_images: nImages,
              budget_reason: decision?.reason ?? "passthrough",
              last_role: decision?.lastRole ?? "none",
              last_tool: decision?.lastTool ?? "-",
              budget: decision?.budget ?? null,
              prompt_n: promptN,
              cache_n: cacheN,
              prompt_ms: promptMs,
              prompt_tps: promptTps,
              predicted_n: predN,
              predicted_ms: predMs,
              gen_tps: genTps,
              draft_n: t.draft_n ?? null,
              draft_acc: t.draft_n_accepted ?? null,
              ttft_ms: ttftMs,
              total_ms: Number(totalMs.toFixed(2)),
            });
          }
          res.end();
        });

        upRes.on("error", (err) => {
          if (!res.headersSent) {
            res.writeHead(502, { "content-type": "application/json" });
          }
          res.end(JSON.stringify({ error: { message: `Upstream stream error: ${err.message}` } }));
        });
      }
    );

    upReq.on("socket", (sock) => sock.setNoDelay(true));
    upReq.on("error", (err) => {
      if (!res.headersSent) {
        res.writeHead(502, { "content-type": "application/json" });
      }
      res.end(JSON.stringify({ error: { message: `Upstream connect error: ${err.message}` } }));
    });

    req.on("aborted", () => upReq.destroy());
    upReq.end(outBody);
  });
});

server.listen(LISTEN_PORT, LISTEN_HOST, () => {
  process.stdout.write(
    `[inkling-proxy] http://${LISTEN_HOST}:${LISTEN_PORT} -> ${UPSTREAM_URL.origin} | ` +
      `budgets: high=${HIGH_BUDGET}, low=${LOW_BUDGET}, high_effort_low=${HIGH_EFFORT_LOW_BUDGET}, compact=${COMPACT_BUDGET} | log=${LOG_PATH}\n`
  );
});

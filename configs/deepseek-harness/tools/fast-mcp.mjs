#!/usr/bin/env node
/**
 * fast-mcp.mjs — Ultra-lean, zero-dependency MCP stdio server for Inkling-2-Mini + DeepSeek Harness (dsh).
 *
 * Eliminates the 3 biggest agentic latency bottlenecks observed in live sessions:
 *   1. `browser`    — Persistent warm headless Chromium via raw CDP (0s cold boot after first call,
 *                     compound actions + console/exception/network capture + inline resized JPEG
 *                     returned in ONE tool call; eliminates ephemeral bwrap Chrome restarts & follow-up read_image turns).
 *   2. `view_image` — Fast resized/cropped local image reader (cuts 1440x810/4K PNG vision tokens & CPU
 *                     encode latency 3-5x while supporting pixel-crisp clip zooms).
 *   3. `multi_edit` — Atomic multi-replacement surgical file editor with built-in instant syntax check
 *                     (eliminates inline `python3 - <<EOF` bash patches & dsh PKG_EXECPATH `node` interception).
 */

import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";

const NODE_BIN = process.execPath;
const WORKSPACE_ROOT = fs.realpathSync(process.cwd());
const TMP_ROOT = fs.realpathSync(os.tmpdir());

const CHROME_CANDIDATES = [
  process.env.FAST_MCP_CHROME,
  path.join(os.homedir(), ".agent-browser/browsers/chrome-157.0.8103.78/chrome"),
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
].filter(Boolean);

function findChrome() {
  for (const p of CHROME_CANDIDATES) {
    if (p && fs.existsSync(p)) return p;
  }
  const abDir = path.join(os.homedir(), ".agent-browser/browsers");
  if (fs.existsSync(abDir)) {
    for (const entry of fs.readdirSync(abDir)) {
      const bin = path.join(abDir, entry, "chrome");
      if (fs.existsSync(bin)) return bin;
    }
  }
  throw new Error("No Chromium/Chrome binary found. Set FAST_MCP_CHROME or install via agent-browser.");
}

function resolveSafePath(inputPath, { allowTmp = false, mustExist = true } = {}) {
  const abs = path.isAbsolute(inputPath) ? path.resolve(inputPath) : path.resolve(WORKSPACE_ROOT, inputPath);
  const checkTarget = mustExist ? fs.realpathSync(abs) : path.join(fs.realpathSync(path.dirname(abs)), path.basename(abs));
  const inWorkspace = checkTarget === WORKSPACE_ROOT || checkTarget.startsWith(WORKSPACE_ROOT + path.sep);
  const inTmp = allowTmp && (checkTarget === TMP_ROOT || checkTarget.startsWith(TMP_ROOT + path.sep) || checkTarget.startsWith("/tmp/"));
  if (!inWorkspace && !inTmp) {
    throw new Error(`Path outside allowed workspace: ${inputPath}`);
  }
  return checkTarget;
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function httpJson(port, urlPath) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host: "127.0.0.1", port, path: urlPath, timeout: 3000 }, (res) => {
      let data = "";
      res.on("data", (c) => (data += c));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("CDP HTTP timeout")));
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ─── Persistent Headless Chromium CDP Controller ───────────────────────────

class PersistentBrowser {
  constructor() {
    this.proc = null;
    this.port = null;
    this.userDir = null;
    this.ws = null;
    this.msgId = 0;
    this.pending = new Map();
    this.viewport = { width: 0, height: 0, mobile: false };
    this.consoleErrors = [];
    this.consoleWarns = [];
    this.exceptions = [];
    this.failedRequests = [];
    this.reqUrls = new Map();
    this.loadResolvers = [];
    this.idleTimer = null;
  }

  touchIdleTimer() {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      this.close().catch(() => {});
    }, 10 * 60 * 1000);
    if (typeof this.idleTimer.unref === "function") this.idleTimer.unref();
  }

  async setRafPaused(paused) {
    if (!this.isAlive()) return;
    try {
      await this.evalJs(`window.__FMCP_SET_PAUSED && window.__FMCP_SET_PAUSED(${paused ? "true" : "false"})`, false);
    } catch {}
  }

  resetLogs() {
    this.consoleErrors = [];
    this.consoleWarns = [];
    this.exceptions = [];
    this.failedRequests = [];
    this.reqUrls.clear();
  }

  isAlive() {
    return (
      this.proc !== null &&
      this.proc.exitCode === null &&
      !this.proc.killed &&
      this.ws !== null &&
      this.ws.readyState === WebSocket.OPEN
    );
  }

  async ensureRunning() {
    if (this.isAlive()) return false;
    await this.close();

    const chrome = findChrome();
    this.port = await findFreePort();
    this.userDir = `/tmp/dsh-ab-${process.pid}-${Date.now().toString(36)}`;
    fs.mkdirSync(this.userDir, { recursive: true });

    this.proc = spawn(
      chrome,
      [
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-crash-reporter",
        "--disable-breakpad",
        "--disable-background-networking",
        "--disable-default-apps",
        "--disable-extensions",
        "--disable-sync",
        "--metrics-recording-only",
        "--mute-audio",
        "--no-first-run",
        "--disable-gpu",
        "--enable-unsafe-swiftshader",
        `--remote-debugging-port=${this.port}`,
        `--user-data-dir=${this.userDir}`,
        "--window-size=960,540",
        "about:blank",
      ],
      {
        stdio: ["ignore", "ignore", "pipe"],
        env: { ...process.env, TMPDIR: "/tmp", XDG_RUNTIME_DIR: "/tmp", HOME: "/tmp" },
      }
    );

    let stderrTail = "";
    this.proc.stderr.on("data", (d) => {
      stderrTail = (stderrTail + d.toString()).slice(-1200);
    });
    this.proc.on("exit", () => {
      this.ws = null;
      this.proc = null;
    });

    let page = null;
    for (let i = 0; i < 100; i++) {
      if (!this.proc || this.proc.exitCode !== null) {
        throw new Error(`Chromium exited early: ${stderrTail}`);
      }
      try {
        const targets = await httpJson(this.port, "/json/list");
        page = Array.isArray(targets) ? targets.find((t) => t.type === "page") : null;
        if (page && page.webSocketDebuggerUrl) break;
      } catch {}
      await sleep(35);
    }
    if (!page) {
      await this.close();
      throw new Error(`Chromium CDP did not start on port ${this.port}: ${stderrTail}`);
    }

    this.ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("CDP WebSocket open timeout")), 8000);
      this.ws.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
      this.ws.onerror = (e) => {
        clearTimeout(timer);
        reject(new Error(`CDP WebSocket error: ${e.message || "unknown"}`));
      };
    });

    this.ws.onmessage = (evt) => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject, timer } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        clearTimeout(timer);
        if (msg.error) reject(new Error(`CDP ${msg.error.message || JSON.stringify(msg.error)}`));
        else resolve(msg.result || {});
        return;
      }
      if (msg.method === "Page.loadEventFired") {
        const rs = this.loadResolvers.splice(0);
        for (const r of rs) r();
      } else if (msg.method === "Runtime.consoleAPICalled") {
        const type = msg.params?.type;
        if (type === "error" || type === "warning") {
          const text = (msg.params.args || [])
            .map((a) => (a.value !== undefined ? String(a.value) : a.description || a.type || ""))
            .join(" ")
            .slice(0, 320);
          const list = type === "error" ? this.consoleErrors : this.consoleWarns;
          if (list.length < 15 && !list.includes(text)) list.push(text);
        }
      } else if (msg.method === "Runtime.exceptionThrown") {
        const det = msg.params?.exceptionDetails || {};
        const desc = det.exception?.description || det.text || "Uncaught exception";
        const loc = det.url ? ` (${det.url}:${det.lineNumber ?? 0})` : "";
        const entry = `${desc}${loc}`.slice(0, 360);
        if (this.exceptions.length < 15 && !this.exceptions.includes(entry)) this.exceptions.push(entry);
      } else if (msg.method === "Network.requestWillBeSent") {
        if (msg.params?.requestId && msg.params?.request?.url) {
          this.reqUrls.set(msg.params.requestId, msg.params.request.url);
        }
      } else if (msg.method === "Network.responseReceived") {
        const st = msg.params?.response?.status;
        if (typeof st === "number" && st >= 400 && this.failedRequests.length < 12) {
          this.failedRequests.push(`${st} ${msg.params.response.url}`.slice(0, 220));
        }
      } else if (msg.method === "Network.loadingFailed") {
        if (!msg.params?.canceled && this.failedRequests.length < 12) {
          const u = this.reqUrls.get(msg.params.requestId) || "(unknown)";
          this.failedRequests.push(`FAILED(${msg.params.errorText || "net"}) ${u}`.slice(0, 220));
        }
      }
    };

    this.ws.onclose = () => {
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error("CDP WebSocket closed"));
      }
      this.pending.clear();
      this.ws = null;
    };

    await this.send("Page.enable");
    await this.send("Runtime.enable");
    await this.send("Network.enable");
    await this.send("Network.setCacheDisabled", { cacheDisabled: true });
    await this.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `(() => {
        if (window.__FMCP_HOOKED) return;
        window.__FMCP_HOOKED = true;
        const origRAF = window.requestAnimationFrame.bind(window);
        const origCAF = window.cancelAnimationFrame.bind(window);
        let paused = false;
        let nextFakeId = 1000000;
        const queued = new Map();
        window.requestAnimationFrame = function(cb) {
          if (!paused) return origRAF(cb);
          const fid = nextFakeId++;
          queued.set(fid, cb);
          return fid;
        };
        window.cancelAnimationFrame = function(id) {
          if (queued.delete(id)) return;
          return origCAF(id);
        };
        window.__FMCP_SET_PAUSED = function(p) {
          p = !!p;
          if (paused === p) return;
          paused = p;
          if (!paused && queued.size > 0) {
            const cbs = Array.from(queued.values());
            queued.clear();
            for (const cb of cbs) origRAF(cb);
          }
        };
      })();`,
    });
    this.viewport = { width: 0, height: 0, mobile: false };
    this.touchIdleTimer();
    return true;
  }

  send(method, params = {}, timeoutMs = 20000) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("CDP not connected"));
    }
    return new Promise((resolve, reject) => {
      const id = ++this.msgId;
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout on ${method}`));
        }
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  waitForLoad(timeoutMs = 10000) {
    return new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(finish, timeoutMs);
      this.loadResolvers.push(finish);
    });
  }

  async setViewport(width, height, mobile) {
    if (this.viewport.width === width && this.viewport.height === height && this.viewport.mobile === mobile) {
      return;
    }
    await this.send("Emulation.setDeviceMetricsOverride", {
      width,
      height,
      deviceScaleFactor: 1,
      mobile: !!mobile,
    });
    this.viewport = { width, height, mobile: !!mobile };
  }

  async evalJs(expression, awaitPromise = true) {
    const res = await this.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise,
    });
    if (res.exceptionDetails) {
      const msg = res.exceptionDetails.exception?.description || res.exceptionDetails.text || "Eval error";
      return { error: msg.slice(0, 400) };
    }
    const r = res.result || {};
    return { value: r.value !== undefined ? r.value : r.description ?? null };
  }

  async selectorCenter(selector) {
    const js = `(() => {
      const el = document.querySelector(${JSON.stringify(selector)});
      if (!el) return null;
      el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' });
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height };
    })()`;
    const out = await this.evalJs(js, false);
    if (out.error) throw new Error(`Selector lookup failed: ${out.error}`);
    if (!out.value) throw new Error(`Selector not found: ${selector}`);
    return out.value;
  }

  async runAction(act) {
    const type = act.type;
    if (type === "wait") {
      const rawMs = Number(act.ms ?? act.duration_ms);
      await sleep(Math.min(Math.max(Number.isFinite(rawMs) ? rawMs : 150, 0), 10000));
      return null;
    }
    if (type === "wait_for") {
      const rawMs = Number(act.ms ?? act.duration_ms);
      const deadline = Date.now() + Math.min(Math.max(Number.isFinite(rawMs) ? rawMs : 5000, 100), 15000);
      while (Date.now() < deadline) {
        const r = await this.evalJs(`Boolean(document.querySelector(${JSON.stringify(act.selector || "")}))`, false);
        if (r.value) return { waited_for: act.selector, found: true };
        await sleep(80);
      }
      return { waited_for: act.selector, found: false };
    }
    if (type === "click" || type === "hover") {
      let x = Number(act.x);
      let y = Number(act.y);
      if (act.selector) {
        const pt = await this.selectorCenter(act.selector);
        x = pt.x;
        y = pt.y;
      }
      if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error(`${type} requires selector or x,y`);
      await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
      if (type === "click") {
        await this.send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", buttons: 1, clickCount: 1 });
        await this.send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", buttons: 0, clickCount: 1 });
      }
      if (act.ms !== 0) await sleep(Math.min(Math.max(Number(act.ms) || 120, 0), 5000));
      return null;
    }
    if (type === "drag") {
      const x1 = Number(act.x);
      const y1 = Number(act.y);
      const x2 = Number(act.to_x);
      const y2 = Number(act.to_y);
      if (![x1, y1, x2, y2].every(Number.isFinite)) throw new Error("drag requires x, y, to_x, to_y");
      const steps = Math.min(Math.max(Number(act.steps) || 10, 2), 40);
      const stepDelay = Math.min(Math.max(Math.round((Number(act.duration_ms) || 240) / steps), 10), 150);
      await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: x1, y: y1 });
      await this.send("Input.dispatchMouseEvent", { type: "mousePressed", x: x1, y: y1, button: "left", buttons: 1, clickCount: 1 });
      for (let i = 1; i <= steps; i++) {
        const xt = x1 + ((x2 - x1) * i) / steps;
        const yt = y1 + ((y2 - y1) * i) / steps;
        await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: xt, y: yt, buttons: 1 });
        await sleep(stepDelay);
      }
      await this.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: x2, y: y2, button: "left", buttons: 0, clickCount: 1 });
      if (act.ms !== 0) await sleep(Math.min(Math.max(Number(act.ms) || 180, 0), 5000));
      return null;
    }
    if (type === "type") {
      if (act.selector) {
        const pt = await this.selectorCenter(act.selector);
        await this.send("Input.dispatchMouseEvent", { type: "mousePressed", x: pt.x, y: pt.y, button: "left", buttons: 1, clickCount: 1 });
        await this.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: pt.x, y: pt.y, button: "left", buttons: 0, clickCount: 1 });
        if (act.clear) {
          await this.evalJs(`(() => { const el = document.querySelector(${JSON.stringify(act.selector)}); if (el && 'value' in el) { el.value = ''; el.dispatchEvent(new Event('input', {bubbles:true})); } })()`, false);
        }
      }
      await this.send("Input.insertText", { text: String(act.text ?? "") });
      if (act.ms) await sleep(Math.min(Number(act.ms), 3000));
      return null;
    }
    if (type === "key") {
      const key = String(act.text || act.key || "Enter");
      await this.send("Input.dispatchKeyEvent", { type: "keyDown", key });
      await this.send("Input.dispatchKeyEvent", { type: "keyUp", key });
      if (act.ms !== 0) await sleep(Math.min(Math.max(Number(act.ms) || 80, 0), 3000));
      return null;
    }
    if (type === "scroll") {
      const dx = Number(act.x) || 0;
      const dy = Number(act.y) || 400;
      await this.evalJs(`window.scrollBy(${dx}, ${dy})`, false);
      if (act.ms) await sleep(Math.min(Number(act.ms), 3000));
      return null;
    }
    if (type === "eval") {
      return await this.evalJs(String(act.js || ""), true);
    }
    throw new Error(`Unknown action type: ${type}`);
  }

  async getSnapshot() {
    const js = `(() => {
      const vis = (el) => {
        const s = getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
        const r = el.getBoundingClientRect();
        return r.width > 1 && r.height > 1;
      };
      const items = [];
      const sels = 'h1,h2,h3,button,a,input,select,textarea,[role="button"],canvas,svg';
      for (const el of document.querySelectorAll(sels)) {
        if (!vis(el)) continue;
        const r = el.getBoundingClientRect();
        const tag = el.tagName.toLowerCase();
        const id = el.id ? '#' + el.id : '';
        const txt = (el.innerText || el.getAttribute('aria-label') || el.placeholder || '').trim().replace(/\\s+/g, ' ').slice(0, 60);
        items.push(\`\${tag}\${id}[\${Math.round(r.left)},\${Math.round(r.top)},\${Math.round(r.width)}x\${Math.round(r.height)}]\${txt ? ' "' + txt + '"' : ''}\`);
        if (items.length >= 45) break;
      }
      const bodyText = (document.body?.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 600);
      return { elements: items, text_preview: bodyText };
    })()`;
    const r = await this.evalJs(js, false);
    return r.value || null;
  }

  async close() {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    const ws = this.ws;
    const proc = this.proc;
    const dir = this.userDir;
    this.ws = null;
    this.proc = null;
    this.userDir = null;
    this.viewport = { width: 0, height: 0, mobile: false };
    try {
      if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    } catch {}
    if (proc && proc.exitCode === null) {
      try {
        proc.kill("SIGTERM");
        await sleep(120);
        if (proc.exitCode === null) proc.kill("SIGKILL");
      } catch {}
    }
    if (dir) {
      try {
        fs.rmSync(dir, { recursive: true, force: true });
      } catch {}
    }
  }
}

const browser = new PersistentBrowser();

// ─── Tool Handlers ─────────────────────────────────────────────────────────

async function handleBrowser(args) {
  const t0 = Date.now();
  if (args.close) {
    await browser.close();
    return { content: [{ type: "text", text: "Browser closed." }] };
  }

  const cold = await browser.ensureRunning();
  browser.touchIdleTimer();
  await browser.setRafPaused(false);

  try {
    const isMobile = Boolean(args.mobile);
    const width = Math.min(Math.max(Number(args.width) || (isMobile ? 390 : 960), 240), 1920);
    const height = Math.min(Math.max(Number(args.height) || (isMobile ? 844 : 540), 240), 1440);
    await browser.setViewport(width, height, isMobile);

    if (args.url) {
      const u = String(args.url);
      if (u.startsWith("file://")) {
        resolveSafePath(new URL(u).pathname, { allowTmp: true, mustExist: true });
      }
      browser.resetLogs();
      const curMeta = await browser.evalJs("location.href", false);
      const sameUrl = typeof curMeta.value === "string" && curMeta.value === u;
      const loadP = browser.waitForLoad(12000);
      if (sameUrl && args.reload) {
        await browser.send("Page.reload", { ignoreCache: true });
      } else {
        await browser.send("Page.navigate", { url: u });
      }
      await loadP;
      const waitMs = args.wait_ms !== undefined ? Number(args.wait_ms) : 250;
      if (waitMs > 0) await sleep(Math.min(waitMs, 10000));
    } else if (args.reload) {
      browser.resetLogs();
      const loadP = browser.waitForLoad(12000);
      await browser.send("Page.reload", { ignoreCache: true });
      await loadP;
      const waitMs = args.wait_ms !== undefined ? Number(args.wait_ms) : 250;
      if (waitMs > 0) await sleep(Math.min(waitMs, 10000));
    }

    const evalResults = [];
    const actions = Array.isArray(args.actions) ? args.actions : [];
    for (let i = 0; i < actions.length; i++) {
      const res = await browser.runAction(actions[i]);
      if (res !== null) evalResults.push({ step: i, ...res });
    }

    const metaRes = await browser.evalJs(
      "({ url: location.href, title: document.title, hasCanvas: Boolean(document.querySelector('canvas')) })",
      false
    );
    const meta = metaRes.value || {};
    const snap = args.snapshot ? await browser.getSnapshot() : null;

    const doShot = args.screenshot !== false;
    let imageBlock = null;
    let shotInfo = "none";

    if (doShot) {
      const format = args.format === "jpeg" ? "jpeg" : "png";
      const quality = Math.min(Math.max(Number(args.quality) || 92, 40), 98);
      const params = format === "jpeg" ? { format: "jpeg", quality } : { format: "png" };
      if (args.clip && typeof args.clip === "object") {
        const cx = Math.max(0, Number(args.clip.x) || 0);
        const cy = Math.max(0, Number(args.clip.y) || 0);
        const cw = Math.max(16, Number(args.clip.width) || 300);
        const ch = Math.max(16, Number(args.clip.height) || 300);
        params.clip = { x: cx, y: cy, width: cw, height: ch, scale: 1 };
      }
      const shot = await browser.send("Page.captureScreenshot", params);
      const b64 = shot.data;
      const buf = Buffer.from(b64, "base64");
      const mimeType = format === "png" ? "image/png" : "image/jpeg";
      if (args.save_path) {
        const outPath = resolveSafePath(String(args.save_path), { allowTmp: true, mustExist: false });
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, buf);
      }
      const dimStr = params.clip ? `${params.clip.width}x${params.clip.height} clip` : `${width}x${height}`;
      shotInfo = `${dimStr} ${format}, ${(buf.length / 1024).toFixed(1)} KB${args.save_path ? ` (saved: ${args.save_path})` : ""}`;
      imageBlock = { type: "image", data: b64, mimeType };
    }

    const summary = {
      url: meta.url || "",
      title: meta.title || "",
      viewport: `${width}x${height}${isMobile ? " mobile" : ""}`,
      cold_boot: cold,
      elapsed_ms: Date.now() - t0,
      screenshot: shotInfo,
    };
    if (meta.hasCanvas) {
      summary.renderer_env = "headless-swiftshader-cpu (software WebGL; FPS is not representative of real GPUs)";
    }
    if (browser.consoleErrors.length) summary.console_errors = browser.consoleErrors;
    if (browser.exceptions.length) summary.exceptions = browser.exceptions;
    if (browser.consoleWarns.length) summary.console_warns = browser.consoleWarns;
    if (browser.failedRequests.length) summary.failed_requests = browser.failedRequests;
    if (evalResults.length) summary.results = evalResults;
    if (snap) summary.snapshot = snap;

    const content = [{ type: "text", text: JSON.stringify(summary, null, 2).slice(0, 8000) }];
    if (imageBlock) content.push(imageBlock);
    return { content };
  } finally {
    await browser.setRafPaused(true);
    browser.touchIdleTimer();
  }
}

async function handleViewImage(args) {
  const t0 = Date.now();
  const safePath = resolveSafePath(String(args.file_path || ""), { allowTmp: true, mustExist: true });
  const maxDim = Math.min(Math.max(Number(args.max_dim) || 960, 128), 1600);
  const format = args.format === "png" ? "png" : "jpeg";
  const quality = Math.min(Math.max(Number(args.quality) || 84, 45), 95);
  const clipJson = args.clip ? JSON.stringify(args.clip) : "";

  const pyScript = `
import base64, io, json, sys
from PIL import Image
path, max_dim, fmt, quality, clip_s = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
im = Image.open(path)
orig_w, orig_h = im.size
if clip_s:
    c = json.loads(clip_s)
    x, y = max(0, int(c.get("x", 0))), max(0, int(c.get("y", 0)))
    w, h = max(16, int(c.get("width", orig_w))), max(16, int(c.get("height", orig_h)))
    im = im.crop((x, y, min(orig_w, x + w), min(orig_h, y + h)))
if max(im.size) > max_dim:
    im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
out_w, out_h = im.size
buf = io.BytesIO()
if fmt == "png":
    im.save(buf, format="PNG")
    mime = "image/png"
else:
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.save(buf, format="JPEG", quality=quality)
    mime = "image/jpeg"
raw = buf.getvalue()
print(json.dumps({
    "orig": f"{orig_w}x{orig_h}",
    "sent": f"{out_w}x{out_h}",
    "bytes": len(raw),
    "mime": mime,
    "b64": base64.b64encode(raw).decode("ascii")
}))
`;

  const out = execFileSync("/usr/bin/python3", ["-c", pyScript, safePath, String(maxDim), format, String(quality), clipJson], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
    timeout: 15000,
  });
  const res = JSON.parse(out);
  const info = `file=${safePath} orig=${res.orig} -> sent=${res.sent} (${res.mime}, ${(res.bytes / 1024).toFixed(1)} KB, ${Date.now() - t0} ms)`;
  return {
    content: [
      { type: "text", text: info },
      { type: "image", data: res.b64, mimeType: res.mime },
    ],
  };
}

const _binCache = new Map();
function findBinOnPath(name) {
  if (_binCache.has(name)) return _binCache.get(name);
  const dirs = (process.env.PATH || "/usr/local/bin:/usr/bin:/bin").split(path.delimiter);
  for (const d of dirs) {
    if (!d) continue;
    const full = path.join(d, name);
    try {
      fs.accessSync(full, fs.constants.X_OK);
      _binCache.set(name, full);
      return full;
    } catch {}
  }
  _binCache.set(name, null);
  return null;
}

function findNearestFile(startDir, fileName) {
  let cur = startDir;
  while (cur && cur.startsWith(WORKSPACE_ROOT)) {
    const cand = path.join(cur, fileName);
    if (fs.existsSync(cand)) return cand;
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  return null;
}

function runAutoSyntaxCheck(safePath) {
  const ext = path.extname(safePath).toLowerCase();
  const checks = [];
  try {
    if (ext === ".js" || ext === ".mjs" || ext === ".cjs") {
      execFileSync(NODE_BIN, ["--check", safePath], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 10000 });
      checks.push("syntax: OK (node --check)");
    } else if (ext === ".py") {
      execFileSync("/usr/bin/python3", ["-m", "py_compile", safePath], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 10000 });
      checks.push("syntax: OK (py_compile)");
      const ruffBin = findBinOnPath("ruff");
      if (ruffBin) {
        try {
          execFileSync(ruffBin, ["check", "--output-format", "concise", "--no-fix", safePath], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
            timeout: 2500,
          });
          checks.push("lint: OK (ruff)");
        } catch (rErr) {
          const out = ((rErr.stdout || "") + "\n" + (rErr.stderr || "")).trim().split("\n").slice(0, 10).join("\n").slice(0, 800);
          checks.push(`lint: FAILED (ruff)\n${out}`);
        }
      }
    } else if (ext === ".ts" || ext === ".tsx") {
      const tsconfig = findNearestFile(path.dirname(safePath), "tsconfig.json");
      const localTsc = tsconfig ? path.join(path.dirname(tsconfig), "node_modules", ".bin", "tsc") : null;
      const tscBin = localTsc && fs.existsSync(localTsc) ? localTsc : findBinOnPath("tsc");
      if (tscBin && tsconfig) {
        try {
          execFileSync(tscBin, ["--noEmit", "--pretty", "false", "-p", tsconfig], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
            timeout: 5000,
          });
          checks.push("types: OK (tsc)");
        } catch (tErr) {
          const out = ((tErr.stdout || "") + "\n" + (tErr.stderr || "")).trim().split("\n").slice(0, 12).join("\n").slice(0, 900);
          checks.push(`types: FAILED (tsc)\n${out}`);
        }
      } else {
        checks.push("syntax: skipped (no tsc/tsconfig)");
      }
    } else if (ext === ".json") {
      JSON.parse(fs.readFileSync(safePath, "utf8"));
      checks.push("syntax: OK (JSON.parse)");
    } else {
      checks.push("syntax: skipped (unsupported ext)");
    }
  } catch (err) {
    const detail = (err.stderr || err.stdout || err.message || "").toString().trim().slice(0, 800);
    checks.push(`syntax: FAILED\n${detail}`);
  }
  return checks.join(" | ");
}

function runVerifyCmd(cmd, timeoutMs) {
  const t0 = Date.now();
  const capMs = Math.min(Math.max(Number(timeoutMs) || 15000, 1000), 60000);
  const envPath = `${path.dirname(NODE_BIN)}${path.delimiter}${process.env.PATH || "/usr/local/bin:/usr/bin:/bin"}`;
  try {
    const out = execFileSync("/bin/bash", ["-lc", cmd], {
      cwd: WORKSPACE_ROOT,
      env: { ...process.env, PATH: envPath },
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: capMs,
      maxBuffer: 2 * 1024 * 1024,
    });
    const tail = out.trim().slice(-2000);
    return `verify: OK (exit 0, ${Date.now() - t0} ms)${tail ? `\n${tail}` : ""}`;
  } catch (err) {
    const code = err.status ?? (err.killed ? "TIMEOUT" : 1);
    const combined = ((err.stdout || "") + "\n" + (err.stderr || "")).trim();
    const tail = (combined || err.message || "").slice(-2500);
    return `verify: FAILED (exit ${code}, ${Date.now() - t0} ms)\n${tail}`;
  }
}

async function handleReadBatch(args) {
  const items = Array.isArray(args.items) ? args.items : [];
  if (items.length === 0) throw new Error("items array must be non-empty");
  if (items.length > 10) throw new Error("items array capped at 10 entries per call");

  const maxTotalBytes = Math.min(Math.max(Number(args.max_total_bytes) || 28000, 2000), 48000);
  const perItemCap = Math.max(1200, Math.floor(maxTotalBytes / items.length));
  const sections = [];
  let totalBytes = 0;

  for (const item of items) {
    if (!item || typeof item !== "object" || !item.file_path) {
      sections.push("=== (invalid item: missing file_path) ===");
      continue;
    }
    try {
      const safePath = resolveSafePath(String(item.file_path), { allowTmp: true, mustExist: true });
      const raw = fs.readFileSync(safePath, "utf8");
      const lines = raw.split(/\r?\n/);
      const offset = Math.max(1, Number(item.offset) || 1);
      const limit = Math.min(Math.max(Number(item.limit) || 200, 1), 400);
      const startIdx = Math.min(offset - 1, lines.length);
      const endIdx = Math.min(startIdx + limit, lines.length);
      const rel = safePath.startsWith(WORKSPACE_ROOT + path.sep)
        ? safePath.slice(WORKSPACE_ROOT.length + 1)
        : safePath;
      let sec = `=== ${rel} (L${startIdx + 1}-${endIdx} of ${lines.length}) ===\n`;
      for (let i = startIdx; i < endIdx; i++) {
        const lineStr = `L${i + 1}: ${lines[i].slice(0, 1500)}\n`;
        if (sec.length + lineStr.length > perItemCap) {
          sec += `... [truncated at ${perItemCap}B cap; pass narrower offset/limit]\n`;
          break;
        }
        sec += lineStr;
      }
      if (totalBytes + sec.length > maxTotalBytes) {
        const rem = Math.max(0, maxTotalBytes - totalBytes);
        if (rem > 120) sections.push(sec.slice(0, rem) + "\n... [total_bytes cap reached]");
        break;
      }
      totalBytes += sec.length;
      sections.push(sec.trimEnd());
    } catch (err) {
      sections.push(`=== ${item.file_path} (ERROR: ${err.message || String(err)}) ===`);
    }
  }

  return { content: [{ type: "text", text: sections.join("\n\n") }] };
}

async function handleMultiEdit(args) {
  const safePath = resolveSafePath(String(args.file_path || ""), { allowTmp: false, mustExist: true });
  const edits = Array.isArray(args.edits) ? args.edits : [];
  if (edits.length === 0) throw new Error("edits array must be non-empty");
  if (edits.length > 30) throw new Error("edits array capped at 30 items per call");

  const orig = fs.readFileSync(safePath, "utf8");
  let cur = orig;
  const applied = [];

  for (let i = 0; i < edits.length; i++) {
    const e = edits[i];
    const oldStr = String(e.old_string ?? "");
    const newStr = String(e.new_string ?? "");
    const replaceAll = Boolean(e.replace_all);
    if (!oldStr) throw new Error(`edit[${i}]: old_string must be non-empty`);

    const first = cur.indexOf(oldStr);
    if (first === -1) {
      throw new Error(`edit[${i}]: old_string not found (no changes written). First 80 chars: ${JSON.stringify(oldStr.slice(0, 80))}`);
    }
    if (!replaceAll) {
      const second = cur.indexOf(oldStr, first + oldStr.length);
      if (second !== -1) {
        throw new Error(`edit[${i}]: old_string matched multiple times; add more surrounding context or set replace_all=true (no changes written).`);
      }
      const lineNo = cur.slice(0, first).split("\n").length;
      cur = cur.slice(0, first) + newStr + cur.slice(first + oldStr.length);
      applied.push(`#${i}@L${lineNo}`);
    } else {
      const parts = cur.split(oldStr);
      const count = parts.length - 1;
      cur = parts.join(newStr);
      applied.push(`#${i}(x${count})`);
    }
  }

  fs.writeFileSync(safePath, cur, "utf8");
  const checkMode = args.check === "none" ? "none" : "auto";
  const checkMsg = checkMode === "auto" ? runAutoSyntaxCheck(safePath) : "syntax: skipped";
  const verifyMsg =
    typeof args.verify_cmd === "string" && args.verify_cmd.trim()
      ? `\n${runVerifyCmd(args.verify_cmd.trim(), args.verify_timeout_ms)}`
      : "";
  const deltaBytes = Buffer.byteLength(cur, "utf8") - Buffer.byteLength(orig, "utf8");
  return {
    content: [
      {
        type: "text",
        text: `Updated ${safePath}: ${edits.length} edits applied (${applied.join(", ")}), delta=${deltaBytes >= 0 ? "+" : ""}${deltaBytes}B | ${checkMsg}${verifyMsg}`,
      },
    ],
  };
}

// ─── Tool Definitions (Lean GBNF-Friendly Draft-07 Schemas) ────────────────

const TOOLS = [
  {
    name: "browser",
    description:
      "Persistent warm headless Chromium verifier (HTTP cache disabled, pauses rAF when idle). Reuses one browser across turns, executes compound actions in one call, reports console/JS/network errors, and returns a lossless 960x540 PNG screenshot inline (no read_image needed; 390x844 when mobile=true). Use clip for detail zooms.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string", description: "URL to navigate to (omit to act on current page)." },
        reload: { type: "boolean", description: "Hard-reload current page first." },
        width: { type: "integer", description: "Viewport width (default 960; mobile default 390)." },
        height: { type: "integer", description: "Viewport height (default 540; mobile default 844)." },
        mobile: { type: "boolean", description: "Mobile viewport mode (defaults to 390x844)." },
        wait_ms: { type: "integer", description: "Settle ms after navigate/reload (default 250)." },
        actions: {
          type: "array",
          description: "Ordered CDP actions before screenshot.",
          items: {
            type: "object",
            properties: {
              type: {
                type: "string",
                enum: ["wait", "wait_for", "click", "drag", "hover", "type", "key", "scroll", "eval"],
              },
              selector: { type: "string" },
              x: { type: "number" },
              y: { type: "number" },
              to_x: { type: "number" },
              to_y: { type: "number" },
              steps: { type: "integer" },
              duration_ms: { type: "integer" },
              text: { type: "string" },
              clear: { type: "boolean" },
              js: { type: "string" },
              ms: { type: "integer" },
            },
            required: ["type"],
          },
        },
        screenshot: { type: "boolean", description: "Return inline screenshot (default true)." },
        format: { type: "string", enum: ["png", "jpeg"], description: "Screenshot format (default lossless png)." },
        quality: { type: "integer", description: "JPEG quality 40-98 when format=jpeg (default 92)." },
        clip: {
          type: "object",
          description: "Crop region {x,y,width,height} for zoomed detail.",
          properties: {
            x: { type: "number" },
            y: { type: "number" },
            width: { type: "number" },
            height: { type: "number" },
          },
          required: ["x", "y", "width", "height"],
        },
        save_path: { type: "string", description: "Optional disk path to save screenshot." },
        snapshot: { type: "boolean", description: "Include visible DOM/element outline." },
        close: { type: "boolean", description: "Close persistent browser." },
      },
    },
  },
  {
    name: "view_image",
    description:
      "Fast local image viewer. Resizes large images to max_dim (default 960px) JPEG to cut vision tokens/encode time 3-5x, or crops a pixel-crisp clip.",
    inputSchema: {
      type: "object",
      properties: {
        file_path: { type: "string" },
        max_dim: { type: "integer", description: "Max side px (default 960, max 1600)." },
        format: { type: "string", enum: ["jpeg", "png"] },
        quality: { type: "integer", description: "JPEG quality 45-95 (default 84)." },
        clip: {
          type: "object",
          description: "Optional crop {x,y,width,height} before resize.",
          properties: {
            x: { type: "integer" },
            y: { type: "integer" },
            width: { type: "integer" },
            height: { type: "integer" },
          },
          required: ["x", "y", "width", "height"],
        },
      },
      required: ["file_path"],
    },
  },
  {
    name: "multi_edit",
    description:
      "Atomic multi-edit file patcher + instant syntax/lint/type check (node/py_compile/ruff/tsc/JSON) + optional verify_cmd to edit and run a test in ONE turn.",
    inputSchema: {
      type: "object",
      properties: {
        file_path: { type: "string" },
        edits: {
          type: "array",
          items: {
            type: "object",
            properties: {
              old_string: { type: "string" },
              new_string: { type: "string" },
              replace_all: { type: "boolean" },
            },
            required: ["old_string", "new_string"],
          },
        },
        check: { type: "string", enum: ["auto", "none"] },
        verify_cmd: { type: "string", description: "Optional shell command (e.g. pytest -q test_x.py) run immediately after edit in one turn." },
        verify_timeout_ms: { type: "integer", description: "Timeout ms for verify_cmd (default 15000, max 60000)." },
      },
      required: ["file_path", "edits"],
    },
  },
  {
    name: "read_batch",
    description:
      "Read multiple files or line ranges in ONE turn (replaces sequential read loops). Bounded to 28 KB total.",
    inputSchema: {
      type: "object",
      properties: {
        items: {
          type: "array",
          items: {
            type: "object",
            properties: {
              file_path: { type: "string" },
              offset: { type: "integer", description: "1-based start line (default 1)." },
              limit: { type: "integer", description: "Max lines (default 200, max 400)." },
            },
            required: ["file_path"],
          },
        },
      },
      required: ["items"],
    },
  },
];

// ─── Minimal MCP JSON-RPC 2.0 Stdio Loop ───────────────────────────────────

function writeMsg(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

async function handleRequest(msg) {
  const { id, method, params } = msg;
  if (method === "initialize") {
    const clientVer = params?.protocolVersion || "2024-11-05";
    writeMsg({
      jsonrpc: "2.0",
      id,
      result: {
        protocolVersion: clientVer,
        capabilities: { tools: {} },
        serverInfo: { name: "dsh-fast-mcp", version: "1.0.0" },
      },
    });
    return;
  }
  if (method === "notifications/initialized" || method === "notifications/cancelled") {
    return;
  }
  if (method === "ping") {
    writeMsg({ jsonrpc: "2.0", id, result: {} });
    return;
  }
  if (method === "tools/list") {
    writeMsg({ jsonrpc: "2.0", id, result: { tools: TOOLS } });
    return;
  }
  if (method === "tools/call") {
    const name = params?.name;
    const args = params?.arguments || {};
    try {
      let result;
      if (name === "browser") result = await handleBrowser(args);
      else if (name === "view_image") result = await handleViewImage(args);
      else if (name === "multi_edit") result = await handleMultiEdit(args);
      else if (name === "read_batch") result = await handleReadBatch(args);
      else throw new Error(`Unknown tool: ${name}`);
      writeMsg({ jsonrpc: "2.0", id, result });
    } catch (err) {
      writeMsg({
        jsonrpc: "2.0",
        id,
        result: {
          isError: true,
          content: [{ type: "text", text: `Error in ${name}: ${err.message || String(err)}` }],
        },
      });
    }
    return;
  }
  if (id !== undefined) {
    writeMsg({
      jsonrpc: "2.0",
      id,
      error: { code: -32601, message: `Method not found: ${method}` },
    });
  }
}

let chain = Promise.resolve();
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try {
    msg = JSON.parse(trimmed);
  } catch {
    return;
  }
  chain = chain.then(() => handleRequest(msg)).catch((err) => {
    process.stderr.write(`[fast-mcp] unhandled: ${err.stack || err}\n`);
  });
});

async function shutdown() {
  await browser.close();
  process.exit(0);
}

rl.on("close", () => {
  chain.finally(() => shutdown());
});
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);

#!/usr/bin/env node
// Regression test for inkling-proxy.mjs: when the client disconnects, the upstream
// request must be dropped too (streaming, before headers, and non-streaming), and
// complete requests, keep-alive, telemetry and the 502 path must keep working.
// Uses a fake upstream on :18090 and a proxy on :18081; no GPU or model needed.
// usage: node tools/proxy-cancel-test.mjs [path/to/inkling-proxy.mjs] [telemetry.jsonl]
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const PROXY_SRC = process.argv[2] ||
  path.join(path.dirname(new URL(import.meta.url).pathname), "..", "configs", "deepseek-harness", "tools", "inkling-proxy.mjs");
const UP_PORT = 18090, PROXY_PORT = 18081, N = 24, GAP = 50;
const LOG = process.argv[3] || path.join(fs.mkdtempSync(path.join(os.tmpdir(), "inkling-proxy-test-")), "telemetry.jsonl");
try { fs.unlinkSync(LOG); } catch {}

let last = null;
const upstream = http.createServer((req, res) => {
  const st = { sent: 0, finished: false, closedEarly: false, tClose: null, t0: performance.now() };
  last = st;
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", async () => {
    let p = {};
    try { p = JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch {}
    let timer = null;
    res.on("close", () => {
      if (!st.finished) { st.closedEarly = true; st.tClose = performance.now() - st.t0; if (timer) clearInterval(timer); }
    });
    if (p.hdr_delay_ms) await sleep(p.hdr_delay_ms);
    if (st.closedEarly) return;
    if (p.stream) {
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
      timer = setInterval(() => {
        if (st.sent >= N) { clearInterval(timer); st.finished = true; res.end("data: [DONE]\n\n"); return; }
        st.sent++;
        res.write("data: " + JSON.stringify({ choices: [{ index: 0, delta: { content: "tok" + st.sent } }] }) + "\n\n");
      }, GAP);
    } else {
      // sin streaming: "genera" N pasos y recien entonces responde
      timer = setInterval(() => {
        st.sent++;
        if (st.sent >= N) {
          clearInterval(timer); st.finished = true;
          const body = JSON.stringify({ choices: [{ message: { role: "assistant", content: "x".repeat(N) } }], usage: { prompt_tokens: 3, completion_tokens: N } });
          res.writeHead(200, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
          res.end(body);
        }
      }, GAP);
    }
  });
});

// abortAt: null | "first" (tras el primer dato) | numero de ms tras enviar
function request(port, extra, abortAt, agent) {
  return new Promise((resolve) => {
    const body = JSON.stringify({ model: "x", messages: [{ role: "user", content: "hola" }], ...extra });
    const req = http.request(
      { host: "127.0.0.1", port, method: "POST", path: "/v1/chat/completions", agent,
        headers: { "content-type": "application/json", "content-length": Buffer.byteLength(body) } },
      (res) => {
        let buf = "", got = 0;
        res.on("data", (c) => { got++; buf += c; if (abortAt === "first" && got === 1) { req.destroy(); resolve({ status: res.statusCode, buf }); } });
        res.on("end", () => resolve({ status: res.statusCode, buf }));
        res.on("error", () => {});
      });
    req.on("error", (e) => resolve({ status: 0, buf: "", err: e.code || e.message }));
    req.end(body);
    if (typeof abortAt === "number") setTimeout(() => { req.destroy(); resolve({ status: -1, buf: "" }); }, abortAt);
  });
}

await new Promise((r) => upstream.listen(UP_PORT, "127.0.0.1", r));
const proxy = spawn(process.execPath, [PROXY_SRC], {
  env: { ...process.env, PROXY_PORT: String(PROXY_PORT), UPSTREAM_URL: "http://127.0.0.1:" + UP_PORT, INKLING_PROXY_LOG: LOG },
  stdio: ["ignore", "pipe", "pipe"],
});
let proxyOut = "", proxyErr = "", proxyExit = null;
proxy.stdout.on("data", (d) => (proxyOut += d));
proxy.stderr.on("data", (d) => (proxyErr += d));
proxy.on("exit", (c) => (proxyExit = c));
for (let i = 0; i < 50 && !proxyOut.includes("[inkling-proxy]"); i++) await sleep(100);

let fails = 0;
const check = (name, ok, detail) => { if (!ok) fails++; console.log((ok ? "OK    " : "FALLA ") + name + " -> " + detail); };
const nData = (b) => b.split("\n").filter((l) => l.startsWith("data: {")).length;
const WAIT = N * GAP + 500;

// --- peticiones completas ---
{
  const r = await request(PROXY_PORT, { stream: true }, null);
  check("streaming completo", r.status === 200 && nData(r.buf) === N && r.buf.includes("[DONE]"), nData(r.buf) + "/" + N + " [DONE]=" + r.buf.includes("[DONE]"));
}
{
  const r = await request(PROXY_PORT, { stream: false }, null);
  let ok = false; try { ok = JSON.parse(r.buf).usage.completion_tokens === N; } catch {}
  check("sin streaming completo", r.status === 200 && ok, "status " + r.status + ", JSON valido=" + ok);
}
{
  const agent = new http.Agent({ keepAlive: true, maxSockets: 1 });
  let all = true;
  for (let i = 0; i < 3; i++) { const r = await request(PROXY_PORT, { stream: i !== 1 }, null, agent); all = all && r.status === 200 && r.buf.length > 0 && last.finished; }
  agent.destroy();
  check("3 seguidas por la misma conexion (keep-alive)", all, "todas completas=" + all);
}

// --- cancelaciones ---
for (const [name, extra, abortAt, port] of [
  ["directo, streaming (referencia)", { stream: true }, "first", UP_PORT],
  ["proxy, streaming tras el 1er fragmento", { stream: true }, "first", PROXY_PORT],
  ["proxy, streaming antes de las cabeceras", { stream: true, hdr_delay_ms: 600 }, 200, PROXY_PORT],
  ["proxy, sin streaming a mitad de la espera", { stream: false }, 300, PROXY_PORT],
]) {
  last = null;
  await request(port, extra, abortAt);
  await sleep(WAIT + (extra.hdr_delay_ms || 0));
  const st = last;
  check("cancelar: " + name, st.closedEarly && st.sent < N / 2,
    "upstream " + st.sent + "/" + N + (st.closedEarly ? ", cerrado a los " + st.tClose.toFixed(0) + " ms" : ", nadie lo cerro"));
}

// --- tras cancelar, todo sigue funcionando ---
{
  const r = await request(PROXY_PORT, { stream: true }, null);
  check("streaming completo despues de cancelar", nData(r.buf) === N && r.buf.includes("[DONE]"), nData(r.buf) + "/" + N);
}
{
  const r = await new Promise((resolve) => http.get({ host: "127.0.0.1", port: PROXY_PORT, path: "/proxy/stats" }, (res) => {
    let b = ""; res.on("data", (c) => (b += c)); res.on("end", () => resolve(b)); }));
  const rows = fs.readFileSync(LOG, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const canc = rows.filter((x) => x.cancelled);
  let statsOk = false; try { statsOk = JSON.parse(r).count === rows.length; } catch {}
  check("telemetria", canc.length === 3 && rows.length === 9 && statsOk,
    rows.length + " filas, " + canc.length + " canceladas (status " + canc.map((x) => x.status).join(",") + "), /proxy/stats valido=" + statsOk);
}

// --- upstream caido: el cliente vivo sigue recibiendo 502 ---
await new Promise((r) => { upstream.closeAllConnections(); upstream.close(r); });
{
  const r = await request(PROXY_PORT, { stream: true }, null);
  check("upstream caido devuelve 502", r.status === 502 && r.buf.includes("Upstream connect error"), "status " + r.status);
}

await sleep(200);
check("el proxy sigue vivo y sin errores en stderr", proxyExit === null && proxyErr.trim() === "", "exit=" + proxyExit + ", stderr=" + JSON.stringify(proxyErr.slice(0, 200)));
proxy.kill("SIGTERM");
await sleep(200);
console.log(fails === 0 ? "RESULTADO: todo OK" : "RESULTADO: " + fails + " fallas");
process.exit(fails === 0 ? 0 : 1);

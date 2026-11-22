#!/usr/bin/env python3
"""Pruebas funcionales del servidor: chat, vision, JSON schema (gramatica -> muestreo CPU), tool call, logprobs.

Uso: server-smoke.py <puerto>     (el servidor debe tener --mmproj para la prueba de vision; start-server.sh lo carga por defecto)
Termina con "RESULTADO: TODO OK" o lista los fallos.
"""
import base64, json, struct, sys, time, urllib.request, zlib
port = int(sys.argv[1])
def post(path, payload, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
def png(w, h, fn):
    raw = b"".join(b"\x00" + b"".join(bytes(fn(x, y)) for x in range(w)) for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
ok = True
def check(name, cond, detail):
    global ok
    ok = ok and cond
    print("[%s] %s: %s" % ("OK" if cond else "FALLO", name, detail), flush=True)

# 1) chat simple
d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Responde solo con el resultado: cuanto es 17 * 23?"}], "max_tokens": 600})
m = d["choices"][0]["message"]; txt = (m.get("content") or "")
check("chat", "391" in txt, "respuesta=%r | razonamiento=%d chars | %.1f t/s" % (txt[:80], len(m.get("reasoning_content") or ""), d["timings"]["predicted_per_second"]))

# 2) vision: mitad izquierda roja, mitad derecha azul
img = png(256, 256, lambda x, y: (220, 20, 20) if x < 128 else (20, 20, 220))
uri = "data:image/png;base64," + base64.b64encode(img).decode()
try:
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "The image has two halves with different colors. Answer in one short sentence: which color is on the left and which on the right?"},
        {"type": "image_url", "image_url": {"url": uri}}]}], "max_tokens": 800, "chat_template_kwargs": {"reasoning_effort": "low"}})
    txt = (d["choices"][0]["message"].get("content") or "").lower()
    check("vision", "red" in txt and "blue" in txt and txt.find("red") < txt.find("blue"), "respuesta=%r | prompt %d tok" % (txt[:120], d["timings"]["prompt_n"]))
except Exception as e:
    check("vision", False, "excepcion: %s" % e)

# 3) JSON schema -> gramatica -> el servidor debe caer al muestreo en CPU sin romperse
schema = {"type": "object", "properties": {"ciudad": {"type": "string"}, "poblacion_millones": {"type": "number"}}, "required": ["ciudad", "poblacion_millones"]}
try:
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Dame la capital de Argentina y su poblacion aproximada en millones."}], "max_tokens": 700,
                                      "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}}, "chat_template_kwargs": {"reasoning_effort": "low"}})
    txt = d["choices"][0]["message"].get("content") or ""
    j = json.loads(txt)
    check("json_schema", "ciudad" in j and "buenos" in str(j["ciudad"]).lower(), "json=%s | %.1f t/s" % (j, d["timings"]["predicted_per_second"]))
except Exception as e:
    check("json_schema", False, "excepcion: %s" % e)

# 4) tool call
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
try:
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is the weather in Rosario right now? Use the tool."}], "tools": tools, "max_tokens": 700, "chat_template_kwargs": {"reasoning_effort": "low"}})
    m = d["choices"][0]["message"]; tc = m.get("tool_calls") or []
    args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
    check("tool_call", bool(tc) and tc[0]["function"]["name"] == "get_weather" and "rosario" in str(args.get("city", "")).lower(), "tool_calls=%s | %.1f t/s" % (json.dumps(tc)[:140], d["timings"]["predicted_per_second"]))
except Exception as e:
    check("tool_call", False, "excepcion: %s" % e)

# 5) logprobs (n_probs) con muestreo en backend activo
try:
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Di hola."}], "max_tokens": 40, "logprobs": True, "top_logprobs": 3, "chat_template_kwargs": {"reasoning_effort": "none"}})
    lp = d["choices"][0].get("logprobs") or {}
    c = lp.get("content") or []
    check("logprobs", len(c) > 0 and len(c[0].get("top_logprobs") or []) == 3, "n=%d primer=%s" % (len(c), json.dumps(c[0])[:120] if c else None))
except Exception as e:
    check("logprobs", False, "excepcion: %s" % e)
print("RESULTADO:", "TODO OK" if ok else "HAY FALLOS")

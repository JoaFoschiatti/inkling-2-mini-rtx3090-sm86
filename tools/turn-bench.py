#!/usr/bin/env python3
"""Simula turnos cortos de un agente sobre un contexto ya cacheado y mide el coste por turno.

Uso: turn-bench.py --port P [--ctx-chars N] [--turns K] [--max-tokens M] [--temp T] [--effort none|medium]
Cada turno agrega la respuesta anterior y un mensaje corto del usuario. Mide por turno:
  wall (ms), ttft (ms hasta el primer delta), prompt_n / prompt_ms, predicted_n / predicted_ms, aceptacion de borradores.
"""
import argparse, hashlib, json, statistics as st, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--ctx-chars", type=int, default=48000)
ap.add_argument("--turns", type=int, default=12)
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--effort", default="none")
ap.add_argument("--corpus", nargs="*", default=[])
ap.add_argument("--label", default="")
ap.add_argument("--skip", type=int, default=2, help="turnos iniciales que no entran en la mediana")
ap.add_argument("--json", default="")
args = ap.parse_args()

text = ""
for path in args.corpus:
    with open(path, "r", errors="replace") as f:
        text += f.read() + "\n"
text = text[: args.ctx_chars]

QUESTIONS = [
    "Resume en una frase de que trata el texto.",
    "Nombra dos herramientas o comandos que aparezcan en el texto.",
    "Que opcion de linea de comandos controla el tamano del contexto?",
    "Escribe una funcion corta en C que sume dos enteros.",
    "Indica en una linea que hace el servidor al recibir una peticion.",
    "Da un ejemplo breve de JSON con dos claves.",
    "Cual es la diferencia entre lote y sublote segun el texto? Responde breve.",
    "Escribe un bucle for en Python que imprima del 0 al 4.",
    "Menciona un backend de aceleracion que aparezca en el texto.",
    "Explica en una frase que es la cuantizacion.",
    "Escribe la firma de una funcion C++ que reciba un vector de enteros.",
    "Que hace la opcion de capas en GPU? Una frase.",
    "Da un comando de ejemplo para compilar con CMake.",
    "Escribe un struct en C con dos campos.",
    "En una frase: para que sirve una plantilla de chat?",
    "Lista tres palabras clave del texto separadas por comas.",
]

def call(messages):
    payload = {
        "model": "inkling", "messages": messages, "max_tokens": args.max_tokens, "stream": True,
        "temperature": args.temp, "seed": args.seed, "cache_prompt": True,
        "chat_template_kwargs": {"reasoning_effort": args.effort},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; timings = None; content = ""; reasoning = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if "timings" in d:
                timings = d["timings"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                c = delta.get("content") or ""; rc = delta.get("reasoning_content") or ""
                if (c or rc) and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                content += c; reasoning += rc
    wall = (time.perf_counter() - t0) * 1000
    return wall, ttft, timings, content, reasoning

messages = [{"role": "system", "content": "Eres un asistente de programacion. Responde de forma breve."},
            {"role": "user", "content": "Documento de referencia:\n\n" + text + "\n\nConfirma con una palabra que lo has leido."}]
wall, ttft, t, content, reasoning = call(messages)
print("[%s] prefill: %d tok en %.0f ms (%.0f t/s), wall %.0f ms" % (args.label, t["prompt_n"], t["prompt_ms"], t.get("prompt_per_second", 0), wall), flush=True)
rows = []
for k in range(args.turns):
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    messages.append(msg)
    messages.append({"role": "user", "content": "Turno %d. %s" % (k + 1, QUESTIONS[k % len(QUESTIONS)])})
    wall, ttft, t, content, reasoning = call(messages)
    row = {"turn": k + 1, "wall": wall, "ttft": ttft or 0.0, "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"],
           "pred_n": t["predicted_n"], "pred_ms": t["predicted_ms"], "cache_n": t.get("cache_n", 0),
           "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0),
           "h": hashlib.sha1((reasoning + "|" + content).encode()).hexdigest()[:10]}
    rows.append(row)
    print("  turno %2d: wall %7.1f  ttft %6.1f  prompt %3d tok %6.1f ms  gen %3d tok %6.1f ms (%.1f ms/tok)  borr %d/%d" % (
        row["turn"], row["wall"], row["ttft"], row["prompt_n"], row["prompt_ms"], row["pred_n"], row["pred_ms"],
        row["pred_ms"] / max(1, row["pred_n"]), row["draft_acc"], row["draft_n"]), flush=True)

use = rows[args.skip:]
med = lambda key: st.median(r[key] for r in use)
summary = {"label": args.label, "turns": len(use), "wall": med("wall"), "ttft": med("ttft"), "prompt_ms": med("prompt_ms"),
           "pred_ms": med("pred_ms"), "prompt_n": med("prompt_n"), "pred_n": med("pred_n"),
           "overhead": st.median(r["wall"] - r["prompt_ms"] - r["pred_ms"] for r in use),
           "sum_wall": sum(r["wall"] for r in use),
           "draft_acc": sum(r["draft_acc"] for r in use), "draft_n": sum(r["draft_n"] for r in use),
           "tokens": [r["pred_n"] for r in rows], "out_hash": hashlib.sha1("".join(r["h"] for r in rows).encode()).hexdigest()[:12]}
print("[%s] mediana: wall %.1f ms | ttft %.1f | prompt %.1f ms (%d tok) | gen %.1f ms (%d tok) | resto %.1f ms | suma wall %.0f ms | borr %d/%d" % (
    args.label, summary["wall"], summary["ttft"], summary["prompt_ms"], summary["prompt_n"], summary["pred_ms"], summary["pred_n"],
    summary["overhead"], summary["sum_wall"], summary["draft_acc"], summary["draft_n"]), "| salida", summary["out_hash"], flush=True)
if args.json:
    with open(args.json, "a") as f:
        f.write(json.dumps(summary) + "\n")

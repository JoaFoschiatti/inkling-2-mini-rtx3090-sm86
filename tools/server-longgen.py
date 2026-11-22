#!/usr/bin/env python3
"""Generacion larga por streaming: t/s total y por ventanas de 256 tokens (detecta costes que crecen con la longitud o con la temperatura).

Uso: server-longgen.py <puerto> <max_tokens> [reasoning_effort=high]
"""
import json, sys, time, urllib.request
port = int(sys.argv[1]); n = int(sys.argv[2]); effort = sys.argv[3] if len(sys.argv) > 3 else "high"
payload = {"model": "inkling", "max_tokens": n, "stream": True, "chat_template_kwargs": {"reasoning_effort": effort},
           "messages": [{"role": "user", "content": "Escribe una guia tecnica muy extensa (varios miles de palabras) sobre el diseno de bases de datos distribuidas: consenso, replicacion, particionado, transacciones y recuperacion ante fallos. Incluye ejemplos de codigo."}]}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
ts = []; timings = None
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            d = json.loads(line[6:])
            if "timings" in d: timings = d["timings"]
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                ts.append(time.time())
print("total: %d tok, %.2f t/s (server timings) | chunks recibidos: %d" % (timings["predicted_n"], timings["predicted_per_second"], len(ts)))
w = 256
print("t/s por ventana de %d chunks:" % w, " ".join("%.1f" % (w / (ts[i + w] - ts[i])) for i in range(0, len(ts) - w, w)))

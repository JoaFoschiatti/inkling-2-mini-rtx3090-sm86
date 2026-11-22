#!/usr/bin/env python3
"""Mide t/s reales por la API del servidor (ruta completa: muestreo + streaming).

Uso: server-bench.py <puerto> <max_tokens> <repeticiones> ['<json extra>']
  ./server-bench.py 8080 512 3 '{"_stream": true, "chat_template_kwargs": {"reasoning_effort": "high"}}'
El JSON extra se mezcla en la peticion; "_stream" y "_prompt" son claves propias de este script.
Por defecto usa reasoning_effort=none y sin streaming. Comparar siempre con la GPU a la misma temperatura inicial.
"""
import json, sys, time, urllib.request
port = int(sys.argv[1]); n = int(sys.argv[2]); reps = int(sys.argv[3]); extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
stream = extra.pop("_stream", False)
prompt = extra.pop("_prompt", "Escribe un ensayo largo y detallado sobre la historia de la computacion, desde el abaco hasta los transformers.")
def one():
    payload = {"model": "inkling", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": stream,
               "chat_template_kwargs": {"reasoning_effort": "none"}}
    payload.update(extra)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    timings = None; ntok = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        if stream:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    d = json.loads(line[6:])
                    if "timings" in d: timings = d["timings"]
        else:
            d = json.loads(r.read().decode()); timings = d.get("timings")
    return timings, time.time() - t0
res = []
for i in range(reps):
    t, wall = one()
    res.append(t["predicted_per_second"])
    print("  rep %d: %d tok, %.2f t/s (%.3f ms/tok), prompt %d tok @ %.0f t/s, wall %.2fs" % (i, t["predicted_n"], t["predicted_per_second"], t["predicted_per_token_ms"], t["prompt_n"], t.get("prompt_per_second", 0), wall), flush=True)
res.sort()
print("  mediana: %.2f t/s  max: %.2f t/s" % (res[len(res)//2], res[-1]))

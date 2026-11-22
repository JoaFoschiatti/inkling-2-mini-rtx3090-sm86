#!/usr/bin/env python3
"""Muestrea 1 token N veces (semillas 0..N-1) para varios contextos y guarda los ids.

Uso (con el servidor ya levantado en <puerto>):
    BACKEND_SAMPLING=1 PORT=18089 ../start-server.sh &   ->  ./sampler-test.py 18089 1000 gpu.json
    BACKEND_SAMPLING=0 PORT=18089 ../start-server.sh &   ->  ./sampler-test.py 18089 1000 cpu.json
    ./sampler-compare.py gpu.json cpu.json
Con la misma semilla, el muestreo en GPU y en CPU deben elegir el mismo token.
"""
import json, sys, time, urllib.request
port = int(sys.argv[1]); n = int(sys.argv[2]); out = sys.argv[3]
PROMPTS = {
    "amplio_en": "Once upon a time, in a small village by the sea, there lived a",
    "amplio_es": "La mejor forma de aprender a programar es",
    "pico_en":   "The capital of France is",
    "codigo":    "def fibonacci(n):\n    if n <",
}
def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())
res = {}
for name, prompt in PROMPTS.items():
    base = {"prompt": prompt, "n_predict": 1, "cache_prompt": True, "return_tokens": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}
    post(dict(base, seed=123456))  # deja el prompt en cache; a partir de aqui cada peticion re-evalua solo el ultimo token
    post(dict(base, seed=123457))
    toks = []
    t0 = time.time()
    for i in range(n):
        d = post(dict(base, seed=i))
        t = d.get("tokens") or []
        toks.append(t[0] if t else -1)
    res[name] = toks
    print("%-10s %d muestras en %.1fs | distintos: %d | mas frecuente: %s" % (name, n, time.time() - t0, len(set(toks)), max(set(toks), key=toks.count)), flush=True)
    time.sleep(8)
# distribucion exacta (CPU, logits pre-muestreo) de los 40 tokens mas probables
probs = {}
for name, prompt in PROMPTS.items():
    d = post({"prompt": prompt, "n_predict": 1, "cache_prompt": True, "n_probs": 40, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 1})
    cp = d.get("completion_probabilities") or []
    if cp:
        top = cp[0].get("top_logprobs") or cp[0].get("top_probs") or []
        probs[name] = [(t.get("id"), t.get("token"), t.get("logprob", None), t.get("prob", None)) for t in top]
json.dump({"samples": res, "probs": probs}, open(out, "w"))

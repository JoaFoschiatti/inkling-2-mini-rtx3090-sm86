#!/usr/bin/env python3
"""Generaciones largas por regimen (codigo, razonamiento, tool call, reescritura) para medir t/s y aceptacion."""
import argparse, json, sys, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--cases", default="code,think,tool,rewrite")
ap.add_argument("--ctx-file", default="")
ap.add_argument("--ctx-chars", type=int, default=0)
ap.add_argument("--label", default="")
ap.add_argument("--json", default="")
args = ap.parse_args()

ctx = ""
if args.ctx_file and args.ctx_chars > 0:
    ctx = open(args.ctx_file, errors="replace").read()[: args.ctx_chars]

CODE_SNIPPET = '''
#include <vector>
#include <string>
#include <unordered_map>

struct Item { std::string name; int qty; double price; };

double total(const std::vector<Item> & items) {
    double sum = 0.0;
    for (const auto & it : items) {
        sum += it.qty * it.price;
    }
    return sum;
}

std::unordered_map<std::string, int> count_by_name(const std::vector<Item> & items) {
    std::unordered_map<std::string, int> out;
    for (const auto & it : items) {
        out[it.name] += it.qty;
    }
    return out;
}

std::vector<Item> filter_expensive(const std::vector<Item> & items, double min_price) {
    std::vector<Item> out;
    for (const auto & it : items) {
        if (it.price >= min_price) {
            out.push_back(it);
        }
    }
    return out;
}
'''

TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Escribe un archivo de texto en disco.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]

CASES = {
    "code":    {"effort": "none",   "prompt": "Escribe en Python una clase LRUCache completa con metodos get y put, con docstrings y un pequeno bloque de pruebas al final. Solo el codigo."},
    "think":   {"effort": "medium", "prompt": "Analiza con detalle las ventajas y desventajas de usar un sistema de tipos gradual en un lenguaje dinamico grande, con ejemplos concretos, y da una recomendacion razonada."},
    "tool":    {"effort": "none",   "prompt": "Usa la herramienta write_file para crear el archivo utils/strings.py con cuatro funciones utiles de manejo de cadenas (slugify, truncate, is_palindrome, count_words), bien documentadas.", "tools": TOOLS},
    "rewrite": {"effort": "none",   "prompt": "Reescribe el siguiente codigo C++ renombrando Item a Product y qty a quantity, sin cambiar nada mas. Devuelve solo el codigo completo.\n" + CODE_SNIPPET},
}

def call(case):
    c = CASES[case]
    user = (("Documento de referencia:\n\n" + ctx + "\n\n") if ctx else "") + c["prompt"]
    payload = {"model": "inkling", "messages": [{"role": "user", "content": user}], "max_tokens": args.max_tokens, "stream": True,
               "temperature": args.temp, "seed": args.seed, "chat_template_kwargs": {"reasoning_effort": c["effort"]}}
    if "tools" in c:
        payload["tools"] = c["tools"]
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); timings = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                d = json.loads(line[6:])
                if "timings" in d:
                    timings = d["timings"]
    return timings, (time.perf_counter() - t0) * 1000

for case in args.cases.split(","):
    t, wall = call(case)
    acc = t.get("draft_n_accepted", 0); dn = t.get("draft_n", 0)
    row = {"label": args.label, "case": case, "pred_n": t["predicted_n"], "tps": t["predicted_per_second"], "ms_tok": t["predicted_per_token_ms"],
           "prompt_n": t["prompt_n"], "draft_acc": acc, "draft_n": dn, "wall": wall}
    print("[%s] %-8s %4d tok  %7.2f t/s  (%.2f ms/tok)  prompt %d tok  borradores %d/%d (%.0f%%)" % (
        args.label, case, row["pred_n"], row["tps"], row["ms_tok"], row["prompt_n"], acc, dn, 100.0 * acc / max(1, dn)), flush=True)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(row) + "\n")

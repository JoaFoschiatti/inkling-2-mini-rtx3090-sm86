#!/usr/bin/env python3
"""Compara dos salidas de sampler-test.py (mismas semillas) y hace chi-cuadrado contra las probabilidades exactas.

Uso: ./sampler-compare.py gpu.json cpu.json
"""
import json, math, sys
g = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2]))
tot_same = tot = 0
for name in g["samples"]:
    a, b = g["samples"][name], c["samples"].get(name)
    if not b: continue
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    same = sum(1 for x, y in zip(a, b) if x == y); tot_same += same; tot += n
    print("%-10s misma semilla -> mismo token: %d/%d (%.3f%%)" % (name, same, n, 100.0 * same / n))
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y][:5]
    if diff: print("    primeras diferencias (semilla, A, B):", diff)
    pr = c["probs"].get(name) or g["probs"].get(name) or []
    for lab, s in (("A", a), ("B", b)):
        chi = 0.0; df = 0; mass = 0.0; cnt_in = 0
        for tid, tok, lp, p in pr:
            p = p if p is not None else math.exp(lp)
            e = n * p
            if e < 25: continue
            o = s.count(tid); chi += (o - e) ** 2 / e; df += 1; mass += p; cnt_in += o
        e_rest = n * (1 - mass); o_rest = n - cnt_in
        if e_rest >= 25: chi += (o_rest - e_rest) ** 2 / e_rest; df += 1
        df -= 1
        z = (chi - df) / math.sqrt(2 * df) if df > 0 else 0
        print("    chi2 %s vs prob. exactas: chi2=%.1f con %d gl (z=%+.2f; |z|<3 compatible) | masa cubierta=%.4f" % (lab, chi, df, z, mass))
if tot: print("TOTAL: %d/%d iguales (%.4f%%)" % (tot_same, tot, 100.0 * tot_same / tot))

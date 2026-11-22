#!/usr/bin/env python3
"""Lector minimo de GGUF (sin numpy): metadatos + lista de tensores."""
import struct, sys, collections
TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16'}
def rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))
def rstr(f):
    (n,) = rd(f, '<Q'); return f.read(n).decode('utf-8', 'replace')
def rval(f, t):
    m = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
    if t in m: return rd(f, m[t])[0]
    if t == 8: return rstr(f)
    if t == 9:
        (et,) = rd(f, '<I'); (n,) = rd(f, '<Q')
        out = [rval(f, et) for _ in range(n)]
        return out if n <= 16 else f"<array len={n} type={et}> {out[:6]}..."
    raise ValueError(t)
path = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else 'all'
with open(path, 'rb') as f:
    magic = f.read(4); (ver,) = rd(f, '<I'); (nt,) = rd(f, '<Q'); (nkv,) = rd(f, '<Q')
    print(f"magic={magic} ver={ver} tensors={nt} kv={nkv}")
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, '<I'); v = rval(f, t)
        if mode in ('all', 'kv'):
            s = str(v)
            if 'chat_template' in k or len(s) > 300: s = s[:160] + '...'
            print(f"  {k} = {s}")
    tens = []
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f, '<I'); dims = rd(f, '<' + 'Q'*nd); (tt,) = rd(f, '<I'); (off,) = rd(f, '<Q')
        tens.append((name, dims, tt, off))
    if mode in ('all', 'tensors'):
        tens_sorted = sorted(tens, key=lambda x: x[3])
        sizes = {}
        for i, t in enumerate(tens_sorted):
            nxt = tens_sorted[i+1][3] if i+1 < len(tens_sorted) else None
            sizes[t[0]] = (nxt - t[3]) if nxt is not None else -1
        bytype = collections.Counter(); cnt = collections.Counter()
        for name, dims, tt, off in tens:
            tn = TYPES.get(tt, f'T{tt}')
            bytype[tn] += max(sizes[name], 0); cnt[tn] += 1
        print("--- bytes por tipo ---")
        for k, v in bytype.most_common(): print(f"  {k:8s} n={cnt[k]:5d}  {v/2**30:8.3f} GiB")
        print("--- tensores (capa 0 / 1 y globales) ---")
        for name, dims, tt, off in tens:
            if name.startswith('blk.') and not (name.startswith('blk.0.') or name.startswith('blk.5.')): continue
            print(f"  {name:48s} {str(dims):28s} {TYPES.get(tt, tt)!s:8s} {sizes[name]/2**20:9.2f} MiB")
        print("n blk:", len({n.split('.')[1] for n, *_ in tens if n.startswith('blk.')}))

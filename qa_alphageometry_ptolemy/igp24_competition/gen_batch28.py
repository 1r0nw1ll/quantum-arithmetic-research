#!/usr/bin/env python3
"""
Batch28: New (T,r) pairs including r=0 from complex degree-8/12 seeds.

Strategies:
A. polsubcyclo(15,8) × S3 cubic → degree-24, r=0, T=(C4×C2)×S3 [new T, new r=0]
B. polsubcyclo(24,8) × S3 cubic → degree-24, r=0, T=(C2³)×S3 [new T, new r=0]
C. polcompositum(D4-0real×S3_f12, x²-d) → degree-24, r=0, T=(S3×D4)×C2 [new r=0]
D. polcompositum(S3×V4_f12, x²-d) → degree-24, r=24, T=(S3×V4)×C2 [new T]
E. More cyclotomic polsubcyclo(m,12) shift trick with new m values
F. Fallback S3×D4
"""
import subprocess, json, sys

def gp(cmd, timeout=300):
    script = f"default(parisizemax,800000000);\n{cmd}\nquit\n"
    r = subprocess.run(['gp','-q'], input=script, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def parse_vec(s):
    start = s.rfind('['); end = s.rfind(']')
    if start < 0: return None
    return [c.strip() for c in s[start+1:end].split(',')]

polys = []
seen_coeffs = set()

def add(label, coeffs):
    key = tuple(coeffs)
    if key in seen_coeffs: return False
    seen_coeffs.add(key)
    polys.append({'label': label, 'coeffs': coeffs})
    return True

C_WIDE = list(range(-60, 61))
D_VALS = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 23, 26, 29, 30, 31, 33]

def try_shift(seed_pv, label_prefix, c_vals, max_per_seed=None):
    found_r = set()
    for c in c_vals:
        if len(polys) >= 100: break
        if max_per_seed and len(found_r) >= max_per_seed: break
        out = gp(f"""
h={seed_pv};
g24=subst(h,'x,x^2+({c}));
if(poldegree(g24)!=24, quit);
if(!polisirreducible(g24), quit);
r=polsturm(g24,-10^9,10^9);
print("OK|",r,"|",Vecrev(g24));
""")
        if not out.startswith("OK|"): continue
        parts = out.split("|",2)
        try: r_val = int(parts[1])
        except: continue
        if r_val in found_r: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"{label_prefix}_c{c}_r{r_val}", coeffs):
                found_r.add(r_val)
                print(f"  {label_prefix} c={c:>4} r={r_val:>2} total={len(polys)}", file=sys.stderr)
    return found_r

def try_compositum24(f_pv, label, d_vals, max_per_seed=5):
    found_r = set()
    for d in d_vals:
        if len(polys) >= 100: break
        if len(found_r) >= max_per_seed: break
        out = gp(f"""
f={f_pv};
comp=polcompositum(f,x^2-{d});
if(#comp==0, quit);
f24=comp[1];
if(poldegree(f24)!=24||!polisirreducible(f24), quit);
r=polsturm(f24,-10^9,10^9);
print("OK|",r,"|",Vecrev(f24));
""")
        if not out.startswith("OK|"): continue
        parts = out.split("|",2)
        try: r_val = int(parts[1])
        except: continue
        if r_val in found_r: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"{label}_d{d}_r{r_val}", coeffs):
                found_r.add(r_val)
                print(f"  {label} d={d} r={r_val} total={len(polys)}", file=sys.stderr)
    return found_r

S3_CUBICS = [("x^3-4*x-1","a"),("x^3-5*x-1","b"),("x^3-7*x-1","c"),
             ("x^3-7*x+7","d"),("x^3-5*x+3","e"),("x^3-11*x-4","f"),
             ("x^3-13*x-4","g"),("x^3-8*x-4","h"),("x^3-16*x-4","i")]

def get_totally_real_subcyclo(m, d):
    out = gp(f"""
v=polsubcyclo({m},{d},'x);
if(type(v)=="t_POL",v=[v]);
if(type(v)!="t_VEC", quit);
apply(p->if(poldegree(p)=={d}&&polisirreducible(p)&&polsturm(p,-10^9,10^9)=={d},print("FOUND|",Vecrev(p))),v);
""")
    for line in out.split('\n'):
        line = line.strip()
        if not line.startswith("FOUND|"): continue
        coeffs = parse_vec(line.split("|",1)[1])
        if coeffs and len(coeffs) == d+1:
            return f"Polrev([{','.join(coeffs)}])"
    return None

# ================================================================
# STRATEGY A: polsubcyclo(15,8) × S3 → r=0, T=(C4×C2)×S3 (new T+r)
# polsubcyclo(15,8) has 0 real roots, Gal over Q = C4×C2 (abelian, order 8)
# × S3 cubic (3 real roots) → degree-24 with 0 real roots = r=0
# ================================================================
print("=== A: polsubcyclo(15,8) x S3 cubics (r=0) ===", file=sys.stderr)

v15_out = gp("v=polsubcyclo(15,8,'x); print(Vecrev(v));")
v15_coeffs = parse_vec(v15_out)
if v15_coeffs and len(v15_coeffs)==9:
    v15_pv = f"Polrev([{','.join(v15_coeffs)}])"
    print(f"  polsubcyclo(15,8) OK, deg=8", file=sys.stderr)
    for cubic, cname in S3_CUBICS:
        if len(polys) >= 9: break
        out = gp(f"""
v8={v15_pv};
f3={cubic};
if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3, quit);
comp=polcompositum(v8,f3);
if(#comp==0, quit);
f24=comp[1];
if(poldegree(f24)!=24||!polisirreducible(f24), quit);
r=polsturm(f24,-10^9,10^9);
print("OK|",r,"|",Vecrev(f24));
""")
        if not out.startswith("OK|"): continue
        parts = out.split("|",2)
        try: r_val = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1':
            if add(f"A_C4C2_{cname}_r{r_val}", coeffs):
                print(f"  A C4C2×S3 {cname} r={r_val} total={len(polys)}", file=sys.stderr)
else:
    print(f"  polsubcyclo(15,8) failed: {v15_out[:40]}", file=sys.stderr)

print(f"After A: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: polsubcyclo(24,8) × S3 → r=0, T=(C2³)×S3 (new T+r)
# polsubcyclo(24,8) has 0 real roots, Gal = C2×C2×C2 (E8, order 8)
# ================================================================
print("\n=== B: polsubcyclo(24,8) x S3 cubics (r=0) ===", file=sys.stderr)

v24_out = gp("v=polsubcyclo(24,8,'x); print(Vecrev(v));")
v24_coeffs = parse_vec(v24_out)
if v24_coeffs and len(v24_coeffs)==9:
    v24_pv = f"Polrev([{','.join(v24_coeffs)}])"
    print(f"  polsubcyclo(24,8) OK, deg=8", file=sys.stderr)
    for cubic, cname in S3_CUBICS:
        if len(polys) >= 18: break
        out = gp(f"""
v8={v24_pv};
f3={cubic};
if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3, quit);
comp=polcompositum(v8,f3);
if(#comp==0, quit);
f24=comp[1];
if(poldegree(f24)!=24||!polisirreducible(f24), quit);
r=polsturm(f24,-10^9,10^9);
print("OK|",r,"|",Vecrev(f24));
""")
        if not out.startswith("OK|"): continue
        parts = out.split("|",2)
        try: r_val = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1':
            if add(f"B_E8_{cname}_r{r_val}", coeffs):
                print(f"  B E8×S3 {cname} r={r_val} total={len(polys)}", file=sys.stderr)
else:
    print(f"  polsubcyclo(24,8) failed", file=sys.stderr)

print(f"After B: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: polcompositum(D4-0real×S3_f12, x²-d) → r=0
# T = (S3×D4)×C2 at r=0 (batch27 had r=12 and r=24 for this T)
# ================================================================
print("\n=== C: D4-0real x S3 compositum -> r=0 ===", file=sys.stderr)

d4_0real_quartics = ["x^4+x^2+2", "x^4+2*x^2+2", "x^4-x^2+3", "x^4+3*x^2+3"]

for d4q in d4_0real_quartics:
    if len(polys) >= 30: break
    for cubic, cname in S3_CUBICS[:3]:
        if len(polys) >= 30: break
        out = gp(f"""
f3={cubic}; f4={d4q};
if(!polisirreducible(f3)||!polisirreducible(f4), quit);
if(polsturm(f3,-10^9,10^9)!=3||polsturm(f4,-10^9,10^9)!=0, quit);
comp=polcompositum(f3,f4);
f12=comp[1];
if(poldegree(f12)!=12||!polisirreducible(f12), quit);
nr=polsturm(f12,-10^9,10^9);
print("OK|",nr,"|",Vecrev(f12));
""")
        if not out.startswith("OK|"): continue
        parts = out.split("|",2)
        nr_seed = int(parts[1])
        sc = parse_vec(parts[2])
        if not sc or len(sc)!=13: continue
        f12_pv = f"Polrev([{','.join(sc)}])"
        print(f"  C {cname}×{d4q[:8]} (nr12={nr_seed})", file=sys.stderr)
        r_found = try_compositum24(f12_pv, f"C_{cname}_{d4q[3:5]}", D_VALS[:5], max_per_seed=2)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After C: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: polcompositum(S3×V4_f12, x²-d) → r=24, T=(S3×V4)×C2 (new T)
# ================================================================
print("\n=== D: S3xV4 f12 compositum with x^2-d (r=24, new T) ===", file=sys.stderr)

v4_pairs = [(2,5), (2,7), (3,5), (2,11)]
for d1, d2 in v4_pairs:
    if len(polys) >= 40: break
    cubic = "x^3-4*x-1"
    out = gp(f"""
f3={cubic}; q1=x^2-{d1}; q2=x^2-{d2};
comp1=polcompositum(f3,q1);
f6=comp1[1];
comp2=polcompositum(f6,q2);
f12=comp2[1];
if(poldegree(f12)!=12||!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12, quit);
print("OK|",Vecrev(f12));
""")
    if not out.startswith("OK|"): continue
    sc = parse_vec(out.split("|",1)[1])
    if not sc or len(sc)!=13: continue
    f12_pv = f"Polrev([{','.join(sc)}])"
    print(f"  D S3V4 d={d1},{d2}: seed OK", file=sys.stderr)
    r_found = try_compositum24(f12_pv, f"D_S3V4_{d1}x{d2}", D_VALS[:4], max_per_seed=2)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After D: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY E: More cyclotomic degree-12 seeds (new m values)
# ================================================================
print("\n=== E: More cyclotomic degree-12 seeds ===", file=sys.stderr)

# batch26 used: 35,39,52,72,78,84,90,91,95,104,111,126,140,180
# batch27 used: 65,156,195,260,312,390,420,455,504,520,546,630
# New m values (highly composite, phi divisible by 12):
new_m_vals = [660, 780, 840, 910, 924, 1092, 1155, 1260, 1365, 1404, 1540, 1848, 2340, 2520, 3003, 3360]

for m in new_m_vals:
    if len(polys) >= 95: break
    seed_pv = get_totally_real_subcyclo(m, 12)
    if seed_pv is None:
        print(f"  E m={m}: none", file=sys.stderr)
        continue
    print(f"  E m={m}: found, shifting...", file=sys.stderr)
    r_found = try_shift(seed_pv, f"E_cyclo{m}", C_WIDE, max_per_seed=5)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After E: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY F: Fallback S3×D4
# ================================================================
if len(polys) < 100:
    print("\n=== F: Fallback S3xD4 ===", file=sys.stderr)
    d4_q = "x^4-6*x^2+2"
    f_cubics = ["x^3-23*x-4", "x^3-24*x-8", "x^3-25*x-7", "x^3-26*x-8",
                "x^3-27*x-9", "x^3-28*x-4"]
    for cubic in f_cubics:
        if len(polys) >= 100: break
        out = gp(f"""
f3={cubic}; f4={d4_q};
if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3, quit);
comp=polcompositum(f3,f4);
f12=comp[1];
if(poldegree(f12)!=12||!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12, quit);
print("OK|",Vecrev(f12));
""")
        if not out.startswith("OK|"): continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  F {cubic[:10]}: seed OK", file=sys.stderr)
        r_found = try_shift(seed_pv, f"F_{cubic[3:7]}", C_WIDE, max_per_seed=5)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch28_polys.json','w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch28_submission.txt','w') as f:
    for p in polys:
        f.write(','.join(p['coeffs'])+'\n')
print(f"Written to /tmp/batch28_submission.txt", file=sys.stderr)

from collections import Counter
sc_ctr = Counter()
rc = Counter()
for p in polys:
    label = p['label']
    for k in ['A_','B_','C_','D_','E_','F_']:
        if label.startswith(k):
            sc_ctr[k.strip('_')] += 1
            break
    for part in label.split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print("Breakdown:", dict(sc_ctr), file=sys.stderr)
print("r:", dict(sorted(rc.items())), file=sys.stderr)

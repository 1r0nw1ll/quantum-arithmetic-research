#!/usr/bin/env python3
"""
Batch24: Full r-value coverage for the new T-numbers discovered in batch23.

Strategy: use the same S3xD4 and C3xD4 compositum seeds from batch23,
but this time cover ALL available r-values (no max_new limit).
Also add: more diverse seeds from different cubic/quartic combinations,
and the well-known T87 seeds for any remaining open T14746 r-values.

We'll also try more seeds systematically to hit T14749, T14751 etc.
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

# Extended c-values covering all transition zones
C_VALS_FULL = list(range(-60, 61))  # integers from -60 to 60

def try_shift_full(seed_polrev, c_vals, label_prefix):
    """Apply shift trick for all c-vals, collect all distinct r-values."""
    found_r = set()
    for c in c_vals:
        if len(polys) >= 100: break
        out = gp(f"""
h={seed_polrev};
g24=subst(h,'x,x^2+({c}));
if(poldegree(g24)!=24, print("DEG"); quit);
if(!polisirreducible(g24), print("IRRED"); quit);
r=polsturm(g24,-10^9,10^9);
print("OK|",r,"|",Vecrev(g24));
""")
        if not out.startswith("OK|"): continue
        parts = out.split("|",2)
        try: r = int(parts[1])
        except: continue
        if r in found_r: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"{label_prefix}_c{c}_r{r}", coeffs):
                found_r.add(r)
                print(f"  {label_prefix} c={c:>4} r={r:>2} OK (total={len(polys)})", file=sys.stderr)
    return found_r

# ================================================================
# BATCH23 SEEDS (re-use for full r-value coverage)
# ================================================================

# S3 cubics (verified: Gal=S3 from disc check in PARI, totally real)
S3_CUBICS = [
    "x^3-4*x-1",   # disc=229 prime → S3
    "x^3-5*x-1",   # disc=473 → S3
    "x^3-7*x-1",   # disc=1345 → S3
    "x^3-5*x+3",   # disc=257 prime → S3
]

# C3 cubics (totally real, Gal=C3, simpler)
C3_CUBICS = [
    "x^3+x^2-2*x-1",  # C3: subfield of Q(zeta_7)
    "x^3-3*x+1",      # C3: subfield of Q(zeta_9)
    "x^3-x^2-2*x+1",  # C3
]

# D4 totally real quartics (splitting degree 8, not 24)
D4_QUARTICS = [
    "x^4-4*x^2+2",    # D4 totally real
    "x^4-4*x^2+1",    # D4 totally real
    "x^4-6*x^2+2",    # D4? totally real
    "x^4-10*x^2+1",   # D4? totally real
    "x^4-8*x^2+8",    # D4? totally real
]

# S4 totally real quartics with Gal=S4 (splitting degree 24)
# These are harder to find; try x^4+bx+c with 4 real roots
# Actually most x^4-bx^2+c type are D4 or smaller
# For S4: need quartic with non-biquadratic form AND 4 real roots
TR_S4_QUARTICS = [
    # x^4-bx^2+c type (check if Gal gives nfsplitting degree != 8)
    "x^4-x^3-3*x^2+x+1",   # check: might be C5 subfield?
    "x^4-x^3-4*x^2+4*x+1",  # check
    "x^4+x^3-3*x^2-x+1",    # check
    "x^4-3*x^3+x^2+x-1",    # check
]

print("=== A: S3 x D4 composita (full r-value coverage) ===", file=sys.stderr)
a_found = 0

for cubic in S3_CUBICS[:3]:
    for quartic in D4_QUARTICS[:3]:
        if len(polys) >= 45: break
        out = gp(f"""
f3={cubic};
f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), print("NOTIRRED"); quit);
nr3=polsturm(f3,-10^9,10^9);
nr4=polsturm(f4,-10^9,10^9);
if(nr3!=3||nr4!=4, print("NR|",nr3,nr4); quit);
comp=polcompositum(f3,f4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
if(!polisirreducible(f12), print("IRRED12"); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR12|",nr); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  A {cubic[:10]} x {quartic[:10]}: {out[:40]}", file=sys.stderr)
            continue
        seed_coeffs = parse_vec(out.split("|",1)[1])
        if not seed_coeffs or len(seed_coeffs)!=13: continue
        seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
        print(f"  A {cubic[:10]} x {quartic[:10]}: seed OK", file=sys.stderr)
        r_found = try_shift_full(seed_polrev, C_VALS_FULL, f"A_{cubic[:5]}_{quartic[:5]}")
        a_found += len(r_found)
        if r_found: print(f"    r-values: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy A: {a_found} polys, total={len(polys)}", file=sys.stderr)

print("\n=== B: C3 x D4 composita (full r-value coverage) ===", file=sys.stderr)
b_found = 0

for cubic in C3_CUBICS[:3]:
    for quartic in D4_QUARTICS[:3]:
        if len(polys) >= 75: break
        out = gp(f"""
f3={cubic};
f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), print("NOTIRRED"); quit);
nr3=polsturm(f3,-10^9,10^9);
nr4=polsturm(f4,-10^9,10^9);
if(nr3!=3||nr4!=4, print("NR|",nr3,nr4); quit);
comp=polcompositum(f3,f4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
if(!polisirreducible(f12), print("IRRED12"); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR12|",nr); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  B {cubic[:10]} x {quartic[:10]}: {out[:40]}", file=sys.stderr)
            continue
        seed_coeffs = parse_vec(out.split("|",1)[1])
        if not seed_coeffs or len(seed_coeffs)!=13: continue
        seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
        print(f"  B {cubic[:10]} x {quartic[:10]}: seed OK", file=sys.stderr)
        r_found = try_shift_full(seed_polrev, C_VALS_FULL, f"B_{cubic[:5]}_{quartic[:5]}")
        b_found += len(r_found)
        if r_found: print(f"    r-values: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy B: {b_found} polys, total={len(polys)}", file=sys.stderr)

print("\n=== C: S3 x new D4 quartics ===", file=sys.stderr)
c_found = 0

# More quartics with 4 real roots
more_quartics = [
    "x^4-5*x^2+3",     # Check splitting degree
    "x^4-6*x^2+7",
    "x^4-7*x^2+11",
    "x^4-8*x^2+14",
    "x^4-3*x^2+1",     # Check
    "x^4-5*x^2+2",     # Check
]

for quartic in more_quartics:
    if len(polys) >= 95: break
    cubic = "x^3-4*x-1"  # Use one S3 cubic
    out = gp(f"""
f3={cubic};
f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), print("NOTIRRED"); quit);
nr3=polsturm(f3,-10^9,10^9);
nr4=polsturm(f4,-10^9,10^9);
if(nr3!=3||nr4!=4, print("NR|",nr3,nr4); quit);
comp=polcompositum(f3,f4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
if(!polisirreducible(f12), print("IRRED12"); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR12|",nr); quit);
print("SEED|",Vecrev(f12));
""")
    if not out.startswith("SEED|"):
        print(f"  C {quartic[:12]}: {out[:40]}", file=sys.stderr)
        continue
    seed_coeffs = parse_vec(out.split("|",1)[1])
    if not seed_coeffs or len(seed_coeffs)!=13: continue
    seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
    print(f"  C {quartic[:12]}: seed OK", file=sys.stderr)
    r_found = try_shift_full(seed_polrev, C_VALS_FULL, f"C_{quartic[:8]}")
    c_found += len(r_found)
    if r_found: print(f"    r-values: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy C: {c_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# Fill to 100 with T87/T89 seeds if we're short
# T87r24 = y^12-132y^11+... (all real, roots spread 0.055..54)
# These likely give T14746 (D6-family) which is "fully scored" but
# might not be by us if we haven't submitted those specific r-values.
# ================================================================
if len(polys) < 100:
    print(f"\n=== D: T87/T89 shift families (fill remaining) ===", file=sys.stderr)
    T87_r24 = "Polrev([729,32076,514188,3844260,14581539,27842616,24546456,9280872,1620171,142380,6348,132,1])"
    T87_r0  = "Polrev([729,-32076,514188,-3844260,14581539,-27842616,24546456,-9280872,1620171,-142380,6348,-132,1])"
    T89_r24 = "Polrev([8,2112,91696,1385632,8299628,17725376,14466032,5103008,892214,83280,4172,104,1])"
    T89_r0  = "Polrev([8,-2112,91696,-1385632,8299628,-17725376,14466032,-5103008,892214,-83280,4172,-104,1])"

    for seed_name, seed in [("T87r24", T87_r24), ("T87r0", T87_r0),
                              ("T89r24", T89_r24), ("T89r0", T89_r0)]:
        if len(polys) >= 100: break
        r_found = try_shift_full(seed, C_VALS_FULL, seed_name)
        if r_found:
            print(f"  {seed_name}: r-values {sorted(r_found)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)

with open('/tmp/batch24_polys.json', 'w') as f:
    json.dump(polys, f)
with open('/tmp/batch24_submission.txt', 'w') as f:
    for p in polys:
        f.write(','.join(p['coeffs']) + '\n')
print(f"Written to /tmp/batch24_submission.txt", file=sys.stderr)

from collections import Counter
strat_counts = Counter()
r_dist = Counter()
for p in polys:
    strat_counts[p['label'].split('_')[0]] += 1
    # Extract r-value from label
    for part in p['label'].split('_'):
        if part.startswith('r'):
            try: r_dist[int(part[1:])] += 1
            except: pass
print("\nStrategy breakdown:", file=sys.stderr)
for k,v in sorted(strat_counts.items()):
    print(f"  {k}: {v}", file=sys.stderr)
print("\nr-value distribution:", file=sys.stderr)
for r,v in sorted(r_dist.items()):
    print(f"  r={r:2d}: {v}", file=sys.stderr)

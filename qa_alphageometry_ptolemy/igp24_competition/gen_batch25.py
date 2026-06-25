#!/usr/bin/env python3
"""
Batch25: New T-numbers via triple-compositum and cyclotomic seeds.

Key insight from batch23/24:
- All S3xD4 composita (regardless of which S3 cubic and D4 quartic) give
  the SAME T-number (same transitive group type in S12).
- Similarly C3xD4 composita give ONE other T-number.
- We need DIFFERENT Galois group types for new T-numbers.

New strategies:
A: S3xV4 seeds = polcompositum(polcompositum(S3-cubic, x^2-d1), x^2-d2)
   Gal = S3xV4 (order 24, different from S3xD4 order 48 AND C3xD4 order 24)
   Roots spread in [-6,6] -> 9-10 distinct r-values per seed

B: polsubcyclo(m,12) = totally real degree-12 abelian field seeds
   Gal = abelian group of order 12 (different for each m)
   Roots in [-2,2] -> ~5 distinct integer-c r-values per seed

C: polsubcyclo(m,24) direct (no shift trick needed)
   Gives one poly per m, r=0, different T-number per m

D: Fill with T87/T89 seeds
"""
import subprocess, json, sys
from itertools import product as iproduct

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

def try_shift(seed_polrev, label_prefix, c_vals, max_polys=None):
    found_r = set()
    for c in c_vals:
        if len(polys) >= 100: break
        if max_polys and len(found_r) >= max_polys: break
        out = gp(f"""
h={seed_polrev};
g24=subst(h,'x,x^2+({c}));
if(poldegree(g24)!=24, print("DEG"); quit);
if(!polisirreducible(g24), print("IRRED"); quit);
r=polsturm(g24,-10^9,10^9);
if(r%2!=0, print("ODD"); quit);
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
                print(f"  {label_prefix} c={c:>4} r={r:>2} total={len(polys)}", file=sys.stderr)
    return found_r

C_WIDE = list(range(-60, 61))
C_NARROW = list(range(-8, 9))

# ================================================================
# STRATEGY A: S3 x V4 triple-compositum
# K12 = Q(alpha, sqrt(d1), sqrt(d2)) where Q(alpha)=S3-cubic field
# Gal(K12/Q) = S3 x V4 (order 24), distinct from S3xD4 (order 48)
# Roots spread ~[-6,6] -> good r-value coverage
# ================================================================
print("=== A: S3 x V4 triple-compositum seeds ===", file=sys.stderr)
a_found = 0

S3_CUBICS = [
    ("x^3-4*x-1", "S3a"),   # disc=229 (prime) -> S3
    ("x^3-5*x-1", "S3b"),   # disc=473 -> S3
    ("x^3-7*x-1", "S3c"),   # disc=1345 -> S3
]
# Pairs of distinct squarefree d1, d2 -> V4 = Q(sqrt(d1)) x Q(sqrt(d2)) / trivial
# Need d1, d2, d1*d2 all squarefree and distinct from each other and discriminant of cubic
V4_PAIRS = [
    (2, 5),     # Q(sqrt(2)) x Q(sqrt(5)): different from disc(S3)
    (2, 7),
    (3, 5),
    (2, 11),
    (5, 7),
]

for (cubic, cname), (d1, d2) in zip(S3_CUBICS[:3], V4_PAIRS[:4]):
    if len(polys) >= 45: break
    out = gp(f"""
f3={cubic};
q1=x^2-{d1};
q2=x^2-{d2};
if(!polisirreducible(f3), print("NOTIRRED3"); quit);
nr3=polsturm(f3,-10^9,10^9);
if(nr3!=3, print("NR3|",nr3); quit);
\\ Step 1: compositum of S3-cubic and sqrt(d1)
comp1=polcompositum(f3,q1);
if(#comp1==0, print("EMPTY1"); quit);
f6=comp1[1];
if(poldegree(f6)!=6, print("DEG6|",poldegree(f6)); quit);
nr6=polsturm(f6,-10^9,10^9);
if(nr6!=6, print("NR6|",nr6); quit);
\\ Step 2: compositum with sqrt(d2)
comp2=polcompositum(f6,q2);
if(#comp2==0, print("EMPTY2"); quit);
f12=comp2[1];
if(poldegree(f12)!=12, print("DEG12|",poldegree(f12)); quit);
if(!polisirreducible(f12), print("IRRED12"); quit);
nr12=polsturm(f12,-10^9,10^9);
if(nr12!=12, print("NR12|",nr12); quit);
print("SEED|",Vecrev(f12));
""")
    if not out.startswith("SEED|"):
        print(f"  A {cname} d=({d1},{d2}): {out[:60]}", file=sys.stderr)
        continue
    seed_coeffs = parse_vec(out.split("|",1)[1])
    if not seed_coeffs or len(seed_coeffs)!=13:
        print(f"  A {cname} d=({d1},{d2}): parse error", file=sys.stderr)
        continue
    seed_pv = f"Polrev([{','.join(seed_coeffs)}])"
    print(f"  A {cname} d=({d1},{d2}): seed OK", file=sys.stderr)
    r_found = try_shift(seed_pv, f"A_{cname}_d{d1}d{d2}", C_WIDE)
    a_found += len(r_found)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy A: {a_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: polsubcyclo(m,12) - totally real abelian degree-12 seeds
# These give abelian Gal groups of order 12 (different per m)
# Roots bounded in [-2,2] -> limited r-values from integer c
# But: different m -> different T-numbers in competition
# ================================================================
print("\n=== B: polsubcyclo(m,12) totally real abelian seeds ===", file=sys.stderr)
b_found = 0

# m values with phi(m)=24 -> max real subfield has degree 12
# phi(35)=phi(5)*phi(7)=4*6=24
# phi(39)=phi(3)*phi(13)=2*12=24
# phi(52)=phi(4)*phi(13)=2*12=24
# phi(72)=phi(8)*phi(9)=4*6=24
# phi(78)=phi(2)*phi(3)*phi(13)=1*2*12=24
# phi(84)=phi(4)*phi(3)*phi(7)=2*2*6=24
m_vals_12 = [35, 39, 52, 72, 78, 84, 90]

for m in m_vals_12:
    if len(polys) >= 75: break
    out = gp(f"""
p=polsubcyclo({m},12,'x);
if(type(p)!="t_POL", print("NOTPOL|",type(p)); quit);
if(poldegree(p)!=12, print("DEG|",poldegree(p)); quit);
if(!polisirreducible(p), print("NOTIRRED"); quit);
nr=polsturm(p,-10^9,10^9);
if(nr!=12, print("NR|",nr); quit);
print("SEED|",Vecrev(p));
""")
    if not out.startswith("SEED|"):
        print(f"  B m={m}: {out[:60]}", file=sys.stderr)
        continue
    seed_coeffs = parse_vec(out.split("|",1)[1])
    if not seed_coeffs or len(seed_coeffs)!=13:
        print(f"  B m={m}: parse error", file=sys.stderr)
        continue
    seed_pv = f"Polrev([{','.join(seed_coeffs)}])"
    print(f"  B m={m}: seed OK, trying shift...", file=sys.stderr)
    # Roots in [-2,2]: integer c from -4 to 4 covers most cases
    r_found = try_shift(seed_pv, f"B_cyclo{m}", C_NARROW)
    b_found += len(r_found)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)
    else: print(f"    no polys found", file=sys.stderr)

print(f"Strategy B: {b_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: polsubcyclo(m,24) direct - one poly per m, r=0
# These are abelian degree-24 Galois extensions (totally complex)
# Each different m gives a different T-number
# ================================================================
print("\n=== C: polsubcyclo(m,24) direct polys ===", file=sys.stderr)
c_found = 0

# More m values: phi(m) divisible by 24, or phi(m)=24
m_vals_24 = [
    35, 39, 72, 78, 84, 90,     # phi=24 (the full Q(zeta_m) has degree 24)
    # phi(m)>24 but degree-24 subfields exist
    65, 91, 95, 119, 143, 120,
    # different structure: m with multiple prime factors
    130, 156, 168, 180,
]

for m in m_vals_24:
    if len(polys) >= 92: break
    out = gp(f"""
p=polsubcyclo({m},24,'x);
if(type(p)!="t_POL", print("NOTPOL|",type(p)); quit);
if(poldegree(p)!=24, print("DEG|",poldegree(p)); quit);
if(!polisirreducible(p), print("NOTIRRED"); quit);
r=polsturm(p,-10^9,10^9);
print("OK|",r,"|",Vecrev(p));
""")
    if not out.startswith("OK|"):
        print(f"  C m={m}: {out[:40]}", file=sys.stderr)
        continue
    parts = out.split("|",2)
    try: r = int(parts[1])
    except: continue
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
        if add(f"subcyclo{m}_r{r}", coeffs):
            c_found += 1
            print(f"  C m={m} r={r} total={len(polys)}", file=sys.stderr)
    else:
        print(f"  C m={m}: invalid coeffs", file=sys.stderr)

print(f"Strategy C: {c_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: Fill with T87/T89 shift (T14746 family)
# Only if still < 100 polys
# ================================================================
if len(polys) < 100:
    print(f"\n=== D: Fill with T87/T89 ===", file=sys.stderr)
    T87_r24 = "Polrev([729,32076,514188,3844260,14581539,27842616,24546456,9280872,1620171,142380,6348,132,1])"
    T87_r0  = "Polrev([729,-32076,514188,-3844260,14581539,-27842616,24546456,-9280872,1620171,-142380,6348,-132,1])"
    T89_r24 = "Polrev([8,2112,91696,1385632,8299628,17725376,14466032,5103008,892214,83280,4172,104,1])"

    for seed_name, seed in [("T87r0",T87_r0), ("T89r24",T89_r24), ("T87r24",T87_r24)]:
        if len(polys) >= 100: break
        r_found = try_shift(seed, seed_name, C_WIDE)
        if r_found: print(f"  D {seed_name}: r={sorted(r_found)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch25_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch25_submission.txt', 'w') as f:
    for p in polys:
        f.write(','.join(p['coeffs']) + '\n')
print(f"Written to /tmp/batch25_submission.txt", file=sys.stderr)

from collections import Counter
sc = Counter()
rc = Counter()
for p in polys:
    sc[p['label'].split('_')[0]] += 1
    for part in p['label'].split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print("Strategy breakdown:", file=sys.stderr)
for k,v in sorted(sc.items()):
    print(f"  {k}: {v}", file=sys.stderr)
print("r-distribution:", file=sys.stderr)
for r,v in sorted(rc.items()):
    print(f"  r={r:2d}: {v}", file=sys.stderr)

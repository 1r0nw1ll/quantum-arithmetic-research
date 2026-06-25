#!/usr/bin/env python3
"""
Fill batch23 to 100 polys using strategies that actually work:
1. degree-3 x degree-4 totally real compositum -> degree-12 totally real seed -> shift trick
2. More S4 quartic nfsplitting (with totally real quartics for r=24)
3. More Dic3 variants (search LMFDB or try PARI-constructed seeds)
4. Direct degree-24 from S4 totally real quartic nfsplitting
5. polsubcyclo approach for larger groups
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

# Load existing polys
with open('/tmp/batch23_polys.json') as f:
    polys = json.load(f)
seen_coeffs = set(tuple(p['coeffs']) for p in polys)
print(f"Starting with {len(polys)} polys", file=sys.stderr)

def add(label, coeffs):
    key = tuple(coeffs)
    if key in seen_coeffs: return False
    seen_coeffs.add(key)
    polys.append({'label': label, 'coeffs': coeffs})
    return True

C_VALS = [-2000,-500,-100,-50,-20,-10,-5,-3,-2,-1,0,1,2,3,5,10,20,50,100,500,2000]

def try_shift(seed_polrev, c_vals, label_prefix, max_new=None):
    found_r = set()
    for c in c_vals:
        if max_new is not None and len(found_r) >= max_new: break
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
                print(f"  {label_prefix} c={c:>6} r={r:>2} OK (total={len(polys)})", file=sys.stderr)
    return found_r

# ================================================================
# STRATEGY G: degree-3 x degree-4 totally real compositum
# polcompositum(cubic3, quartic4) -> degree-12 totally real
# Gal = G3 x G4 (novel!)
# ================================================================
print("\n=== Strategy G: degree-3 x degree-4 totally real compositions ===", file=sys.stderr)
g_found = 0

# S3 cubics (totally real, Gal=S3, disc non-square):
s3_cubics = [
    "x^3-4*x-1",      # disc=229 prime (S3) ✓
    "x^3-5*x-1",      # disc=473=11*43 (S3) ✓
    "x^3-7*x-1",      # disc=1345=5*269 (S3) ✓
    "x^3-5*x+3",      # disc=257 prime (S3) ✓
    "x^3-6*x+4",      # disc=432 non-square (S3?) ✓
    "x^3-4*x+1",      # disc=229? check
    "x^3-7*x+7",      # check
    "x^3-8*x-4",      # check
]

# Totally real quartics (4 real roots):
# Try to find some with Gal=S4 (so nfsplitting=24) OR Gal=D4/A4 (nfsplitting<24)
tr_quartics = [
    "x^4-5*x^2+3",    # roots^2=(5+/-sqrt(13))/2, both positive
    "x^4-7*x^2+11",   # roots^2=(7+/-sqrt(5))/2, both positive
    "x^4-6*x^2+7",    # roots^2=3+/-sqrt(2), both positive
    "x^4-8*x^2+14",   # roots^2=4+/-sqrt(2), both positive
    "x^4-5*x^2+5",    # cyclotomic related (C4)
    "x^4-6*x^2+6",    # check
    "x^4-8*x^2+13",   # check
    "x^4-4*x^2+1",    # roots +/-2cos(pi/5) etc
    "x^4-3*x^2+1",    # check
    "x^4-7*x^2+6",    # check
]

for cubic in s3_cubics[:4]:
    for quartic in tr_quartics[:6]:
        if len(polys) >= 80: break
        out = gp(f"""
f3={cubic};
f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), print("NOTIRRED"); quit);
nr3=polsturm(f3,-10^9,10^9);
nr4=polsturm(f4,-10^9,10^9);
if(nr3!=3, print("NR3|",nr3); quit);
if(nr4!=4, print("NR4|",nr4); quit);
comp=polcompositum(f3,f4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
deg=poldegree(f12);
if(deg!=12, print("DEG|",deg); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR12|",nr); quit);
if(!polisirreducible(f12), print("IRRED12"); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  G {cubic[:12]} x {quartic[:12]}: {out[:50]}", file=sys.stderr)
            continue
        seed_coeffs = parse_vec(out.split("|",1)[1])
        if not seed_coeffs or len(seed_coeffs)!=13: continue
        seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
        print(f"  G {cubic[:12]} x {quartic[:12]}: seed OK (deg-12 totally real)", file=sys.stderr)
        r_found = try_shift(seed_polrev, C_VALS, f"G_{cubic[:6]}_{quartic[:6]}", max_new=3)
        g_found += len(r_found)
        if r_found:
            print(f"    r-values: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy G: {g_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY H: Totally real S4 quartic nfsplitting (degree 24, r=24)
# x^4+bx+c with polgalois=S4 and 4 real roots
# These give r=24 splitting field polys with NEW T-number
# ================================================================
print("\n=== Strategy H: Totally real quartic nfsplitting ===", file=sys.stderr)
h_found = 0

# For totally real quartics, need discriminant > 0 and 4 real roots
# Try x^4-px^2+q with p,q positive and (p/2)^2 > q (for positive roots^2)
# AND irreducible AND Gal=S4 (nfsplitting degree=24)
tr_quartics_h = [
    "x^4-5*x^2+3",
    "x^4-7*x^2+11",
    "x^4-6*x^2+7",
    "x^4-8*x^2+14",
    "x^4-9*x^2+19",
    "x^4-10*x^2+23",
    "x^4-7*x^2+7",
    "x^4-6*x^2+4",
    "x^4-8*x^2+11",
    "x^4-9*x^2+16",
    "x^4-5*x^2+4",
    "x^4-6*x^2+8",
    "x^4-7*x^2+9",
    "x^4-8*x^2+12",
    "x^4-9*x^2+14",
    "x^4-10*x^2+18",
    "x^4-11*x^2+26",
    "x^4-12*x^2+32",
    "x^4-7*x^2+3",
    "x^4-9*x^2+21",
]

for qpoly in tr_quartics_h:
    if len(polys) >= 90: break
    out = gp(f"""
f={qpoly};
if(!polisirreducible(f), print("NOTIRRED"); quit);
nr=polsturm(f,-10^9,10^9);
if(nr!=4, print("NR|",nr); quit);
\\ nfsplitting degree must be 24 (Gal=S4)
h=nfsplitting(f);
deg=poldegree(h);
if(deg!=24, print("NOTDEG24|",deg); quit);
if(!polisirreducible(h), print("NOTIRRED24"); quit);
r=polsturm(h,-10^9,10^9);
print("OK|",r,"|",Vecrev(h));
""")
    if not out.startswith("OK|"):
        print(f"  H {qpoly}: {out[:50]}", file=sys.stderr)
        continue
    parts = out.split("|",2)
    try: r = int(parts[1])
    except: continue
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
        if add(f"S4TRnfs_{qpoly[:15]}_r{r}", coeffs):
            h_found += 1
            print(f"  H {qpoly}: TR nfsplitting r={r} OK (total={len(polys)})", file=sys.stderr)

print(f"Strategy H: {h_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY I: More diverse degree-12 totally real seeds
# Use polsubcyclo(m,6) x quadratic (degree-6 x degree-2 = degree-12)
# C6 x C2 -> C12? No: the shift of C12 gives T14744 (already scored)
# But: if the degree-6 is NON-CYCLOTOMIC totally real, Gal(6-poly) != C6
# Need totally real DEGREE-6 polys with non-cyclic Galois group
# ================================================================
print("\n=== Strategy I: non-cyclic degree-6 x quadratic ===", file=sys.stderr)
i_found = 0

# Totally real degree-6 polys from polcompositum(C3 cubic, real quadratic)
# C3-cubic x quadratic -> degree-6 Gal=C3xC2=C6 (cyclic, boring)
# BUT: S3-cubic x real quadratic -> degree-6 Gal=D6=S3xC2 (if disjoint)
# S3xC2 has order 12 = same as D6 = T14746. Already scored!
#
# For NEW T: need degree-6 totally real with Gal of order > 12
# Gal of degree-6 totally real: can be C6(6), S3(6), A3=C3(6), D3(6),
# C2xC2xC2... wait degree-6 is 6 elements.
# For Gal order > 6 and degree-6: impossible (Gal is transitive subgroup of S6
# of order ≥ 6 BUT degree-6 can have Gal like A5 (order 60) or S5 (order 120)
# Actually: for degree 6 irreducible, Gal is a transitive subgroup of S6
# Possible orders: 6,12,18,24,36,48,60,72,120,360,720

# A4 (order 12) in degree-6 action: This requires a degree-6 field whose Galois
# closure has degree 12 with Gal=A4. Such fields exist.

# For now: try totally real degree-6 polys from LMFDB with Gal=A5 (order 60):
# These are very high-group fields. Shift trick -> NEW T-number!

# Let's try: polsubcyclo(31,6) x polsubcyclo(7,3) - both degree 6 and 3
# Actually let me try degree-3 x degree-4 = degree-12 (totally real)
# Using pairs from LMFDB minimal polys

# Specifically: C3 cubic (totally real) x D4 quartic (totally real)
# Compositum: degree 3x4=12 if disjoint

d4_quartics = [
    "x^4-2",          # Gal=D4, but complex roots
    "x^4-4*x^2+2",    # D4, totally real
    "x^4-4*x^2+1",    # D4? check if totally real, roots^2=(4+-sqrt(12))/2=2+-sqrt(3)>0 ✓
    "x^4-6*x^2+2",    # check
    "x^4-6*x^2+1",    # roots^2=3+-sqrt(8)/2... 3+sqrt(8)>0 ✓, 3-sqrt(8)=3-2.83>0 ✓ TR!
    "x^4-8*x^2+4",    # roots^2=4+-sqrt(12)>0 ✓ TR! Gal=D4?
    "x^4-10*x^2+1",   # roots^2=5+-sqrt(24), 5-sqrt(24)=5-4.9>0 TR! Gal=D4?
]

c3_cubics = [
    "x^3+x^2-2*x-1",  # C3 cubic (7th cyclotomic subfield)
    "x^3-3*x+1",      # C3 cubic
    "x^3-x^2-2*x+1",  # C3 cubic?
]

for cubic in c3_cubics:
    for quartic in d4_quartics:
        if len(polys) >= 97: break
        out = gp(f"""
f3={cubic};
f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), print("NOTIRRED"); quit);
nr3=polsturm(f3,-10^9,10^9);
nr4=polsturm(f4,-10^9,10^9);
if(nr3!=3, print("NR3|",nr3); quit);
if(nr4!=4, print("NR4|",nr4); quit);
comp=polcompositum(f3,f4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
deg=poldegree(f12);
if(deg!=12, print("DEG|",deg); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR12|",nr); quit);
if(!polisirreducible(f12), print("IRRED12"); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  I {cubic[:12]} x {quartic[:12]}: {out[:50]}", file=sys.stderr)
            continue
        seed_coeffs = parse_vec(out.split("|",1)[1])
        if not seed_coeffs or len(seed_coeffs)!=13: continue
        seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
        print(f"  I {cubic[:12]} x {quartic[:12]}: seed OK", file=sys.stderr)
        r_found = try_shift(seed_polrev, C_VALS, f"I_C3xD4", max_new=3)
        i_found += len(r_found)
        if r_found:
            print(f"    r-values: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy I: {i_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY J: Fill remaining with polsubcyclo(m,12) x imag quadratic
# These give r=0 polys from complex C12 seeds, diverse T-numbers
# polsubcyclo(13,12) x Q(sqrt(-d)) -> degree-24, r=0
# Actually these are all Gal=C12xC2 so SAME T-number. Not diverse.
# Instead: use polsubcyclo(m,24) for various m with phi(m)=24
# phi(m)=24: m=25,35,39,45,50,52,56,60,66,70,72,78,84,90
# polsubcyclo(m,24) has degree 24 if phi(m) is divisible by 24
# phi(25)=20, phi(35)=24 -> 24-subfield of Q(zeta_35)!
# phi(35)=phi(5)phi(7)=4*6=24 -> YES!
# polsubcyclo(35,24): Galois group is C24? Or cyclic of order 24?
# Q(zeta_35)=Q(zeta_5,zeta_7) with Gal=C4xC6=C12xC2? No: C4xC6 has order 24.
# Wait: gcd(5,7)=1 so Gal(Q(zeta_35)/Q)=Gal(Q(zeta_5)/Q)xGal(Q(zeta_7)/Q)=C4xC6=C12? No: C4xC6 is not cyclic unless gcd(4,6)=1 which it's not. C4xC6 has order 24 and is isomorphic to C12xC2? No: C4xC6=C4xC6 which has elements of max order lcm(4,6)=12, but C24 has elements of order 24. So Gal(Q(zeta_35))=C4xC6 is not cyclic! -> polsubcyclo(35,24) is a degree-24 poly with NON-CYCLIC Galois group C4xC6!
# This gives a NEW T-number different from T14744!
# ================================================================
print("\n=== Strategy J: polsubcyclo(m,24) for various m ===", file=sys.stderr)
j_found = 0

# m values where phi(m) >= 24 and m has a degree-24 subfield
# phi(35)=24, phi(39)=24, phi(45)=24, phi(52)=24, phi(56)=24, phi(60)=16? phi(60)=phi(4)phi(3)phi(5)=2*2*4=16. No.
# phi(65)=phi(5)phi(13)=4*12=48, phi(35)=24, phi(77)=phi(7)phi(11)=6*10=60
# phi(45)=phi(9)phi(5)=6*4=24 ✓
# phi(39)=phi(3)phi(13)=2*12=24 ✓
# phi(52)=phi(4)phi(13)=2*12=24 ✓
# phi(56)=phi(8)phi(7)=4*6=24 ✓

m_vals_24 = [35, 39, 45, 52, 56, 84]  # phi(m) = 24 or multiple

for m in m_vals_24:
    if len(polys) >= 100: break
    out = gp(f"""
p=polsubcyclo({m},24,'x);
if(type(p)!="t_POL", print("NOT_POL|",type(p)); quit);
if(poldegree(p)!=24, print("DEG|",poldegree(p)); quit);
if(!polisirreducible(p), print("NOTIRRED"); quit);
r=polsturm(p,-10^9,10^9);
print("OK|",r,"|",Vecrev(p));
""")
    if not out.startswith("OK|"):
        print(f"  J m={m}: {out[:50]}", file=sys.stderr)
        continue
    parts = out.split("|",2)
    try: r = int(parts[1])
    except: continue
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
        if add(f"subcyclo_{m}_r{r}", coeffs):
            j_found += 1
            print(f"  J m={m} r={r} OK (total={len(polys)})", file=sys.stderr)

# Also try polsubcyclo for degree-24 from m with phi(m) >= 48:
for m in [65, 77, 85, 91, 95]:
    if len(polys) >= 100: break
    out = gp(f"""
p=polsubcyclo({m},24,'x);
if(type(p)!="t_POL", print("NOT_POL|",type(p)); quit);
if(poldegree(p)!=24, print("DEG|",poldegree(p)); quit);
if(!polisirreducible(p), print("NOTIRRED"); quit);
r=polsturm(p,-10^9,10^9);
print("OK|",r,"|",Vecrev(p));
""")
    if not out.startswith("OK|"):
        print(f"  J m={m}: {out[:50]}", file=sys.stderr)
        continue
    parts = out.split("|",2)
    try: r = int(parts[1])
    except: continue
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
        if add(f"subcyclo_{m}_r{r}", coeffs):
            j_found += 1
            print(f"  J m={m} r={r} OK (total={len(polys)})", file=sys.stderr)

print(f"Strategy J: {j_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY K: nfsubfields/polredabs of S4 degree-4 composita
# polcompositum(x^4-x-1, x^4+x-1) -> degree-16 or less? try
# But simpler: use resolvent construction.
# Actually: nfsplitting of S4 quartic but with different QUARTICS
# including those with 4 real roots - try more varieties
# ================================================================
print("\n=== Strategy K: More diverse quartic nfsplitting ===", file=sys.stderr)
k_found = 0

# More quartics with x^4+bx^2+c type (check nr=4 real roots)
quartics_more = [
    # Total real quartics (nr=4)
    "x^4-7*x^2+3", "x^4-8*x^2+8", "x^4-9*x^2+18",
    "x^4-10*x^2+20", "x^4-11*x^2+28",
    # Totally complex quartics (nr=0) - diverse T-numbers
    "x^4+x+3", "x^4+2*x+3", "x^4+3*x+5", "x^4+x^2+x+1",
    "x^4+x^3+x^2+x+1",  # 5th cyclotomic
    "x^4+2*x^3+2*x+2",
    "x^4-x^3+x^2-x+1",  # 10th cyclotomic
    "x^4+x^3+2*x^2+x+1",
    "x^4+2*x^2+2", "x^4+3*x^2+3",
    "x^4+x^3+x^2+2",
    # Mixed real/complex (nr=2)
    "x^4-x^2-1", "x^4-2*x^2-2", "x^4-x^2-2",
    "x^4-3*x^2-3", "x^4-4*x^2-3",
]

for qpoly in quartics_more:
    if len(polys) >= 100: break
    out = gp(f"""
f={qpoly};
if(!polisirreducible(f), print("NOTIRRED"); quit);
h=nfsplitting(f);
if(poldegree(h)!=24, print("NOTDEG24|",poldegree(h)); quit);
if(!polisirreducible(h), print("NOTIRRED24"); quit);
r=polsturm(h,-10^9,10^9);
print("OK|",r,"|",Vecrev(h));
""")
    if not out.startswith("OK|"):
        print(f"  K {qpoly}: {out[:50]}", file=sys.stderr)
        continue
    parts = out.split("|",2)
    try: r = int(parts[1])
    except: continue
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
        if add(f"S4nfs2_{qpoly[:12]}_r{r}", coeffs):
            k_found += 1
            print(f"  K {qpoly}: nfsplitting r={r} OK (total={len(polys)})", file=sys.stderr)

print(f"Strategy K: {k_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)

outfile = '/tmp/batch23_polys.json'
with open(outfile, 'w') as f_out:
    json.dump(polys, f_out)
print(f"Saved to {outfile}", file=sys.stderr)

subfile = '/tmp/batch23_submission.txt'
with open(subfile, 'w') as f_out:
    for p in polys:
        f_out.write(','.join(p['coeffs']) + '\n')
print(f"Wrote {len(polys)} lines to {subfile}", file=sys.stderr)

from collections import Counter
strat_counts = Counter()
for p in polys:
    strat_counts[p['label'].split('_')[0]] += 1
print("\nStrategy breakdown:", file=sys.stderr)
for k,v in sorted(strat_counts.items()):
    print(f"  {k}: {v}", file=sys.stderr)

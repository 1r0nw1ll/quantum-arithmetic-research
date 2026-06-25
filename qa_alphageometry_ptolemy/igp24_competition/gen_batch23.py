#!/usr/bin/env python3
"""
Batch23: New T-numbers via diverse algebraic constructions.

Strategy:
A. Dic3 (12T5) seed from LMFDB + shift trick -> likely NEW T-number (5th order-12 group)
B. polcompositum of TWO different C6 cyclotomic polys -> degree-12 Gal=C6xC6 (order 36) -> shift -> NEW T
C. polcompositum of C6 + C6 (other primes) composita -> more Gal=C6xC6 variants -> SAME T as B
D. S4 quartic nfsplitting -> degree-24 directly -> NEW T-number
E. polcompositum of C6 with totally-real S3-cubic -> degree-12 Gal=C6xS3 (order 36) -> shift -> NEW T
F. More diverse degree-12 seeds (polcompositum of 2 totally real cubics, etc)
"""
import subprocess, json, sys

def gp(cmd, timeout=240):
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

def try_shift(seed_polrev, c_vals, label_prefix, max_new=None):
    """Apply shift trick h(x^2+c) for various c values. Returns r-values found."""
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

C_VALS = [-2000,-500,-100,-50,-20,-10,-5,-3,-2,-1,0,1,2,3,5,10,20,50,100,500,2000]

# ================================================================
# STRATEGY A: Dic3 (12T5) totally real seed from LMFDB
# x^12 - 24x^10 - 10x^9 + 216x^8 + 180x^7 - 844x^6 - 1080x^5
#   + 1056x^4 + 2200x^3 + 720x^2 - 240x - 80
# Verified: irreducible, 12 real roots, 12T5=Dic3
# ================================================================
print("=== Strategy A: Dic3 (12T5) seed ===", file=sys.stderr)
DIC3_SEED = "Polrev([-80,-240,720,2200,1056,-1080,-844,180,216,-10,-24,0,1])"
r_vals = try_shift(DIC3_SEED, C_VALS, "Dic3")
print(f"Dic3: r-values found: {sorted(r_vals)}", file=sys.stderr)

# Fractional shifts for r=2,4,6 (c between 2.83 and 2.27):
# Roots span [-2.60 ... 2.83]. Need c in (2.77,2.83) for r=2,
# (2.57,2.77) for r=4, (2.27,2.57) for r=6.
# Use q=2: c = a/2. Try a/2 = 2.5 -> c=5/2 for r=4 (between 2.27 and 2.57)
# F(y) = 2^24 * h((y/2)^2 + a/2) -- monic integer poly via y=2x
print("  Trying fractional shifts (b*x substitution)...", file=sys.stderr)
frac_shifts = [
    # (p, q) meaning c=p/q, shift to integer via y=qx
    # For r=4: c ~ 2.4 = 12/5 -> q=5, p=12
    # For r=6: c ~ 2.3 = 23/10 -> q=10, p=23
    # For r=2: c ~ 2.80 = 14/5 -> too large coefficients, skip
    # Better: c = 5/2=2.5 (r=4), c=11/4=2.75 (r=2? too close to boundary)
]
# Actually: for r=2, need c in (2.77,2.83). Smallest denominator fraction: 2.8 = 14/5
# For q=5: F(y) coefficients involve 5^24 ~ 6e16 -> too large for 100KB submission
# Skip fractional shifts for now (coefficient explosion)

# Try a SECOND Dic3 field from LMFDB (different discriminant -> different seed poly)
# Search LMFDB for another 12T5 totally real field
print("  Trying to find more Dic3 seeds via PARI...", file=sys.stderr)
# Construct Dic3 via: Dic3 = <a,b | a^6=1, b^2=a^3, b^-1*a*b=a^-1>
# A totally real Dic3 degree-12 poly can come from:
# nfinit of Dic3 extension of Q via class field theory
# Instead: use polcompositum of polsubcyclo(7,3) (C3, degree 3) and ... hard.
# For now, just use the one known Dic3 seed.

# ================================================================
# STRATEGY B: polcompositum of TWO different C6 cyclotomic polys
# polsubcyclo(13,6) x polsubcyclo(37,6) -> degree-12, Gal=C6xC6 (order 36)
# -> shift trick -> NEW T-number (group order 2^12 * 36 = 147456)
# ================================================================
print("\n=== Strategy B: C6xC6 composite seeds ===", file=sys.stderr)
b_found = 0

# Prime pairs p1,p2 with p1,p2 prime, p1≡p2≡1(mod 6), p1≠p2, gcd(disc)=1
c6_prime_pairs = [
    (13, 37), (13, 43), (13, 61), (13, 67),
    (37, 43), (37, 61), (43, 61),
    (13, 97), (37, 97), (13, 103)
]

for (p1, p2) in c6_prime_pairs:
    if len(polys) >= 55: break
    out = gp(f"""
c6a=polsubcyclo({p1},6,'x);
c6b=polsubcyclo({p2},6,'x);
comp=polcompositum(c6a,c6b);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR|",nr); quit);
print("SEED|",Vecrev(f12));
""")
    if not out.startswith("SEED|"):
        print(f"  B p1={p1} p2={p2}: {out[:60]}", file=sys.stderr)
        continue
    seed_coeffs = parse_vec(out.split("|",1)[1])
    if not seed_coeffs or len(seed_coeffs)!=13: continue
    seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
    print(f"  B p1={p1} p2={p2}: seed OK", file=sys.stderr)

    r_found = try_shift(seed_polrev, C_VALS, f"C6xC6_p{p1}x{p2}", max_new=3)
    b_found += len(r_found)
    if r_found:
        print(f"    r-values: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy B: {b_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: polcompositum C6 + totally real S3 cubic -> degree-12
# S3-cubic (totally real) x C6 cyclotomic -> Gal = C6 x S3 (order 36)?
# Or Gal = something else (depends on if fields disjoint)
# ================================================================
print("\n=== Strategy C: C6 x S3-cubic composite ===", file=sys.stderr)
c_found = 0

# Totally real S3 cubics (Gal=S3, 3 real roots): x^3 - 3x + 1 (C3?),
# x^3 - 3x - 1 (S3), x^3 + x^2 - 2x - 1 (C7 subfield?), etc.
# Standard totally real S3 cubics with small discriminant:
s3_cubics = [
    "x^3+x^2-2*x-1",       # 7th cyclotomic - Gal=C3, nr=3, disc=49
    "x^3-3*x+1",            # Gal=C3, nr=3, disc=81? Actually this is C3!
    "x^3-x^2-2*x+1",        # Gal=C3?
    "x^3+x^2-4*x+1",        # Check
    "x^3-4*x-1",             # S3, nr=3
    "x^3-4*x+1",             # Check
    "x^3-3*x-1",             # S3, nr=3
    "x^3+x^2-3*x-2",        # Check
    "x^3-x^2-3*x+1",        # Check
    "x^3+x^2-4*x-2",        # Check
    "x^3-5*x+1",             # Check
    "x^3+x^2-5*x+2",        # Check
]

for cubic in s3_cubics:
    if len(polys) >= 70: break
    # Check cubic is irreducible and totally real
    out_check = gp(f"""
f3={cubic};
if(!polisirreducible(f3), print("NOTIRRED"); quit);
nr3=polsturm(f3,-10^9,10^9);
if(nr3!=3, print("NR3|",nr3); quit);
print("OK3");
""")
    if out_check.strip() != "OK3":
        print(f"  C {cubic}: {out_check[:40]}", file=sys.stderr)
        continue

    # Compositum with polsubcyclo(13,6)
    for c6_prime in [13, 37]:
        if len(polys) >= 70: break
        out = gp(f"""
f3={cubic};
c6=polsubcyclo({c6_prime},6,'x);
comp=polcompositum(f3,c6);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR|",nr); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  C {cubic[:20]} x C6_{c6_prime}: {out[:50]}", file=sys.stderr)
            continue
        seed_coeffs = parse_vec(out.split("|",1)[1])
        if not seed_coeffs or len(seed_coeffs)!=13: continue
        seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
        print(f"  C {cubic[:20]} x C6_{c6_prime}: seed OK", file=sys.stderr)

        r_found = try_shift(seed_polrev, C_VALS, f"C6S3_c6{c6_prime}", max_new=3)
        c_found += len(r_found)
        if r_found:
            print(f"    r-values: {sorted(r_found)}", file=sys.stderr)
        break  # One c6_prime per cubic

print(f"Strategy C: {c_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: S4 quartic nfsplitting -> direct degree-24 poly
# Takes a quartic with Gal=S4, computes its splitting field poly (degree 24)
# This bypasses the shift trick entirely -> potentially NEW T-number
# ================================================================
print("\n=== Strategy D: S4 quartic nfsplitting ===", file=sys.stderr)
d_found = 0

# Generic quartics with potentially Gal=S4 (need polgalois check)
# x^4 + a*x + b often has Gal=S4 for prime discriminant
# disc(x^4+px+q) = -27p^4 + 256q^3; for prime disc -> Gal=S4
s4_quartics = [
    "x^4-x-1",
    "x^4+x-1",
    "x^4-2*x-1",
    "x^4-3*x-1",
    "x^4-x+2",
    "x^4+x+2",
    "x^4-2*x+2",
    "x^4+2*x-2",
    "x^4-x^2-x-1",
    "x^4+x^2-x+1",
    "x^4-x^2+2*x-1",
    "x^4+x^3-x-1",
    "x^4-x^3+x-1",
    "x^4+x^3-x^2-1",
    "x^4-x^3+x^2+x-1",
]

for qpoly in s4_quartics:
    if len(polys) >= 82: break
    out = gp(f"""
f={qpoly};
if(!polisirreducible(f), print("NOTIRRED"); quit);
\\ Check nfsplitting degree (want 24 = S4 regular representation)
h=nfsplitting(f);
if(poldegree(h)!=24, print("NOTDEG24|",poldegree(h)); quit);
if(!polisirreducible(h), print("NOTIRRED24"); quit);
r=polsturm(h,-10^9,10^9);
print("OK|",r,"|",Vecrev(h));
""")
    if not out.startswith("OK|"):
        print(f"  D {qpoly}: {out[:50]}", file=sys.stderr)
        continue
    parts = out.split("|",2)
    try: r = int(parts[1])
    except: continue
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
        if add(f"S4nfs_{qpoly[:15]}_r{r}", coeffs):
            d_found += 1
            print(f"  D {qpoly}: nfsplitting r={r} OK (total={len(polys)})", file=sys.stderr)

print(f"Strategy D: {d_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY E: polcompositum of two totally real S3 cubics -> degree-12
# Gal = S3 x S3 (order 36) -> shift trick -> NEW T-number
# ================================================================
print("\n=== Strategy E: S3 x S3 composite degree-12 seeds ===", file=sys.stderr)
e_found = 0

# Need TWO different S3 totally real cubics with disjoint splitting fields
# Common S3 cubics: x^3-3x+1 (C3!), x^3-4x-1, x^3-3x-1, x^3+3x-1, ...
# Check: polgalois(x^3-4*x-1) -> probably S3
s3_cubic_pairs = [
    ("x^3-4*x-1", "x^3-3*x-1"),
    ("x^3-4*x-1", "x^3+x^2-2*x-1"),  # second might be C3
    ("x^3-3*x-1", "x^3-x^2-3*x+1"),
    ("x^3-4*x+1", "x^3-4*x-1"),
    ("x^3-x^2-4*x+3", "x^3-4*x-1"),
]

for (c3_a, c3_b) in s3_cubic_pairs:
    if len(polys) >= 93: break
    out = gp(f"""
f3a={c3_a};
f3b={c3_b};
if(!polisirreducible(f3a)||!polisirreducible(f3b), print("NOTIRRED"); quit);
nra=polsturm(f3a,-10^9,10^9);
nrb=polsturm(f3b,-10^9,10^9);
if(nra!=3||nrb!=3, print("NR|",nra,nrb); quit);
comp=polcompositum(f3a,f3b);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
nr=polsturm(f12,-10^9,10^9);
if(nr!=12, print("NR12|",nr); quit);
print("SEED|",Vecrev(f12));
""")
    if not out.startswith("SEED|"):
        print(f"  E {c3_a[:15]} x {c3_b[:15]}: {out[:50]}", file=sys.stderr)
        continue
    seed_coeffs = parse_vec(out.split("|",1)[1])
    if not seed_coeffs or len(seed_coeffs)!=13: continue
    seed_polrev = f"Polrev([{','.join(seed_coeffs)}])"
    print(f"  E {c3_a[:15]} x {c3_b[:15]}: seed OK", file=sys.stderr)

    r_found = try_shift(seed_polrev, C_VALS, f"S3xS3", max_new=3)
    e_found += len(r_found)
    if r_found:
        print(f"    r-values: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy E: {e_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY F: Fill to 100 with polsubcyclo(m,12) x quadratic seeds
# These give new T-numbers different from T14744 (C12 standalone)
# polsubcyclo(13,12) x Q(sqrt(d)) -> Gal = C12 x C2 (order 24!)
# ================================================================
print("\n=== Strategy F: C12 x C2 composite (order-24 Galois group) ===", file=sys.stderr)
f_found = 0

for m in [13, 37, 61, 73, 97]:
    if len(polys) >= 100: break
    for d in [2, 3, 5, 7, 11, 13, 17, 19]:
        if len(polys) >= 100: break
        out = gp(f"""
c12=polsubcyclo({m},12,'x);
if(poldegree(c12)!=12, print("DEG|",poldegree(c12)); quit);
qd=x^2-{d};
comp=polcompositum(c12,qd);
if(#comp==0, print("EMPTY"); quit);
f=comp[1];
if(poldegree(f)!=24, print("DEG24|",poldegree(f)); quit);
if(!polisirreducible(f), print("IRRED"); quit);
r=polsturm(f,-10^9,10^9);
print("OK|",r,"|",Vecrev(f));
""")
        if not out.startswith("OK|"):
            print(f"  F m={m} d={d}: {out[:50]}", file=sys.stderr)
            continue
        parts = out.split("|",2)
        try: r = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"C12xC2_m{m}_d{d}_r{r}", coeffs):
                f_found += 1
                print(f"  F m={m} d={d} r={r} OK (total={len(polys)})", file=sys.stderr)

print(f"Strategy F: {f_found} polys, total={len(polys)}", file=sys.stderr)

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

# Also print summary of strategies used
from collections import Counter
strat_counts = Counter()
for p in polys:
    strat_counts[p['label'].split('_')[0]] += 1
print("\nStrategy breakdown:", file=sys.stderr)
for k,v in sorted(strat_counts.items()):
    print(f"  {k}: {v}", file=sys.stderr)

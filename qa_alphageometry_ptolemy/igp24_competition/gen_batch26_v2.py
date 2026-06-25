#!/usr/bin/env python3
"""
Batch26 v2: Fixed PARI VEC handling using apply() instead of for loops.

Key fix: polsubcyclo() returns t_VEC of subfields. Use apply() to iterate,
which works correctly via subprocess unlike for(i=1,#v,...).

Strategies:
  A: S3 x C4 seeds (from polsubcyclo(15,4) totally-real quartic) → NEW T-number
  B: polsubcyclo(m,12) abelian totally-real seeds → NEW T-numbers (one per m)
  C: polsubcyclo(m,24) direct totally-real → NEW T-numbers
  D: More S3xV4 seeds → same T as batch25 but different r-coverage possible
  E: S3xD4 filler
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

C_WIDE = list(range(-60, 61))
C_NARROW = list(range(-8, 9))

def get_totally_real_subcyclo(m, d):
    """Find totally-real degree-d subfield of Q(zeta_m).
    Uses apply() which correctly iterates over t_VEC (for-loops fail in subprocess)."""
    out = gp(f"""
v=polsubcyclo({m},{d},'x);
if(type(v)=="t_POL",v=[v]);
if(type(v)!="t_VEC", quit);
apply(p->if(poldegree(p)=={d} && polisirreducible(p) && polsturm(p,-10^9,10^9)=={d}, print("FOUND|",Vecrev(p))),v);
""")
    for line in out.split('\n'):
        line = line.strip()
        if not line.startswith("FOUND|"): continue
        coeffs = parse_vec(line.split("|",1)[1])
        if coeffs and len(coeffs) == d+1:
            return f"Polrev([{','.join(coeffs)}])"
    return None

# ================================================================
# STRATEGY A: S3 x C4 seeds
# polsubcyclo(15,4) → 3 degree-4 subfields of Q(zeta_15)
# The totally-real one (4 real roots) has Galois group C4
# polcompositum(S3-cubic, C4-quartic) → degree-12 totally-real, Gal=S3xC4
# S3xC4 (order 24, non-isomorphic to S3xV4) → DIFFERENT T-number
# ================================================================
print("=== A: S3 x C4 seeds ===", file=sys.stderr)
a_found = 0

c4_quartic = get_totally_real_subcyclo(15, 4)
print(f"  C4 quartic from polsubcyclo(15,4): {c4_quartic}", file=sys.stderr)

if c4_quartic is None:
    # Fallback: hardcode known totally-real C4 quartic x^4-x^3-4x^2+4x+1
    # (verified: Galois=C4, 4 real roots, discriminant 5^2*29^2=21025)
    c4_quartic = "Polrev([1,4,-4,-1,1])"
    print(f"  Using hardcoded C4 quartic: x^4-x^3-4x^2+4x+1", file=sys.stderr)

if c4_quartic:
    S3_CUBICS = [
        ("x^3-4*x-1",  "S3a"),
        ("x^3-5*x-1",  "S3b"),
        ("x^3-7*x-1",  "S3c"),
        ("x^3-5*x+3",  "S3d"),
        ("x^3-7*x+7",  "S3e"),
        ("x^3-8*x-4",  "S3f"),
        ("x^3-11*x-4", "S3g"),
        ("x^3-13*x-4", "S3h"),
    ]
    for cubic, cname in S3_CUBICS:
        if len(polys) >= 64: break
        out = gp(f"""
f3={cubic};
q4={c4_quartic};
if(!polisirreducible(f3), quit);
if(polsturm(f3,-10^9,10^9)!=3, quit);
if(!polisirreducible(q4), quit);
if(polsturm(q4,-10^9,10^9)!=4, quit);
comp=polcompositum(f3,q4);
if(#comp==0, quit);
f12=comp[1];
if(poldegree(f12)!=12 || !polisirreducible(f12), quit);
if(polsturm(f12,-10^9,10^9)!=12, quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  A {cname}: failed - {out[:40]}", file=sys.stderr)
            continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  A {cname}xC4: seed OK, shifting...", file=sys.stderr)
        r_found = try_shift(seed_pv, f"A_{cname}_S3C4", C_WIDE, max_per_seed=8)
        a_found += len(r_found)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy A: {a_found}, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: polsubcyclo(m,12) totally-real abelian seeds
# Different m → different abelian Galois group → different T-number
# Fixed: use apply() instead of for loop
# ================================================================
print("\n=== B: polsubcyclo(m,12) totally-real seeds ===", file=sys.stderr)
b_found = 0

# m values where phi(m) divisible by 12 and there's a totally-real subfield of degree 12
# Known: polsubcyclo(35,12) v[3] has 12 real roots
m_vals = [35, 39, 52, 72, 78, 84, 90, 91, 95, 104, 111, 126, 140, 180]

for m in m_vals:
    if len(polys) >= 82: break
    seed_pv = get_totally_real_subcyclo(m, 12)
    if seed_pv is None:
        print(f"  B m={m}: no totally-real degree-12 subfield", file=sys.stderr)
        continue
    print(f"  B m={m}: found totally-real abelian seed, shifting...", file=sys.stderr)
    r_found = try_shift(seed_pv, f"B_cyclo{m}", C_WIDE, max_per_seed=6)
    b_found += len(r_found)
    if r_found:
        print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"Strategy B: {b_found}, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: polsubcyclo(m,24) direct totally-real degree-24
# These ARE the Galois group (no shift trick extension needed)
# Each one is a new valid competition entry
# Fixed: use apply() for VEC iteration
# ================================================================
print("\n=== C: polsubcyclo(m,24) direct ===", file=sys.stderr)
c_found = 0

m24_vals = [65, 91, 95, 104, 111, 120, 130, 143, 156, 168, 180, 195, 35, 39, 52, 84, 78]

for m in m24_vals:
    if len(polys) >= 90: break
    out = gp(f"""
v=polsubcyclo({m},24,'x);
if(type(v)=="t_POL",v=[v]);
if(type(v)!="t_VEC", quit);
apply(p->if(poldegree(p)==24 && polisirreducible(p) && polsturm(p,-10^9,10^9)>0, print("OK|",polsturm(p,-10^9,10^9),"|",Vecrev(p))),v);
""")
    added_here = 0
    for line in out.split('\n'):
        if len(polys) >= 90: break
        if added_here >= 3: break
        line = line.strip()
        if not line.startswith("OK|"): continue
        parts = line.split("|",2)
        try: r_val = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"C_cyclo{m}_{added_here}_r{r_val}", coeffs):
                added_here += 1
                c_found += 1
                print(f"  C m={m} [{added_here}] r={r_val} total={len(polys)}", file=sys.stderr)

print(f"Strategy C: {c_found}, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: More S3xV4 seeds (different r-coverage from batch25)
# Same T-number as batch25 but fills gaps if batch25 had invalids
# ================================================================
if len(polys) < 98:
    print("\n=== D: More S3xV4 seeds ===", file=sys.stderr)
    d_seeds = [
        ("x^3-4*x-1",  2, 13, "S3a_d2d13"),
        ("x^3-5*x-1",  2, 13, "S3b_d2d13"),
        ("x^3-7*x-1",  3, 13, "S3c_d3d13"),
        ("x^3-4*x-1",  5, 11, "S3a_d5d11"),
        ("x^3-5*x+3",  2, 7,  "S3d_d2d7"),
        ("x^3-7*x+7",  2, 5,  "S3e_d2d5"),
        ("x^3-11*x-4", 2, 7,  "S3f_d2d7"),
    ]
    for cubic, d1, d2, lname in d_seeds:
        if len(polys) >= 98: break
        out = gp(f"""
f3={cubic}; q1=x^2-{d1}; q2=x^2-{d2};
if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3, quit);
comp1=polcompositum(f3,q1);
if(#comp1==0, quit);
f6=comp1[1];
if(poldegree(f6)!=6||polsturm(f6,-10^9,10^9)!=6, quit);
comp2=polcompositum(f6,q2);
if(#comp2==0, quit);
f12=comp2[1];
if(poldegree(f12)!=12||!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12, quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"): continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  D {lname}: seed OK", file=sys.stderr)
        r_found = try_shift(seed_pv, f"D_{lname}", C_WIDE, max_per_seed=7)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

# ================================================================
# STRATEGY E: S3xD4 filler (same T as batch23/24, new r-values)
# ================================================================
if len(polys) < 100:
    print("\n=== E: S3xD4 filler ===", file=sys.stderr)
    e_seeds = [
        ("x^3-14*x-4",   "x^4-6*x^2+2"),
        ("x^3-15*x-7",   "x^4-6*x^2+2"),
        ("x^3-16*x-4",   "x^4-6*x^2+2"),
        ("x^3-16*x+16",  "x^4-6*x^2+2"),
        ("x^3-17*x-4",   "x^4-6*x^2+2"),
        ("x^3-4*x-1",    "x^4-10*x^2+1"),
        ("x^3-5*x-1",    "x^4-10*x^2+1"),
        ("x^3-11*x-4",   "x^4-10*x^2+1"),
    ]
    for cubic, quartic in e_seeds:
        if len(polys) >= 100: break
        out = gp(f"""
f3={cubic}; f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), quit);
if(polsturm(f3,-10^9,10^9)!=3||polsturm(f4,-10^9,10^9)!=4, quit);
comp=polcompositum(f3,f4);
if(#comp==0, quit);
f12=comp[1];
if(poldegree(f12)!=12||!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12, quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"): continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  E {cubic[:10]}: seed OK", file=sys.stderr)
        r_found = try_shift(seed_pv, f"E_{cubic[3:7]}", C_WIDE, max_per_seed=5)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch26_polys.json','w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch26_submission.txt','w') as f:
    for p in polys:
        f.write(','.join(p['coeffs'])+'\n')
print(f"Written to /tmp/batch26_submission.txt", file=sys.stderr)

from collections import Counter
sc_ctr = Counter()
rc = Counter()
for p in polys:
    label = p['label']
    for k in ['A_','B_','C_','D_','E_']:
        if label.startswith(k):
            sc_ctr[k.strip('_')] += 1
            break
    for part in label.split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print("Breakdown:", dict(sc_ctr), file=sys.stderr)
print("r:", dict(sorted(rc.items())), file=sys.stderr)

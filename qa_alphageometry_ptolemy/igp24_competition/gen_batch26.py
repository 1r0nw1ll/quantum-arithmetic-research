#!/usr/bin/env python3
"""
Batch26: New T-numbers via S3xC4 and polsubcyclo totally-real degree-12 seeds.

Key findings from batch25:
- polsubcyclo(m,12) returns t_VEC (multiple subfields) — need to pick totally-real one
- polsubcyclo(m,24) also returns t_VEC for some m
- S3xV4 construction works well (31+28=59 polys from 8 seeds)

New strategies:
A: S3xC4 seeds = polcompositum(S3-cubic, polsubcyclo(15,4))
   polsubcyclo(15,4) is the maximal real subfield of Q(zeta_15), Gal=C4
   S3xC4 has order 24, DIFFERENT from S3xV4 (also order 24) -> different T-number!

B: polsubcyclo(m,12) VEC fix — find the totally-real subfield by polsturm check
   Each m gives a different abelian Gal group (order 12) -> different T-numbers

C: More S3xV4 seeds (fewer c-values, more seeds for different T-number coverage)

D: Fill with diverse cyclotomic direct degree-24 polys
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
if(poldegree(g24)!=24, print("DEG"); quit);
if(!polisirreducible(g24), print("IRRED"); quit);
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

def get_totally_real_subcyclo12(m):
    """Get the totally-real degree-12 subfield of Q(zeta_m) if it exists."""
    out = gp(f"""
v=polsubcyclo({m},12,'x);
if(type(v)=="t_POL",
   p=v;
   if(poldegree(p)==12 && polisirreducible(p) && polsturm(p,-10^9,10^9)==12,
      print("FOUND|",Vecrev(p)));
   quit);
if(type(v)!="t_VEC", quit);
for(i=1,#v,
   p=v[i];
   if(poldegree(p)!=12, next);
   if(!polisirreducible(p), next);
   if(polsturm(p,-10^9,10^9)!=12, next);
   print("FOUND|",Vecrev(p)));
""")
    for line in out.split('\n'):
        line = line.strip()
        if not line.startswith("FOUND|"): continue
        coeffs = parse_vec(line.split("|",1)[1])
        if coeffs and len(coeffs)==13:
            return f"Polrev([{','.join(coeffs)}])"
    return None

C_WIDE = list(range(-60, 61))
C_NARROW = list(range(-8, 9))

# ================================================================
# STRATEGY A: S3 x C4 — polsubcyclo(15,4) as the C4 quartic
# polsubcyclo(15,4) = max real subfield of Q(zeta_15), Gal=C4, totally real
# S3 x C4 has order 24, different T-number from S3xV4 (also order 24)
# ================================================================
print("=== A: S3 x C4 seeds (polsubcyclo(15,4) as quartic) ===", file=sys.stderr)
a_found = 0

# First get the C4 quartic from polsubcyclo(15,4)
c4_quartic_pv = None
out = gp("""
q=polsubcyclo(15,4,'x);
print("C4Q|",type(q),"|",Vecrev(q));
""")
if out.startswith("C4Q|"):
    parts = out.split("|")
    qtype = parts[1]
    print(f"  polsubcyclo(15,4) type: {qtype}", file=sys.stderr)
    if qtype == "t_POL":
        qcoeffs = parse_vec(parts[2])
        if qcoeffs:
            c4_quartic_pv = f"Polrev([{','.join(qcoeffs)}])"
            print(f"  C4 quartic: {c4_quartic_pv[:60]}", file=sys.stderr)
    elif qtype == "t_VEC":
        # Pick first irreducible degree-4 poly
        out2 = gp("""
v=polsubcyclo(15,4,'x);
for(i=1,#v,
   p=v[i];
   if(poldegree(p)==4 && polisirreducible(p),
      print("FOUND|",Vecrev(p));
      break));
""")
        for line in out2.split('\n'):
            line = line.strip()
            if line.startswith("FOUND|"):
                qcoeffs = parse_vec(line.split("|",1)[1])
                if qcoeffs:
                    c4_quartic_pv = f"Polrev([{','.join(qcoeffs)}])"
                break

if c4_quartic_pv:
    # Verify it's totally real
    out = gp(f"q={c4_quartic_pv}; print(polsturm(q,-10^9,10^9),\"|\"poldegree(q));")
    print(f"  C4 quartic check: {out}", file=sys.stderr)

    S3_CUBICS = [
        ("x^3-4*x-1", "S3a"),
        ("x^3-5*x-1", "S3b"),
        ("x^3-7*x-1", "S3c"),
        ("x^3-4*x+1", "S3d"),  # check if Gal=S3
    ]
    for cubic, cname in S3_CUBICS:
        if len(polys) >= 52: break
        out = gp(f"""
f3={cubic};
q4={c4_quartic_pv};
if(!polisirreducible(f3), print("FAIL3"); quit);
if(polsturm(f3,-10^9,10^9)!=3, print("NR3"); quit);
comp=polcompositum(f3,q4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
if(!polisirreducible(f12), print("IRRED"); quit);
if(polsturm(f12,-10^9,10^9)!=12, print("NR12|",polsturm(f12,-10^9,10^9)); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  A {cname}: {out[:50]}", file=sys.stderr)
            continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  A {cname}: seed OK, shifting...", file=sys.stderr)
        r_found = try_shift(seed_pv, f"A_{cname}_C4", C_WIDE)
        a_found += len(r_found)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)
else:
    print("  WARNING: Could not get polsubcyclo(15,4) as polynomial", file=sys.stderr)
    # Fallback: try the 5th cyclotomic real subfield directly
    # x^4+x^3-3x^2-x+1 is the 5th cyclotomic real polynomial (minimal poly of 2cos(2pi/15)?)
    # Actually: let's just use x^4-x^3-4x^2+4x+1 (check this)
    # Or polsubcyclo(20,4) = real subfield of Q(zeta_20), also Gal=C4
    for alt_m in [20, 24, 16]:
        out_a = gp(f"""
v=polsubcyclo({alt_m},4,'x);
if(type(v)=="t_POL",
   p=v;
   nr=polsturm(p,-10^9,10^9);
   print("ALT|",Vecrev(p),"|nr=",nr);
   quit);
if(type(v)!="t_VEC", quit);
for(i=1,#v,
   p=v[i];
   if(poldegree(p)!=4 || !polisirreducible(p), next);
   nr=polsturm(p,-10^9,10^9);
   print("ALT|",Vecrev(p),"|nr=",nr);
   break);
""")
        print(f"  polsubcyclo({alt_m},4): {out_a[:80]}", file=sys.stderr)
        if out_a.startswith("ALT|"):
            pparts = out_a.split("|")
            nr = int(pparts[2].replace("nr=",""))
            if nr == 4:  # totally real
                qcoeffs = parse_vec(pparts[1])
                if qcoeffs:
                    c4_quartic_pv = f"Polrev([{','.join(qcoeffs)}])"
                    print(f"  Using polsubcyclo({alt_m},4) as C4 quartic", file=sys.stderr)
                    break

print(f"Strategy A: {a_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: polsubcyclo(m,12) totally-real VEC fix
# For each m with phi(m)>=24, find totally-real degree-12 subfield
# ================================================================
print("\n=== B: polsubcyclo(m,12) totally-real subfields ===", file=sys.stderr)
b_found = 0

# phi(m) must be divisible by 2 (since we want degree-12 subfield = phi(m)/2 * something)
# phi(m)=24: m in {35,39,52,72,78,84,90}
# phi(m)=48: m in {65,78,100,104,...} — degree-12 is a proper subfield
# phi(m)=36: m in {111,37,...} — degree-12 is degree/3 subfield
m_vals = [35, 39, 52, 72, 78, 84, 90, 91, 95, 104, 111, 126, 140, 180]

for m in m_vals:
    if len(polys) >= 75: break
    seed_pv = get_totally_real_subcyclo12(m)
    if seed_pv is None:
        print(f"  B m={m}: no totally-real degree-12 subfield", file=sys.stderr)
        continue
    print(f"  B m={m}: found totally-real seed, shifting...", file=sys.stderr)
    r_found = try_shift(seed_pv, f"B_cyclo{m}", C_NARROW, max_per_seed=6)
    b_found += len(r_found)
    if r_found:
        print(f"    r: {sorted(r_found)}", file=sys.stderr)
    else:
        print(f"    no r-values found (roots maybe too narrow)", file=sys.stderr)

print(f"Strategy B: {b_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: polsubcyclo(m,24) with VEC fix
# Many m values have multiple degree-24 subfields
# ================================================================
print("\n=== C: polsubcyclo(m,24) VEC-aware ===", file=sys.stderr)
c_found = 0

more_m24 = [65, 91, 95, 104, 111, 120, 130, 143, 156, 168, 180, 195,
             52, 84, 126, 140, 210, 252, 360]

for m in more_m24:
    if len(polys) >= 92: break
    out = gp(f"""
v=polsubcyclo({m},24,'x);
if(type(v)=="t_POL",
   p=v;
   if(poldegree(p)==24 && polisirreducible(p),
      r=polsturm(p,-10^9,10^9);
      print("OK|",r,"|",Vecrev(p)));
   quit);
if(type(v)!="t_VEC", quit);
for(i=1,#v,
   p=v[i];
   if(poldegree(p)!=24, next);
   if(!polisirreducible(p), next);
   r=polsturm(p,-10^9,10^9);
   print("OK|",r,"|",Vecrev(p)));
""")
    added = 0
    for line in out.split('\n'):
        if len(polys) >= 92: break
        line = line.strip()
        if not line.startswith("OK|"): continue
        parts = line.split("|",2)
        try: r_val = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"subcyclo{m}_{c_found+added}_r{r_val}", coeffs):
                added += 1
                print(f"  C m={m} r={r_val} total={len(polys)}", file=sys.stderr)
    c_found += added

print(f"Strategy C: {c_found} polys, total={len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: Fill with more S3xD4 (different S3 cubics, covers gaps)
# ================================================================
if len(polys) < 100:
    print(f"\n=== D: Fill ===", file=sys.stderr)
    d_cubics_quartics = [
        ("x^3-11*x-4", "x^4-6*x^2+2"),
        ("x^3-11*x+7", "x^4-6*x^2+2"),
        ("x^3-12*x-8", "x^4-6*x^2+2"),
        ("x^3-13*x-4", "x^4-6*x^2+2"),
        ("x^3-4*x-1", "x^4-8*x^2+8"),
    ]
    for cubic, quartic in d_cubics_quartics:
        if len(polys) >= 100: break
        out = gp(f"""
f3={cubic}; f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), print("FAIL"); quit);
if(polsturm(f3,-10^9,10^9)!=3||polsturm(f4,-10^9,10^9)!=4, print("NR"); quit);
comp=polcompositum(f3,f4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12||!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12,
   print("FAIL12"); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"): continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  D {cubic[:10]}: seed OK", file=sys.stderr)
        r_found = try_shift(seed_pv, f"D_{cubic[3:7]}", C_WIDE, max_per_seed=5)
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
sc = Counter()
rc = Counter()
for p in polys:
    sc[p['label'].split('_')[0]] += 1
    for part in p['label'].split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
for k,v in sorted(sc.items()): print(f"  {k}: {v}", file=sys.stderr)
print("r:", dict(sorted(rc.items())), file=sys.stderr)

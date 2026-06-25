#!/usr/bin/env python3
"""Fill batch25 from 63 to 100 polys."""
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
with open('/tmp/batch25_polys.json') as f:
    polys = json.load(f)
seen_coeffs = {tuple(p['coeffs']) for p in polys}
print(f"Loaded {len(polys)} existing polys", file=sys.stderr)

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

C_WIDE = list(range(-60, 61))
C_NARROW = list(range(-8, 9))

# ================================================================
# MORE S3xV4 seeds (4th and 5th V4 pairs)
# ================================================================
print("=== S3xV4 extra seeds ===", file=sys.stderr)

extra_seeds = [
    ("x^3-4*x-1", 2, 11, "S3a_d2d11"),
    ("x^3-5*x-1", 3, 7, "S3b_d3d7"),
    ("x^3-4*x-1", 5, 7, "S3a_d5d7"),
    ("x^3-7*x-1", 2, 11, "S3c_d2d11"),
    ("x^3-4*x-1", 3, 11, "S3a_d3d11"),
]

for cubic, d1, d2, lname in extra_seeds:
    if len(polys) >= 85: break
    out = gp(f"""
f3={cubic};
q1=x^2-{d1};
q2=x^2-{d2};
if(!polisirreducible(f3), print("FAIL"); quit);
nr3=polsturm(f3,-10^9,10^9);
if(nr3!=3, print("NR3|",nr3); quit);
comp1=polcompositum(f3,q1);
if(#comp1==0, print("EMPTY1"); quit);
f6=comp1[1];
if(poldegree(f6)!=6, print("DEG6|",poldegree(f6)); quit);
nr6=polsturm(f6,-10^9,10^9);
if(nr6!=6, print("NR6|",nr6); quit);
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
        print(f"  {lname}: {out[:50]}", file=sys.stderr)
        continue
    sc = parse_vec(out.split("|",1)[1])
    if not sc or len(sc)!=13: continue
    seed_pv = f"Polrev([{','.join(sc)}])"
    print(f"  {lname}: seed OK, shifting...", file=sys.stderr)
    r_found = try_shift(seed_pv, lname, C_WIDE, max_per_seed=7)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After extra S3xV4: {len(polys)}", file=sys.stderr)

# ================================================================
# Fix polsubcyclo: handle t_VEC case by iterating over subfields
# ================================================================
print("\n=== polsubcyclo VEC handler ===", file=sys.stderr)

# For t_VEC: polsubcyclo returns a vector of degree-d polys
# Pick the ones that are totally real (nr==d/2 * 2 = d for degree-12 subfields)
m_vals_try = [35, 39, 52, 72, 78, 84, 90, 91, 104, 111]

for m in m_vals_try:
    if len(polys) >= 95: break
    # Try degree-12 subfields (totally real)
    out = gp(f"""
v=polsubcyclo({m},12,'x);
if(type(v)=="t_POL",
   if(polisirreducible(v) && polsturm(v,-10^9,10^9)==12,
      print("POLY|",Vecrev(v))); quit);
if(type(v)!="t_VEC", quit);
forvec(idx=[1,#v],
   p=v[idx[1]];
   if(!polisirreducible(p), next);
   if(polsturm(p,-10^9,10^9)!=12, next);
   print("POLY|",Vecrev(p));
   break);
""")
    for line in out.split('\n'):
        line = line.strip()
        if not line.startswith("POLY|"): continue
        seed_coeffs = parse_vec(line.split("|",1)[1])
        if not seed_coeffs or len(seed_coeffs)!=13: continue
        seed_pv = f"Polrev([{','.join(seed_coeffs)}])"
        print(f"  m={m}: totally real degree-12 subfield found, shifting...", file=sys.stderr)
        r_found = try_shift(seed_pv, f"cyclo{m}", C_NARROW, max_per_seed=5)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)
        break

print(f"After cyclo VEC: {len(polys)}", file=sys.stderr)

# ================================================================
# More polsubcyclo(m,24) direct - iterate over VEC
# ================================================================
print("\n=== More polsubcyclo(m,24) direct ===", file=sys.stderr)

more_m24 = [65, 91, 95, 104, 120, 130, 143, 156, 168, 180, 195,
            35, 39, 52, 72, 78]

for m in more_m24:
    if len(polys) >= 100: break
    out = gp(f"""
v=polsubcyclo({m},24,'x);
if(type(v)=="t_POL",
   if(poldegree(v)==24 && polisirreducible(v),
      r=polsturm(v,-10^9,10^9);
      print("OK|",r,"|",Vecrev(v))); quit);
if(type(v)!="t_VEC", quit);
forvec(idx=[1,#v],
   p=v[idx[1]];
   if(poldegree(p)!=24, next);
   if(!polisirreducible(p), next);
   r=polsturm(p,-10^9,10^9);
   print("OK|",r,"|",Vecrev(p)));
""")
    for line in out.split('\n'):
        if len(polys) >= 100: break
        line = line.strip()
        if not line.startswith("OK|"): continue
        parts = line.split("|",2)
        try: r_val = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"subcyclo{m}_r{r_val}", coeffs):
                print(f"  m={m} r={r_val} total={len(polys)}", file=sys.stderr)

print(f"After cyclo24: {len(polys)}", file=sys.stderr)

# ================================================================
# Final fill with more C3xD4 seeds (might give same T-number as batch23/24
# but at r-values not yet scored)
# ================================================================
if len(polys) < 100:
    print("\n=== Fill: additional S3xD4 seeds ===", file=sys.stderr)
    # Use S3 cubics NOT used in batch24 (batch24 used x^3-4,-5,-7,-5+3)
    extra_cubics = [
        ("x^3-8*x-4", "x^4-6*x^2+2"),
        ("x^3-8*x+4", "x^4-6*x^2+2"),
        ("x^3-9*x-1", "x^4-6*x^2+2"),
        ("x^3-11*x-4", "x^4-6*x^2+2"),
    ]
    for cubic, quartic in extra_cubics:
        if len(polys) >= 100: break
        out = gp(f"""
f3={cubic};
f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), print("FAIL"); quit);
if(polsturm(f3,-10^9,10^9)!=3||polsturm(f4,-10^9,10^9)!=4, print("NR"); quit);
comp=polcompositum(f3,f4);
if(#comp==0, print("EMPTY"); quit);
f12=comp[1];
if(poldegree(f12)!=12, print("DEG|",poldegree(f12)); quit);
if(!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12, print("FAIL12"); quit);
print("SEED|",Vecrev(f12));
""")
        if not out.startswith("SEED|"):
            print(f"  {cubic[:8]}: {out[:40]}", file=sys.stderr)
            continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  {cubic[:8]}: seed OK", file=sys.stderr)
        r_found = try_shift(seed_pv, f"S3D4_{cubic[3:7]}", C_WIDE, max_per_seed=4)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)

with open('/tmp/batch25_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch25_submission.txt', 'w') as f:
    for p in polys:
        f.write(','.join(p['coeffs']) + '\n')
print(f"Written to /tmp/batch25_submission.txt", file=sys.stderr)

from collections import Counter
rc = Counter()
for p in polys:
    for part in p['label'].split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print("r-distribution:", file=sys.stderr)
for r,v in sorted(rc.items()):
    print(f"  r={r:2d}: {v}", file=sys.stderr)

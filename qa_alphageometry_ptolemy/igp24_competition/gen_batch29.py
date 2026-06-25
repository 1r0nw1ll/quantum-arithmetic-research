#!/usr/bin/env python3
"""
Batch29: Maximum new T-numbers via cyclotomic seeds with large m.
Last daily submission — focus on breadth (many new T-numbers).

Each new m gives a different abelian degree-12 Galois group → different T-number.
Shift trick gives 5 r-values per T-number → up to 100 new scored (T,r) pairs.

Also test polsubcyclo(40,8) for potentially-real degree-8 field with C4xC2 Gal.
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

def try_shift(seed_pv, label_prefix, c_vals, max_per_seed=5):
    found_r = set()
    for c in c_vals:
        if len(polys) >= 100: break
        if len(found_r) >= max_per_seed: break
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
# BONUS A: polsubcyclo(40,8) × S3 if totally-real
# phi(40)=16, maximal totally real subfield has degree 8
# ================================================================
print("=== Bonus A: polsubcyclo(40,8) ===", file=sys.stderr)
v40_out = gp("""
v=polsubcyclo(40,8,'x);
if(type(v)=="t_POL",
   print("POL|",poldegree(v),"|",polsturm(v,-10^9,10^9),"|",Vecrev(v)));
if(type(v)=="t_VEC",
   apply(p->print("VEC|",poldegree(p),"|",polsturm(p,-10^9,10^9),"|",Vecrev(p)),v));
""")
print(f"  polsubcyclo(40,8): {v40_out[:80]}", file=sys.stderr)

# Check if any degree-8 with 8 real roots
v40_seed = None
for line in v40_out.split('\n'):
    line = line.strip()
    if not line.startswith(("POL|","VEC|")): continue
    parts = line.split("|",3)
    if len(parts) < 4: continue
    try:
        deg, nr = int(parts[1]), int(parts[2])
    except: continue
    if deg==8 and nr==8:
        v40_coeffs = parse_vec(parts[3])
        if v40_coeffs and len(v40_coeffs)==9:
            v40_seed = f"Polrev([{','.join(v40_coeffs)}])"
            print(f"  Found totally-real degree-8 in polsubcyclo(40,8)!", file=sys.stderr)
            break

if v40_seed:
    S3_CUBICS = [("x^3-4*x-1","a"),("x^3-5*x-1","b"),("x^3-7*x-1","c"),
                 ("x^3-7*x+7","d"),("x^3-5*x+3","e")]
    for cubic, cname in S3_CUBICS:
        if len(polys) >= 5: break
        out = gp(f"""
v8={v40_seed};
f3={cubic};
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
            if add(f"BA40_{cname}_r{r_val}", coeffs):
                print(f"  BA40 {cname} r={r_val} total={len(polys)}", file=sys.stderr)
else:
    print(f"  No totally-real degree-8 in polsubcyclo(40,8)", file=sys.stderr)

# ================================================================
# MAIN: Large m cyclotomic degree-12 seeds
# batches 26-28 used m up to ~2520
# Try larger m values systematically
# ================================================================
print("\n=== Main: cyclotomic degree-12 seeds (large m) ===", file=sys.stderr)

# Strategy: highly composite numbers with phi(m) divisible by 12
# phi(m) must be divisible by 12 AND the degree-12 subfield must be totally real
# The totally-real part has degree phi(m)/2, so phi(m)/2 must be divisible by 12 → phi(m) div by 24

# All m with phi(m) div by 24 and reasonably small:
m_candidates = [
    # batch26: 35,39,52,72,78,84,90,91,95,104,111,126,140,180
    # batch27: 65,156,195,260,312,390,420,455,504,520,546,630
    # batch28: 780,840,910,924,1092,1155,1260,1365,1404,1540,1848,2340,2520
    # NEW (batch29):
    3003, 3360, 3465, 3780, 4095, 4290, 4620, 4641, 4680, 5005, 5040, 5460,
    5544, 5720, 6006, 6160, 6435, 6720, 7140, 7280, 7560, 7644, 7980, 8008,
    8190, 8580, 9009, 9240, 9360, 10010, 10395, 10920, 11004, 11220, 11440,
    12012, 12285, 12376, 12870, 13020, 13860, 14280
]

for m in m_candidates:
    if len(polys) >= 100: break
    seed_pv = get_totally_real_subcyclo(m, 12)
    if seed_pv is None:
        continue  # Skip silently for speed
    print(f"  m={m}: found totally-real seed", file=sys.stderr)
    r_found = try_shift(seed_pv, f"E_cyclo{m}", C_WIDE, max_per_seed=5)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After cyclo: {len(polys)}", file=sys.stderr)

# ================================================================
# Fill with S3xC4 (new S3 cubics not used in batch26)
# ================================================================
if len(polys) < 100:
    print("\n=== Fill: S3xC4 ===", file=sys.stderr)
    c4_quartic_pv = "Polrev([1,4,-4,-1,1])"
    fill_cubics = [
        ("x^3-14*x-4","S3_14a"), ("x^3-14*x+4","S3_14b"), ("x^3-15*x-7","S3_15"),
        ("x^3-16*x-4","S3_16a"), ("x^3-17*x-4","S3_17"), ("x^3-18*x-4","S3_18"),
        ("x^3-19*x-4","S3_19"), ("x^3-20*x-8","S3_20"),
    ]
    for cubic, cname in fill_cubics:
        if len(polys) >= 100: break
        out = gp(f"""
f3={cubic}; q4={c4_quartic_pv};
if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3, quit);
comp=polcompositum(f3,q4);
f12=comp[1];
if(poldegree(f12)!=12||!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12, quit);
print("OK|",Vecrev(f12));
""")
        if not out.startswith("OK|"): continue
        sc = parse_vec(out.split("|",1)[1])
        if not sc or len(sc)!=13: continue
        seed_pv = f"Polrev([{','.join(sc)}])"
        print(f"  Fill {cname}: seed OK", file=sys.stderr)
        r_found = try_shift(seed_pv, f"Fill_{cname}", C_WIDE, max_per_seed=5)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch29_polys.json','w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch29_submission.txt','w') as f:
    for p in polys:
        f.write(','.join(p['coeffs'])+'\n')
print(f"Written to /tmp/batch29_submission.txt", file=sys.stderr)

from collections import Counter
rc = Counter()
tc = Counter()
for p in polys:
    label = p['label']
    for part in label.split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
    prefix = label.split('_')[0]
    tc[prefix] += 1
print("T-number groups:", dict(sorted(tc.items())), file=sys.stderr)
print("r:", dict(sorted(rc.items())), file=sys.stderr)

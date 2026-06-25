#!/usr/bin/env python3
"""
Batch27: New T-numbers via compositum approach (not shift trick).

Strategy overview:
A. polcompositum(f12, x²-d) where f12 has various real root counts
   - f12 with nr=0: degree-24 with r=0
   - f12 with nr=6: degree-24 with r=12
   - f12 with nr=12 (totally real): degree-24 with r=24
   Each gives T-number = G×C2 where G=Gal(f12/Q). NEW T-numbers vs shift trick!

B. polsubcyclo(17,8) × S3 cubic → degree-24 directly (C8×S3, r=24, new T)

C. Splitting field of D4-quartic × S3 → degree-24 (D4_regular×S3, r=24, new T)

D. Fallback: S3×D4 with more S3 cubics (same T as batch23/24, different r)
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

def try_compositum24(f_pv, label, d_vals, max_per_seed=10):
    """polcompositum(f, x^2-d) -> degree-24 poly directly."""
    found_r = set()
    for d in d_vals:
        if len(polys) >= 100: break
        if len(found_r) >= max_per_seed: break
        out = gp(f"""
f={f_pv};
q2=x^2-{d};
comp=polcompositum(f,q2);
if(#comp==0, quit);
f24=comp[1];
if(poldegree(f24)!=24, quit);
if(!polisirreducible(f24), quit);
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

D_VALS = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 23, 26, 29, 30, 31, 33, 34]

# ================================================================
# STRATEGY A: polcompositum(f12, x²-d) where f12 has various signatures
# Gives T-number = Gal(f12)×C2 (NEW vs shift-trick T-numbers)
# ================================================================
print("=== A: f12 compositum with quadratics ===", file=sys.stderr)

# A1: Totally-real S3×D4 degree-12 seeds (nr=12) → r=24
print("  A1: S3xD4 seeds (r=24)", file=sys.stderr)
s3_cubics = [("x^3-4*x-1","a"), ("x^3-5*x-1","b"), ("x^3-7*x-1","c"),
             ("x^3-5*x+3","d"), ("x^3-11*x-4","e")]
d4_quartic = "x^4-6*x^2+2"

for cubic, cname in s3_cubics:
    if len(polys) >= 15: break
    out = gp(f"""
f3={cubic}; f4={d4_quartic};
comp=polcompositum(f3,f4);
f12=comp[1];
if(poldegree(f12)!=12||!polisirreducible(f12), quit);
if(polsturm(f12,-10^9,10^9)!=12, quit);
print("OK|",Vecrev(f12));
""")
    if not out.startswith("OK|"): continue
    sc = parse_vec(out.split("|",1)[1])
    if not sc or len(sc)!=13: continue
    f12_pv = f"Polrev([{','.join(sc)}])"
    r_found = try_compositum24(f12_pv, f"A1_{cname}_S3D4", D_VALS, max_per_seed=3)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

# A2: NOT-totally-real degree-12 seeds for r=0 and r=12
# Use D4-quartic with 2 real roots × S3 cubic (3 real) → f12 with 6 real → r=12
# Use D4-quartic with 0 real roots × S3 cubic (3 real) → f12 with 0 real → r=0
print("  A2: Mixed-signature f12 seeds (r=0,12)", file=sys.stderr)

# D4-quartics with 2 real roots: need polgalois=D4 (order 8) and exactly 2 real roots
# x^4-2x^2-2: roots x^2=1±√3, x^2=1+√3≈2.73>0(2 real), x^2=1-√3<0(0 real) → 2 real roots
# Check if D4
out_chk = gp("f=x^4-2*x^2-2; print(polgalois(f)[1],\"|\",polgalois(f)[4],\"|\",polsturm(f,-10^9,10^9));")
print(f"  x^4-2x^2-2: {out_chk}", file=sys.stderr)

# x^4-x^2-1: roots x^2=(1±√5)/2, (1+√5)/2≈1.618>0(2 real), (1-√5)/2<0(0) → 2 real
out_chk2 = gp("f=x^4-x^2-1; if(polisirreducible(f), print(polgalois(f)[1],\"|\",polgalois(f)[4],\"|\",polsturm(f,-10^9,10^9)), print(\"reducible\"));")
print(f"  x^4-x^2-1: {out_chk2}", file=sys.stderr)

# x^4+x^2-1: roots x^2=(-1±√5)/2. (-1+√5)/2≈0.618>0(2 real), (-1-√5)/2<0(0) → 2 real
out_chk3 = gp("f=x^4+x^2-1; if(polisirreducible(f), print(polgalois(f)[1],\"|\",polgalois(f)[4],\"|\",polsturm(f,-10^9,10^9)), print(\"reducible\"));")
print(f"  x^4+x^2-1: {out_chk3}", file=sys.stderr)

# x^4-3x^2-3: roots x^2=(3±√21)/2, (3+√21)/2≈4.79>0(2real), (3-√21)/2≈-0.79<0(0) → 2 real
out_chk4 = gp("f=x^4-3*x^2-3; if(polisirreducible(f), print(polgalois(f)[1],\"|\",polgalois(f)[4],\"|\",polsturm(f,-10^9,10^9)), print(\"reducible\"));")
print(f"  x^4-3x^2-3: {out_chk4}", file=sys.stderr)

# x^4-4x^2-2: roots x^2=2±√6, (2+√6)>0(2 real), (2-√6)<0(0) → 2 real
out_chk5 = gp("f=x^4-4*x^2-2; if(polisirreducible(f), print(polgalois(f)[1],\"|\",polgalois(f)[4],\"|\",polsturm(f,-10^9,10^9)), print(\"reducible\"));")
print(f"  x^4-4x^2-2: {out_chk5}", file=sys.stderr)

# Pick D4 quartics with 2 real roots
d4_2real = []
for fstr, nm in [("x^4-2*x^2-2","d4_2r_a"),("x^4-4*x^2-2","d4_2r_b"),("x^4-3*x^2-3","d4_2r_c")]:
    out = gp(f"f={fstr}; if(polisirreducible(f)&&polgalois(f)[1]==8&&polsturm(f,-10^9,10^9)==2, print(\"D4_2R\"), print(\"no\"));")
    if "D4_2R" in out:
        d4_2real.append((fstr, nm))
        print(f"  Found D4/2real: {fstr}", file=sys.stderr)

for d4q, d4n in d4_2real:
    if len(polys) >= 30: break
    for cubic, cname in s3_cubics[:3]:
        if len(polys) >= 30: break
        out = gp(f"""
f3={cubic}; f4={d4q};
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
        print(f"  A2 {cname}×{d4n} (nr={nr_seed})", file=sys.stderr)
        r_found = try_compositum24(f12_pv, f"A2_{cname}_{d4n}", D_VALS, max_per_seed=3)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After A: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: polsubcyclo(17,8) × S3 cubic → degree-24 directly
# Gal = C8 × S3 (order 48) — new T-number
# All r=24 (both totally real)
# ================================================================
print("\n=== B: polsubcyclo(17,8) x S3 cubics ===", file=sys.stderr)

v17_out = gp("v=polsubcyclo(17,8,'x); print(Vecrev(v));")
v17_coeffs = parse_vec(v17_out)
if v17_coeffs:
    v17_pv = f"Polrev([{','.join(v17_coeffs)}])"
    b_cubics = [("x^3-4*x-1","a"),("x^3-5*x-1","b"),("x^3-7*x-1","c"),
                ("x^3-7*x+7","d"),("x^3-5*x+3","e"),("x^3-11*x-4","f"),
                ("x^3-13*x-4","g"),("x^3-8*x-4","h")]
    for cubic, cname in b_cubics:
        if len(polys) >= 50: break
        out = gp(f"""
v8={v17_pv};
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
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"B_C8S3_{cname}_r{r_val}", coeffs):
                print(f"  B C8×S3 {cname} r={r_val} total={len(polys)}", file=sys.stderr)

print(f"After B: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: Splitting field of D4-quartic × S3 cubic
# Splitting field of x^4-6x^2+2 = polcompositum(x^4-6x^2+2, x^2-2)[1] (degree 8)
# polcompositum(f8_split, S3_cubic) → degree-24, Gal=D4_regular × S3, r=24
# ================================================================
print("\n=== C: Splitting field D4 x S3 cubics ===", file=sys.stderr)

out = gp("f4=x^4-6*x^2+2; sp=polcompositum(f4,x^2-2)[1]; print(Vecrev(sp));")
sp_coeffs = parse_vec(out)
if sp_coeffs and len(sp_coeffs)==9:
    sp_pv = f"Polrev([{','.join(sp_coeffs)}])"
    c_cubics = [("x^3-4*x-1","a"),("x^3-5*x-1","b"),("x^3-7*x-1","c"),
                ("x^3-7*x+7","d"),("x^3-5*x+3","e"),("x^3-11*x-4","f"),
                ("x^3-13*x-4","g"),("x^3-8*x-4","h")]
    for cubic, cname in c_cubics:
        if len(polys) >= 65: break
        out = gp(f"""
f8={sp_pv};
f3={cubic};
if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3, quit);
comp=polcompositum(f8,f3);
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
        if coeffs and len(coeffs)==25 and coeffs[-1]=='1' and coeffs[0]!='0':
            if add(f"C_D4sp_{cname}_r{r_val}", coeffs):
                print(f"  C D4split×S3 {cname} r={r_val} total={len(polys)}", file=sys.stderr)

print(f"After C: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: S3×C4 compositum with quadratics (Gal=(S3×C4)×C2, new T)
# ================================================================
print("\n=== D: S3xC4 f12 compositum with x^2-d ===", file=sys.stderr)

c4_quartic_pv = "Polrev([1,4,-4,-1,1])"  # x^4-x^3-4x^2+4x+1 from polsubcyclo(15,4)

for cubic, cname in s3_cubics:
    if len(polys) >= 75: break
    out = gp(f"""
f3={cubic};
q4={c4_quartic_pv};
comp=polcompositum(f3,q4);
f12=comp[1];
if(poldegree(f12)!=12||!polisirreducible(f12)||polsturm(f12,-10^9,10^9)!=12, quit);
print("OK|",Vecrev(f12));
""")
    if not out.startswith("OK|"): continue
    sc = parse_vec(out.split("|",1)[1])
    if not sc or len(sc)!=13: continue
    f12_pv = f"Polrev([{','.join(sc)}])"
    r_found = try_compositum24(f12_pv, f"D_{cname}_S3C4", D_VALS[:6], max_per_seed=2)
    if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After D: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY E: polsubcyclo(m,12) × shift trick for m NOT in batch26
# Batch26 used m=35,39,52,72,78,84,90,91,95,104,111,126,140,180
# Try new m values
# ================================================================
if len(polys) < 90:
    print("\n=== E: More cyclotomic degree-12 seeds ===", file=sys.stderr)
    new_m_vals = [65, 156, 195, 260, 312, 390, 420, 455, 504, 520, 546, 630]

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

    for m in new_m_vals:
        if len(polys) >= 90: break
        seed_pv = get_totally_real_subcyclo(m, 12)
        if seed_pv is None:
            print(f"  E m={m}: none", file=sys.stderr)
            continue
        print(f"  E m={m}: found, shifting...", file=sys.stderr)
        r_found = try_shift(seed_pv, f"E_cyclo{m}", C_WIDE, max_per_seed=5)
        if r_found: print(f"    r: {sorted(r_found)}", file=sys.stderr)

print(f"After E: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY F: Fallback S3×D4 with new S3 cubics
# ================================================================
if len(polys) < 100:
    print("\n=== F: S3xD4 filler ===", file=sys.stderr)
    f_seeds = [
        ("x^3-18*x-4",  d4_quartic),
        ("x^3-19*x-4",  d4_quartic),
        ("x^3-20*x-8",  d4_quartic),
        ("x^3-21*x-7",  d4_quartic),
        ("x^3-22*x-8",  d4_quartic),
    ]
    for cubic, quartic in f_seeds:
        if len(polys) >= 100: break
        out = gp(f"""
f3={cubic}; f4={quartic};
if(!polisirreducible(f3)||!polisirreducible(f4), quit);
if(polsturm(f3,-10^9,10^9)!=3||polsturm(f4,-10^9,10^9)!=4, quit);
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
with open('/tmp/batch27_polys.json','w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch27_submission.txt','w') as f:
    for p in polys:
        f.write(','.join(p['coeffs'])+'\n')
print(f"Written to /tmp/batch27_submission.txt", file=sys.stderr)

from collections import Counter
sc_ctr = Counter()
rc = Counter()
for p in polys:
    label = p['label']
    for k in ['A1','A2','B_','C_','D_','E_','F_']:
        if label.startswith(k):
            sc_ctr[k.strip('_')] += 1
            break
    for part in label.split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print("Breakdown:", dict(sc_ctr), file=sys.stderr)
print("r:", dict(sorted(rc.items())), file=sys.stderr)

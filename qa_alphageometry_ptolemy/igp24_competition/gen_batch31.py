#!/usr/bin/env python3
"""
Batch31: Maximize r=24 and r=0 coverage.

Leaderboard insight:
- r=24: 18,680 unclaimed pairs (25.3%) — LARGEST absolute unclaimed bucket
- r=0:  14,716 unclaimed pairs (40.8%)
- r=22: SKIP (100% saturated)

Fix from batch30: c=-5 instead of c=-3 for totally-real seeds → ensures r=24
for most seeds (avoids r=22 edge case where min root < -3 for some composite m).

Strategy:
A: polsubcyclo(m,8) complex × S3 → r=0 (new m values not in batch30)
B: cyclotomic totally-real deg-12 seeds × c=-5 → r=24 (c < min_root for most seeds)
C: cyclotomic totally-real deg-12 seeds × c=-7 → r=24 backup
D: fill with more fresh m values for strategy A

PARI rule: ALL if() statements on SINGLE LINES. No multi-line blocks from stdin.
polsubcyclo(m,8) WITHOUT 'x variable arg.
"""
import subprocess, json, sys
from collections import Counter

def gp(cmd, timeout=300):
    script = "default(parisizemax,800000000);\n" + cmd + "\nquit\n"
    r = subprocess.run(['gp','-q'], input=script, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or '').strip()

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

S3_CUBICS = [
    ("x^3-4*x-1",   "c01"),
    ("x^3-5*x-1",   "c02"),
    ("x^3-7*x-1",   "c03"),
    ("x^3-7*x+7",   "c04"),
    ("x^3-5*x+3",   "c05"),
    ("x^3-8*x-4",   "c06"),
    ("x^3-11*x-4",  "c07"),
    ("x^3-13*x-4",  "c08"),
    ("x^3-14*x-4",  "c09"),
    ("x^3-15*x-7",  "c10"),
    ("x^3-16*x-4",  "c11"),
    ("x^3-17*x-4",  "c12"),
]

def get_complex_deg8(m):
    out = gp(
        "v=polsubcyclo(" + str(m) + ",8);"
        "if(type(v)==\"t_POL\"&&poldegree(v)==8&&polsturm(v,-10^9,10^9)==0,print(\"OK|\",Vecrev(v)));"
        "if(type(v)==\"t_VEC\",for(i=1,#v,if(poldegree(v[i])==8&&polsturm(v[i],-10^9,10^9)==0,print(\"OK|\",Vecrev(v[i]));break)));"
    )
    for line in out.split('\n'):
        line = line.strip()
        if line.startswith("OK|"):
            coeffs = parse_vec(line[3:])
            if coeffs and len(coeffs) == 9:
                return "Polrev([" + ",".join(coeffs) + "])"
    return None

def degree8_cross_s3(m, tag):
    f8_pv = get_complex_deg8(m)
    if f8_pv is None:
        print(f"  {tag}: no complex deg-8 for m={m}", file=sys.stderr)
        return 0
    print(f"  {tag}: f8 OK for m={m}", file=sys.stderr)
    added = 0
    for cubic, cname in S3_CUBICS:
        if len(polys) >= 100: break
        out = gp(
            "f8=" + f8_pv + ";"
            "f3=" + cubic + ";"
            "if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3,quit);"
            "comp=polcompositum(f8,f3);"
            "if(#comp==0,quit);"
            "f24=comp[1];"
            "if(poldegree(f24)!=24||!polisirreducible(f24),quit);"
            "r=polsturm(f24,-10^9,10^9);"
            "print(\"OK|\",r,\"|\",Vecrev(f24));"
        )
        if not out.startswith("OK|"): continue
        parts = out.split("|", 2)
        try: r_val = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs) == 25 and coeffs[-1] == '1':
            if add(f"{tag}_{cname}_r{r_val}", coeffs):
                added += 1
                print(f"    {tag} {cname} r={r_val} total={len(polys)}", file=sys.stderr)
    return added

def get_totally_real_deg12(m):
    out = gp(
        "v=polsubcyclo(" + str(m) + ",12);"
        "if(type(v)==\"t_POL\"&&poldegree(v)==12&&polsturm(v,-10^9,10^9)==12,print(\"FOUND|\",Vecrev(v)));"
        "if(type(v)==\"t_VEC\",for(i=1,#v,if(poldegree(v[i])==12&&polsturm(v[i],-10^9,10^9)==12,print(\"FOUND|\",Vecrev(v[i]));break)));"
    )
    for line in out.split('\n'):
        line = line.strip()
        if line.startswith("FOUND|"):
            coeffs = parse_vec(line[6:])
            if coeffs and len(coeffs) == 13:
                return "Polrev([" + ",".join(coeffs) + "])"
    return None

def cyclo_shift(m, c_vals, tag, max_per_m=1):
    seed_pv = get_totally_real_deg12(m)
    if seed_pv is None:
        return 0
    added = 0
    for c in c_vals:
        if len(polys) >= 100: break
        if added >= max_per_m: break
        out = gp(
            "h=" + seed_pv + ";"
            "g24=subst(h,x,x^2+(" + str(c) + "));"
            "if(poldegree(g24)!=24,quit);"
            "if(!polisirreducible(g24),quit);"
            "r=polsturm(g24,-10^9,10^9);"
            "if(r==22,quit);"  # skip r=22, fully saturated
            "print(\"OK|\",r,\"|\",Vecrev(g24));"
        )
        if not out.startswith("OK|"): continue
        parts = out.split("|", 2)
        try: r_val = int(parts[1])
        except: continue
        if r_val == 22: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs) == 25 and coeffs[-1] == '1' and coeffs[0] != '0':
            if add(f"{tag}_c{c}_r{r_val}", coeffs):
                added += 1
                print(f"  {tag} m={m} c={c} r={r_val} total={len(polys)}", file=sys.stderr)
    return added

# ================================================================
# STRATEGY A: Fresh m values for polsubcyclo(m,8) × S3 → r=0
# Batch30 used: 15,16,20,24,30,35,39,41,45,52,55,56,70,72,73,
#               75,78,82,84,87,88,89,90,91,95,100
# Use next set from scan (m=102..300)
# ================================================================
print("=== A: polsubcyclo(m,8) × S3 (fresh m values) ===", file=sys.stderr)
A_M_VALS = [
    102, 104, 105, 110, 111, 112, 115, 116, 117, 119, 120, 123,
    128, 130, 132, 135, 136, 137, 140, 143, 144, 145, 146, 148,
    150, 152, 153, 155, 156, 159, 160, 164, 165, 168, 170, 174,
]

for m in A_M_VALS:
    if len(polys) >= 60: break  # reserve 40 for B/C
    degree8_cross_s3(m, f"A_s{m}")

print(f"\nAfter A: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: cyclotomic seeds × c=-5 → r=24 (target biggest unclaimed bucket)
# c=-5 is far enough below -2 to clear min roots of most seeds
# Filter: skip r=22 inside PARI
# ================================================================
print("\n=== B: cyclotomic c=-5 → r=24 (target) ===", file=sys.stderr)
B_M_VALS = [
    35, 65, 91, 156, 260, 390, 420, 546, 630,
    780, 840, 910, 924, 1092, 1155, 1260, 1365,
    1540, 1848, 2340, 2520, 3003, 3360, 3465,
    3780, 4095, 4290, 4620, 4641, 5040, 5460,
    5544, 6006, 6720, 7140, 7560, 8008, 8190, 8580,
    9009, 9240, 10010, 10395, 11220, 12012, 13860,
]
for m in B_M_VALS:
    if len(polys) >= 85: break
    cyclo_shift(m, [-5, -6, -7, -8, -10], f"B_m{m}")

print(f"\nAfter B: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: fill with more strategy A or cyclotomic c=3 (r=0)
# ================================================================
print("\n=== C: fill ===", file=sys.stderr)
for m in A_M_VALS[12:]:  # continue from where A left off
    if len(polys) >= 100: break
    degree8_cross_s3(m, f"C_s{m}")

# Also try more cyclotomic c=3 for r=0
C_M_VALS = [175, 180, 182, 195, 200, 203, 204, 205, 210, 215, 220, 225]
for m in C_M_VALS:
    if len(polys) >= 100: break
    degree8_cross_s3(m, f"C2_s{m}")

print(f"\nAfter C: {len(polys)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch31_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch31_submission.txt', 'w') as f:
    for p in polys:
        f.write(','.join(p['coeffs']) + '\n')

rc = Counter()
sc = Counter()
for p in polys:
    label = p['label']
    sc[label.split('_')[0]] += 1
    for part in label.split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print(f"r-dist: {dict(sorted(rc.items()))}", file=sys.stderr)
print(f"strategy: {dict(sorted(sc.items()))}", file=sys.stderr)
print(f"Written /tmp/batch31_submission.txt", file=sys.stderr)

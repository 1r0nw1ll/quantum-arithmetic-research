#!/usr/bin/env python3
"""
Batch32: Continue r=24 and r=0 expansion with fresh m values.

Strategy:
A: polsubcyclo(m,8) × S3, m=175..350 range (fresh m values)
B: cyclotomic c=-5..-10 scan for r=24, fresh large m values
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

def degree8_cross_s3(m, tag, max_s3=12):
    f8_pv = get_complex_deg8(m)
    if f8_pv is None:
        return 0
    print(f"  {tag}: f8 OK m={m}", file=sys.stderr)
    added = 0
    for cubic, cname in S3_CUBICS[:max_s3]:
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

def cyclo_shift_r24(m, tag):
    """Try to get r=24 by shifting with c far enough below all roots."""
    seed_pv = get_totally_real_deg12(m)
    if seed_pv is None:
        return 0
    for c in [-5, -6, -7, -8, -10, -12, -15, -20]:
        if len(polys) >= 100: break
        out = gp(
            "h=" + seed_pv + ";"
            "g24=subst(h,x,x^2+(" + str(c) + "));"
            "if(poldegree(g24)!=24,quit);"
            "if(!polisirreducible(g24),quit);"
            "r=polsturm(g24,-10^9,10^9);"
            "if(r!=24,quit);"  # only r=24
            "print(\"OK|24|\",Vecrev(g24));"
        )
        if out.startswith("OK|24|"):
            parts = out.split("|", 2)
            coeffs = parse_vec(parts[2])
            if coeffs and len(coeffs) == 25 and coeffs[-1] == '1' and coeffs[0] != '0':
                if add(f"{tag}_c{c}_r24", coeffs):
                    print(f"  {tag} m={m} c={c} r=24 total={len(polys)}", file=sys.stderr)
                    return 1
    return 0

# ================================================================
# STRATEGY A: polsubcyclo(m,8) × S3, fresh m values from scan
# Batch30 used m<=100. Batch31 used m=102..156.
# Use m=175..400 range
# ================================================================
print("=== A: polsubcyclo(m,8) × S3 m=175..400 ===", file=sys.stderr)
A_M_VALS = [
    175, 180, 182, 183, 184, 185, 187, 190, 192, 195,
    200, 203, 204, 205, 208, 210, 212, 215, 216, 219,
    220, 221, 222, 224, 225, 228, 230, 231, 232, 233,
    234, 235, 238, 240, 244, 245, 246, 247, 248, 252,
]

for m in A_M_VALS:
    if len(polys) >= 50: break
    degree8_cross_s3(m, f"A_s{m}", max_s3=5)  # max 5 S3 per m to cover more T-numbers

print(f"\nAfter A: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: cyclotomic c scan for r=24, large fresh m values
# Use m values NOT used in batches 26-31
# ================================================================
print("\n=== B: cyclotomic → r=24, fresh large m ===", file=sys.stderr)
B_M_VALS = [
    8580, 9009, 9240, 9360, 10010, 10395, 10920, 11004,
    11220, 11440, 12012, 12285, 12870, 13020, 13860, 14280,
    15015, 15120, 16380, 17160, 18018, 18480, 19019, 20020,
    21021, 21840, 22260, 23100, 24024, 25025, 26040, 27027,
    28028, 29029, 30030, 31920, 32760, 34320, 36036, 37800,
    40040, 42042, 43680, 45045, 46200, 48048, 50050, 55440,
]
for m in B_M_VALS:
    if len(polys) >= 85: break
    cyclo_shift_r24(m, f"B_m{m}")

print(f"\nAfter B: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: More polsubcyclo(m,8) × S3 to fill
# ================================================================
print("\n=== C: fill with more deg8×S3 ===", file=sys.stderr)
for m in [255, 256, 259, 260, 261, 264, 265, 267, 270, 272, 273, 274, 275, 276, 280, 281, 285, 286, 287, 288]:
    if len(polys) >= 100: break
    degree8_cross_s3(m, f"C_s{m}", max_s3=4)

print(f"\nAfter C: {len(polys)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch32_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch32_submission.txt', 'w') as f:
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
print(f"Written /tmp/batch32_submission.txt", file=sys.stderr)

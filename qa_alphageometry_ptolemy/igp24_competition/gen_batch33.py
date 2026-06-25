#!/usr/bin/env python3
"""
Batch33: Push r=24 hard with aggressive c-scan, and expand r=0 m range.

Focus: maximize r=24 (18,680 unclaimed, lowest coverage rate among large buckets).
Strategy B now tries c=-5..-30 to guarantee r=24 for any cyclotomic seed.
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
    ("x^3-4*x-1", "c01"), ("x^3-5*x-1", "c02"), ("x^3-7*x-1", "c03"),
    ("x^3-7*x+7", "c04"), ("x^3-5*x+3", "c05"), ("x^3-8*x-4", "c06"),
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

def degree8_cross_s3(m, tag, max_s3=6):
    f8_pv = get_complex_deg8(m)
    if f8_pv is None:
        return 0
    added = 0
    for cubic, cname in S3_CUBICS[:max_s3]:
        if len(polys) >= 100: break
        out = gp(
            "f8=" + f8_pv + ";f3=" + cubic + ";"
            "if(!polisirreducible(f3)||polsturm(f3,-10^9,10^9)!=3,quit);"
            "comp=polcompositum(f8,f3);if(#comp==0,quit);"
            "f24=comp[1];if(poldegree(f24)!=24||!polisirreducible(f24),quit);"
            "r=polsturm(f24,-10^9,10^9);print(\"OK|\",r,\"|\",Vecrev(f24));"
        )
        if not out.startswith("OK|"): continue
        parts = out.split("|", 2)
        try: r_val = int(parts[1])
        except: continue
        coeffs = parse_vec(parts[2])
        if coeffs and len(coeffs) == 25 and coeffs[-1] == '1':
            if add(f"{tag}_{cname}_r{r_val}", coeffs):
                added += 1
    if added:
        print(f"  {tag} m={m}: {added} polys total={len(polys)}", file=sys.stderr)
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
    seed_pv = get_totally_real_deg12(m)
    if seed_pv is None:
        return 0
    for c in [-5, -6, -7, -8, -10, -12, -15, -20, -25, -30]:
        if len(polys) >= 100: break
        out = gp(
            "h=" + seed_pv + ";"
            "g24=subst(h,x,x^2+(" + str(c) + "));"
            "if(poldegree(g24)!=24,quit);if(!polisirreducible(g24),quit);"
            "r=polsturm(g24,-10^9,10^9);if(r!=24,quit);"
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
# STRATEGY A: polsubcyclo(m,8) × S3, m=290..500
# Batch32 used m=175..264
# ================================================================
print("=== A: polsubcyclo(m,8) × S3 m=290..500 ===", file=sys.stderr)
A_M_VALS = [
    290, 291, 292, 295, 296, 299, 300, 303, 304, 305,
    306, 308, 310, 312, 313, 315, 318, 319, 320, 323,
    325, 327, 328, 330, 333, 335, 336, 339, 340, 344,
    345, 348, 350, 351, 352, 355, 356, 357, 360, 364,
    365, 366, 368, 369, 370, 371, 372, 374, 375, 376,
]
for m in A_M_VALS:
    if len(polys) >= 50: break
    degree8_cross_s3(m, f"A_s{m}", max_s3=4)

print(f"\nAfter A: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: r=24 via aggressive c-scan, very large m values
# Use m values beyond batch31/32 range
# ================================================================
print("\n=== B: cyclo → r=24, very large m ===", file=sys.stderr)
B_M_VALS = [
    55440, 60060, 65520, 69300, 72072, 75075, 78078, 80080,
    83160, 85085, 90090, 92820, 95095, 100100, 105105, 110110,
    112112, 115115, 120120, 125125, 130130, 135135, 140140,
    145145, 150150, 155155, 160160, 165165, 170170, 175175,
    180180, 185185, 190190, 195195, 200200, 210210, 225225,
    240240, 255255, 270270, 285285, 300300, 315315, 330330,
    345345, 360360, 375375, 390390, 405405, 420420,
]
for m in B_M_VALS:
    if len(polys) >= 90: break
    cyclo_shift_r24(m, f"B_m{m}")

print(f"\nAfter B: {len(polys)}", file=sys.stderr)

# Fill
print("\n=== C: fill ===", file=sys.stderr)
for m in [377, 380, 384, 385, 388, 390, 391, 392, 395, 396, 399, 400]:
    if len(polys) >= 100: break
    degree8_cross_s3(m, f"C_s{m}", max_s3=4)

print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch33_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch33_submission.txt', 'w') as f:
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
print(f"Written /tmp/batch33_submission.txt", file=sys.stderr)

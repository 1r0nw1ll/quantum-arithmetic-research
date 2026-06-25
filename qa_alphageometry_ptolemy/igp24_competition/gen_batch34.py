#!/usr/bin/env python3
"""
Batch34: Target r=16 and r=20 (lowest coverage rates per leaderboard).

Method:
- Compute roots of each totally-real deg-12 cyclotomic seed numerically
- Find c values that give exactly 8 roots > c (→ r=16) or 10 roots > c (→ r=20)
- Also include r=0 and r=24 fills

Leaderboard:
- r=20: 16.4% coverage (lowest!) → 9,069 unclaimed
- r=16: 20.4% coverage → 17,055 unclaimed
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
    ("x^3-11*x-4", "c07"), ("x^3-13*x-4", "c08"),
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
            c = parse_vec(line[3:])
            if c and len(c) == 9:
                return "Polrev([" + ",".join(c) + "])"
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
        c = parse_vec(parts[2])
        if c and len(c) == 25 and c[-1] == '1':
            if add(f"{tag}_{cname}_r{r_val}", c):
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
            c = parse_vec(line[6:])
            if c and len(c) == 13:
                return "Polrev([" + ",".join(c) + "])"
    return None

def cyclo_target_r(m, target_r, tag):
    """Find c giving exactly target_r real roots for cyclotomic deg-12 seed."""
    seed_pv = get_totally_real_deg12(m)
    if seed_pv is None:
        return 0
    # Get sorted real roots of the degree-12 seed
    out = gp(
        "h=" + seed_pv + ";"
        "rts=polroots(h);"
        "real_rts=List();"
        "for(i=1,#rts,if(abs(imag(rts[i]))<10^(-5),listput(real_rts,real(rts[i]))));"
        "real_rts=vecsort(Vec(real_rts));"
        "if(#real_rts!=12,quit);"
        "print(\"ROOTS|\",real_rts);"
    )
    if not out.startswith("ROOTS|"): return 0
    roots_str = out[6:]
    # Parse roots as floats
    roots_str = roots_str.strip()[1:-1]  # remove [ ]
    try:
        roots = [float(x.strip()) for x in roots_str.split(',')]
    except: return 0
    if len(roots) != 12: return 0
    roots.sort()  # ascending

    # For r=2k real roots: need exactly k roots > c
    # i.e., c is between roots[12-k-1] and roots[12-k]
    k = target_r // 2
    if k < 1 or k > 11: return 0
    # c between roots[11-k] and roots[12-k] (0-indexed)
    # roots[12-k-1] < c < roots[12-k]
    # (root at index 11-k) < c < (root at index 12-k)
    lo = roots[11-k]
    hi = roots[12-k] if k < 12 else roots[11-k] + 100
    c = (lo + hi) / 2  # midpoint
    c_int = int(c)  # use integer c for clean poly
    # Verify r value
    for c_try in [c_int, c_int + 1, c_int - 1]:
        if len(polys) >= 100: break
        out2 = gp(
            "h=" + seed_pv + ";"
            "g24=subst(h,x,x^2+(" + str(c_try) + "));"
            "if(poldegree(g24)!=24,quit);if(!polisirreducible(g24),quit);"
            "r=polsturm(g24,-10^9,10^9);if(r!=" + str(target_r) + ",quit);"
            "print(\"OK|\",r,\"|\",Vecrev(g24));"
        )
        if out2.startswith("OK|"):
            parts = out2.split("|", 2)
            try: r_val = int(parts[1])
            except: continue
            coeffs = parse_vec(parts[2])
            if coeffs and len(coeffs) == 25 and coeffs[-1] == '1' and coeffs[0] != '0':
                if add(f"{tag}_c{c_try}_r{r_val}", coeffs):
                    print(f"  {tag} m={m} c={c_try} r={r_val} total={len(polys)}", file=sys.stderr)
                    return 1
    return 0

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

# m values with totally-real deg-12 subfields, sorted by size
R20_M_VALS = [
    # Small m → simpler fields, smaller discriminants → better score if contested
    35, 65, 91, 156, 260, 390, 420, 546, 630,
    780, 840, 910, 924, 1092, 1155, 1260, 1365,
    1540, 1848, 2340, 2520, 3003, 3360, 3465,
    3780, 4095, 4290, 4620, 4641, 5040, 5460,
]

R16_M_VALS = [
    35, 65, 91, 156, 260, 390, 420, 546, 630,
    780, 840, 910, 924, 1092, 1155, 1260, 1365,
    1540, 1848, 2340, 2520, 3003, 3360, 3465,
]

# ================================================================
# STRATEGY A: Target r=20 (16.4% — LOWEST coverage)
# ================================================================
print("=== A: target r=20 ===", file=sys.stderr)
for m in R20_M_VALS:
    if len(polys) >= 35: break
    cyclo_target_r(m, 20, f"A20_m{m}")

print(f"\nAfter A (r=20): {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: Target r=16 (20.4% coverage)
# ================================================================
print("\n=== B: target r=16 ===", file=sys.stderr)
for m in R16_M_VALS:
    if len(polys) >= 65: break
    cyclo_target_r(m, 16, f"B16_m{m}")

print(f"\nAfter B (r=16): {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: r=24 fill
# ================================================================
print("\n=== C: r=24 fill ===", file=sys.stderr)
C24_M_VALS = [
    5544, 6006, 6720, 7140, 7560, 8008, 8190, 8580,
    9009, 9240, 10010, 10395, 11220, 12012, 13860, 14280,
    15015, 16380, 17160, 18018, 19019, 20020,
]
for m in C24_M_VALS:
    if len(polys) >= 85: break
    cyclo_shift_r24(m, f"C_m{m}")

print(f"\nAfter C (r=24): {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: r=0 fill with fresh m values
# ================================================================
print("\n=== D: r=0 fill ===", file=sys.stderr)
for m in [399, 400, 403, 404, 405, 406, 407, 408, 409, 410, 411, 415, 416, 420, 424, 425]:
    if len(polys) >= 100: break
    degree8_cross_s3(m, f"D_s{m}", max_s3=4)

print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch34_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch34_submission.txt', 'w') as f:
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
print(f"Written /tmp/batch34_submission.txt ({len(polys)} polys)", file=sys.stderr)

#!/usr/bin/env python3
"""
Batch35: Target r=8 and r=12 (32.7% and 29.4% coverage).

Leaderboard snapshot:
- r=8:  32.7% → 15,863 unclaimed
- r=12: 29.4% → 14,074 unclaimed
- r=20: 16.4% → 9,069 unclaimed (still targeting)
- r=16: 20.4% → 17,055 unclaimed (still targeting)

Method: cyclo_target_r(m, 8) and cyclo_target_r(m, 12) — find c between 4th/5th
and 6th/7th largest roots of the deg-12 seed respectively.
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

def degree8_cross_s3(m, tag, max_s3=4):
    f8_pv = get_complex_deg8(m)
    if f8_pv is None:
        return 0
    added = 0
    for cubic, cname in [("x^3-4*x-1","c01"),("x^3-5*x-1","c02"),("x^3-7*x-1","c03"),("x^3-7*x+7","c04")][:max_s3]:
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
    out = gp(
        "h=" + seed_pv + ";"
        "rts=polroots(h);"
        "rr=List();"
        "for(i=1,#rts,if(abs(imag(rts[i]))<10^(-5),listput(rr,real(rts[i]))));"
        "rr=vecsort(Vec(rr));"
        "if(#rr!=12,quit);"
        "print(\"ROOTS|\",rr);"
    )
    if not out.startswith("ROOTS|"): return 0
    roots_str = out[6:].strip()[1:-1]
    try:
        roots = sorted([float(x.strip()) for x in roots_str.split(',')])
    except: return 0
    if len(roots) != 12: return 0

    k = target_r // 2
    if k < 1 or k > 11: return 0
    lo = roots[11-k]
    hi = roots[12-k] if k < 12 else roots[11-k] + 100
    c_mid = (lo + hi) / 2

    for c_try in [int(c_mid), int(c_mid) + 1, int(c_mid) - 1, round(c_mid)]:
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

M_VALS = [
    35, 65, 91, 156, 260, 390, 420, 546, 630,
    780, 840, 910, 924, 1092, 1155, 1260, 1365,
    1540, 1848, 2340, 2520, 3003, 3360, 3465,
    3780, 4095, 4290, 4620, 4641, 5040, 5460,
    5544, 6006, 6720, 7140, 7560, 8008, 8190, 8580,
    9009, 9240, 10010, 10395, 11220, 12012,
]

# ================================================================
# STRATEGY A: r=8 (32.7%)
# ================================================================
print("=== A: target r=8 ===", file=sys.stderr)
for m in M_VALS:
    if len(polys) >= 30: break
    cyclo_target_r(m, 8, f"A8_m{m}")

print(f"\nAfter A (r=8): {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: r=12 (29.4%)
# ================================================================
print("\n=== B: target r=12 ===", file=sys.stderr)
for m in M_VALS:
    if len(polys) >= 60: break
    cyclo_target_r(m, 12, f"B12_m{m}")

print(f"\nAfter B (r=12): {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: r=20 more (16.4%)
# ================================================================
print("\n=== C: target r=20 ===", file=sys.stderr)
for m in M_VALS[15:]:
    if len(polys) >= 80: break
    cyclo_target_r(m, 20, f"C20_m{m}")

print(f"\nAfter C (r=20): {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY D: r=0 fill
# ================================================================
print("\n=== D: r=0 fill ===", file=sys.stderr)
for m in [427, 429, 430, 432, 435, 436, 438, 440, 442, 444, 445, 447, 448, 450, 451, 452]:
    if len(polys) >= 100: break
    degree8_cross_s3(m, f"D_s{m}", max_s3=3)

print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch35_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch35_submission.txt', 'w') as f:
    for p in polys:
        f.write(','.join(p['coeffs']) + '\n')

rc = Counter()
sc = Counter()
for p in polys:
    label = p['label']
    sc[label.split('_')[0][:3]] += 1
    for part in label.split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print(f"r-dist: {dict(sorted(rc.items()))}", file=sys.stderr)
print(f"strategy: {dict(sorted(sc.items()))}", file=sys.stderr)
print(f"Written /tmp/batch35_submission.txt ({len(polys)} polys)", file=sys.stderr)

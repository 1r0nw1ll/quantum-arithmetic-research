#!/usr/bin/env python3
"""
Batch30 v3: r=0 AND r=24 polys from two complementary strategies.

Key fixes:
- All PARI if() on SINGLE LINES (multi-line from stdin breaks PARI parser)
- polsubcyclo(m,8) WITHOUT 'x var arg (avoids tick-quote parser issue)
- Apply user insight: c=3 → r=0, c=-3 → r=24 for cyclotomic totally-real seeds

Leaderboard insight (from competition site):
- r=22: SKIP (100% saturated, no new pairs)
- r=24: 18,680 unclaimed pairs — HIGH VALUE
- r=0:  14,716 unclaimed pairs — HIGH VALUE
- r=20: 9,069 unclaimed pairs, lowest coverage rate 16.4% — MEDIUM VALUE

Strategy:
A: polsubcyclo(m,8) complex × S3 totally-real cubics → r=0 (12 S3 cubics per m)
B: cyclotomic totally-real deg-12 seeds × shift c=-3 → r=24 (1 per m, new direction)
C: cyclotomic totally-real deg-12 seeds × shift c=3 → r=0 (fill if needed)
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
    """Return Polrev string for first nr=0 degree-8 subfield of Q(zeta_m). Single-line ifs."""
    out = gp(
        "v=polsubcyclo(" + str(m) + ",8);"
        "if(type(v)==\"t_POL\"&&poldegree(v)==8&&polsturm(v,-10^9,10^9)==0,print(\"OK|\",Vecrev(v)));"
        "if(type(v)==\"t_VEC\",for(i=1,#v,p=v[i];if(poldegree(p)==8&&polsturm(p,-10^9,10^9)==0,print(\"OK|\",Vecrev(p));break)));"
    )
    for line in out.split('\n'):
        line = line.strip()
        if line.startswith("OK|"):
            coeffs = parse_vec(line[3:])
            if coeffs and len(coeffs) == 9:
                return "Polrev([" + ",".join(coeffs) + "])"
    return None

def degree8_cross_s3(m, tag):
    """polsubcyclo(m,8) complex × S3 cubics → degree-24, r=0."""
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
    """Return Polrev string for totally-real deg-12 subfield of Q(zeta_m). Single-line ifs."""
    out = gp(
        "v=polsubcyclo(" + str(m) + ",12);"
        "if(type(v)==\"t_POL\"&&poldegree(v)==12&&polsturm(v,-10^9,10^9)==12,print(\"FOUND|\",Vecrev(v)));"
        "if(type(v)==\"t_VEC\",for(i=1,#v,p=v[i];if(poldegree(p)==12&&polsturm(p,-10^9,10^9)==12,print(\"FOUND|\",Vecrev(p));break)));"
    )
    for line in out.split('\n'):
        line = line.strip()
        if line.startswith("FOUND|"):
            coeffs = parse_vec(line[6:])
            if coeffs and len(coeffs) == 13:
                return "Polrev([" + ",".join(coeffs) + "])"
    return None

def cyclo_shift(m, c, tag):
    """Shift totally-real deg-12 seed by c: subst(h, x, x^2+c) → degree-24."""
    seed_pv = get_totally_real_deg12(m)
    if seed_pv is None:
        print(f"  {tag}: no totally-real deg-12 for m={m}", file=sys.stderr)
        return 0
    out = gp(
        "h=" + seed_pv + ";"
        "g24=subst(h,x,x^2+(" + str(c) + "));"
        "if(poldegree(g24)!=24,quit);"
        "if(!polisirreducible(g24),quit);"
        "r=polsturm(g24,-10^9,10^9);"
        "print(\"OK|\",r,\"|\",Vecrev(g24));"
    )
    if not out.startswith("OK|"): return 0
    parts = out.split("|", 2)
    try: r_val = int(parts[1])
    except: return 0
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs) == 25 and coeffs[-1] == '1' and coeffs[0] != '0':
        if add(f"{tag}_c{c}_r{r_val}", coeffs):
            print(f"  {tag} m={m} c={c} r={r_val} total={len(polys)}", file=sys.stderr)
            return 1
    return 0

# ================================================================
# STRATEGY A: polsubcyclo(m,8) complex × S3 cubics → r=0
# m values with confirmed POL-type complex degree-8 subfield
# ================================================================
print("=== A: polsubcyclo(m,8) × S3 cubics ===", file=sys.stderr)
A_M_VALS = [15, 16, 20, 24, 30, 35, 39, 41, 45, 52, 55, 56, 70, 72, 73,
             75, 78, 82, 84, 87, 88, 89, 90, 91, 95, 100]

for m in A_M_VALS:
    if len(polys) >= 70: break  # reserve 30 for B/C
    degree8_cross_s3(m, f"A_s{m}")

print(f"\nAfter A: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY B: cyclotomic totally-real deg-12 seeds, c=-3 → r=24
# For any cyclotomic totally-real seed, roots in [-2,2], c=-3 < -2
# so ALL 24 roots of subst(h,x,x^2-3) are REAL → r=24
# ================================================================
print("\n=== B: cyclotomic c=-3 → r=24 ===", file=sys.stderr)
# m values from successful batch26-29 runs (known to have totally-real deg-12 subfields)
B_M_VALS = [
    35, 65, 91, 156, 260, 390, 420, 546, 630,
    780, 840, 910, 924, 1092, 1155, 1260, 1365,
    1540, 1848, 2340, 2520, 3003, 3360, 3465,
    3780, 4095, 4290, 4620, 4641, 5040, 5460,
    5544, 6006, 6720, 7140, 7560,
]
for m in B_M_VALS:
    if len(polys) >= 90: break  # reserve 10 for C
    cyclo_shift(m, -3, f"B_m{m}")

print(f"\nAfter B: {len(polys)}", file=sys.stderr)

# ================================================================
# STRATEGY C: cyclotomic totally-real deg-12 seeds, c=+3 → r=0
# (same seeds as B, different c value)
# ================================================================
print("\n=== C: cyclotomic c=+3 → r=0 ===", file=sys.stderr)
for m in B_M_VALS:
    if len(polys) >= 100: break
    cyclo_shift(m, 3, f"C_m{m}")

print(f"\nAfter C: {len(polys)}", file=sys.stderr)

# ================================================================
print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch30_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch30_submission.txt', 'w') as f:
    for p in polys:
        f.write(','.join(p['coeffs']) + '\n')

rc = Counter()
strategy_c = Counter()
for p in polys:
    label = p['label']
    strategy_c[label.split('_')[0]] += 1
    for part in label.split('_'):
        if part.startswith('r'):
            try: rc[int(part[1:])] += 1
            except: pass
print(f"r-dist: {dict(sorted(rc.items()))}", file=sys.stderr)
print(f"strategy: {dict(sorted(strategy_c.items()))}", file=sys.stderr)
print(f"Written /tmp/batch30_submission.txt", file=sys.stderr)

#!/usr/bin/env python3
"""
Batch36: ALL cyclotomic seeds — the only strategy that gives non-baseline T-numbers.

CRITICAL FINDING from batch30:
- polsubcyclo(m,8)×S3 → 24T27/30/32 (LMFDB baseline) → mostly 0 points
- cyclotomic seed × shift → 24T14744/14745/10258 (non-baseline) → 1.0 point each
- Efficiency: cyclotomic 44x better than polsubcyclo×S3

Target underserved r-buckets NOT yet covered in batches 30-35:
- r=4:  covered in batch30 (2 polys), need more
- r=6:  covered in batch30 (3 polys), need more
- r=10: NOT YET targeted
- r=14: NOT YET targeted
- r=18: only 2 polys so far

Strategy: for each m, hit 3-4 different r-buckets using cyclo_target_r.
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

def cyclo_target_r(seed_pv, target_r, tag):
    """Find c giving exactly target_r real roots."""
    out = gp(
        "h=" + seed_pv + ";"
        "rts=polroots(h);rr=List();"
        "for(i=1,#rts,if(abs(imag(rts[i]))<10^(-5),listput(rr,real(rts[i]))));"
        "rr=vecsort(Vec(rr));if(#rr!=12,quit);print(\"ROOTS|\",rr);"
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
    hi = roots[12-k] if k < 12 else roots[11] + 100
    c_mid = (lo + hi) / 2

    for c_try in [int(c_mid), int(c_mid)+1, int(c_mid)-1, round(c_mid), int(c_mid)+2, int(c_mid)-2]:
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
                    print(f"  {tag} c={c_try} r={r_val} total={len(polys)}", file=sys.stderr)
                    return 1
    return 0

def cyclo_shift_exact(seed_pv, c, target_r, tag):
    """Direct shift with c, accept only if r == target_r (not r==22)."""
    out = gp(
        "h=" + seed_pv + ";"
        "g24=subst(h,x,x^2+(" + str(c) + "));"
        "if(poldegree(g24)!=24,quit);if(!polisirreducible(g24),quit);"
        "r=polsturm(g24,-10^9,10^9);"
        "if(r==22,quit);"  # skip
        "print(\"OK|\",r,\"|\",Vecrev(g24));"
    )
    if not out.startswith("OK|"): return 0
    parts = out.split("|", 2)
    try: r_val = int(parts[1])
    except: return 0
    if target_r is not None and r_val != target_r: return 0
    coeffs = parse_vec(parts[2])
    if coeffs and len(coeffs) == 25 and coeffs[-1] == '1' and coeffs[0] != '0':
        if add(f"{tag}_c{c}_r{r_val}", coeffs):
            print(f"  {tag} c={c} r={r_val} total={len(polys)}", file=sys.stderr)
            return 1
    return 0

# All m values with totally-real degree-12 subfields (tried and tested)
M_VALS = [
    35, 65, 91, 156, 260, 390, 420, 546, 630,
    780, 840, 910, 924, 1092, 1155, 1260, 1365,
    1540, 1848, 2340, 2520, 3003, 3360, 3465,
    3780, 4095, 4290, 4620, 4641, 5040, 5460,
    5544, 6006, 6720, 7140, 7560, 8008, 8190, 8580,
    9009, 9240, 10010, 10395, 11220, 12012, 13860,
    14280, 15015, 15120, 16380, 17160,
]

# Target r values and their allocations
TARGETS = [
    (4,  12, "r04"),   # r=4
    (6,  12, "r06"),   # r=6
    (10, 12, "r10"),   # r=10 — NOT YET TARGETED
    (14, 12, "r14"),   # r=14 — NOT YET TARGETED
    (18, 12, "r18"),   # r=18 — only 2 polys so far
    (24, 10, "r24"),   # r=24
    (0,  10, "r00"),   # r=0
    (20, 10, "r20"),   # r=20 (fill)
    (8,  10, "r08"),   # r=8 (fill)
]

# Pre-compute seeds for m values
seeds = {}
print("=== Pre-computing seeds ===", file=sys.stderr)
for m in M_VALS:
    s = get_totally_real_deg12(m)
    if s:
        seeds[m] = s
print(f"  Seeds: {len(seeds)} m values", file=sys.stderr)

print("\n=== Targeting all r buckets ===", file=sys.stderr)
for target_r, alloc, rtag in TARGETS:
    print(f"\n--- target r={target_r} (alloc={alloc}) ---", file=sys.stderr)
    added = 0
    for m in M_VALS:
        if added >= alloc: break
        if m not in seeds: continue
        seed_pv = seeds[m]
        if target_r == 0:
            result = cyclo_shift_exact(seed_pv, 3, 0, f"{rtag}_m{m}")
        elif target_r == 24:
            result = 0
            for c in [-5,-6,-7,-8,-10,-12,-15,-20,-25,-30]:
                result = cyclo_shift_exact(seed_pv, c, 24, f"{rtag}_m{m}")
                if result: break
        else:
            result = cyclo_target_r(seed_pv, target_r, f"{rtag}_m{m}")
        added += result
    print(f"  Got {added} for r={target_r}", file=sys.stderr)

print(f"\n=== After targeted: {len(polys)} polys ===", file=sys.stderr)

# Fill if short
if len(polys) < 100:
    print("\n=== Fill: r=12 ===", file=sys.stderr)
    for m in M_VALS:
        if len(polys) >= 100: break
        if m not in seeds: continue
        cyclo_target_r(seeds[m], 12, f"fill12_m{m}")

print(f"\n=== FINAL: {len(polys)} polys ===", file=sys.stderr)
with open('/tmp/batch36_polys.json', 'w') as f:
    json.dump(polys, f, indent=2)
with open('/tmp/batch36_submission.txt', 'w') as f:
    for p in polys:
        f.write(','.join(p['coeffs']) + '\n')

rc = Counter()
for p in polys:
    for part in p['label'].split('_'):
        if part.startswith('r') and len(part) == 3:
            try: rc[int(part[1:])] += 1
            except: pass
        elif part.startswith('r') and not part.startswith('r0'):
            try: rc[int(part[1:])] += 1
            except: pass
# Count by actual r in label
rc2 = Counter()
for p in polys:
    parts = p['label'].split('_')
    for pt in parts:
        if pt.startswith('r') and pt[1:].isdigit():
            rc2[int(pt[1:])] += 1
            break
print(f"r-dist: {dict(sorted(rc2.items()))}", file=sys.stderr)
print(f"Written /tmp/batch36_submission.txt ({len(polys)} polys)", file=sys.stderr)

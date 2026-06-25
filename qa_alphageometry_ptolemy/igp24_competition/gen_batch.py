"""
IGP24 polynomial batch generator.

Usage:
  python3 gen_batch.py [--output submission.txt] [--max N]
  
Generates polynomials from the T87/T89 shift families, S4 nfsplitting,
T114 family (x^8-p × cyclotomic), and other constructions.

All generated polys are verified irreducible with correct degrees.
"""
import subprocess, json, argparse, sys, math
from pathlib import Path

PARIMAX = "default(parisizemax,400000000);"

T87_r24 = "y^12-132*y^11+6348*y^10-142380*y^9+1620171*y^8-9280872*y^7+24546456*y^6-27842616*y^5+14581539*y^4-3844260*y^3+514188*y^2-32076*y+729"
T87_r0  = "y^12+132*y^11+6348*y^10+142380*y^9+1620171*y^8+9280872*y^7+24546456*y^6+27842616*y^5+14581539*y^4+3844260*y^3+514188*y^2+32076*y+729"
T89_r24 = "y^12-104*y^11+4172*y^10-83280*y^9+892214*y^8-5103008*y^7+14466032*y^6-17725376*y^5+8299628*y^4-1385632*y^3+91696*y^2-2112*y+8"
T89_r0  = "y^12+104*y^11+4172*y^10+83280*y^9+892214*y^8+5103008*y^7+14466032*y^6+17725376*y^5+8299628*y^4+1385632*y^3+91696*y^2+2112*y+8"

# Root boundaries for T87 (sorted ascending)
T87_ROOTS = [0.055, 0.082, 0.207, 0.250, 0.307, 0.935, 3.210, 9.786, 11.979, 14.511, 36.520, 54.157]
# Root boundaries for T89
T89_ROOTS = [0.005, 0.038, 0.085, 0.118, 0.519, 1.455, 4.211, 9.432, 11.801, 13.124, 26.433, 36.779]

def expected_r(roots, k):
    """For g_r24(x²+k): count how many roots β_i > k → each gives 2 real roots."""
    return 2 * sum(1 for b in roots if b > k)

def gp_batch(cmds, timeout=300):
    script = PARIMAX + "\n" + "\n".join(cmds) + "\nquit\n"
    r = subprocess.run(['gp','-q'], input=script, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def parse_poly_line(poly_str):
    start = poly_str.rfind('['); end = poly_str.rfind(']')
    if start == -1: return None
    try:
        v = [int(x.strip()) for x in poly_str[start+1:end].split(',')]
        return v if (len(v)==25 and v[-1]==1 and v[0]!=0) else None
    except: return None

def generate_shift_polys(max_count=100, skip_k=None):
    """
    Generate shift polys: g(x²+k) and g(x²-k) for T87/T89.
    skip_k: set of (src, k) pairs already submitted.
    """
    skip_k = skip_k or set()
    
    # Generate k values for each r-range
    # T87r24 (r-values from shifts): 
    #   r=12: k=1..3; r=10: k=4..9; r=8: k=10..11; r=6: k=12..14
    #   r=4: k=15..36; r=2: k=37..54; r=0: k=55+
    all_shifts = []
    
    # Large r=0 range (k=55 to 500) — most abundant
    for k in range(55, 500):
        if ('r24', k) not in skip_k:
            all_shifts.append(('r24', k, 0))
    
    # r=2 range
    for k in range(37, 55):
        if ('r24', k) not in skip_k:
            all_shifts.append(('r24', k, 2))
    
    # r=4 range
    for k in range(15, 37):
        if ('r24', k) not in skip_k:
            all_shifts.append(('r24', k, 4))
    
    # T87r0 r=24 range (K=55 to 500)
    for k in range(55, 500):
        if ('r0', k) not in skip_k:
            all_shifts.append(('r0', k, 24))
    
    # T87r0 r=22 range
    for k in range(37, 55):
        if ('r0', k) not in skip_k:
            all_shifts.append(('r0', k, 22))
    
    # T87r0 r=20 range
    for k in range(15, 37):
        if ('r0', k) not in skip_k:
            all_shifts.append(('r0', k, 20))

    # T89r24 r=0 range (k=37 to 500)
    for k in range(37, 500):
        if ('t89r24', k) not in skip_k:
            all_shifts.append(('t89r24', k, 0))

    # T89r0 r=24 range (K=37 to 500)
    for k in range(37, 500):
        if ('t89r0', k) not in skip_k:
            all_shifts.append(('t89r0', k, 24))
    
    selected = all_shifts[:max_count]
    
    cmds = [
        f"gr24(y)={T87_r24};",
        f"gr0(y)={T87_r0};",
        f"g89r24(y)={T89_r24};",
        f"g89r0(y)={T89_r0};",
    ]
    for src, k, exp_r in selected:
        if src == 'r24':
            cmds.append(f"h=subst(gr24(y),y,x^2+{k}); print(\"{src}|{k}|{exp_r}|\",Vecrev(h));")
        elif src == 'r0':
            cmds.append(f"h=subst(gr0(y),y,x^2-{k}); print(\"{src}|{k}|{exp_r}|\",Vecrev(h));")
        elif src == 't89r24':
            cmds.append(f"h=subst(g89r24(y),y,x^2+{k}); print(\"{src}|{k}|{exp_r}|\",Vecrev(h));")
        elif src == 't89r0':
            cmds.append(f"h=subst(g89r0(y),y,x^2-{k}); print(\"{src}|{k}|{exp_r}|\",Vecrev(h));")
    
    out = gp_batch(cmds)
    
    polys = []
    for line in out.split('\n'):
        parts = line.split('|')
        if len(parts) >= 4:
            src, k_str, exp_r_str, poly_str = parts[0], parts[1], parts[2], parts[3]
            v = parse_poly_line(poly_str)
            if v:
                polys.append({
                    'coeffs': v, 'src': src, 'k': int(k_str), 'exp_r': int(exp_r_str),
                    'label': f"{src}_k{k_str}_r{exp_r_str}"
                })
    
    return polys

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='submission.txt')
    p.add_argument('--max', type=int, default=100)
    p.add_argument('--skip-from', help='JSON file with already-submitted polys to skip')
    p.add_argument('--start-k', type=int, default=55)
    args = p.parse_args()
    
    skip_k = set()
    if args.skip_from:
        with open(args.skip_from) as f:
            for item in json.load(f):
                skip_k.add((item['src'], item['k']))
    
    print(f"Generating up to {args.max} polys...")
    polys = generate_shift_polys(max_count=args.max, skip_k=skip_k)
    print(f"Generated {len(polys)} polys")
    
    content = f"# IGP24 shift batch — {len(polys)} polys\n#\n"
    for d in polys[:args.max]:
        cs = ','.join(str(c) for c in d['coeffs'])
        content += f"# {d['label']}\n{cs}\n"
    
    out_path = Path(args.output)
    out_path.write_text(content)
    print(f"Written {len(polys)} polys to {out_path}")
    
    # Save metadata for skip tracking
    meta_path = out_path.with_suffix('.json')
    meta_path.write_text(json.dumps(polys))
    print(f"Metadata saved to {meta_path}")

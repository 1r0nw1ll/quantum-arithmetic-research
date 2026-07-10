# Does QA's orbit->E8 correspondence IS the icosian/E8 grounding, or ad-hoc?
# Standalone analysis (touches nothing in production). SageMath 10.7.
#
# Two "E8 from QA" constructions in the repo:
#   (I)  icosian ring E8 (qa_icosian_order.py): the definite quaternion order over
#        Q(sqrt5); 120 units = binary icosahedral 2I = 600-cell; ring ~ E8. A prime-5
#        / sqrt5 object (rigorously grounded in Voight GTM 288 / Conway-Sloane SPLAG).
#   (II) orbit->E8-Type2 map (qa_e8_orbit_packet.py): 7 consecutive Cosmos-orbit
#        b-values -> parity bits u_k = b_k mod 2 -> Type-2 E8 root 1/2(-1)^{u_k}.
#
# Question: are these the SAME structure?
#   COMPUTED here: the orbit->E8-Type2 map reads only b mod 2 (Fibonacci mod 2, parity
#     period | 3) and reaches only 4 of the 128 Type-2 roots; the icosian units carry
#     golden (phi/sqrt5) coordinates.
#   CITED: the icosian ring is E8 over Q(sqrt5) (qa_icosian_order 7/7; Voight GTM 288;
#     Conway-Sloane SPLAG); the production qa_core.e8_alignment uses no E8 (scoping).
#   INTERPRETATION (not proved here): the two are the (unique) E8 lattice addressed at
#     different primes -- the orbit map a prime-2 (Fibonacci-mod-2) slice, the icosian
#     ring the prime-5 golden-order full E8, which is the fundamental grounding.

from itertools import combinations
M = 24

def qa_step(b, e, m=M): return e, ((b + e - 1) % m) + 1
def orbit(b0, e0, m=M):
    pts, (b, e) = [], (b0, e0)
    while True:
        pts.append((b, e)); b, e = qa_step(b, e, m)
        if (b, e) == pts[0]: return pts

def cosmos_orbits(m=M):
    seen = set(); orbs = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen: continue
            o = orbit(b, e, m); seen.update(o)
            if len(o) == m: orbs.append(o)
    return orbs

def packet_type2(orb, start):
    n = len(orb); bs = [orb[(start + k) % n][0] for k in range(7)]
    u = [b % 2 for b in bs]; u.append(sum(u) % 2)
    return tuple(-1 if uk else 1 for uk in u)          # sign pattern (x2 = Type-2 root)

# ---------------------------------------------------------------------------
print("="*70)
print("QA orbit->E8 correspondence vs the icosian/E8 grounding")
print("="*70)

# (1) COMPUTED (this file): the orbit->E8 map reads only b mod 2 = Fibonacci mod 2
orbs = cosmos_orbits(M)
b0 = [p[0] for p in orbs[0]]
def parity_period(orb):
    s = [p[0] % 2 for p in orb]
    return next(p for p in range(1, len(s)+1) if all(s[i] == s[i % p] for i in range(len(s))))
from collections import Counter
per = Counter(parity_period(o) for o in orbs)
print(f"\n[1] COMPUTED: {len(orbs)} Cosmos orbits at m={M}; orbit b-values are Fibonacci mod {M}")
print(f"    ({b0[:10]}...). b mod 2 is Fibonacci mod 2, so parity period DIVIDES 3:")
print(f"    period distribution {dict(per)} (nonconstant orbits period 3 = Pisano pi(2)=3;")
print(f"    {per.get(1,0)} constant-parity). The map reads only b mod 2 -> a prime-2 structure.")
addressed = set()
for orb in orbs:
    for s in range(len(orb)): addressed.add(packet_type2(orb, s))
print(f"    all Cosmos orbits together address only {len(addressed)} distinct Type-2 sign")
print(f"    patterns of the 128 Type-2 E8 roots -- a thin prime-2 (Fibonacci-mod-2) slice.")

# (2) the icosian unit coordinates are golden (sqrt5) data
K.<w> = NumberField(x^2 - x - 1)
phi = w
sample = vector([0, 1/2, (phi-1)/2, phi/2])         # a 96-type icosian unit
nrd = sum(c*c for c in sample)
print(f"\n[2] COMPUTED: a 96-type icosian unit (0, 1/2, (phi-1)/2, phi/2) has reduced norm")
print(f"    {nrd} = 1, with coordinates in phi = (1+sqrt5)/2 -- golden / prime-5 data, not")
print(f"    rational. CITED (qa_icosian_order.py, 7/7 + Voight GTM 288 / Conway-Sloane SPLAG")
print(f"    Ch.8): these 120 units are the binary icosahedral group 2I = the 600-cell, and")
print(f"    the icosian ring is (isometric to) E8 -- the genuine full sqrt5-grounded E8.")

# (3) verdict (computed facts + their interpretation, kept separate)
print("\n" + "="*70)
print("VERDICT.  COMPUTED here: the orbit->E8-Type2 map reads only b mod 2 (Fibonacci mod")
print(f"2, parity period | 3) and reaches only {len(addressed)} of the 128 Type-2 E8 roots -- a thin")
print("slice, not a rich E8 structure. The icosian units are golden (phi/sqrt5) data.")
print("CITED (qa_icosian_order 7/7 + Voight/SPLAG; scoping of qa_core.py): the icosian ring")
print("is the genuine full E8 over Q(sqrt5); the production qa_core.e8_alignment does not use")
print("E8 at all (single hardcoded Fibonacci vector [1,1,2,3,0,0,0,0], not a root).")
print("INTERPRETATION: the two repo 'E8' constructions address the (unique) E8 lattice at")
print("different primes -- the orbit map is a prime-2 (Fibonacci-mod-2) slice, the icosian")
print("ring the prime-5 (golden-order) full E8. Neither existing 'E8 alignment' is a rich,")
print("grounded E8 structure; the icosian sqrt5-E8 is. So a properly grounded E8 alignment")
print("(task 2) should align QA tuples to the ICOSIAN E8, not the prime-2 parity slice or")
print("the single Fibonacci vector.")

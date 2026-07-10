# From-scratch Brandt matrices reproducing a NON-CM Hilbert newform over Q(sqrt5).
# Target: the NON-CM Hecke eigenvalue system of LMFDB 2.2.5.1-31.x-a (level norm 31,
# weight [2,2], NON-CM, dim 1, rational eigenvalues; the two forms 31.1-a/31.2-a over
# the two primes above 31 share this system). Level = a prime p31 over 31, which is
# SPLIT UNRAMIFIED -> O_K/p31 = F_31 (a field), so the splitting is direct (no Newton
# lift; simpler than the p5^3 CM case). Run: sage qa_brandt_level31_compute.sage.
#
# This is the strong test: CM eigenvalues are structured/"easy"; reproducing a
# NON-CM form's genuinely irregular Hecke eigenvalues from Brandt matrices is the
# real validation of the quaternion realization.

K.<w> = NumberField(x^2 - x - 1)                 # w = phi
nf = pari('nfinit(y^2 - y - 1)'); A = pari.alginit(nf, [-1, -1]); n = 8
def col(v): return pari([int(c) for c in v]).Col()
def nrdK(v):
    z = A.algnorm(col(v), 0).lift(); return QQ(z.polcoef(0)) + QQ(z.polcoef(1))*w
def qmulZ(u, v): return tuple(int(x) for x in A.algmul(col(list(u)), col(list(v))))

# --- maximal order norm form (validate det 5^4, 120 units) ---
qs = [nrdK([1 if t==r else 0 for t in range(n)]) for r in range(n)]
G = matrix(ZZ, n, n)
for a in range(n):
    for b in range(n):
        s = [0]*n; s[a]+=1; s[b]+=1
        G[a,b] = QQ((nrdK(s) - qs[a] - qs[b]).trace())
assert G.det() == 5^4 and pari(G).qfminim(4, 200)[0] == 120
print("[1] maximal order over Z[phi]: det 5^4, 120 units = 2I  [ok]")

# --- O_K-basis + structure constants over O_K ---
wvec = vector(ZZ, [0,0,1,0,0,0,0,0])
def zmul(u, v): return vector(ZZ, [ZZ(x) for x in A.algmul(col(list(u)), col(list(v)))])
Phi = matrix(ZZ, [zmul(wvec, [1 if t==i else 0 for t in range(n)]) for i in range(n)]).transpose()
cur = []; fs = []
for i in range(n):
    ei = vector(ZZ, [1 if t==i else 0 for t in range(n)])
    cand = cur + [ei, Phi*ei]
    if matrix(ZZ, cand).rank() == len(cand): cur = cand; fs.append(ei)
    if len(fs) == 4: break
Bm = matrix(ZZ, [fs[a//2] if a%2==0 else Phi*fs[a//2] for a in range(8)]); Bi = Bm.inverse()
def to_ok(x):
    c = vector(QQ, x)*Bi; return [(ZZ(c[2*i]), ZZ(c[2*i+1])) for i in range(4)]
FS = [list(f) for f in fs]
COK = [[to_ok(zmul(FS[a], FS[b])) for b in range(4)] for a in range(4)]
IDok = to_ok(vector(ZZ,[1 if t==0 else 0 for t in range(n)]))
print("[2] O_K-basis + structure constants over Z[phi]  [ok]")

# --- R = O_K/p31 = F_31, one of the two primes over 31 (phi -> 19, i.e. sqrt5=6 mod 31;
# the conjugate prime has phi -> 13, sqrt5 = -6). The two primes give the Galois-conjugate
# forms 31.1-a / 31.2-a, which share the SAME Hecke eigenvalue SYSTEM (they differ only by
# which prime carries which of a split pair). We reproduce that shared non-CM system; we do
# NOT here pin the specific LMFDB ideal label 31.1 vs 31.2 (that needs ordered prime->a_q). ---
P = 31; F = GF(P); PHIRED = F(19)      # phi mod this prime over 31 (19^2-19-1 = 0 mod 31)
assert PHIRED^2 - PHIRED - 1 == 0
def red(ab): return F(ab[0]) + F(ab[1])*PHIRED             # O_K -> F_31
def dmul_F(u, v):                                          # multiply in O_max (x) F_31
    r = [F(0)]*4
    for a in range(4):
        for b in range(4):
            if u[a] == 0 or v[b] == 0: continue
            co = u[a]*v[b]
            for c in range(4):
                p, q = COK[a][b][c]; r[c] += co*red((p, q))
    return r
DID = [red(IDok[i]) for i in range(4)]
def x_to_DF(xz):
    ok = vector(QQ, xz)*Bi; return [red((ZZ(ok[2*i]), ZZ(ok[2*i+1]))) for i in range(4)]

# rank-1 idempotent in M_2(F_31): find splitting element (min poly splits), form e
gens = [[F(1) if t==i else F(0) for t in range(4)] for i in range(4)]
def dsub(u, v): return [u[i]-v[i] for i in range(4)]
def dscal(s, u): return [s*u[i] for i in range(4)]
e = None
import itertools
for co in itertools.product(range(P), repeat=4):
    if not any(co): continue
    x = [F(c) for c in co]
    x2 = dmul_F(x, x)
    # solve x2 = alpha x + beta*1
    Msys = matrix(F, [list(x), list(DID)]).transpose(); bb = vector(F, list(x2))
    try: sol = Msys.solve_right(bb)
    except Exception: continue
    if Msys*sol != bb: continue
    al, be = sol; disc = al*al + 4*be
    if not disc.is_square(): continue
    sq = disc.sqrt(); inv2 = F(2)^(-1); l1 = (al+sq)*inv2; l2 = (al-sq)*inv2
    if l1 == l2: continue
    ei = dscal((l1-l2)^(-1), dsub(x, dscal(l2, DID)))
    if dmul_F(ei, ei) == ei and ei != DID and any(ei): e = ei; break
ep = dsub(DID, e)
print(f"[3] splitting O_max (x) F_31 -> M_2(F_31): rank-1 idempotent found  [ok]")

# --- Eichler order O = {x : (1-e) x e = 0 in M_2(F_31)}, index 31 (= N(p31)) ---
rows = []
for i in range(8):
    y = dmul_F(dmul_F(ep, x_to_DF([1 if t==i else 0 for t in range(8)])), e)
    rows.append([ZZ(y[k]) for k in range(4)])
Pm = matrix(F, rows).transpose()
ker = Pm.right_kernel().basis_matrix()                    # over F: kernel of x -> (1-e)xe
# lift to a Z-lattice: O = { integer x : (1-e)xbar e == 0 mod p31 }
Mmap = matrix(ZZ, [[ZZ(Pm[r, c]) for c in range(8)] for r in range(4)])
L31 = P                                                   # the congruence modulus structure
Big = block_matrix(ZZ, [[Mmap, -P*identity_matrix(ZZ, 4)]])
kz = Big.right_kernel().basis_matrix()[:, :8]
Olat = (ZZ^8).span(kz.rows()); BO = matrix(ZZ, Olat.basis())
idx = abs(BO.det())
det_expect = 5^4 * P^2                                    # disc(O) = disc(O_max)*N(p31)^2
assert idx == P and (BO*G*BO.transpose()).det() == det_expect
print(f"[4] Eichler order level p31: index [O_max:O] = {idx} (=31), Gram det = "
      f"{(BO*G*BO.transpose()).det()} (= 5^4*31^2 = {det_expect})  [ok]")

# --- 5. class number via O_max^1-orbits on P^1(F_31); build full rho ---
gg = [[F(1) if t==i else F(0) for t in range(4)] for i in range(4)]
E12 = next(c for c in (dmul_F(dmul_F(e, g), ep) for g in gg) if any(c))
kk = next(k for k in range(4) if e[k] != 0)
for hh in gg:
    E21c = dmul_F(dmul_F(ep, hh), e)
    if not any(E21c): continue
    cc = dmul_F(E12, E21c)[kk] / e[kk]
    if cc != 0: E21 = dscal(cc^(-1), E21c); break
def cof(y, U):
    k = next(k for k in range(4) if U[k] != 0); return y[k] / U[k]
def rho(xz):
    xd = x_to_DF(xz)
    return ((cof(dmul_F(dmul_F(e, xd), e), e),  cof(dmul_F(dmul_F(e, xd), ep), E12)),
            (cof(dmul_F(dmul_F(ep, xd), e), E21), cof(dmul_F(dmul_F(ep, xd), ep), ep)))
assert rho([1 if t==0 else 0 for t in range(8)]) == ((F(1), F(0)), (F(0), F(1)))
Vmin = pari(G).qfminim(4, 200)[2]
units = [tuple(int(Vmin[r, c]) for r in range(8)) for c in range(Vmin.ncols())]
units += [tuple(-x for x in u) for u in units]
# P^1(F_31): [1:t] for t in F, plus [0:1]
P1 = [(F(1), F(t)) for t in range(P)] + [(F(0), F(1))]
def norml(a, b): return (F(1), b/a) if a != 0 else (F(0), F(1))
def act(M, line):
    a, b = line
    return norml(M[0][0]*a + M[0][1]*b, M[1][0]*a + M[1][1]*b)
rhos = [rho(list(u)) for u in units]
idx = {l: i for i, l in enumerate(P1)}; par = list(range(len(P1)))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i, l in enumerate(P1):
    for Mr in rhos:
        j = idx[act(Mr, l)]; ri, rj = find(i), find(j)
        if ri != rj: par[ri] = rj
orbit_of = {P1[i]: find(i) for i in range(len(P1))}
orbmap = {}
for i in range(len(P1)): orbmap.setdefault(find(i), []).append(i)
orbids = sorted(orbmap); sizes = [len(orbmap[o]) for o in orbids]
weights = [ZZ(60 // s) if 60 % s == 0 else QQ(60)/s for s in sizes]
mass = sum(QQ(1)/wt for wt in weights)
assert len(orbids) == 2 and sorted(weights) == [3, 5] and mass == QQ(8)/15
print(f"[5] class number h = {len(orbids)} (=2)  orbit sizes {sizes}  weights {weights} "
      f"(={{3,5}})  mass {mass} (= 8/15)  [ok]")

# --- 6. Brandt matrices T(q) for good primes; compare to LMFDB 2.2.5.1-31.1-a ---
reps = {}
for i in range(len(P1)):
    o = find(i)
    if o not in reps: reps[o] = P1[i]
repline = [reps[o] for o in orbids]
def totpos_gen(p):
    if p % 5 in (2, 3): return p*K(1)
    for a in range(-15, 16):
        for b in range(-15, 16):
            g = a + b*w
            if g.norm() == p and g > 0 and (QQ(a) + QQ(b)*((1-sqrt(5))/2)) > 0: return g
    return None
def brandt(beta):
    V = pari(G).qfminim(2*beta.trace(), 40000, flag=2)[2]
    elts = []
    for c in range(V.ncols()):
        v = [ZZ(V[r, c]) for r in range(8)]
        if nrdK(v) == beta: elts += [tuple(v), tuple(-x for x in v)]
    used = set(); ra = []
    for al in elts:
        if al in used: continue
        ra.append(al)
        for u in units: used.add(qmulZ(u, al))
    h = len(orbids); T = matrix(ZZ, h, h)
    for ii in range(h):
        for al in ra:
            T[orbids.index(orbit_of[act(rho(list(al)), repline[ii])]), ii] += 1
    # verify #neighbours = N(q)+1 and that N(q)+1 is a genuine eigenvalue (Eisenstein):
    Nq = ZZ(beta.norm().abs())
    assert len(ra) == Nq + 1, (len(ra), Nq + 1)
    assert all(sum(T[r, c] for r in range(h)) == Nq + 1 for c in range(h))   # column sums
    assert (Nq + 1) in T.eigenvalues()                                       # Eisenstein eig
    return T, Nq
print("[6] Brandt matrices T(q) vs the NON-CM Hecke system of 2.2.5.1-31.x-a, good primes to norm 49:")
# a_q of the shared non-CM eigenvalue system (level prime 31 excluded). inert p -> single
# a_q; split p -> the split PAIR {a,a'} carried by the two primes over p (31.1/31.2 assign
# the pair oppositely, so we check unordered membership -- pinning the ideal label needs
# ordered prime->a_q, not done here).
INERT = {2: -3, 3: 2, 5: -2, 7: 2}                    # norms 4,9,5,49
SPLIT = {11: (4, -4), 19: (-4, 4), 29: (-2, -2), 41: (-6, -6)}
def cusp_eig(T, Nq):                                  # the non-Eisenstein (cusp) eigenvalue
    evs = list(T.eigenvalues()); evs.remove(Nq + 1); return evs[0]
allok = True
for p in sorted(INERT):
    T, Nq = brandt(totpos_gen(p)); ce = ZZ(cusp_eig(T, Nq)); tgt = INERT[p]
    ok = (ce == tgt); allok = allok and ok
    print(f"    inert p={p:2d} N={Nq:2d}: cusp eigenvalue {ce:+d}  a_q {tgt:+d}  {'MATCH' if ok else 'MISMATCH'}")
for p in sorted(SPLIT):
    T, Nq = brandt(totpos_gen(p)); ce = ZZ(cusp_eig(T, Nq))
    ok = (ce in SPLIT[p]); allok = allok and ok
    print(f"    split p={p:2d} N={Nq:2d}: cusp eigenvalue {ce:+d}  a_q in {SPLIT[p]}  {'MATCH' if ok else 'MISMATCH'}")
assert allok
print("\n[ok] ALL good primes to norm 49 reproduce the NON-CM eigenvalue system of")
print("2.2.5.1-31.x-a. The 2x2 Brandt matrices of the level-p31 Eichler order give its")
print("irregular non-CM Hecke eigenvalues -- the machinery is not special to the CM case.")

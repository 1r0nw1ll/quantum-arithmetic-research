# From-scratch computation of the 3x3 Brandt matrices of the level-125 Eichler order
# over Q(sqrt5), realizing the CM Hilbert newform LMFDB 2.2.5.1-125.1-a.
# Run: sage qa_brandt_level125_compute.sage   (SageMath 10.7; uses PARI alginit).
#
# Pipeline (all exact; each stage validated):
#   1. B=(-1,-1|Q(sqrt5)); PARI maximal order O_max; norm form det 5^4, 120 units = 2I.
#   2. O_K-basis + structure constants over O_K=Z[phi].
#   3. R = O_K/p5^3 (p5=(sqrt5), |R|=125); rank-1 idempotent via mod-p5 seed + Newton lift;
#      matrix units -> splitting rho: O_max (x) R  ->  M_2(R).
#   4. Eichler order O = {x : (1-e) x e = 0} ; validated reduced disc = 5^10 (level p5^3).
#   5. class number h = #(O_max^1-orbits on P^1(R)); weights from orbit sizes; mass.
#   6. Brandt matrix T(q) for good primes q: the 12,32,50 q-neighbour elements alpha
#      (nrd(alpha) a totally-positive generator of q) act by rho(alpha) on P^1; the
#      orbit each rep lands in gives the matrix. Char polys checked vs LMFDB.

K.<w> = NumberField(x^2 - x - 1)                 # w = phi, O_K = Z[phi]
nf = pari('nfinit(y^2 - y - 1)'); A = pari.alginit(nf, [-1, -1]); n = 8
def col(v): return pari([int(c) for c in v]).Col()
def nrdK(v):
    z = A.algnorm(col(v), 0).lift(); return QQ(z.polcoef(0)) + QQ(z.polcoef(1))*w
def qmulZ(u, v): return tuple(int(x) for x in A.algmul(col(list(u)), col(list(v))))

# --- 1. maximal-order norm form ---
qs = [nrdK([1 if t==r else 0 for t in range(n)]) for r in range(n)]
G = matrix(ZZ, n, n)
for a in range(n):
    for b in range(n):
        s = [0]*n; s[a]+=1; s[b]+=1
        G[a,b] = QQ((nrdK(s) - qs[a] - qs[b]).trace())
assert G.det() == 5^4 and pari(G).qfminim(4, 200)[0] == 120
print("[1] B totally definite; maximal order norm form det 5^4, 120 units = 2I  [ok]")

# --- 2. O_K-basis {f_i} with {f_i, phi f_i} a Z-basis; structure constants over O_K ---
wvec = vector(ZZ, [0,0,1,0,0,0,0,0])          # phi = basis element 2 (verified w^2=w+1)
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
print("[2] O_K-basis (unimodular), structure constants over O_K=Z[phi]  [ok]")

# --- 3. R = O_K/p5^3 and the splitting rho ---
L = matrix(ZZ, [[-5,10],[10,5]]); H = L.hermite_form(); Hi = H.inverse()
def rr(ab):
    v = vector(QQ, ab); k = (v*Hi).apply_map(lambda t: t.round()); return tuple(ZZ(x) for x in (v-k*H))
def radd(x,y): return rr((x[0]+y[0], x[1]+y[1]))
def rsub(x,y): return rr((x[0]-y[0], x[1]-y[1]))
def rmul(x,y):
    a,b=x; c,e=y; return rr((a*c+b*e, a*e+b*c+b*e))
R0=(0,0); R1=(1,0)
Rset = sorted(set(rr((a,b)) for a in range(30) for b in range(30))); assert len(Rset)==125
def is_unit(r): return (r[0]+3*r[1])%5 != 0
INV = {}
for r in Rset:
    if is_unit(r):
        INV[r] = next(s for s in Rset if rmul(r,s)==R1)
def rinv(r): return INV[r]
def dmul(u,v):
    r=[R0,R0,R0,R0]
    for a in range(4):
        for b in range(4):
            if u[a]==R0 or v[b]==R0: continue
            co=rmul(u[a],v[b])
            for c in range(4):
                p,q=COK[a][b][c]; r[c]=radd(r[c],rmul(co,(p,q)))
    return r
def dsub(u,v): return [rsub(u[i],v[i]) for i in range(4)]
def dscal(s,u): return [rr((s*u[i][0], s*u[i][1])) for i in range(4)]
DID = [IDok[i] for i in range(4)]
e = [(3,0),(0,0),(3,0),(3,0)]                    # idempotent mod p5 (F5), then Newton-lift
for _ in range(4):
    e2 = dmul(e,e); e = dsub(dscal(3,e2), dscal(2,dmul(e2,e)))
assert dmul(e,e) == e and e != DID and e != [R0]*4
ep = dsub(DID, e)
gens = [[R1 if t==i else R0 for t in range(4)] for i in range(4)]
E12 = next(c for c in (dmul(dmul(e,g),ep) for g in gens) if c != [R0]*4)
kk = next(k for k in range(4) if is_unit(e[k]))
for hh in gens:
    E21c = dmul(dmul(ep,hh),e)
    if E21c == [R0]*4: continue
    cc = rmul(dmul(E12,E21c)[kk], rinv(e[kk]))
    if is_unit(cc): E21 = [rmul(rinv(cc),E21c[i]) for i in range(4)]; break
assert dmul(E12,E21)==e and dmul(E21,E12)==ep
def x_to_D(xz):
    ok = vector(QQ, xz)*Bi; return [rr((ZZ(ok[2*i]), ZZ(ok[2*i+1]))) for i in range(4)]
def cof(y,U):
    k = next(k for k in range(4) if is_unit(U[k])); return rmul(y[k], rinv(U[k]))
def rho(xz):
    xd = x_to_D(xz)
    return ((cof(dmul(dmul(e,xd),e),e),  cof(dmul(dmul(e,xd),ep),E12)),
            (cof(dmul(dmul(ep,xd),e),E21), cof(dmul(dmul(ep,xd),ep),ep)))
assert rho([1 if t==0 else 0 for t in range(8)]) == ((R1,R0),(R0,R1))
print("[3] splitting rho: O_max (x) O_K/p5^3 -> M_2(R) built and validated  [ok]")

# --- 4. Eichler order O = { x : (1-e) x e = 0 }, validate reduced disc 5^10 ---
imgs = []
for i in range(8):
    y = dmul(dmul(ep, x_to_D([1 if t==i else 0 for t in range(8)])), e)
    imgs.append(vector(ZZ, [y[k][j] for k in range(4) for j in range(2)]))
Pm = matrix(ZZ, imgs).transpose(); L4 = block_diagonal_matrix([H]*4)
ker = block_matrix(ZZ, [[Pm, -L4.transpose()]]).right_kernel().basis_matrix()[:, :8]
Olat = (ZZ^8).span(ker.rows()); BO = matrix(ZZ, Olat.basis())
assert abs(BO.det()) == 125 and (BO*G*BO.transpose()).det() == 5^10
print("[4] Eichler order level p5^3: index 125, reduced-norm form det 5^10  [ok]")

# --- 5. class number = O_max^1-orbits on P^1(R); weights; mass ---
# the 120 norm-1 units are the minimal vectors of the maximal-order norm form
Vmin = pari(G).qfminim(4, 200)[2]
units = [tuple(int(Vmin[r,c]) for r in range(8)) for c in range(Vmin.ncols())]
units += [tuple(-x for x in u) for u in units]
def normline(na, nb):
    return (R1, rmul(nb, rinv(na))) if is_unit(na) else (rmul(na, rinv(nb)), R1)
def act(P, line):
    a,b = line
    na = radd(rmul(P[0][0],a), rmul(P[0][1],b)); nb = radd(rmul(P[1][0],a), rmul(P[1][1],b))
    return normline(na, nb)
P1 = []; seen = set()
for a in Rset:
    for b in Rset:
        if not (is_unit(a) or is_unit(b)): continue
        rep = (R1, rmul(b, rinv(a))) if is_unit(a) else (rmul(a, rinv(b)), R1)
        if rep not in seen: seen.add(rep); P1.append(rep)
assert len(P1) == 150
rhos = [rho(list(u)) for u in units]
idxmap = {l:i for i,l in enumerate(P1)}; par = list(range(150))
def find(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for i,l in enumerate(P1):
    for P in rhos:
        j = idxmap[act(P,l)]; ri,rj = find(i),find(j)
        if ri!=rj: par[ri]=rj
orbit_of = {P1[i]: find(i) for i in range(150)}
orbmap = {}
for i in range(150): orbmap.setdefault(find(i), []).append(i)
orbids = sorted(orbmap); sizes = [len(orbmap[o]) for o in orbids]
# weight w_i = |Gamma|/|orbit_i| with Gamma = 120 units mod +-1 = 60 acting on P^1
weights = [ZZ(60 // sz) if (60 % sz == 0) else QQ(60)/sz for sz in sizes]
mass = sum(QQ(1)/wt for wt in weights)
print(f"[5] class number h = {len(orbids)}  orbit sizes {sizes}  weights {weights}  mass {mass}")
assert len(orbids) == 3 and mass == QQ(5)/2

# --- 6. Brandt matrices T(q) via q-neighbours ---
reps = {}
for i in range(150):
    o = find(i)
    if o not in reps: reps[o] = P1[i]
repline = [reps[o] for o in orbids]
def brandt(beta, label, expect):
    V = pari(G).qfminim(2*beta.trace(), 60000, flag=2)[2]
    elts = []
    for c in range(V.ncols()):
        v = [ZZ(V[r,c]) for r in range(8)]
        if nrdK(v) == beta: elts += [tuple(v), tuple(-x for x in v)]
    used = set(); ra = []
    for al in elts:
        if al in used: continue
        ra.append(al)
        for u in units: used.add(qmulZ(u, al))
    h = len(orbids); T = matrix(ZZ, h, h)
    for ii in range(h):
        for al in ra:
            oj = orbit_of[act(rho(list(al)), repline[ii])]
            T[orbids.index(oj), ii] += 1
    cp = T.charpoly().factor()
    Nq = ZZ(beta.norm().abs())
    print(f"[6] T({label})  N(q)={Nq}  #neighbours={len(ra)} (=N(q)+1)  char poly = {cp}")
    print(T)
    assert all(sum(T[r,c] for r in range(h)) == Nq+1 for c in range(h))
    assert str(cp) == expect, (str(cp), expect)
    return T
brandt(3+w,   "p11", "(x - 12) * (x^2 + x - 31)")
brandt(5+2*w, "p31", "(x - 32) * (x^2 + 11*x - 1)")
brandt(7*K(1),"p7-inert(N=49)", "(x - 50) * x^2")

# --- 7. full Hecke-system check: every good prime up to norm 100 vs LMFDB ---
# LMFDB 2.2.5.1-125.1-a cusp-factor targets (S,P) with cusp factor x^2 - S x + P:
TARGETS = {2:(0,0), 3:(0,0), 11:(-1,-31), 19:(0,0), 29:(0,0), 31:(-11,-1),
           41:(9,-11), 7:(0,0), 59:(0,0), 61:(-1,-31), 71:(19,59), 79:(0,0), 89:(0,0)}
def totpos_gen(p):
    if p % 5 in (2,3): return p*K(1)                     # inert: nrd = p, N(q)=p^2
    for a in range(-15,16):
        for b in range(-15,16):
            g = a+b*w
            if g.norm()==p and g>0 and (QQ(a)+QQ(b)*((1-sqrt(5))/2))>0: return g
    for a in range(-15,16):
        for b in range(-15,16):
            g = a+b*w
            if abs(g.norm())==p:
                for u in [1,w,w^2,-w,-w^2,w^3,w^4]:
                    h = g*u
                    if h.norm()==p and h>0 and (QQ(h[0])+QQ(h[1])*((1-sqrt(5))/2))>0: return h
    return None
def cusp_of(T, Nq): return (ZZ(T.trace()-(Nq+1)), ZZ(T.det()/(Nq+1)))
print("\n[7] full Hecke system: Brandt cusp factor vs LMFDB, every good prime to norm 100")
allok = True
for p in sorted(TARGETS):
    beta = totpos_gen(p); Nq = ZZ(beta.norm().abs())
    V = pari(G).qfminim(2*beta.trace(), 200000, flag=2)[2]
    elts = []
    for c in range(V.ncols()):
        v = [ZZ(V[r,c]) for r in range(8)]
        if nrdK(v) == beta: elts += [tuple(v), tuple(-x for x in v)]
    used = set(); ra = []
    for al in elts:
        if al in used: continue
        ra.append(al)
        for u in units: used.add(qmulZ(u, al))
    T = matrix(ZZ, 3, 3)
    for ii in range(3):
        for al in ra:
            T[orbids.index(orbit_of[act(rho(list(al)), repline[ii])]), ii] += 1
    comp = cusp_of(T, Nq); tgt = TARGETS[p]; ok = (comp == tgt and len(ra) == Nq+1)
    allok = allok and ok
    print(f"    p={p:3d}  N(q)={Nq:4d}  cusp (S,P) computed {str(comp):>12s}  LMFDB {str(tgt):>12s}  {'OK' if ok else 'MISMATCH'}")
assert allok
print("\nALL 13 GOOD PRIMES TO NORM 100 REPRODUCE LMFDB 2.2.5.1-125.1-a (full Hecke system).")

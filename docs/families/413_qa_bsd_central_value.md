<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Birch-Swinnerton-Dyer (1965) doi:10.1515/crll.1965.218.79, Rohrlich (1984) doi:10.1007/BF01388742 -->
# [413] QA BSD Central Value — Euler Factor Trichotomy at s=½

**Cert family**: `qa_bsd_central_value_cert_v1`
**Claim**: At the analytic center s=½, the GL₄/ℚ AI Euler factors split into three arithmetic classes:

| Class | Condition | P_p(p^{−½}) | Type |
|---|---|---|---|
| Inert | p ≡ 2,3 mod 5 | **2** | Fraction — certifiable |
| Ramified | p = 5 | **1** | Integer — certifiable |
| Split | p ≡ 1,4 mod 5 | 4+N/p **− 2T/√p** | Irrational — observer projection |

## The Three Cases

### Inert primes (C1)

P_p^{inert}(Y) = 1 + p²Y⁴. At Y = p^{−½}:

```
Y⁴ = (p^{-1/2})⁴ = p^{-2} = Fraction(1, p²)
P_p(p^{-1/2}) = 1 + p² · Fraction(1, p²) = 1 + 1 = 2
```

Every inert prime contributes **exactly 2** — a Fraction, not a float.
The local Euler factor at the center is `(1/2)`.

### Ramified prime (C2)

P_5^{ram}(Y) = 1 (from cert [411]). At any Y: P_5^{ram} = **1**. Local factor = 1.

### Split primes (C3)

P_p(Y) = 1 − TY + (N+2p)Y² − pTY³ + p²Y⁴. At Y = p^{−½}:

```
P_p(p^{-1/2}) = Fraction(4p+N, p)   −   2T/√p
                ↑ rational part             ↑ irrational part
```

The rational part `Fraction(4p+N, p)` is certifiable. The coefficient **−2T** of `1/√p` is
non-zero for all 22 split primes (T≠0 verified), and `√p` is irrational for every prime p.
Therefore the full center value is irrational — a **continuous observer projection** under
Theorem NT. It cannot enter the QA discrete layer.

## Inert Partial Euler Product (C4)

The rational spine of L(½, AI(f)) from inert primes alone:

```
∏_{p inert, p≤193} P_p(p^{-1/2})^{-1} = (½)^22 = Fraction(1, 2^22) = 1/4194304
```

Exact Fraction arithmetic — no floating point.

## BSD Connection

The full central value decomposes as:

```
L(½, AI(f)) = L_rat  ×  L_split  ×  L_∞
               ↓          ↓           ↓
           (rational)  (irrational) (Gamma values)
           certified   observer     observer
```

**BSD conjecture** (Birch–Swinnerton-Dyer): ord_{s=½} L(s, AI(f)) = rank A_f(ℚ)

- If L(½) ≠ 0: BSD predicts **rank A_f(ℚ) = 0** (no rational points of infinite order)
- Rohrlich (1984): for CM forms over ℚ(ζ₅)-towers, L(½) ≠ 0 in generic twist families
- The integer prediction: **r_alg = 0** (the abelian surface A_f has only finitely many rational points)

The **irrational part L_split** is the only piece that can drive L(½) to zero — it is also
the only piece that is an observer projection. The rational spine L_rat = 1/4194304 is never
zero. Theorem NT: the zero/non-zero decision lives in the continuous layer.

## Theorem NT Boundary

| Object | Layer | Reason |
|---|---|---|
| P_p^{inert}(p^{-½}) = 2 | QA (Fraction) | p^{-2}·p² = 1, exact |
| P_5^{ram}(5^{-½}) = 1 | QA (int) | trivial polynomial |
| Rational part Fraction(4p+N,p) | QA (Fraction) | integer division |
| T ≠ 0 check | QA (int) | integer comparison |
| −2T/√p coefficient | Observer | √p is irrational |
| L(½, AI(f)) itself | Observer | infinite product |
| ε (root number) | Observer | Gauss sum |

## Checks

- **C1**: P_p^{inert}(p^{−½}) = Fraction(2) for 22/22 inert primes — PASS
- **C2**: P_5^{ram}(5^{−½}) = 1 — PASS
- **C3**: T≠0 for 22/22 split primes; rational part = Fraction(4p+N,p); irrational remainder confirmed — PASS
- **C4**: Inert partial product = Fraction(1, 2^22) = Fraction(1, 4194304) — PASS

## Chain

- Builds directly on [404] (split Euler poly), [409] (inert Euler poly), [411] (ramified = 1)
- The trichotomy at s=½ exactly mirrors [410] (Dedekind ζ_F split/inert/ramified)
- BSD rank prediction r=0 is the integer output; all analytic evidence is observer projection

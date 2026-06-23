# [261] QA Orbit Stratification Cert

## What this is

Machine-checkable form of the two-layer QA Orbit Stratification Theorem, proven 2026-04-20. Closes the gap between the ad-hoc orbit enumeration in `qa_orbital_dynamics.py` and a formal algebraic classification of every `(b,e)` pair in `(Z/mZ)²`.

**Part I — ⟨σ,μ⟩-orbit content-ideal classes**

For any modulus m and prime-power factor p^k | m, define the content-ideal invariant `J(b,e) := min(v_p(b), v_p(e))` where v_p is the p-adic valuation. The ⟨σ,μ⟩-orbit structure exactly equals the level sets:

```
L_j = { (b,e) ∈ {1..m}² : min(v_p(b), v_p(e)) = j }
```

Sizes: `|L_j| = p^{2(k-j-1)}·(p²-1)` for `j < k`; `|L_k| = 1` (the singularity at (p^k, p^k) when m=p^k).  
For composite m: `|orbit| = ∏ |L_{j_i}|` via CRT, where each j_i is the level at prime factor p_i.

**Part II — σ-only orbit refinement by Legendre symbol**

Within each level set L_0, σ-only orbits refine by how x²−x−1 factors mod p:

- **Case A** (inert, Legendre(5,p) = −1): every L_0 orbit has uniform length π(p^n)
- **Case B** (split, Legendre(5,p) = +1): φ-eigenspace orbits of length ord_p(φ), ψ-eigenspace orbits of length ord_p(ψ), plus generic orbits of length π(p)
- **Case C** (ramified, p = 5, n ≥ 2): exactly 5^{n-1} orbits of length π(5^n) and 5^{n-1} orbits of length π(5^{n-1})

**Bridge**: μ (the pair-swap operator (b,e)→(e,b)) is the collapse operator: it quotients Part II's Frobenius/Jordan refinement back to Part I's content-ideal classes.

**Primary sources**:
- Nielsen, J. (1924). Über die Isomorphismengruppe der freien Gruppen. *Math. Annalen* 91:169–209. Generation of GL₂(ℤ) by elementary matrices T and Tᵀ — grounds that σ and μ together generate the full orbit structure.
- Lang, S. (2002). *Algebra*, 3rd ed., Ch. III. Springer. Local-ring module classification via elementary divisors — grounds the content-ideal class decomposition.
- Wall, D.D. (1960). Fibonacci series modulo m. *American Math. Monthly* 67(6):525–532. Fibonacci Pisano periodicity π(p^k) — grounds Case A/B/C orbit length formulas.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_orbit_stratification_cert_v1/qa_orbit_stratification_cert_validate.py` |
| Pass fixture | `qa_orbit_stratification_cert_v1/fixtures/qos_pass_core.json` |
| Fail fixture | `qa_orbit_stratification_cert_v1/fixtures/qos_fail_wrong_component_count.json` |
| Mapping ref | `qa_orbit_stratification_cert_v1/mapping_protocol_ref.json` |
| Theory | `docs/theory/QA_ORBIT_STRATIFICATION_THEOREM.md` |
| Synthesis | `docs/theory/QA_ORBIT_THEOREM_SYNTHESIS.md` |
| Audit trail | `docs/theory/QA_GENERATOR_REACHABILITY.md` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_orbit_stratification_cert_v1
python qa_orbit_stratification_cert_validate.py --self-test
```

## Semantics

- **QOS_1**: `schema_version` = `"QA_ORBIT_STRATIFICATION_CERT.v1"`
- **QOS_A**: J(b,e) := min(v_p(b), v_p(e)) is invariant under both σ and μ at every (b,e) in {1..m}² — verified exhaustively for each declared prime-power modulus
- **QOS_B_SZ**: per-level sizes match closed form p^{2(k-j-1)}·(p²-1) for j<k, 1 for j=k
- **QOS_B_CC**: ⟨σ,μ⟩-orbit components equal the level sets (no mixed-J components, each level = one component)
- **QOS_C**: CRT factorization — for composite m, component sizes equal products of per-factor level sizes
- **QOS_II_A**: for inert p (Legendre(5,p)=−1) on p^n, every L_0 σ-only orbit has length π(p^n)
- **QOS_II_B**: for split p (Legendre(5,p)=+1, n=1), L_0 orbit length-counts match eigenspace + generic formula
- **QOS_II_C**: for p=5, n≥2: length-count matches 5^{n-1} orbits of length π(5^n) + 5^{n-1} orbits of length π(5^{n-1})
- **QOS_F**: `fail_ledger` is a well-formed list

## QA Axiom Compliance

- **A1**: all witnesses in {1..m}; never 0
- **A2**: d=b+e and a=b+2e derived as expressions only, not independently assigned
- **T1**: orbit length = integer path count k under σ
- **T2**: pure-integer validator; no floats; v_p, orbit enumeration, size formulas all in Z
- **S1**: uses b*b not b**2 in level-size formula
- **S2**: b, e are int throughout; no np.zeros or float state

## Relation to other certs

- **[128] `qa_fibonacci_matrix_orbit_periods_cert_v1`** — Part II Case A uses π(p^n) as the uniform orbit length; cert [128] anchors Fibonacci orbit period arithmetic.
- **[191] `qa_tiered_reachability_theorem_cert`** — substrate for reachability structure; [261] stratification refines [191]'s level sets into explicit orbit-class decomposition.
- **[262] `qa_unequal_k_ccr_invariant_cert_v1`** — companion; cert [262] uses the orbit-class structure supplied by [261] (σ-only Cosmos/Satellite/Singularity) for the Lehmann trace formula verification.
- **[263] `qa_failure_density_enumeration_cert_v1`** — uses reachability ratios 1/81, 8/81, 72/81; [261] provides the algebraic proof that exactly these three orbit families exist.

## Scope boundary

**The cert does NOT:**
- Cover m=24 in Part II (Cases A/B/C require m=p^k prime-power; 24=8×3 is composite; Part I CRT covers composite m for the content-ideal structure)
- Prove that σ and μ generate all of GL₂(Z/mZ) — only that their orbits = content-ideal classes
- Address non-Fibonacci step operators (cert is scoped to T_F = σ dynamics)

**The cert DOES:**
- Provide a complete algebraic classification of all (b,e) orbits under ⟨σ,μ⟩ for any m
- Give exact closed-form orbit-size predictions from p^k alone (no enumeration required)
- Prove the tripartition 72+8+1=81 for m=9 algebraically, not just by enumeration

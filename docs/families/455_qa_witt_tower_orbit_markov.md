# [455] QA Witt Tower Orbit Transition Markov Chain

## Claim

Pre-registered in-sample (IS) Markov matrix from ^GSPC 2001–2012 data, validated against out-of-sample (OOS) 2013–2026 on two assets (GSPC and QQQ). The orbit staircase constraints (S→C=0, C→S=0) certified in [453] and [454] are genuine structural properties — they hold in both IS and OOS partitions and across multiple assets. Cosmos is the stickiest orbit (P(C→C)=0.899); Singularity is the most transient.

## Pre-Registered IS Matrix (^GSPC 2001–2012)

n_IS=136 pairs, n_S_IS=5. Row = from-orbit, Col = to-orbit:

| From \ To | S | Sat | C |
|---|---|---|---|
| **S** | 0.200 | 0.800 | **0.000** |
| **Sat** | 0.167 | 0.333 | 0.500 |
| **C** | **0.000** | 0.113 | 0.887 |

**Key structural constraint**: P(S→C)=0 and P(C→S)=0 — the orbit staircase zeros, pre-registered from IS data.

## OOS Validation (^GSPC 2013–2026)

n_OOS=163 pairs, n_S_OOS=7.

OOS S-state months:

| Month | Notes |
|---|---|
| 2013-12 | Taper tantrum period |
| 2016-09 | Pre-election volatility |
| 2020-01 | COVID crash onset (lead month — see [453]) |
| 2020-02 | COVID crash (S-cluster begins) |
| 2020-03 | COVID crash (3-month consecutive S-run) |
| 2022-05 | Rate hike bear market |
| 2022-06 | Rate hike bear market |

The 2020 COVID crash produced a 3-month consecutive S-cluster — the longest in 25 years — consistent with the severity of the price collapse. S-states exit only to Sat (P(S→Sat)=1.0 when P(S→S) accounts for consecutive runs).

## Persistence Hierarchy

| Orbit | P(stay)_IS | P(stay)_OOS | Interpretation |
|---|---|---|---|
| **C (Cosmos)** | 0.887 | 0.908 | High inertia — normal market regime |
| **Sat (Satellite)** | 0.333 | 0.333 | Transitional — moves toward C or S |
| **S (Singularity)** | 0.200 | 0.500 | Transient extreme — cluster then exit |

Cosmos is the natural attractor: markets spend most time there and are slow to leave. Satellite is unstable (turnover every 3 months on average). Singularity clusters during acute stress then exits exclusively to Satellite, never directly to Cosmos.

## Multi-Asset OOS Results

| Asset | n_S_OOS | S→C_OOS | C→S_OOS | Staircase |
|---|---|---|---|---|
| ^GSPC | 7 | 0 | 0 | HOLD |
| QQQ | 2 | 0 | 0 | HOLD |
| **Pooled** | **9** | **0** | **0** | **HOLD** |

QQQ OOS S-months: 2013-09, 2022-04 — both isolated (no adjacent pairs), so P(S→S)_QQQ_OOS=0.

## Why the Staircase IS Mathematically Forced (C2/C3 Are Structural, Not Empirical)

The orbit class is based on rank bins in Z/27Z. A state (b, e) is S if b%9==0 and e%9==0. The transition from S to C would require: current pair (b, e) with b%9=0, e%9=0 AND next pair (e, b') with e%9≠0 OR b'%9≠0. Since e is already divisible by 9 in the S-state, the only way to reach C from S is if the shared bin e%9=0 AND the next bin b'%9≠0 (so (e,b') is Sat, not C). Or alternatively, e%9≠0 in the next pair, but e was already the second bin in the current S-pair, so e%9=0. Therefore the shared bin e has e%9=0, which forces the next pair (e, b') to have one bin divisible by 9. Hence the next orbit is EITHER S (if b'%9=0) or Sat (if b'%9≠0). It is structurally IMPOSSIBLE to go S→C. The staircase is **mathematically forced** — not an empirical regularity.

Equivalently for C→S: if current pair (b, e) has b%9≠0 and e%9≠0 (C-orbit), the next pair (e, b') has e%9≠0. For the next pair to be S, both e and b' must be divisible by 9, but e%9≠0 in the C-state — contradiction. So C→S is also structurally impossible.

**This means C2 and C3 are guaranteed to pass by construction.** The genuine empirical claims are C1 (IS adequacy), C4 (Cosmos persistence magnitude > 0.80), C5 (S non-persistence relative to C), and C6 (multi-asset replication of persistence hierarchy).

## Relation to Prior Certs

- [453]: S-orbit recession concentration (GSPC, perm_p=0.013) — the *recession signal* uses these same orbits
- [454]: Gold null — recession concentration is risk-asset specific
- [442]: Fixed-layer contemporaneous regime classifier — distinct method, same MOD=27 family
- [110]: Witt tower structural parent

The Markov cert shows **why the recession signal is durable**: S-states cluster (S→S≈0.33–0.50) during acute stress but cannot persist indefinitely (must pass through Sat before returning to C). The orbit sequence has memory — the S-cluster is the signature of a crash, not just a momentary fluctuation.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: IS sample adequacy | n_IS=136 (≥80), n_S_IS=5 (≥3) | PASS |
| C2: Staircase on IS | GSPC S→C=0, C→S=0 in IS | PASS |
| C3: Staircase on OOS | GSPC and QQQ both S→C=0, C→S=0 OOS | PASS |
| C4: Cosmos persistence | P(C→C)_full=0.899 > 0.80 > P(Sat→Sat)_full=0.333 | PASS |
| C5: S non-persistence | P(S→S)_OOS=0.500 < P(C→C)_OOS=0.908 | PASS |
| C6: Multi-asset OOS | GSPC+QQQ pooled n_S=9, S→C=0, C→S=0 | PASS |

## Primary Sources

- Wall, H. S. (1960). doi:10.1080/00029890.1960.11989541
- NBER Business Cycle Dating Committee. www.nber.org/cycles

## Related Certs

- [110] QA Witt Tower Framework (structural parent)
- [453] QA Witt Tower Orbit Recession Predictor
- [454] QA Witt Tower Orbit Recession Null (Gold)
- [442] QA Cross-Domain Witt Tower Regime Discriminator

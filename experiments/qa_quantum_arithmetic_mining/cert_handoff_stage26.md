# Stage 26 Cert Handoff

## Scope

This handoff flags two QA arithmetic geometry targets that moved from empirical
mining signals to proof-grade structural results. It does not scaffold a cert
family. The intended owner for the cert decision is Will.

Decision recorded after review: split these into two separate cert families.
The `D_plus_F_square` result is a closed conic-parametrization theorem. The
`directrix_distance_integer` result is a structural divisibility theorem, with a
separate empirical observation that mod-9 QA orbit membership beat single
generator residue baselines in Stage 21.

## Candidate 1: `D_plus_F_square`

Status: `PROVEN_BY_RATIONAL_CONIC_PARAMETRIZATION`

Source artifact:
`results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_stage24_dplusf_square_proof_closure.json`

Canonical hash:
`d11010614f73693ca55c5f613c484570cb03582e1fe84bc477e1736365e77cd3`

Theorem:
For integers `b,e >= 1`, define `d=b+e`, `a=b+2*e`, `D=d*d`, and `F=a*b`.
Then `D+F` is a square iff there exist integers `t,m,n >= 1` with
`e=t*(m*m - 4*m*n + 2*n*n)>0` such that:

```text
b=t*2*m*n
sqrt(D+F)=t*abs(m*m - 2*n*n)
```

Proof route:
Reduce `D+F=k*k` to `k*k+2*b*b=u*u` with `u=e+2*b`, then parametrize
the rational conic `X*X+2*Y*Y=1` through `(1,0)`.

Audit:
The bounded `b,e<=300` sanity check found `319` brute solutions,
`319` parametrization hits, and `0` misses.

Cert recommendation:
Cert-worthy as a structural theorem. Suggested family label:
`qa_dplusf_square_parametrization_cert_v1`.

## Candidate 2: `directrix_distance_integer`

Status: `PROVEN_STRUCTURAL_DIVISIBILITY_REDUCTION`

Source artifact:
`results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_stage25_directrix_divisibility_closure.json`

Canonical hash:
`cb9a820a9fede0ead32370d76f3fa36d2005a11301f9d0c2d4c1810934bddab5`

Theorem:
For integers `b,e >= 1` and `d=b+e`, the exact directrix integrality target
`e | d*d*d` is equivalent to `e | b*b*b`.

Sharper classifier:
If `kernel3(e)=product p^ceil(v_p(e)/3)`, then
`directrix_distance_integer` holds exactly when `kernel3(e) | b`.

Proof route:
Since `d=b+e`, `d` is congruent to `b` modulo `e`, so `d*d*d` is congruent
to `b*b*b` modulo `e`. The `kernel3` condition follows prime-by-prime from
`e | b*b*b`.

Audit:
The bounded `400x400` audit checked `160000` pairs, found support `5398`,
reduction mismatches `0`, and kernel-classifier mismatches `0`.

Cert recommendation:
Cert-worthy, but smaller and more elementary than Candidate 1. Suggested family
label: `qa_directrix_divisibility_cert_v1`.

Stage 21 orbit note:
The directrix target is no longer an unexplained conic invariant, but Stage 21
did find a QA-specific residue effect: `qa_orbit_family9` and `qa_orbit_id9`
lifted `3.93`, beating `e_only` lift `2.43` and `b_only` lift `2.61`. That
orbit-vs-generator gap should be described as empirical context or a separate
observation, not as part of the divisibility theorem itself.

## Suggested Next Action

Create two separate cert-family specs if Will approves cert conversion:

1. `qa_dplusf_square_parametrization_cert_v1`
2. `qa_directrix_divisibility_cert_v1`

Do not combine them unless Will explicitly reverses this split decision.

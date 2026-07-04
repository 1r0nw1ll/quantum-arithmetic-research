#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=sizing sweep follow-up for cert [515]; sources cited in mapping_protocol_ref.json -->
"""
NTRU parameter-sizing sweep, follow-up to cert [515]'s generalization
(reproduce_fpylll_generalization.py).

Cert [515] established: at the single toy size N=83, the "safe" QA-orbit
construction (gcd(m,3)=1) is statistically indistinguishable from a random
key under plain LLL, but under BKZ BOTH break equally (8/8) -- because N=83
is simply too small for any NTRU instance to resist BKZ. That leaves an
open question this script addresses: does "gcd(m,3)=1 is safe" survive as
N grows toward where random keys actually start resisting BKZ, or is there
a subtler QA-orbit-specific weakness (independent of the mod-3 mechanism)
that only becomes visible once the attack has to work harder?

Real NTRU security parameters (NIST PQC round 3: ntruhps2048677 N=677,
ntruhrss701 N=701, ntruhps4096821 N=821) are far outside what plain-Python
fpylll BKZ can attack interactively (a single BKZ-20+ run on a 1400+
dimensional lattice is an industrial, multi-core, multi-hour+ undertaking
in real cryptanalysis papers, not something this session's compute budget
can produce honest numbers for). This sweep instead covers N=83 (baseline,
from [515]), N=167, N=251 -- large enough to see the trend as the attack
gets relatively harder, small enough to run interactively -- and reports
where the trend is heading rather than claiming to reach real security
margins.

Three constructions compared at each N:
  - random: random ternary keys (baseline)
  - qa_unsafe: QA orbit, m=24 (3|24, the vulnerable "applied" QA modulus)
  - qa_safe: QA orbit, m=80 (gcd(80,3)=1, the "safe" construction)
"""
from __future__ import annotations

import random
import sys
import time

try:
    from fpylll import IntegerMatrix, LLL, BKZ
except ImportError:
    print("fpylll not installed -- `pip install fpylll cysignals` into a venv first.",
          file=sys.stderr)
    raise

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reproduce_fpylll_experiment import (  # noqa: E402
    keygen_random, keygen_qa, check_fg_in_lattice,
    ntru_lattice_basis, target_norm2,
)
from reproduce_fpylll_generalization import find_long_period_seed_pool  # noqa: E402


def run_reduction_timed(basis_rows, dim, use_bkz=False, bkz_block=10):
    A = IntegerMatrix(dim, dim)
    for i, row in enumerate(basis_rows):
        for j, val in enumerate(row):
            A[i, j] = val
    t0 = time.time()
    LLL.reduction(A)
    if use_bkz:
        # mpfr (arbitrary-precision float) avoids a known fplll numerical
        # failure ("infinite loop in babai") that plain double precision
        # can hit on larger dimensions/block sizes.
        BKZ.reduction(A, BKZ.Param(block_size=bkz_block), float_type="mpfr", precision=212)
    elapsed = time.time() - t0
    return A, elapsed


def attack_once_timed(f, g, h, N, q, use_bkz=False, bkz_block=10):
    basis = ntru_lattice_basis(h, N, q)
    A, elapsed = run_reduction_timed(basis, 2 * N, use_bkz=use_bkz, bkz_block=bkz_block)
    target = target_norm2(f, g)
    best = min(sum(A[i, j] * A[i, j] for j in range(A.ncols)) for i in range(A.nrows))
    return best, target, elapsed


def summarize(ratios, times, label, trials):
    broke = sum(1 for r in ratios if r <= 1.5)
    avg = sum(ratios) / len(ratios)
    avg_t = sum(times) / len(times)
    print(f"  {label:14s}: broken {broke}/{trials}  avg(best/target)={avg:10.3f}  avg_time={avg_t:6.1f}s", flush=True)
    return broke, avg


if __name__ == "__main__":
    lll_trials = 5
    bkz_trials = 3
    # (N, q, attack_tiers) -- BKZ-20 dropped at N=251 to keep runtime sane
    # (mpfr-precision BKZ on a 502-dim lattice is expensive); LLL and BKZ-10
    # still run at every size to track the trend.
    sweep = [
        (83, 256, [("LLL only", False, 0, lll_trials), ("BKZ-10", True, 10, bkz_trials), ("BKZ-20", True, 20, bkz_trials)]),
        (167, 512, [("LLL only", False, 0, lll_trials), ("BKZ-10", True, 10, bkz_trials), ("BKZ-20", True, 20, bkz_trials)]),
        (251, 512, [("LLL only", False, 0, lll_trials), ("BKZ-10", True, 10, bkz_trials)]),
    ]

    rng = random.Random(42)
    results = {}

    for N, q, attack_tiers in sweep:
        print(f"\n=== N={N} q={q} ({2*N}-dim lattice) ===", flush=True)
        safe_pool, safe_period = find_long_period_seed_pool(80, trials_scan=6400, limit=lll_trials * 4)
        unsafe_pool, unsafe_period = find_long_period_seed_pool(24, trials_scan=576, limit=lll_trials * 4)
        print(f"  qa_safe seed pool: m=80 period={safe_period}; qa_unsafe seed pool: m=24 period={unsafe_period}", flush=True)

        for tier_label, use_bkz, block, trials in attack_tiers:
            print(f" --- {tier_label} ---", flush=True)
            row = {}
            for cons_label, mod_m, pool in [
                ("random", None, None),
                ("qa_unsafe", 24, unsafe_pool),
                ("qa_safe", 80, safe_pool),
            ]:
                ratios, times = [], []
                for _ in range(trials):
                    if cons_label == "random":
                        f, g, h = keygen_random(N, q, rng)
                    else:
                        f, g, h = keygen_qa(N, q, mod_m, rng, seed_pool=pool)
                    assert check_fg_in_lattice(f, g, h, N, q)
                    best, target, elapsed = attack_once_timed(f, g, h, N, q, use_bkz=use_bkz, bkz_block=block)
                    ratios.append(best / target)
                    times.append(elapsed)
                broke, avg = summarize(ratios, times, cons_label, trials)
                row[cons_label] = {"broken": broke, "avg_ratio": avg, "trials": trials}
            results[(N, tier_label)] = row

    print("\n\n=== Summary: broken/trials at each (N, attack) ===", flush=True)
    print(f"{'N':>5s} {'attack':10s} {'random':>10s} {'qa_unsafe':>12s} {'qa_safe':>10s}", flush=True)
    for N, q, attack_tiers in sweep:
        for tier_label, _, _, trials in attack_tiers:
            row = results[(N, tier_label)]
            print(f"{N:5d} {tier_label:10s} {row['random']['broken']:>3d}/{trials:<3d}    "
                  f"{row['qa_unsafe']['broken']:>3d}/{trials:<3d}      {row['qa_safe']['broken']:>3d}/{trials:<3d}", flush=True)

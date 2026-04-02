#!/usr/bin/env python3
"""
QA Resonance ↔ Modular Geodesic Experiment (local, deterministic)

Entry point: `python qa_resonance_spectral_experiment.py ...`

Goal
----
This script is a *computational harness* to test candidate "QA resonance labels"
against standard modular-surface geodesic data:

  - enumerate elements of PSL_2(Z) by words in generators S,T (optionally T^{-1})
  - filter hyperbolic elements (|tr| > 2)
  - compute closed-geodesic length ℓ(γ) from trace
  - reduce γ mod N (default N=72, lcm(24,9))
  - evaluate candidate label functions that mimic QA-style (mod 24, digital-root mod 9)
  - test invariance under inversion and random conjugation
  - test whether labels factor through reduction mod N inside the sampled set

This does NOT assume any canonical QA→SL_2(Z) bridge; instead it offers multiple
candidate label functions (trace-based, discriminant-based, continued-fraction-based,
and simple column-extraction heuristics) and reports which ones pass basic
well-definedness tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# --- Section: PSL2Z matrix plumbing ---


@dataclass(frozen=True, slots=True)
class Mat2:
    a: int
    b: int
    c: int
    d: int

    def __matmul__(self, other: "Mat2") -> "Mat2":
        return Mat2(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
        )

    def det(self) -> int:
        return self.a * self.d - self.b * self.c

    def tr(self) -> int:
        return self.a + self.d

    def inv(self) -> "Mat2":
        # For SL_2(Z): inverse is [[d,-b],[-c,a]]
        return Mat2(self.d, -self.b, -self.c, self.a)

    def negate(self) -> "Mat2":
        return Mat2(-self.a, -self.b, -self.c, -self.d)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)


I2 = Mat2(1, 0, 0, 1)
S = Mat2(0, -1, 1, 0)  # S^2 = -I in SL2; identity in PSL2
T = Mat2(1, 1, 0, 1)
T_INV = Mat2(1, -1, 0, 1)


def psl_normal_form(m: Mat2) -> Mat2:
    """
    Identify m ~ -m (PSL2) by choosing a deterministic representative.
    Rule: pick the sign where the first nonzero entry in (a,b,c,d) is positive.
    """
    tup = m.to_tuple()
    tup_neg = (-tup[0], -tup[1], -tup[2], -tup[3])

    def key(t: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        for x in t:
            if x != 0:
                return (0,) + t if x > 0 else (1,) + t
        return (0,) + t

    return Mat2(*tup) if key(tup) <= key(tup_neg) else Mat2(*tup_neg)


def mod_psl_normal_form(entries_mod_n: Tuple[int, int, int, int], n: int) -> Tuple[int, int, int, int]:
    """
    Identify M ~ -M in PSL2(Z/nZ) by picking lexicographically minimal of M and -M mod n.
    """
    a, b, c, d = entries_mod_n
    neg = ((-a) % n, (-b) % n, (-c) % n, (-d) % n)
    return entries_mod_n if entries_mod_n <= neg else neg


def reduce_mod_n(m: Mat2, n: int) -> Tuple[int, int, int, int]:
    return mod_psl_normal_form((m.a % n, m.b % n, m.c % n, m.d % n), n=n)


def mul_mod_n(x: Tuple[int, int, int, int], y: Tuple[int, int, int, int], n: int) -> Tuple[int, int, int, int]:
    ax, bx, cx, dx = x
    ay, by, cy, dy = y
    out = (
        (ax * ay + bx * cy) % n,
        (ax * by + bx * dy) % n,
        (cx * ay + dx * cy) % n,
        (cx * by + dx * dy) % n,
    )
    return mod_psl_normal_form(out, n=n)


def inv_mod_n(x: Tuple[int, int, int, int], n: int) -> Tuple[int, int, int, int]:
    a, b, c, d = x
    out = (d % n, (-b) % n, (-c) % n, a % n)
    return mod_psl_normal_form(out, n=n)


def conj_mod_n(g: Tuple[int, int, int, int], x: Tuple[int, int, int, int], n: int) -> Tuple[int, int, int, int]:
    return mul_mod_n(mul_mod_n(g, x, n=n), inv_mod_n(g, n=n), n=n)


# --- Section: enumeration in PSL2Z ---


GENS: Dict[str, Mat2] = {"S": S, "T": T, "U": T_INV}  # U := T^{-1}


def enumerate_psl2z_ball(max_word_len: int, include_t_inv: bool) -> Dict[Mat2, str]:
    """
    BFS enumeration of distinct PSL2Z elements within a word-length ball.

    Returns:
        map element -> one shortest word over {S,T,(U)} that evaluates to it.
    """
    if max_word_len < 0:
        raise ValueError("max_word_len must be >= 0")

    alphabet = ["S", "T"] + (["U"] if include_t_inv else [])
    seen: Dict[Mat2, str] = {psl_normal_form(I2): ""}
    frontier: List[Mat2] = [psl_normal_form(I2)]

    for depth in range(1, max_word_len + 1):
        next_frontier: List[Mat2] = []
        for g in frontier:
            w = seen[g]
            for sym in alphabet:
                h = psl_normal_form(g @ GENS[sym])
                if h in seen:
                    continue
                seen[h] = (w + sym)
                next_frontier.append(h)
        frontier = next_frontier

    return seen


# --- Section: hyperbolic geometry invariants ---


def is_hyperbolic(m: Mat2) -> bool:
    return abs(m.tr()) > 2


def geodesic_length_from_trace(trace: int) -> float:
    return 2.0 * math.acosh(abs(trace) / 2.0)


# --- Section: arithmetic helpers (QA-like mod 24 × digital-root mod 9) ---


def digital_root_9(n: int) -> int:
    # Canonical QA convention: ((n-1) mod 9)+1 for n>0; allow n=0 for convenience
    n = abs(n)
    if n == 0:
        return 0
    return ((n - 1) % 9) + 1


Label = Tuple[int, int]


def label_from_int(x: int) -> Label:
    return (x % 24, digital_root_9(x))


def label_trace(m: Mat2) -> Label:
    return label_from_int(abs(m.tr()))


def label_discriminant(m: Mat2) -> Label:
    # D = tr^2 - 4 for det=1
    t = m.tr()
    return label_from_int(t * t - 4)


def label_trace2_mod72(m: Mat2) -> Label:
    """
    PSL2(Z/72Z)-well-defined class function derived from trace^2 mod 72.
    (Trace changes sign under M ~ -M, but trace^2 does not.)
    """
    t2 = (m.tr() * m.tr()) % 72
    return label_from_int(t2)


def label_disc_mod72(m: Mat2) -> Label:
    """
    PSL2(Z/72Z)-well-defined class function derived from discriminant mod 72:
      disc ≡ tr^2 - 4 (mod 72)
    """
    t2 = (m.tr() * m.tr()) % 72
    disc = (t2 - 4) % 72
    return label_from_int(disc)


def label_col1_a_equals_b_plus_2e(m: Mat2) -> Label:
    """
    Heuristic "QA bridge": treat column 1 as (b,e) and compute a=b+2e.
    This is NOT expected to be conjugacy-invariant; included to falsify quickly.
    """
    b = abs(m.a)
    e = abs(m.c)
    a = b + 2 * e
    return label_from_int(a)


# --- Section: continued fraction period of quadratic irrational fixed point ---


def floor_div_surd(p: int, q: int, d: int) -> int:
    """
    Compute floor((p + sqrt(d))/q) exactly using integer arithmetic.
    Assumptions: q>0, d>0 non-square.
    """
    if q <= 0:
        raise ValueError("q must be > 0")
    r = math.isqrt(d)
    a = (p + r) // q
    # Correct if (a+1) is still <= (p+sqrt(d))/q
    while True:
        lhs = (a + 1) * q - p
        if lhs <= 0:
            a += 1
            continue
        if lhs * lhs <= d:
            a += 1
            continue
        return a


def continued_fraction_quadratic(p0: int, q0: int, d: int, max_steps: int = 10_000) -> Tuple[List[int], List[int]]:
    """
    Continued fraction of x = (p0 + sqrt(d))/q0 with q0>0.

    Returns:
        (preperiod_digits, period_digits)
    """
    if d <= 0:
        raise ValueError("d must be positive")
    r = math.isqrt(d)
    if r * r == d:
        raise ValueError("d must be non-square")
    if q0 <= 0:
        raise ValueError("q0 must be > 0")

    p, q = p0, q0
    digits: List[int] = []
    seen: Dict[Tuple[int, int], int] = {}

    for step in range(max_steps):
        if q == 0:
            raise RuntimeError("continued fraction encountered q=0 (invalid surd state)")
        # Normalize to q>0 for the floor computation.
        if q < 0:
            p, q = -p, -q

        state = (p, q)
        if state in seen:
            start = seen[state]
            return digits[:start], digits[start:]
        seen[state] = len(digits)

        a = floor_div_surd(p, q, d)
        digits.append(a)

        # Transform:
        # x = (p + sqrt(d))/q, a=floor(x)
        # x' = 1/(x-a) = (p' + sqrt(d))/q'
        p_next = a * q - p
        numerator = d - p_next * p_next
        if numerator % q != 0:
            raise RuntimeError(
                "continued fraction integrality failure: expected q | (d - p'^2) "
                f"but got p={p}, q={q}, a={a}, p'={p_next}, d={d}"
            )
        q_next = numerator // q
        p, q = p_next, q_next

    raise RuntimeError("continued fraction did not cycle within max_steps")


def fixed_point_surd_for_hyperbolic(m: Mat2) -> Tuple[int, int, int]:
    """
    Return (p,q,d) for attracting fixed point alpha = (p + sqrt(d))/q of m.

    For m=[[a,b],[c,d]] with c != 0:
      fixed points solve c x^2 + (d-a)x - b = 0.
    One root is:
      x = (a-d + sqrt((a+d)^2 - 4)) / (2c)

    We return p = (a-d), q = 2c, d = (tr^2 - 4).
    Then alpha = (p + sqrt(d))/q. If q < 0, we flip signs.
    """
    if not is_hyperbolic(m):
        raise ValueError("matrix must be hyperbolic")
    if m.c == 0:
        raise ValueError("hyperbolic elements in PSL2Z have c != 0")

    disc = m.tr() * m.tr() - 4
    p = m.a - m.d
    q = 2 * m.c
    if q < 0:
        p, q = -p, -q
    return p, q, disc


def label_cf_period_sum(m: Mat2) -> Label:
    """
    Candidate "bridge": take the periodic part of the continued fraction of the attracting fixed point,
    and use sum(period_digits) as the integer to be reduced to (mod 24, dr_9).

    This is designed to be stable under cyclic shifts and reversal of the period, hence a plausible
    conjugacy/orientation invariant *if* the period extraction is performed consistently.
    """
    p, q, disc = fixed_point_surd_for_hyperbolic(m)
    _, period = continued_fraction_quadratic(p, q, disc)
    s = sum(period)
    return label_from_int(s)


LABEL_FNS: Dict[str, Callable[[Mat2], Label]] = {
    "trace": label_trace,
    "disc": label_discriminant,
    "trace2_mod72": label_trace2_mod72,
    "disc_mod72": label_disc_mod72,
    "col1_a=b+2e": label_col1_a_equals_b_plus_2e,
    "cf_period_sum": label_cf_period_sum,
}


# --- Section: invariance tests ---


def random_word(alphabet: Sequence[str], length: int, rng: random.Random) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def eval_word(word: str) -> Mat2:
    m = I2
    for ch in word:
        m = m @ GENS[ch]
    return psl_normal_form(m)


def conjugation_invariance_check(
    m: Mat2,
    label_fn: Callable[[Mat2], Label],
    trials: int,
    max_conj_word_len: int,
    include_t_inv: bool,
    rng: random.Random,
) -> Tuple[bool, Optional[Label]]:
    """
    Test label_fn(g m g^{-1}) == label_fn(m) over random conjugators g.
    Returns (ok, first_counterexample_label).
    """
    target = label_fn(m)
    alphabet = ["S", "T"] + (["U"] if include_t_inv else [])
    for _ in range(trials):
        length = rng.randint(0, max_conj_word_len)
        g = eval_word(random_word(alphabet, length, rng))
        conj = psl_normal_form(g @ m @ g.inv())
        if label_fn(conj) != target:
            return False, label_fn(conj)
    return True, None


def inversion_invariance_check(m: Mat2, label_fn: Callable[[Mat2], Label]) -> bool:
    return label_fn(m) == label_fn(psl_normal_form(m.inv()))


# --- Section: forced congruence collisions (kernel of reduction mod N) ---


@dataclass(frozen=True, slots=True)
class KernelElementSpec:
    kind: str
    params: Tuple[int, ...]

    def build(self, n: int) -> Mat2:
        if self.kind == "upper_unipotent":
            (m,) = self.params
            return Mat2(1, n * m, 0, 1)
        if self.kind == "lower_unipotent":
            (m,) = self.params
            return Mat2(1, 0, n * m, 1)
        if self.kind == "balanced_hyperbolic":
            # det([[1+n*x, n*x],[-n*x, 1-n*x]]) = 1 for all integers x
            (x,) = self.params
            return Mat2(1 + n * x, n * x, -n * x, 1 - n * x)
        raise ValueError(f"unknown kernel element kind: {self.kind}")


KERNEL_KINDS = ("upper_unipotent", "lower_unipotent", "balanced_hyperbolic")


def random_kernel_element_spec(x_bound: int, rng: random.Random) -> KernelElementSpec:
    kind = rng.choice(KERNEL_KINDS)
    if kind in ("upper_unipotent", "lower_unipotent"):
        m = rng.randint(-x_bound, x_bound)
        if m == 0:
            m = 1
        return KernelElementSpec(kind=kind, params=(m,))
    x = rng.randint(-x_bound, x_bound)
    if x == 0:
        x = 1
    return KernelElementSpec(kind=kind, params=(x,))


def random_kernel_element(n: int, x_bound: int, steps: int, rng: random.Random) -> Tuple[Mat2, List[KernelElementSpec]]:
    """
    Build k in ker(rho_n) by multiplying simple generators that are ≡ I (mod n).
    Returns (k, specs) where specs is a replayable construction.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    k = I2
    specs: List[KernelElementSpec] = []
    for _ in range(steps):
        spec = random_kernel_element_spec(x_bound=x_bound, rng=rng)
        specs.append(spec)
        k = psl_normal_form(k @ spec.build(n))
    return k, specs


def forced_collision_factor_test(
    items: Sequence[Tuple[Mat2, str]],
    modulus_n: int,
    label_name: str,
    label_fn: Callable[[Mat2], Label],
    trials: int,
    kernel_steps: int,
    kernel_x_bound: int,
    rng: random.Random,
) -> Dict[str, object]:
    """
    Adversarial test of: label factors through rho_N.

    Construct forced collisions by left-multiplying by k ∈ ker(rho_N):
      M' = k M
    which guarantees rho_N(M') = rho_N(M) by construction.

    Returns:
      {"status": "pass", ...} or {"status": "fail", witness_fields...}
    """
    if trials <= 0:
        raise ValueError("trials must be > 0")
    if not items:
        return {"schema_id": "QA_CONGRUENCE_LABEL_FACTOR_CERT.v1", "status": "skipped", "reason": "no items"}

    for trial_idx in range(trials):
        m, w = items[rng.randrange(0, len(items))]
        base_red = reduce_mod_n(m, modulus_n)
        base_label = label_fn(m)

        # Ensure the collided element stays hyperbolic (needed for cf-based labels).
        for attempt_idx in range(50):
            k, specs = random_kernel_element(n=modulus_n, x_bound=kernel_x_bound, steps=kernel_steps, rng=rng)
            m2 = psl_normal_form(k @ m)
            if not is_hyperbolic(m2):
                continue
            red2 = reduce_mod_n(m2, modulus_n)
            if red2 != base_red:
                raise RuntimeError("kernel collision construction failed: reductions differ")
            label2 = label_fn(m2)
            if label2 != base_label:
                return {
                    "schema_id": "QA_CONGRUENCE_LABEL_FACTOR_CERT.v1",
                    "status": "fail",
                    "label_name": label_name,
                    "modulus_n": modulus_n,
                    "trial_index": trial_idx,
                    "attempt_index": attempt_idx,
                    "base_word": w,
                    "base_mat": m.to_tuple(),
                    "base_reduction": base_red,
                    "base_label": base_label,
                    "kernel_specs": [{"kind": s.kind, "params": s.params} for s in specs],
                    "collided_mat": m2.to_tuple(),
                    "collided_reduction": red2,
                    "collided_label": label2,
                }
            break

    return {"schema_id": "QA_CONGRUENCE_LABEL_FACTOR_CERT.v1", "status": "pass", "label_name": label_name, "modulus_n": modulus_n, "trials": trials}


def verify_kernel_report(path: str) -> int:
    with open(path) as f:
        report = json.load(f)

    status = report.get("status")
    label_name = report.get("label_name")
    modulus_n_raw = report.get("modulus_n")

    print("=" * 80)
    print("Kernel Report Verification")
    print("=" * 80)
    print(f"path      : {path}")
    print(f"status    : {status}")
    print(f"label     : {label_name}")
    print(f"modulus_n : {modulus_n_raw}")
    print()

    if status in ("pass", "skipped"):
        if status == "pass":
            print(f"trials     : {report.get('trials')}")
            print("VERDICT: NOTE (pass report cannot be verified from a single witness)")
            return 0
        print(f"reason     : {report.get('reason')}")
        print("VERDICT: NOTE (skipped)")
        return 0

    if status != "fail":
        print("VERDICT: NOTE (unknown report status)")
        return 0

    if modulus_n_raw is None:
        raise ValueError("kernel report missing modulus_n")
    modulus_n = int(modulus_n_raw)
    if label_name not in LABEL_FNS:
        raise ValueError(f"unknown label_name in report: {label_name}")
    label_fn = LABEL_FNS[label_name]

    base = Mat2(*report["base_mat"])
    collided = Mat2(*report["collided_mat"])
    base_red = reduce_mod_n(base, modulus_n)
    col_red = reduce_mod_n(collided, modulus_n)
    base_label = label_fn(base)
    col_label = label_fn(collided)

    print(f"rho_N(base)     : {base_red}")
    print(f"rho_N(collided) : {col_red}")
    print(f"label(base)     : {base_label}")
    print(f"label(collided) : {col_label}")
    print()

    if base_red != col_red:
        print("VERDICT: FAIL (reductions differ)")
        return 2

    if base_label == col_label:
        print("VERDICT: FAIL (report claims counterexample but labels match)")
        return 2
    print("VERDICT: OK (counterexample verified)")
    return 0


# --- Section: main experiment ---


def run_experiment(
    max_word_len: int,
    include_t_inv: bool,
    modulus_n: int,
    label_name: str,
    sample_strategy: str,
    conj_trials: int,
    conj_max_len: int,
    seed: int,
    out_csv: Optional[str],
    limit_hyperbolic_rows: Optional[int],
    mod_conj_top_k: int,
    mod_conj_limit: int,
    max_power: int,
    primitive_only: bool,
    weight_t: Optional[float],
    kernel_collision_trials: int,
    kernel_steps: int,
    kernel_x_bound: int,
    kernel_out_json: Optional[str],
) -> int:
    rng = random.Random(seed)
    label_fn = LABEL_FNS[label_name]

    elements = enumerate_psl2z_ball(max_word_len=max_word_len, include_t_inv=include_t_inv)
    all_count = len(elements)
    element_set = set(elements.keys())

    # Detect proper powers within the enumerated set (bounded by max_power).
    # For each element x, record the largest exponent k such that x = r^k for some r in element_set.
    max_k_for: Dict[Mat2, int] = defaultdict(lambda: 1)
    primitive_root_for: Dict[Mat2, Mat2] = {}
    if max_power >= 2:
        for r in element_set:
            p = r
            for k in range(2, max_power + 1):
                p = psl_normal_form(p @ r)
                if p not in element_set:
                    continue
                if k > max_k_for[p]:
                    max_k_for[p] = k
                    primitive_root_for[p] = r

    hyperbolic_all: List[Tuple[Mat2, str]] = [(m, w) for (m, w) in elements.items() if is_hyperbolic(m)]
    if primitive_only:
        hyperbolic_all = [(m, w) for (m, w) in hyperbolic_all if int(max_k_for[m]) == 1]

    if sample_strategy == "smallest_trace":
        hyperbolic_all.sort(key=lambda mw: (abs(mw[0].tr()), len(mw[1]), mw[1]))
        if limit_hyperbolic_rows is not None:
            hyperbolic = hyperbolic_all[:limit_hyperbolic_rows]
        else:
            hyperbolic = hyperbolic_all
    elif sample_strategy == "random":
        if limit_hyperbolic_rows is None or limit_hyperbolic_rows >= len(hyperbolic_all):
            hyperbolic = list(hyperbolic_all)
            rng.shuffle(hyperbolic)
        else:
            hyperbolic = rng.sample(hyperbolic_all, k=limit_hyperbolic_rows)
    else:
        raise ValueError(f"unknown sample_strategy: {sample_strategy}")

    label_counts: Counter[Label] = Counter()
    label_counts_prim: Counter[Label] = Counter()
    inv_fail = 0
    conj_fail = 0
    by_mod_n: Dict[Tuple[int, int, int, int], List[Label]] = defaultdict(list)
    by_mod_n_prim: Dict[Tuple[int, int, int, int], List[Label]] = defaultdict(list)

    weighted_sum = 0.0
    weighted_sum_prim = 0.0

    length_sum_by_label: Dict[Label, float] = defaultdict(float)
    length_min_by_label: Dict[Label, float] = {}
    length_max_by_label: Dict[Label, float] = {}

    rows: List[Dict[str, object]] = []
    for m, w in hyperbolic:
        k_power = int(max_k_for[m])
        is_prim = k_power == 1

        tr = m.tr()
        length = geodesic_length_from_trace(tr)
        red = reduce_mod_n(m, modulus_n)
        label = label_fn(m)

        label_counts[label] += 1
        length_sum_by_label[label] += float(length)
        if label not in length_min_by_label or length < length_min_by_label[label]:
            length_min_by_label[label] = float(length)
        if label not in length_max_by_label or length > length_max_by_label[label]:
            length_max_by_label[label] = float(length)
        by_mod_n[red].append(label)
        if is_prim:
            label_counts_prim[label] += 1
            by_mod_n_prim[red].append(label)

        if not inversion_invariance_check(m, label_fn):
            inv_fail += 1

        ok_conj, _ = conjugation_invariance_check(
            m=m,
            label_fn=label_fn,
            trials=conj_trials,
            max_conj_word_len=conj_max_len,
            include_t_inv=include_t_inv,
            rng=rng,
        )
        if not ok_conj:
            conj_fail += 1

        if weight_t is not None:
            wgt = math.exp(-weight_t * length)
            weighted_sum += wgt
            if is_prim:
                weighted_sum_prim += wgt

        rows.append(
            {
                "word": w,
                "word_len": len(w),
                "a": m.a,
                "b": m.b,
                "c": m.c,
                "d": m.d,
                "trace": tr,
                "length": length,
                f"mod{modulus_n}": red,
                "label_mod24": label[0],
                "label_dr9": label[1],
                "power_k": k_power,
                "is_primitive_in_sample": int(is_prim),
            }
        )

    # Factoring-through-mod-N check within this sample:
    mod_inconsistent = 0
    for red, labels in by_mod_n.items():
        if len(set(labels)) > 1:
            mod_inconsistent += 1

    mod_inconsistent_prim = 0
    for red, labels in by_mod_n_prim.items():
        if len(set(labels)) > 1:
            mod_inconsistent_prim += 1

    if mod_conj_top_k > 0 and hyperbolic:
        s_mod = reduce_mod_n(S, modulus_n)
        t_mod = reduce_mod_n(T, modulus_n)
        t_inv_mod = reduce_mod_n(T_INV, modulus_n)

        print("Conjugacy class sizes in PSL2(Z/NZ) (BFS under conjugation by S,T,T^{-1}):")
        for (m, w) in hyperbolic[:mod_conj_top_k]:
            x_mod = reduce_mod_n(m, modulus_n)
            frontier = [x_mod]
            seen_mod = {x_mod}

            while frontier and len(seen_mod) < mod_conj_limit:
                cur = frontier.pop()
                for g_mod in (s_mod, t_mod, t_inv_mod):
                    nxt = conj_mod_n(g_mod, cur, n=modulus_n)
                    if nxt in seen_mod:
                        continue
                    seen_mod.add(nxt)
                    frontier.append(nxt)

            cap_note = "" if len(seen_mod) < mod_conj_limit else f" (capped at {mod_conj_limit})"
            word_disp = w if w else "∅"
            print(f"  word={word_disp:>10s}  trace={m.tr():>4d}  |ConjClass|≈{len(seen_mod)}{cap_note}")
        print()

    kernel_report: Optional[Dict[str, object]] = None
    if kernel_collision_trials > 0:
        kernel_report = forced_collision_factor_test(
            items=hyperbolic,
            modulus_n=modulus_n,
            label_name=label_name,
            label_fn=label_fn,
            trials=kernel_collision_trials,
            kernel_steps=kernel_steps,
            kernel_x_bound=kernel_x_bound,
            rng=rng,
        )
        print("Forced kernel-collision test (label factors through rho_N):")
        print(f"  trials         : {kernel_collision_trials}")
        print(f"  kernel_steps   : {kernel_steps}")
        print(f"  kernel_x_bound : {kernel_x_bound}")
        print(f"  status         : {kernel_report['status']}")
        if kernel_report["status"] == "fail":
            print("  counterexample : YES")
        print()

    # Output
    print("=" * 80)
    print("QA Resonance ↔ Modular Geodesic Experiment")
    print("=" * 80)
    print(f"Enumerated PSL2Z elements (ball): {all_count}")
    print(f"Hyperbolic elements sampled      : {len(hyperbolic)}")
    print(f"Hyperbolic sampling              : {sample_strategy}")
    print(f"Generator set                    : {{S,T{',T^{-1}' if include_t_inv else ''}}}")
    print(f"Modulus N                        : {modulus_n}")
    print(f"Label function                   : {label_name}")
    print(f"Power detection                  : k<= {max_power} (within enumerated set)")
    if primitive_only:
        print("Primitive filter                 : ON (drops detected proper powers)")
    print()
    print("Well-definedness diagnostics (within sample):")
    print(f"  inversion failures             : {inv_fail}")
    print(f"  random conjugation failures    : {conj_fail}  (trials={conj_trials}, max_conj_len={conj_max_len})")
    print(f"  mod-N collision inconsistencies: {mod_inconsistent} residue-classes with >1 label")
    print(f"  mod-N inconsistencies (primitive only): {mod_inconsistent_prim}")
    if weight_t is not None:
        print()
        print(f"Weighted length sum (g(l)=exp(-t*l), t={weight_t}):")
        print(f"  all hyperbolic   : {weighted_sum:.6g}")
        print(f"  primitive-only   : {weighted_sum_prim:.6g}")
    print()
    print("Top labels (mod24, dr9):")
    for (lab, cnt) in label_counts.most_common(12):
        mean_len = length_sum_by_label[lab] / max(1, cnt)
        lo = length_min_by_label.get(lab, float("nan"))
        hi = length_max_by_label.get(lab, float("nan"))
        print(f"  {lab}: {cnt}")
        print(f"    length mean/min/max          : {mean_len:.6g} / {lo:.6g} / {hi:.6g}")
    if label_counts_prim:
        print()
        print("Top labels (primitive only):")
        for (lab, cnt) in label_counts_prim.most_common(12):
            print(f"  {lab}: {cnt}")
    print()

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "word",
                    "word_len",
                    "a",
                    "b",
                    "c",
                    "d",
                    "trace",
                    "length",
                    f"mod{modulus_n}",
                    "label_mod24",
                    "label_dr9",
                    "power_k",
                    "is_primitive_in_sample",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {out_csv}")

    if kernel_out_json and kernel_report is not None:
        with open(kernel_out_json, "w") as f:
            json.dump(kernel_report, f, indent=2, sort_keys=True)
        print(f"Wrote kernel report JSON: {kernel_out_json}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify_kernel_json", type=str, default=None)
    parser.add_argument("--max_word_len", type=int, default=10)
    parser.add_argument("--include_t_inv", action="store_true")
    parser.add_argument("--modulus_n", type=int, default=72)
    parser.add_argument("--label", type=str, default="cf_period_sum", choices=sorted(LABEL_FNS.keys()))
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="smallest_trace",
        choices=["smallest_trace", "random"],
        help="How to select hyperbolic elements when --limit_hyperbolic_rows is set.",
    )
    parser.add_argument("--conj_trials", type=int, default=64)
    parser.add_argument("--conj_max_len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="qa_resonance_spectral_experiment.csv")
    parser.add_argument("--limit_hyperbolic_rows", type=int, default=400)
    parser.add_argument("--mod_conj_top_k", type=int, default=6)
    parser.add_argument("--mod_conj_limit", type=int, default=200_000)
    parser.add_argument("--max_power", type=int, default=6)
    parser.add_argument("--primitive_only", action="store_true")
    parser.add_argument("--weight_t", type=float, default=None)
    parser.add_argument("--kernel_collision_trials", type=int, default=0)
    parser.add_argument("--kernel_steps", type=int, default=2)
    parser.add_argument("--kernel_x_bound", type=int, default=50)
    parser.add_argument("--kernel_out_json", type=str, default=None)
    args = parser.parse_args(argv)

    if args.verify_kernel_json:
        return verify_kernel_report(args.verify_kernel_json)

    return run_experiment(
        max_word_len=args.max_word_len,
        include_t_inv=args.include_t_inv,
        modulus_n=args.modulus_n,
        label_name=args.label,
        sample_strategy=args.sample_strategy,
        conj_trials=args.conj_trials,
        conj_max_len=args.conj_max_len,
        seed=args.seed,
        out_csv=args.out_csv,
        limit_hyperbolic_rows=args.limit_hyperbolic_rows,
        mod_conj_top_k=args.mod_conj_top_k,
        mod_conj_limit=args.mod_conj_limit,
        max_power=args.max_power,
        primitive_only=args.primitive_only,
        weight_t=args.weight_t,
        kernel_collision_trials=args.kernel_collision_trials,
        kernel_steps=args.kernel_steps,
        kernel_x_bound=args.kernel_x_bound,
        kernel_out_json=args.kernel_out_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())

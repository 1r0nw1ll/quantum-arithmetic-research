# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Lothaire 2002 ISBN 978-0-521-81220-7; Hardy+Wright 2008 ISBN 978-0-19-921986-5) -->
"""Cert [295]: QA Pell Sturmian Bridge.

PRIMARY CLAIM:
  The Pell Stern-Brocot words (cert [294]) and the characteristic Sturmian
  word s_alpha (alpha=sqrt(2)-1) are TWO DIFFERENT ENCODINGS of the same
  arithmetic, bridged by the Pell e-values:

  SB PATH (cert [294]):  encodes sqrt(2) = [1;2,2,2,...]  (value > 1)
    Word: (RLLR)^inf — PERIODIC (period 4), NOT Sturmian
    Alphabet: {L,R}, density 1/2 each

  STURMIAN s_alpha:      encodes sqrt(2)-1 = [0;2,2,2,...]  (value < 1)
    Word: s(n)=floor((n+1)*alpha)-floor(n*alpha), APERIODIC, IS Sturmian
    Complexity: p(n) = n+1  (minimal for aperiodic binary word)
    Gap sizes: {2,3} = {floor(1/alpha), ceil(1/alpha)} = {floor(sqrt(2)+1), ceil(...)}

  BRIDGE:
    The Pell e-values {1,2,5,12,29,70,...} are IDENTICAL to the denominators
    of the CF convergents to alpha=sqrt(2)-1=[0;2,2,...].
    REASON: both satisfy the SAME recurrence q_{n+1}=2q_n+q_{n-1} with
    the SAME initial conditions q_0=1, q_1=2.
    This recurrence arises from the SHARED period-2 CF structure of
    sqrt(2)=[1;2-bar] and sqrt(2)-1=[0;2-bar].

STURMIAN WORD DEFINITION:
  s(n) = floor((n+1)*alpha) - floor(n*alpha)  for alpha = sqrt(2)-1
  This takes values in {0,1}. Density of 1s = alpha = sqrt(2)-1 approx 0.414.
  The word is aperiodic (no finite period exists).
  Complexity p(n) = n+1 (exactly one new subword of each new length).

KEY DISTINCTION:
  (RLLR)^inf has complexity [2,4,4,4,...]:
    p(1)=2 (both L and R appear)
    p(2)=4 (all of RL,LL,LR,RR appear -- collapses immediately, not n+1)
  s_alpha has complexity [2,3,4,5,...]:
    p(n)=n+1 for ALL n (Sturmian, never exceeds n+1)
  Therefore (RLLR)^inf CANNOT be Sturmian.

WHAT IS KNOWN vs NOVEL:
  Known: Sturmian theory (Morse-Hedlund 1938, Lothaire 2002); CF convergents
    and Pell denominators satisfying q_{n+1}=2q_n+q_{n-1} (classical).
  Novel: The explicit identification of Pell e-values as the bridge between
    the PERIODIC SB encoding of sqrt(2) (cert [294]) and the APERIODIC
    Sturmian encoding of sqrt(2)-1; the QA/Koenig framing of this connection.
"""

from __future__ import annotations

from math import floor, sqrt
from typing import List, Tuple

ALPHA = sqrt(2) - 1   # sqrt(2)-1 = [0;2,2,2,...], approx 0.41421356


def _sturmian(n: int, alpha: float = ALPHA) -> int:
    """s(n) = floor((n+1)*alpha) - floor(n*alpha). Values in {0,1}."""
    return floor((n + 1) * alpha) - floor(n * alpha)


def _sturmian_word(length: int) -> List[int]:
    return [_sturmian(n) for n in range(length)]


def _complexity(word: List[int], n: int) -> int:
    """Number of distinct subwords of length n."""
    return len(set(tuple(word[i:i+n]) for i in range(len(word) - n + 1)))


def _word_complexity(word_str: str, n: int) -> int:
    return len(set(word_str[i:i+n] for i in range(len(word_str) - n + 1)))


def _is_periodic(seq: List[int], max_period: int) -> bool:
    n = len(seq)
    for p in range(1, min(max_period + 1, n // 2 + 1)):
        if all(seq[i] == seq[i % p] for i in range(n)):
            return True
    return False


def _gap_sizes(word: List[int]) -> set:
    positions = [i for i, x in enumerate(word) if x == 1]
    if len(positions) < 2:
        return set()
    return set(positions[i+1] - positions[i] for i in range(len(positions)-1))


# ---------------------------------------------------------------------------
# Checks per fixture type
# STURMIAN_COMPLEXITY  — p(n) = n+1 for s_alpha
# PELL_E_RECURRENCE   — Pell e-values satisfy q_{n+1}=2q_n+q_{n-1}
# GAP_SIZES           — gap sizes in s_alpha are subset of {2,3}
# STURMIAN_APERIODIC  — s_alpha has no period ≤ max_period in window
# COMPLEXITY_OF_WORD  — explicit word has given complexity
# ---------------------------------------------------------------------------


def validate_fixture(fixture: dict) -> dict:
    check_type = fixture["check_type"]
    results: dict = {}

    if check_type == "complexity":
        n = fixture["n"]
        expected = fixture["expected_p_n"]
        word = _sturmian_word(max(200, n * 20))
        actual = _complexity(word, n)
        results["COMPLEXITY_OK"] = (actual == expected)

    elif check_type == "pell_e_recurrence":
        vals: List[int] = fixture["pell_e"]
        # Check q_{n+1}=2q_n+q_{n-1} for all consecutive triples
        rec_ok = all(vals[i+2] == 2*vals[i+1] + vals[i]
                     for i in range(len(vals)-2))
        # Check initial conditions q_0=1, q_1=2
        init_ok = (vals[0] == 1 and vals[1] == 2)
        # Check these equal the CF denominators for alpha=[0;2,2,...]:
        q = [1, 2]
        while len(q) < len(vals):
            q.append(2*q[-1] + q[-2])
        cf_ok = (vals == q[:len(vals)])
        results["RECURRENCE_OK"] = rec_ok
        results["INIT_OK"] = init_ok
        results["CF_EQUAL_OK"] = cf_ok

    elif check_type == "gap_sizes":
        length = fixture["word_length"]
        expected_gaps = set(fixture["expected_gaps"])
        word = _sturmian_word(length)
        actual_gaps = _gap_sizes(word)
        results["GAP_SUBSET_OK"] = actual_gaps.issubset(expected_gaps)
        # Also verify no empty gaps and at least 1s exist
        results["HAS_ONES"] = (1 in word)

    elif check_type == "aperiodic":
        window = fixture["window"]
        max_period = fixture["max_period_checked"]
        word = _sturmian_word(window)
        results["APERIODIC_OK"] = not _is_periodic(word, max_period)

    elif check_type == "complexity_of_word":
        word_str = fixture["word"]
        n = fixture["n"]
        expected = fixture["expected_p_n"]
        actual = _word_complexity(word_str, n)
        results["COMPLEXITY_OK"] = (actual == expected)

    else:
        results["UNKNOWN_CHECK"] = False

    return results


def self_test() -> bool:
    failures = []

    # --- Sturmian complexity p(n)=n+1 ---
    word = _sturmian_word(300)
    for n in range(1, 12):
        p = _complexity(word, n)
        if p != n + 1:
            failures.append(f"Complexity p({n})={p}, expected {n+1}")

    # --- Aperiodic ---
    if _is_periodic(word[:100], 50):
        failures.append("Sturmian word appears periodic up to period 50")

    # --- Gap sizes in {2,3} ---
    gaps = _gap_sizes(word)
    if not gaps.issubset({2, 3}):
        failures.append(f"Gap sizes not in {{2,3}}: {gaps}")

    # --- (RLLR)^k is periodic, NOT Sturmian ---
    rllr_word = [1 if c=='L' else 0 for c in 'RLLR' * 20]
    if not _is_periodic(rllr_word[:80], 10):
        failures.append("(RLLR)^20 not detected as periodic")
    rllr_p2 = _complexity(rllr_word[:30], 2)
    if rllr_p2 != 4:
        failures.append(f"(RLLR) p(2)={rllr_p2}, expected 4")
    # Sturmian needs p(2)=3; (RLLR) has p(2)=4 — not Sturmian
    if rllr_p2 == 3:
        failures.append("(RLLR) mistakenly has p(2)=3 (Sturmian condition)")

    # --- Pell e-values = CF denominators for alpha=[0;2,2,...] ---
    pell_e = [1, 2, 5, 12, 29, 70, 169, 408]
    q = [1, 2]
    while len(q) < len(pell_e):
        q.append(2 * q[-1] + q[-2])
    if pell_e != q[:len(pell_e)]:
        failures.append(f"Pell e != CF denominators: {pell_e} vs {q[:len(pell_e)]}")

    # --- Recurrence check ---
    for i in range(len(pell_e) - 2):
        if pell_e[i+2] != 2*pell_e[i+1] + pell_e[i]:
            failures.append(f"Recurrence failed at i={i}: {pell_e[i+2]} != 2*{pell_e[i+1]}+{pell_e[i]}")

    # --- s_alpha density approaches sqrt(2)-1 ---
    count1 = sum(word)
    density = count1 / len(word)
    expected_density = ALPHA
    if abs(density - expected_density) > 0.01:
        failures.append(f"Density {density:.4f} too far from alpha={expected_density:.4f}")

    # --- Fail fixture detection ---
    fail_cases = [
        {"check_type": "complexity_of_word",
         "word": "RLLRRLLRRLLRRLLRRLLRRLLRRLLRRLLR",
         "n": 2, "expected_p_n": 3, "expected": "FAIL"},
        {"check_type": "complexity",
         "n": 5, "expected_p_n": 7, "expected": "FAIL"},
    ]
    for case in fail_cases:
        checks = validate_fixture(case)
        if all(checks.values()):
            failures.append(f"Expected FAIL case passed: {case}")

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
    return len(failures) == 0


FAMILY_ID = 295
CERT_SLUG = "qa_pell_sturmian_bridge_cert_v1"


def validate_cert_family(cert_dir) -> Tuple[bool, List[str]]:
    import json
    from pathlib import Path

    errors: List[str] = []
    fixture_dir = Path(cert_dir) / "fixtures"
    if not fixture_dir.is_dir():
        errors.append("missing fixtures/ directory")
        return False, errors

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        if all_pass == expect_pass:
            pass_count += 1
        else:
            fail_count += 1
            errors.append(
                f"fixture {path.name}: expected={'PASS' if expect_pass else 'FAIL'} "
                f"got={'PASS' if all_pass else 'FAIL'} checks={checks}"
            )

    if fail_count:
        errors.append(f"{fail_count} fixture(s) had wrong outcome")
    return fail_count == 0, errors


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="QA Pell Sturmian Bridge Cert validator [295]"
    )
    parser.add_argument("cert_dir", nargs="?", default=str(Path(__file__).parent))
    parser.add_argument("--self-test", action="store_true", dest="selftest")
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir)
    fixture_dir = cert_dir / "fixtures"

    if args.selftest:
        st_ok = self_test()
        fam_ok, fam_errors = validate_cert_family(cert_dir)
        fix_files = list(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
        pass_files = [f for f in fix_files if "pass_" in f.name]
        fail_files = [f for f in fix_files if "fail_" in f.name]
        errors = ([] if st_ok else ["self_test FAIL"]) + fam_errors
        payload = {
            "ok": st_ok and fam_ok,
            "family_id": FAMILY_ID,
            "slug": CERT_SLUG,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": errors,
        }
        print(json.dumps(payload, sort_keys=True))
        sys.exit(0 if payload["ok"] else 1)

    if not self_test():
        print("SELF_TEST FAIL")
        sys.exit(1)
    print("SELF_TEST PASS")

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        ok = all_pass == expect_pass
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        print(f"{'PASS' if ok else 'FAIL'} {path.name}: {checks}")

    print(f"\nFixtures: {pass_count} PASS, {fail_count} FAIL")
    if fail_count:
        sys.exit(1)

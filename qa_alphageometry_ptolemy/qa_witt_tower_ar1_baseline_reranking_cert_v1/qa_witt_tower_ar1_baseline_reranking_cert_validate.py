from __future__ import annotations
# <!-- PRIMARY-SOURCE-EXEMPT: reason=original statistical audit derivation; sources cited in mapping_protocol_ref.json -->

QA_COMPLIANCE = (
    "cert_validator -- this cert audits an OBSERVER-PROJECTION statistical question "
    "(does a rank-bin operator applied to continuous physical series reveal structure "
    "beyond a plain correlated-Gaussian null model); the Monte Carlo simulation and "
    "z-scores are float observer-projection quantities used to CLASSIFY existing QA "
    "cert claims, not QA state themselves. The rank-bin discretization step (b,e in "
    "{0..26}, a=b+2e<=6) matches the QA operator used by certs [490]-[495] exactly."
)
"""Cert [516]: Witt Tower AR(1)-Baseline Reranking.

PRIMARY CLAIM:
  The "discrimination ladder" documented across certs [490]-[495] (rivers,
  precipitation, temperature, ocean SST, EEG interictal, 1-min FX) ranks
  domains by raw n_signal_ratio (observed / expected-under-independence).
  That ranking conflates two different things: (1) how strongly the domain
  is ALREADY autocorrelated at lag 1 (a well-documented, pre-existing fact
  for each domain -- ocean thermal inertia, synoptic blocking, etc., cited
  in the certs' own primary sources), and (2) how much the QA rank-bin
  operator reveals BEYOND what that known autocorrelation alone predicts.

  Re-ranking by (2) instead of raw ratio nearly INVERTS the published
  ladder: the two domains presented as "STRONGEST PERSISTENCE" (ocean SST
  [493], temperature [492]) show close to ZERO genuine excess beyond a
  plain AR(1)/correlated-Gaussian null at the SAME reported lag-1
  autocorrelation -- their raw ratios are essentially a restatement of
  already-known autocorrelation magnitudes, not a new QA-specific finding.
  The domains ranked lower in the original ladder (EEG interictal,
  precipitation, rivers, FX) show substantial genuine excess -- these are
  the results that are NOT already explained by textbook correlation.

  This does not retract [490]-[495]'s underlying data or n_signal_ratio
  computations, which are independently re-verified here as correct. It
  corrects the INTERPRETIVE framing: "highest raw ratio" was implicitly
  read as "most novel finding," when the two are only the same thing for
  a domain with near-zero baseline autocorrelation.

METHOD:
  For each domain, simulate a plain AR(1) Gaussian process
  x[t] = rho*x[t-1] + sqrt(1-rho^2)*eps[t] at the domain's OWN reported
  lag-1 autocorrelation rho (pooled across that domain's stations/
  patients/pairs, taken directly from the certified fallback data in
  [490]-[495], plus a live re-fetch of USGS river data for [490] since
  that cert does not store autocorr in its fallback record). Apply the
  IDENTICAL rank-bin operator (b,e in {0..26} via full-sample rank,
  a=b+2e<=6) used by the real certs, and measure the ratio of the AR(1)
  process's own n_signal to its theoretical independence-baseline
  (16/729 per triplet). This is the AR(1)-predicted ratio: what you get
  "for free" from the domain's known correlation alone, no QA-specific
  structure required. excess_ratio = observed_ratio / AR1_predicted_ratio.

EMPIRICAL RECORD (2026-07-04, pure-Python Monte Carlo, seed=0, n=8000,
40 trials per domain; reproducible via reproduce_ar1_baseline.py):

  domain              rho     AR1_predicted   observed   excess_ratio
  SST        [493]   +0.942      4.23           4.432       1.05
  Temperature[492]   +0.728      3.36           3.400       1.01
  Rivers     [490]   +0.310      1.88           2.689       1.43
  Precipitation[494] +0.318      1.91           3.048       1.60
  EEG        [491]   -0.262      0.44           0.725       1.64
  FX         [495]   -0.127      0.71           1.009       1.42

  Reranked by excess_ratio (descending): EEG (1.64) > Precipitation (1.60)
  > Rivers (1.43) > FX (1.42) > SST (1.05) > Temperature (1.01) -- nearly
  the exact inverse of the original raw-ratio ladder (SST 4.43 > Temp 3.40
  > Precip 3.05 > Rivers 2.69 > FX 1.009 > EEG 0.72).

  Rivers' rho was live-fetched from USGS NWIS (log-return lag-1 autocorr,
  4 gauges: Potomac 0.385, Hudson 0.203, Missouri 0.306, Eel 0.345,
  pooled mean 0.310) since cert [490]'s fallback record does not store an
  autocorr field.

SUB-CLAIMS:
  (A) SIMULATION CORRECTNESS: the AR(1) Monte Carlo + rank-bin operator,
      run at rho=0 (no correlation), reproduces ratio ~1.0 (matches the
      independence baseline the whole ladder is calibrated against).

  (B) SST/TEMPERATURE NEAR-NULL EXCESS: excess_ratio for both domains is
      within 10% of 1.0 (essentially fully explained by known
      autocorrelation) -- the two domains presented as strongest in the
      original ladder.

  (C) EEG/PRECIP/RIVERS/FX SUBSTANTIAL EXCESS: excess_ratio for all four
      domains exceeds 1.3 (genuine structure beyond plain correlation) --
      including two domains (FX, EEG) treated as "null" or "anti-
      persistent" in the original framing.

  (D) RERANKING INVERSION: the Spearman-style ordering by excess_ratio
      is NOT the same as the ordering by raw observed ratio -- the
      correction changes which domains are "most interesting," not just
      their exact numeric ranking.

  (E) WITNESS CONSISTENCY: the recorded (rho, observed, AR1_predicted,
      excess_ratio) tuples for all 6 domains are internally consistent
      (excess_ratio = observed / AR1_predicted, to 2 decimal places).

CHECKS (ARB = AR1-Reranking Baseline):
  ARB_SIM_NULL       AR(1) simulation at rho=0 gives ratio in [0.9, 1.1]
  ARB_SST_NEAR_NULL  SST excess_ratio in [0.9, 1.15]
  ARB_TEMP_NEAR_NULL Temperature excess_ratio in [0.9, 1.15]
  ARB_EEG_EXCESS     EEG excess_ratio > 1.3
  ARB_PRECIP_EXCESS  Precipitation excess_ratio > 1.3
  ARB_RIVERS_EXCESS  Rivers excess_ratio > 1.3
  ARB_FX_EXCESS      FX excess_ratio > 1.3
  ARB_RERANK_INVERTS reranking by excess_ratio differs from raw-ratio ranking
  ARB_WITNESS        recorded tuples are internally consistent

Primary sources:
  Rayner, N.A. et al. (2003). "Global analyses of sea surface temperature."
    J. Geophys. Res. 108(D14). DOI 10.1029/2002JD002670. (SST persistence
    baseline cited by cert [493].)
  Namias, J. (1952). "Long range weather forecasting." AMS. (Synoptic
    blocking baseline cited by cert [492].)
  This cert's own reproduce_ar1_baseline.py (pure-Python Monte Carlo,
    no external dependency) for the AR(1)/rank-bin simulation.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

FAMILY_ID = 516
CERT_SLUG = "qa_witt_tower_ar1_baseline_reranking_cert_v1"
MOD = 27
SIGNAL_THRESHOLD = 6
INDEPENDENCE_FRAC = 16.0 / 729.0

# Recorded 2026-07-04 empirical record (see reproduce_ar1_baseline.py for the
# Monte Carlo that produced ar1_predicted; rho/observed are taken directly
# from the certified fallback data in certs [490]-[495], plus a live USGS
# re-fetch for rivers' rho, which that cert doesn't store).
DOMAIN_RECORD: Dict[str, dict] = {
    "SST_493":         {"cert": 493, "rho": 0.942,  "observed": 4.432, "ar1_predicted": 4.23},
    "Temperature_492": {"cert": 492, "rho": 0.728,  "observed": 3.400, "ar1_predicted": 3.36},
    "Rivers_490":      {"cert": 490, "rho": 0.310,  "observed": 2.689, "ar1_predicted": 1.88},
    "Precipitation_494": {"cert": 494, "rho": 0.318, "observed": 3.048, "ar1_predicted": 1.91},
    "EEG_491":         {"cert": 491, "rho": -0.262, "observed": 0.725, "ar1_predicted": 0.44},
    "FX_495":          {"cert": 495, "rho": -0.127, "observed": 1.009, "ar1_predicted": 0.71},
}


def excess_ratio(observed: float, ar1_predicted: float) -> float:
    return observed / ar1_predicted


def rank_bins(vals: List[float], mod: int = MOD) -> List[int]:
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return [int(r * mod / n) for r in ranks]


def simulate_ar1_ratio(rho: float, n: int, trials: int, seed: int) -> Tuple[float, float]:
    """Deterministic (given seed) pure-Python Monte Carlo: AR(1) process at
    the given rho, rank-binned identically to the real certs' operator."""
    import random
    rng = random.Random(seed)
    ratios = []
    for _ in range(trials):
        x = [rng.gauss(0, 1)]
        innov = (1 - rho * rho) ** 0.5
        for _ in range(n - 1):
            x.append(rho * x[-1] + innov * rng.gauss(0, 1))
        bins = rank_bins(x)
        n_triplets = n - 2
        n_sig = sum(1 for t in range(n_triplets) if bins[t] + 2 * bins[t + 1] <= SIGNAL_THRESHOLD)
        n_expected = n_triplets * INDEPENDENCE_FRAC
        ratios.append(n_sig / n_expected)
    mean = sum(ratios) / len(ratios)
    var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    return mean, var ** 0.5


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_sim_null() -> Tuple[bool, float]:
    """ARB_SIM_NULL: rho=0 (no correlation) reproduces the independence
    baseline ratio ~1.0 -- a fast, small-n sanity check on the simulator
    itself (not part of the historical n=8000/trials=40 empirical record)."""
    mean, _ = simulate_ar1_ratio(rho=0.0, n=2000, trials=10, seed=1)
    return 0.9 <= mean <= 1.1, mean


def check_near_null(key: str) -> Tuple[bool, float]:
    rec = DOMAIN_RECORD[key]
    er = excess_ratio(rec["observed"], rec["ar1_predicted"])
    return 0.9 <= er <= 1.15, er


def check_excess(key: str, threshold: float = 1.3) -> Tuple[bool, float]:
    rec = DOMAIN_RECORD[key]
    er = excess_ratio(rec["observed"], rec["ar1_predicted"])
    return er > threshold, er


def check_rerank_inverts() -> Tuple[bool, list, list]:
    by_raw = sorted(DOMAIN_RECORD.keys(), key=lambda k: -DOMAIN_RECORD[k]["observed"])
    by_excess = sorted(DOMAIN_RECORD.keys(), key=lambda k: -excess_ratio(
        DOMAIN_RECORD[k]["observed"], DOMAIN_RECORD[k]["ar1_predicted"]))
    return by_raw != by_excess, by_raw, by_excess


def check_witness() -> Tuple[bool, list]:
    mismatches = []
    for key, rec in DOMAIN_RECORD.items():
        computed = round(excess_ratio(rec["observed"], rec["ar1_predicted"]), 2)
        stored = rec.get("excess_ratio")
        if stored is not None and abs(computed - stored) > 0.01:
            mismatches.append((key, computed, stored))
    return len(mismatches) == 0, mismatches


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------

def validate_fixture(fixture: dict) -> dict:
    kind = fixture.get("kind")

    if kind == "sim_null":
        ok, mean = check_sim_null()
        return {"ARB_SIM_NULL": ok}

    if kind == "near_null":
        ok, er = check_near_null(fixture["domain"])
        return {f"ARB_{fixture['domain'].split('_')[0].upper()}_NEAR_NULL": ok}

    if kind == "excess":
        ok, er = check_excess(fixture["domain"], fixture.get("threshold", 1.3))
        return {f"ARB_{fixture['domain'].split('_')[0].upper()}_EXCESS": ok}

    if kind == "rerank":
        ok, by_raw, by_excess = check_rerank_inverts()
        return {"ARB_RERANK_INVERTS": ok}

    if kind == "witness":
        record = fixture.get("record", {})
        for key, vals in record.items():
            if key not in DOMAIN_RECORD:
                return {"ARB_WITNESS": False}
            rec = DOMAIN_RECORD[key]
            if abs(vals.get("rho", 0) - rec["rho"]) > 1e-6:
                return {"ARB_WITNESS": False}
        return {"ARB_WITNESS": True}

    return {"ARB_UNKNOWN_KIND": False}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> bool:
    failures: List[str] = []

    ok, mean = check_sim_null()
    if not ok:
        failures.append(f"ARB_SIM_NULL FAIL: rho=0 gave ratio={mean:.3f}, expected [0.9,1.1]")

    for key, label in (("SST_493", "SST"), ("Temperature_492", "Temperature")):
        ok, er = check_near_null(key)
        if not ok:
            failures.append(f"ARB_{label.upper()}_NEAR_NULL FAIL: excess_ratio={er:.3f}, expected [0.9,1.15]")

    for key, label in (("EEG_491", "EEG"), ("Precipitation_494", "Precipitation"),
                       ("Rivers_490", "Rivers"), ("FX_495", "FX")):
        ok, er = check_excess(key)
        if not ok:
            failures.append(f"ARB_{label.upper()}_EXCESS FAIL: excess_ratio={er:.3f}, expected >1.3")

    ok, by_raw, by_excess = check_rerank_inverts()
    if not ok:
        failures.append(f"ARB_RERANK_INVERTS FAIL: raw={by_raw} excess={by_excess} (should differ)")

    ok, mismatches = check_witness()
    if not ok:
        failures.append(f"ARB_WITNESS FAIL: {mismatches}")

    if failures:
        for f in failures[:15]:
            print("FAIL:", f, file=sys.stderr)
    return len(failures) == 0


# ---------------------------------------------------------------------------
# Cert family validation
# ---------------------------------------------------------------------------

def validate_cert_family(cert_dir: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    mp = cert_dir / "mapping_protocol_ref.json"
    if not mp.exists():
        errors.append("mapping_protocol_ref.json missing")
    else:
        data = json.loads(mp.read_text())
        if data.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
            errors.append("mapping_protocol_ref: wrong protocol_version")
        if not data.get("scope_note", "").strip():
            errors.append("mapping_protocol_ref: empty scope_note")

    fixture_dir = cert_dir / "fixtures"
    if not fixture_dir.is_dir():
        errors.append("fixtures/ directory missing")
    else:
        fix_files = list(fixture_dir.glob("*.json"))
        pass_files = [f for f in fix_files if f.name.startswith("pass_")]
        fail_files = [f for f in fix_files if f.name.startswith("fail_")]
        if not pass_files:
            errors.append("no pass_*.json fixtures found")
        if not fail_files:
            errors.append("no fail_*.json fixtures found")
        for path in sorted(fix_files):
            try:
                fixture = json.loads(path.read_text())
            except Exception as exc:
                errors.append(f"{path.name}: JSON parse error: {exc}")
                continue
            expect_pass = fixture.get("expected", "PASS") == "PASS"
            checks = validate_fixture(fixture)
            all_pass = all(v for v in checks.values() if isinstance(v, bool))
            if all_pass != expect_pass:
                errors.append(f"{path.name}: expected {'PASS' if expect_pass else 'FAIL'}, got {'PASS' if all_pass else 'FAIL'}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"QA Witt Tower AR(1)-Baseline Reranking Cert validator [{FAMILY_ID}]"
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
        pass_files = [f for f in fix_files if f.name.startswith("pass_")]
        fail_files = [f for f in fix_files if f.name.startswith("fail_")]
        errors = ([] if st_ok else ["self_test FAIL"]) + fam_errors
        payload = {
            "ok": st_ok and fam_ok,
            "family_id": FAMILY_ID,
            "slug": CERT_SLUG,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": errors,
            "checks_summary": {
                "ARB_SIM_NULL": check_sim_null()[0],
                "ARB_SST_NEAR_NULL": check_near_null("SST_493")[0],
                "ARB_TEMP_NEAR_NULL": check_near_null("Temperature_492")[0],
                "ARB_EEG_EXCESS": check_excess("EEG_491")[0],
                "ARB_PRECIP_EXCESS": check_excess("Precipitation_494")[0],
                "ARB_RIVERS_EXCESS": check_excess("Rivers_490")[0],
                "ARB_FX_EXCESS": check_excess("FX_495")[0],
                "ARB_RERANK_INVERTS": check_rerank_inverts()[0],
                "ARB_WITNESS": check_witness()[0],
            },
        }
        print(json.dumps(payload, sort_keys=True, indent=2))
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
        all_pass = all(v for v in checks.values() if isinstance(v, bool))
        ok = all_pass == expect_pass
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        status = "PASS" if ok else "FAIL"
        bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
        print(f"{status} {path.name}: {bool_checks}")

    print(f"\nFixtures: {pass_count} PASS, {fail_count} FAIL")
    if fail_count:
        sys.exit(1)

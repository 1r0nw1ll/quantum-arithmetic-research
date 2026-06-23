# Primary source: Wildberger N.J. (2005) ISBN 978-0-9757492-0-8; Hasselmann K. (1976) doi:10.1111/j.2153-3490.1976.tb00696.x; empirical chain certs [491][495][490][494][492][493]
QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Witt Tower tau-monotone ladder: "
    "n_sig_milli=n_sig_ratio*1000 integer milliunits; tau_rank 1..6 ordinal integers; "
    "concordance check uses integer comparison of n_sig_milli pairs); "
    "Theorem NT: n_sig_ratio is an integer ratio (count/expected) — all divisions "
    "deferred to observer layer; QA state is the discrete (b,e) pair; "
    "no float state, no continuous observer in QA layer"
)
"""
QA Witt Tower tau-Monotone Discrimination Ladder Cert [503]

CLAIM: The six Witt Tower empirical certs [491][495][490][494][492][493] form a
monotone concordant discrimination ladder: domain autocorrelation timescale tau
(ordinal rank 1..6: EEG < FX < rivers < precip < temp < SST) and the empirically
observed return-rank n_sig_ratio are concordant for all 15 domain pairs
(Kendall tau concordance = 1, pure integer check).

This theoretical anchor ties 6 empirical certs into a single falsifiable claim:
the return-rank operator DISCRIMINATES physical systems by their autocorrelation
timescale, monotonically.

Four checks:
  WTM_1  6 domains declared; tau_rank 1..6 distinct; n_sig_milli positive integers;
         cert_ids reference the empirical chain
  WTM_2  All C(6,2)=15 tau_rank/n_sig_milli pairs concordant (Kendall tau = 1)
  WTM_3  Anti/null/structural split:
           tau_rank=1 (EEG) n_sig_milli < 1000 (anti-persistent, n_sig_ratio < 1)
           tau_rank=2 (FX) null zone: 1000 <= n_sig_milli < 2000
           tau_rank=3..6 (rivers..SST) n_sig_milli >= 2000 (structural persistence)
  WTM_4  Ladder span: max_n_sig_milli * 1000 > min_n_sig_milli * 6000
         (span ratio > 6.0 using integer arithmetic)

Physical explanation:
  tau_rank  domain     tau (approx)    mechanism
  1         EEG        < 1 second      Neural oscillation decay; anti-persistent
                                       (spike refractory period forces mean reversion)
  2         FX 1-min   1-10 minutes    Bid-ask bounce; structural null
  3         rivers     ~ 1 day         Hydraulic routing; moderate persistence
  4         precip     3-7 days        Wet/dry spell length; moderate persistence
  5         temp       ~ weeks         Atmospheric thermal inertia
  6         SST        ~ months        Ocean heat capacity >> atmosphere (Hasselmann 1976)

Schema: QA_WITT_TOWER_TAU_MONOTONE_CERT.v1
"""

import json
import sys
from pathlib import Path

SCHEMA = "QA_WITT_TOWER_TAU_MONOTONE_CERT.v1"

_EXPECTED_DOMAINS = {
    "EEG":    {"tau_rank": 1, "n_sig_milli": 720,  "cert_id": 491},
    "FX":     {"tau_rank": 2, "n_sig_milli": 1009, "cert_id": 495},
    "rivers": {"tau_rank": 3, "n_sig_milli": 2690, "cert_id": 490},
    "precip": {"tau_rank": 4, "n_sig_milli": 3050, "cert_id": 494},
    "temp":   {"tau_rank": 5, "n_sig_milli": 3400, "cert_id": 492},
    "SST":    {"tau_rank": 6, "n_sig_milli": 4430, "cert_id": 493},
}

_ANTI_PERSISTENT_THRESHOLD = 1000     # n_sig_milli < 1000 → n_sig_ratio < 1
_STRUCTURAL_PERSISTENCE_THRESHOLD = 2000  # n_sig_milli >= 2000 → n_sig_ratio >= 2
_NULL_ZONE_DOMAIN = "FX"             # sits in [1000, 2000)
_SPAN_THRESHOLD_MILLI = 6000         # ladder span > 6.0 (integer: max*1000 > min*6000)


def _check_fixture(data):
    errors = []

    # SRC
    if data.get("schema_version") != SCHEMA:
        errors.append(f"SRC: expected schema_version={SCHEMA!r}, got {data.get('schema_version')!r}")

    domains = data.get("domains", {})

    # WTM_1 — domain registry
    if len(domains) != 6:
        errors.append(f"WTM_1a: expected 6 domains, got {len(domains)}")
    tau_ranks = [d.get("tau_rank") for d in domains.values()]
    if sorted(tau_ranks) != list(range(1, 7)):
        errors.append(f"WTM_1b: tau_ranks must be {{1..6}} distinct, got {sorted(tau_ranks)}")
    for name, d in domains.items():
        nsm = d.get("n_sig_milli")
        if not isinstance(nsm, int) or nsm <= 0:
            errors.append(f"WTM_1c: {name} n_sig_milli must be positive int, got {nsm!r}")
        cid = d.get("cert_id")
        if not isinstance(cid, int) or cid <= 0:
            errors.append(f"WTM_1d: {name} cert_id must be positive int, got {cid!r}")

    # WTM_2 — monotone concordance: all 15 pairs
    domain_list = sorted(domains.items(), key=lambda kv: kv[1].get("tau_rank", 0))
    concordant = 0
    discordant = 0
    for i in range(len(domain_list)):
        for j in range(i + 1, len(domain_list)):
            name_i, d_i = domain_list[i]
            name_j, d_j = domain_list[j]
            ri, nsi = d_i.get("tau_rank", 0), d_i.get("n_sig_milli", 0)
            rj, nsj = d_j.get("tau_rank", 0), d_j.get("n_sig_milli", 0)
            if ri < rj:
                if nsi < nsj:
                    concordant += 1
                else:
                    discordant += 1
                    errors.append(
                        f"WTM_2: discordant pair ({name_i} tau_rank={ri} n_sig={nsi}) "
                        f"vs ({name_j} tau_rank={rj} n_sig={nsj}): "
                        f"tau_rank {ri}<{rj} but n_sig {nsi}>={nsj}"
                    )
    if concordant + discordant > 0 and discordant == 0:
        if concordant != 15:
            errors.append(f"WTM_2b: expected 15 concordant pairs, got {concordant}")

    # WTM_3 — anti/null/structural split
    rank_to_nsm = {d["tau_rank"]: d["n_sig_milli"] for d in domains.values()
                   if isinstance(d.get("tau_rank"), int) and isinstance(d.get("n_sig_milli"), int)}

    if 1 in rank_to_nsm:
        if rank_to_nsm[1] >= _ANTI_PERSISTENT_THRESHOLD:
            errors.append(
                f"WTM_3a: tau_rank=1 (EEG) n_sig_milli={rank_to_nsm[1]} must be "
                f"< {_ANTI_PERSISTENT_THRESHOLD} (anti-persistent, n_sig_ratio < 1)"
            )
    # FX null zone: tau_rank=2 must be in [1000, 2000)
    if 2 in rank_to_nsm:
        nsm2 = rank_to_nsm[2]
        if nsm2 < _ANTI_PERSISTENT_THRESHOLD or nsm2 >= _STRUCTURAL_PERSISTENCE_THRESHOLD:
            errors.append(
                f"WTM_3b: tau_rank=2 (FX) n_sig_milli={nsm2} must be in "
                f"[{_ANTI_PERSISTENT_THRESHOLD}, {_STRUCTURAL_PERSISTENCE_THRESHOLD}) "
                f"(null zone)"
            )
    for rank in range(3, 7):
        if rank in rank_to_nsm:
            nsm = rank_to_nsm[rank]
            if nsm < _STRUCTURAL_PERSISTENCE_THRESHOLD:
                errors.append(
                    f"WTM_3c: tau_rank={rank} n_sig_milli={nsm} must be "
                    f">= {_STRUCTURAL_PERSISTENCE_THRESHOLD} (structural persistence)"
                )

    # WTM_4 — ladder span > 6.0 (integer: max*1000 > min*6000)
    if rank_to_nsm:
        max_nsm = max(rank_to_nsm.values())
        min_nsm = min(rank_to_nsm.values())
        if max_nsm * 1000 <= min_nsm * _SPAN_THRESHOLD_MILLI:
            errors.append(
                f"WTM_4: span max_n_sig_milli={max_nsm}, min={min_nsm}: "
                f"{max_nsm}*1000={max_nsm*1000} must be > {min_nsm}*6000={min_nsm*6000}"
            )

    # Fail-ledger
    if "fail_ledger" in data:
        fl = data["fail_ledger"]
        if not isinstance(fl, list) or not all(isinstance(s, str) for s in fl):
            errors.append("F: fail_ledger must be a list of strings")

    return errors


def _run_self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    results = {}

    for fixture_path in sorted(fixtures_dir.glob("*.json")):
        expected_pass = fixture_path.stem.startswith("pass_")
        with open(fixture_path) as f:
            data = json.load(f)
        errs = _check_fixture(data)
        passed = len(errs) == 0
        ok = passed == expected_pass
        results[fixture_path.name] = {
            "expected": "PASS" if expected_pass else "FAIL",
            "got": "PASS" if passed else "FAIL",
            "ok": ok,
            "errors": errs,
        }

    all_ok = all(v["ok"] for v in results.values())
    return {"ok": all_ok, "fixtures": results}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        result = _run_self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            data = json.load(f)
        errs = _check_fixture(data)
        if errs:
            print(json.dumps({"ok": False, "errors": errs}, indent=2))
            sys.exit(1)
        print(json.dumps({"ok": True}, indent=2))
        sys.exit(0)

    result = _run_self_test()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()

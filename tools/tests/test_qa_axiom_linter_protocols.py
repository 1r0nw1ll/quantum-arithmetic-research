#!/usr/bin/env python3
"""
Regression tests for qa_axiom_linter protocol gates and protocol validators.

Run from repo root:
    python tools/tests/test_qa_axiom_linter_protocols.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LINTER = REPO_ROOT / "tools" / "qa_axiom_linter.py"
EXP_VALIDATOR = REPO_ROOT / "qa_experiment_protocol" / "validator.py"
BENCH_VALIDATOR = REPO_ROOT / "qa_benchmark_protocol" / "validator.py"
SUPPRESSION = "#" + " noqa"


def _qa_header() -> str:
    name = "QA_" + "COMPLIANCE"
    return (
        f"{name} = {{\n"
        "    'observer': 'linter_regression_test',\n"
        "    'state_alphabet': '(b,e) test fixture',\n"
        "}\n"
    )


def _run_linter(body: str) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", prefix="lint_fixture_", delete=False) as f:
        f.write(body)
        path = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(LINTER), str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr
    finally:
        path.unlink(missing_ok=True)


def _run_linter_path(path: Path) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(LINTER), str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def _run_validator(validator: Path, payload: dict) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="protocol_fixture_", delete=False) as f:
        json.dump(payload, f)
        path = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(validator), str(path), "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr
    finally:
        path.unlink(missing_ok=True)


def assert_rule(body: str, rule_id: str, context: str) -> None:
    code, out = _run_linter(body)
    assert code != 0, f"{context}: expected nonzero exit\n{out}"
    assert rule_id in out, f"{context}: expected {rule_id}\n{out}"


def _valid_protocol_payload(kind: str, ledger_name: str) -> dict:
    if kind == "experiment":
        payload = json.loads((REPO_ROOT / "qa_experiment_protocol" / "fixtures" / "valid_min.json").read_text())
    elif kind == "benchmark":
        payload = json.loads((REPO_ROOT / "qa_benchmark_protocol" / "fixtures" / "valid_min.json").read_text())
    else:
        raise ValueError(kind)
    payload["source_mapping"]["theory_doc"] = str(REPO_ROOT / "EXPERIMENT_AXIOMS_BLOCK.md")
    payload["reproducibility"]["results_ledger"] = ledger_name
    return payload


def test_undismissable_rules() -> None:
    assert_rule(
        _qa_header() + f"for _k in range(0, 9):\n    pass  {SUPPRESSION}: A1-2\n",
        "A1-2",
        "A1-2 suppression must be ignored",
    )
    assert_rule(
        f"{SUPPRESSION}: EXP-1\n" + _qa_header()
        + "from scipy.stats import chi2_contingency\n"
        + "result = chi2_contingency([[1, 2], [3, 4]])\n",
        "EXP-1",
        "EXP-1 suppression must be ignored",
    )
    assert_rule(
        _qa_header()
        + "from qa_elements import qa_elements\n"
        + f"def compute_C(b, e, m):  {SUPPRESSION}: ELEM-2\n"
        + "    return 2 * (b + e) * e\n",
        "ELEM-2",
        "ELEM-2 suppression must be ignored",
    )


def test_exp_bench_indicators() -> None:
    assert_rule(
        _qa_header()
        + "from scipy.stats import chi2_contingency\n"
        + "result = chi2_contingency([[1, 2], [3, 4]])\n",
        "EXP-1",
        "chi2_contingency should require an experiment protocol",
    )
    assert_rule(
        _qa_header()
        + "import sklearn.ensemble as ske\n"
        + "score = roc_auc_score(y_true, y_score)\n",
        "BENCH-1",
        "aliased sklearn import with a metric should require a benchmark protocol",
    )
    assert_rule(
        _qa_header()
        + "baselines = ['qa', 'sklearn']\n",
        "BENCH-1",
        "baseline structure should require a benchmark protocol without an import",
    )


def test_t2_b_4_signal_variants() -> None:
    code, out = _run_linter(
        _qa_header()
        + "b = signal + offset\n"
        + "e = continuous_value - 1\n"
        + "E = signal_mean\n"
    )
    assert code != 0, f"signal injection variants should fail\n{out}"
    assert out.count("[T2-b-4]") == 3, f"expected three T2-b-4 hits\n{out}"


def test_protocol_sibling_must_validate() -> None:
    with tempfile.TemporaryDirectory(prefix="protocol_sibling_") as d:
        root = Path(d)
        exp_script = root / "exp_script.py"
        exp_script.write_text(
            _qa_header()
            + "from scipy.stats import chi2_contingency\n"
            + "result = chi2_contingency([[1, 2], [3, 4]])\n",
            encoding="utf-8",
        )
        (root / "experiment_protocol.json").write_text("{}", encoding="utf-8")
        exp_code, exp_out = _run_linter_path(exp_script)
        assert exp_code != 0 and "EXP-1" in exp_out, exp_out

    with tempfile.TemporaryDirectory(prefix="protocol_sibling_") as d:
        root = Path(d)
        bench_script = root / "bench_script.py"
        bench_script.write_text(
            _qa_header()
            + "baselines = ['qa', 'sklearn']\n",
            encoding="utf-8",
        )
        (root / "benchmark_protocol.json").write_text("{}", encoding="utf-8")
        bench_code, bench_out = _run_linter_path(bench_script)
        assert bench_code != 0 and "BENCH-1" in bench_out, bench_out


def test_explicit_protocol_runtime_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="protocol_runtime_") as d:
        root = Path(d)
        (root / "experiment_protocol.json").write_text(
            json.dumps(_valid_protocol_payload("experiment", "runs.jsonl")),
            encoding="utf-8",
        )

        good = root / "good_exp.py"
        good.write_text(
            _qa_header()
            + "EXPERIMENT_PROTOCOL_REF = 'experiment_protocol.json'\n"
            + "from qa_reproducibility import log_run\n"
            + "from scipy.stats import chi2_contingency\n"
            + "def run_ablation():\n"
            + "    return {'ok': True}\n"
            + "def main():\n"
            + "    result = chi2_contingency([[1, 2], [3, 4]])\n"
            + "    run_ablation()\n"
            + "    log_run(EXPERIMENT_PROTOCOL_REF, status='complete', results={'stat': str(result.statistic)})\n"
            + "if __name__ == '__main__':\n"
            + "    main()\n",
            encoding="utf-8",
        )
        good_code, good_out = _run_linter_path(good)
        assert good_code == 0, good_out

        good_runtime_module = root / "good_runtime_module_exp.py"
        good_runtime_module.write_text(
            _qa_header()
            + "EXPERIMENT_PROTOCOL_REF = 'experiment_protocol.json'\n"
            + "from qa_reproducibility import runtime as qrt\n"
            + "from scipy.stats import chi2_contingency\n"
            + "def run_ablation():\n"
            + "    return {'ok': True}\n"
            + "def main():\n"
            + "    result = chi2_contingency([[1, 2], [3, 4]])\n"
            + "    run_ablation()\n"
            + "    qrt.log_run(EXPERIMENT_PROTOCOL_REF, status='complete', results={'stat': str(result.statistic)})\n"
            + "if __name__ == '__main__':\n"
            + "    main()\n",
            encoding="utf-8",
        )
        good_module_code, good_module_out = _run_linter_path(good_runtime_module)
        assert good_module_code == 0, good_module_out

        missing_ablation = root / "missing_ablation.py"
        missing_ablation.write_text(
            _qa_header()
            + "EXPERIMENT_PROTOCOL_REF = 'experiment_protocol.json'\n"
            + "from qa_reproducibility import log_run\n"
            + "from scipy.stats import chi2_contingency\n"
            + "def main():\n"
            + "    chi2_contingency([[1, 2], [3, 4]])\n"
            + "    log_run(EXPERIMENT_PROTOCOL_REF)\n"
            + "if __name__ == '__main__':\n"
            + "    main()\n",
            encoding="utf-8",
        )
        abl_code, abl_out = _run_linter_path(missing_ablation)
        assert abl_code != 0 and "EXP-ABLATION" in abl_out, abl_out

        missing_runtime = root / "missing_runtime.py"
        missing_runtime.write_text(
            _qa_header()
            + "EXPERIMENT_PROTOCOL_REF = 'experiment_protocol.json'\n"
            + "from scipy.stats import chi2_contingency\n"
            + "def run_ablation():\n"
            + "    return {'ok': True}\n"
            + "def main():\n"
            + "    chi2_contingency([[1, 2], [3, 4]])\n"
            + "    run_ablation()\n"
            + "if __name__ == '__main__':\n"
            + "    main()\n",
            encoding="utf-8",
        )
        rt_code, rt_out = _run_linter_path(missing_runtime)
        assert rt_code != 0 and "EXP-RUNTIME" in rt_out, rt_out

        fake_runtime = root / "fake_runtime.py"
        fake_runtime.write_text(
            _qa_header()
            + "EXPERIMENT_PROTOCOL_REF = 'experiment_protocol.json'\n"
            + "from scipy.stats import chi2_contingency\n"
            + "def log_run(*args, **kwargs):\n"
            + "    return None\n"
            + "def run_ablation():\n"
            + "    return {'ok': True}\n"
            + "def main():\n"
            + "    chi2_contingency([[1, 2], [3, 4]])\n"
            + "    run_ablation()\n"
            + "    log_run(EXPERIMENT_PROTOCOL_REF)\n"
            + "if __name__ == '__main__':\n"
            + "    main()\n",
            encoding="utf-8",
        )
        fake_code, fake_out = _run_linter_path(fake_runtime)
        assert fake_code != 0 and "EXP-RUNTIME" in fake_out, fake_out


def test_s2_1_boundaries() -> None:
    clean_code, clean_out = _run_linter(
        _qa_header()
        + "probe = np.zeros(10)\n"
        + "cache = np.ones(10)\n"
        + "prob = np.random.default_rng(7)\n"
    )
    assert clean_code == 0, f"boundary identifiers should not trigger S2-1\n{clean_out}"

    code, out = _run_linter(
        _qa_header()
        + "b = np.random.default_rng(7)\n"
        + "e = np.random.default_rng(8).normal(size=3)\n"
        + "b = np.asarray([1.0], dtype=float)\n"
    )
    assert code != 0, f"state float sources should fail\n{out}"
    assert out.count("[S2-1]") == 3, f"expected three S2-1 hits\n{out}"


def test_elem_2_strict() -> None:
    assert_rule(
        _qa_header()
        + "from qa_elements import qa_elements\n"
        + "def compute_C(b, e, m):\n"
        + "    return 2 * (b + e) * e\n",
        "ELEM-2",
        "canonical import plus local compute_C redefinition should fail",
    )


def test_schema_strict() -> None:
    exp_payload = {
        "protocol_version": "QA_EXPERIMENT_PROTOCOL.v1",
        "experiment_id": "bad_exp",
        "hypothesis": "",
        "null_model": {
            "description": "null",
            "independence_argument": "independent",
        },
        "pre_registration": {
            "seed": 0,
            "date_utc": "not-a-date",
            "n_trials": 1,
        },
        "decision_rules": {
            "accept_criterion": "accept",
            "reject_criterion": "reject",
            "on_unsupportive": "investigate_observer",
        },
        "observer_projection": {
            "description": "observer",
            "state_alphabet": "(b,e)",
        },
        "real_data_status": "pending",
    }
    exp_code, exp_out = _run_validator(EXP_VALIDATOR, exp_payload)
    assert exp_code != 0, f"empty hypothesis and bad date must fail\n{exp_out}"

    missing_data_payload = dict(exp_payload)
    missing_data_payload["hypothesis"] = "valid hypothesis"
    missing_data_payload["pre_registration"] = dict(exp_payload["pre_registration"])
    missing_data_payload["pre_registration"]["date_utc"] = "2026-04-13T00:00:00Z"
    missing_data_payload["real_data_status"] = "missing/no_such_real_data.csv"
    path_code, path_out = _run_validator(EXP_VALIDATOR, missing_data_payload)
    assert path_code != 0, f"real_data_status path must exist when not sentinel\n{path_out}"

    bench_payload = {
        "protocol_version": "QA_BENCHMARK_PROTOCOL.v1",
        "benchmark_id": "bad_bench",
        "qa_method": {
            "name": "qa_method",
            "description": "desc",
            "observer_projection": "projection",
        },
        "baselines": [
            {"name": "DummyClassifier", "implementation_ref": "sklearn.dummy.DummyClassifier"}
        ],
        "datasets": [
            {"name": "synthetic", "source": "synthetic_generator"}
        ],
        "parity_contract": {
            "same_seed_all_methods": True,
            "same_data_split": True,
            "same_preprocessing": True,
        },
        "calibration_provenance": {
            "learned_on": "training",
            "procedure": "none",
            "domain_of_origin": "synthetic",
        },
        "framework_inheritance": {
            "mode": "ported"
        },
        "metrics": [],
    }
    bench_code, bench_out = _run_validator(BENCH_VALIDATOR, bench_payload)
    assert bench_code != 0, f"empty metrics and missing prior_cert must fail\n{bench_out}"

    bad_ref_payload = dict(bench_payload)
    bad_ref_payload["framework_inheritance"] = {
        "mode": "ported",
        "prior_cert": "cert [154]",
    }
    bad_ref_payload["metrics"] = ["AUROC"]
    bad_ref_payload["baselines"] = [
        {"name": "Missing", "implementation_ref": "not_importable_package.Model"}
    ]
    ref_code, ref_out = _run_validator(BENCH_VALIDATOR, bad_ref_payload)
    assert ref_code != 0, f"implementation_ref top-level package must import\n{ref_out}"


def test_meta_validator_json_contract() -> None:
    for validator in (EXP_VALIDATOR, BENCH_VALIDATOR):
        result = subprocess.run(
            [sys.executable, str(validator), "--self-test", "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload.get("ok") is True, result.stdout


TESTS = [
    test_undismissable_rules,
    test_exp_bench_indicators,
    test_t2_b_4_signal_variants,
    test_protocol_sibling_must_validate,
    test_explicit_protocol_runtime_contract,
    test_s2_1_boundaries,
    test_elem_2_strict,
    test_schema_strict,
    test_meta_validator_json_contract,
]


def main() -> int:
    passed = 0
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
        else:
            passed += 1
            print(f"[PASS] {fn.__name__}")
    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
external_validation_prompt_injection_multibench.py

Run real prompt-injection benchmarks through QA guardrail and report metrics for
both targets explicitly:
1) Injection presence detection
2) Attack success prevention (where labels exist)
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qa_guardrail.qa_guardrail import guard, GuardrailContext  # noqa: E402
from qa_guardrail.threat_scanner import (  # noqa: E402
    SCANNER_ID,
    SCANNER_VERSION,
    MALICIOUS_PATTERNS,
    MALFORMED_PATTERNS,
    ADVERSARIAL_PATTERNS,
    get_current_patterns_hash,
)


OUTPUT_DIR = SCRIPT_DIR / "external_validation_certs"
SCRIPT_VERSION = "1.1.0"

# Sources
TENSORTRUST_DETECTION_URL = (
    "https://huggingface.co/datasets/qxcv/tensor-trust/raw/main/"
    "detecting-extractions/v1/prompt_extraction_detection.jsonl"
)
TENSORTRUST_V1_ATTACKS_URL = (
    "https://huggingface.co/datasets/qxcv/tensor-trust/resolve/main/"
    "raw-data/v1/raw_dump_attacks.jsonl.bz2?download=true"
)
TENSORTRUST_V1_DEFENSES_URL = (
    "https://huggingface.co/datasets/qxcv/tensor-trust/resolve/main/"
    "raw-data/v1/raw_dump_defenses.jsonl.bz2?download=true"
)
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_SPLITS_API = "https://datasets-server.huggingface.co/splits"
DEEPSET_SUBSET_PATH = SCRIPT_DIR / "external_validation_data" / "prompt_injection_benchmark_subset.jsonl"

# Baseline detector intentionally simple for "caught-where-baseline-missed" analysis.
BASELINE_PATTERN_SPECS = [
    ("ignore_previous", r"\bignore\b.{0,40}\b(previous|prior|above|all)\b"),
    ("disregard_instruction", r"\bdisregard\b.{0,20}\b(instruction|prompt|rules?)\b"),
    ("forget_rules", r"\bforget\b.{0,30}\b(previous|instructions?|rules?)\b"),
    ("reveal_secret", r"\breveal\b.{0,30}\b(password|secret|system prompt)\b"),
    ("role_takeover", r"\b(i want you to act as|you are now|pretend you are)\b"),
]
BASELINE_PATTERNS = [(name, re.compile(pattern)) for name, pattern in BASELINE_PATTERN_SPECS]

# Strict negative policy for TensorTrust defenses.
STRICT_NEGATIVE_CUE_RE = re.compile(
    r"\b("
    r"ignore|disregard|forget|jailbreak|developer mode|system prompt|reveal|"
    r"act as|you are now|pretend you are|roleplay|override|bypass"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class BenchCase:
    benchmark_id: str
    source_dataset: str
    source_url: str
    split: str
    case_id: str
    prompt_text: str
    expected_injection_present: Optional[bool]
    expected_attack_success: Optional[bool]
    metadata: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _http_get_text(url: str) -> str:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.text


def _download_file(url: str, out_path: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def fetch_hf_splits(dataset: str) -> List[str]:
    payload = _http_get_json(HF_SPLITS_API, params={"dataset": dataset})
    splits = [str(item["split"]) for item in payload.get("splits", [])]
    return sorted(set(splits))


def fetch_hf_rows(dataset: str, split: str, config: str = "default") -> Iterable[Dict[str, Any]]:
    offset = 0
    page_size = 100
    total = None
    while True:
        payload = _http_get_json(
            HF_ROWS_API,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            },
        )
        rows = payload.get("rows", [])
        if total is None:
            total = int(payload.get("num_rows_total", 0))
        for item in rows:
            yield item.get("row", {})
        offset += len(rows)
        if not rows or offset >= total:
            break


def load_tensortrust_detection(max_cases: Optional[int]) -> List[BenchCase]:
    text = _http_get_text(TENSORTRUST_DETECTION_URL)
    cases: List[BenchCase] = []
    for idx, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_text = str(row.get("llm_output", "")).strip()
        if not prompt_text:
            continue
        cases.append(
            BenchCase(
                benchmark_id="tensortrust_prompt_extraction_detection",
                source_dataset="qxcv/tensor-trust",
                source_url=TENSORTRUST_DETECTION_URL,
                split="v1",
                case_id=f"tt-detect-{row.get('sample_id', idx)}",
                prompt_text=prompt_text,
                expected_injection_present=bool(row.get("is_prompt_extraction", False)),
                expected_attack_success=None,
                metadata={
                    "sample_id": row.get("sample_id"),
                    "access_code_sha256": sha256_text(str(row.get("access_code", ""))),
                },
            )
        )
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases


def load_tensortrust_v1_attacks(max_cases: Optional[int], cache_path: Optional[Path]) -> List[BenchCase]:
    if cache_path is None:
        cache_path = OUTPUT_DIR / "tmp_tensortrust_v1_attacks.jsonl.bz2"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        _download_file(TENSORTRUST_V1_ATTACKS_URL, cache_path)

    cases: List[BenchCase] = []
    with bz2.open(cache_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_text = str(row.get("attacker_input", "")).strip()
            if not prompt_text:
                continue
            cases.append(
                BenchCase(
                    benchmark_id="tensortrust_v1_attacks",
                    source_dataset="qxcv/tensor-trust",
                    source_url=TENSORTRUST_V1_ATTACKS_URL,
                    split="raw-data/v1",
                    case_id=f"tt-v1-attack-{row.get('attack_id', idx)}",
                    prompt_text=prompt_text,
                    expected_injection_present=True,
                    expected_attack_success=bool(row.get("output_is_access_granted", False)),
                    metadata={
                        "attack_id": row.get("attack_id"),
                        "llm_choice": row.get("llm_choice"),
                        "is_self_attack": bool(row.get("is_self_attack", False)),
                    },
                )
            )
            if max_cases is not None and len(cases) >= max_cases:
                break
    return cases


def load_tensortrust_v1_defenses(
    max_cases: Optional[int],
    cache_path: Optional[Path],
    negative_policy: str,
) -> List[BenchCase]:
    if cache_path is None:
        cache_path = OUTPUT_DIR / "tmp_tensortrust_v1_defenses.jsonl.bz2"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        _download_file(TENSORTRUST_V1_DEFENSES_URL, cache_path)

    cases: List[BenchCase] = []
    with bz2.open(cache_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_text = str(row.get("opening_defense", "")).strip()
            if not prompt_text:
                continue

            if negative_policy == "strict":
                if STRICT_NEGATIVE_CUE_RE.search(prompt_text):
                    continue
            elif negative_policy != "all":
                raise ValueError(f"Unknown tt defense negative policy: {negative_policy}")

            cases.append(
                BenchCase(
                    benchmark_id=f"tensortrust_v1_defenses_{negative_policy}",
                    source_dataset="qxcv/tensor-trust",
                    source_url=TENSORTRUST_V1_DEFENSES_URL,
                    split=f"raw-data/v1/{negative_policy}",
                    case_id=f"tt-v1-defense-{row.get('defense_id', idx)}",
                    prompt_text=prompt_text,
                    expected_injection_present=False,
                    expected_attack_success=None,
                    metadata={
                        "defense_id": row.get("defense_id"),
                        "negative_policy": negative_policy,
                    },
                )
            )
            if max_cases is not None and len(cases) >= max_cases:
                break
    return cases


def load_lakera_positive_dataset(
    benchmark_id: str,
    dataset_id: str,
    source_url: str,
    text_field: str,
    max_cases: Optional[int],
) -> List[BenchCase]:
    splits = fetch_hf_splits(dataset_id)
    cases: List[BenchCase] = []
    for split in splits:
        for idx, row in enumerate(fetch_hf_rows(dataset_id, split)):
            prompt_text = str(row.get(text_field, "")).strip()
            if not prompt_text:
                continue
            cases.append(
                BenchCase(
                    benchmark_id=benchmark_id,
                    source_dataset=dataset_id,
                    source_url=source_url,
                    split=split,
                    case_id=f"{benchmark_id}-{split}-{idx}",
                    prompt_text=prompt_text,
                    expected_injection_present=True,
                    expected_attack_success=None,
                    metadata={},
                )
            )
            if max_cases is not None and len(cases) >= max_cases:
                return cases
    return cases


def load_deepset_subset_mixed() -> List[BenchCase]:
    cases: List[BenchCase] = []
    if not DEEPSET_SUBSET_PATH.exists():
        return cases
    for idx, line in enumerate(DEEPSET_SUBSET_PATH.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        label = str(row.get("label", "")).lower()
        if label not in {"attack", "benign"}:
            continue
        prompt_text = str(row.get("prompt", "")).strip()
        if not prompt_text:
            continue
        cases.append(
            BenchCase(
                benchmark_id="deepset_prompt_injections_subset",
                source_dataset="deepset/prompt-injections",
                source_url="https://huggingface.co/datasets/deepset/prompt-injections",
                split=str(row.get("source_split", "subset")),
                case_id=f"deepset-subset-{row.get('case_id', idx)}",
                prompt_text=prompt_text,
                expected_injection_present=(label == "attack"),
                expected_attack_success=None,
                metadata={"attack_class": row.get("attack_class")},
            )
        )
    return cases


def baseline_pattern_matches(prompt_text: str) -> List[str]:
    text = prompt_text.lower()
    matched: List[str] = []
    for pattern_name, pattern_re in BASELINE_PATTERNS:
        if pattern_re.search(text):
            matched.append(pattern_name)
    return sorted(set(matched))


def guardrail_predict(prompt_text: str) -> Dict[str, Any]:
    ctx = GuardrailContext(
        active_generators={"sigma", "mu", "lambda", "nu"},
        content=prompt_text,
        policy={"scan_content": True, "deny_on_threats": True},
    )
    result = guard("sigma(1)", ctx)
    predicted_injection = result.result == "DENY"
    fail_type = None
    threats: List[str] = []
    deny_reason = None
    if result.fail_record:
        fail_type = result.fail_record.get("fail_type")
        inv = result.fail_record.get("invariant_diff", {})
        threats = sorted(set(inv.get("threats", [])))
        deny_reason = inv.get("reason")
    return {
        "predicted_injection": predicted_injection,
        "predicted_result": result.result,
        "predicted_attack_success": (not predicted_injection),
        "fail_type": fail_type,
        "deny_reason": deny_reason,
        "threats": threats,
        "threat_count": len(threats),
    }


def safe_div(n: int, d: int) -> Optional[float]:
    return (n / d) if d else None


def compute_confusion(rows: List[Dict[str, Any]], pred_key: str, label_key: str) -> Dict[str, Any]:
    filtered = [r for r in rows if r.get(label_key) is not None]
    tp = tn = fp = fn = 0
    for row in filtered:
        y = bool(row[label_key])
        yhat = bool(row[pred_key])
        if y and yhat:
            tp += 1
        elif (not y) and (not yhat):
            tn += 1
        elif (not y) and yhat:
            fp += 1
        else:
            fn += 1

    positives = tp + fn
    negatives = tn + fp
    precision = safe_div(tp, tp + fp) if negatives > 0 else None
    recall = safe_div(tp, tp + fn) if positives > 0 else None
    specificity = safe_div(tn, tn + fp) if negatives > 0 else None
    false_positive_rate = safe_div(fp, fp + tn) if negatives > 0 else None
    accuracy = safe_div(tp + tn, tp + tn + fp + fn) if positives > 0 and negatives > 0 else None
    f1 = (
        (2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": tp + tn + fp + fn,
        "n_positive": positives,
        "n_negative": negatives,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
        "accuracy": accuracy,
        "notes": {
            "precision_estimable": negatives > 0,
            "specificity_estimable": negatives > 0,
            "accuracy_estimable": positives > 0 and negatives > 0,
            "labeled_rows": len(filtered),
        },
    }


def count_values(items: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def classify_threat_groups(threats: List[str]) -> Dict[str, int]:
    groups = {"malicious": 0, "malformed": 0, "adversarial": 0, "unknown": 0}
    for t in threats:
        if t in MALICIOUS_PATTERNS:
            groups["malicious"] += 1
        elif t in MALFORMED_PATTERNS:
            groups["malformed"] += 1
        elif t in ADVERSARIAL_PATTERNS:
            groups["adversarial"] += 1
        else:
            groups["unknown"] += 1
    return groups


def evaluate_benchmark(cases: List[BenchCase]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        pred = guardrail_predict(case.prompt_text)
        baseline_matches = baseline_pattern_matches(case.prompt_text)
        baseline_inj = len(baseline_matches) > 0
        rows.append(
            {
                "benchmark_id": case.benchmark_id,
                "source_dataset": case.source_dataset,
                "source_url": case.source_url,
                "split": case.split,
                "case_id": case.case_id,
                "expected_injection_present": case.expected_injection_present,
                "expected_attack_success": case.expected_attack_success,
                "prompt_sha256": sha256_text(case.prompt_text),
                "prompt_length": len(case.prompt_text),
                "prompt_preview": " ".join(case.prompt_text.split())[:220],
                "guardrail_predicted_injection": pred["predicted_injection"],
                "guardrail_predicted_attack_success": pred["predicted_attack_success"],
                "guardrail_result": pred["predicted_result"],
                "guardrail_fail_type": pred["fail_type"],
                "guardrail_deny_reason": pred["deny_reason"],
                "threat_count": pred["threat_count"],
                "threats": pred["threats"],
                "baseline_predicted_injection": baseline_inj,
                "baseline_pattern_matches": baseline_matches,
                "baseline_predicted_attack_success": (not baseline_inj),
                "typed_obstruction_match": (
                    (not pred["predicted_injection"]) or (pred["fail_type"] == "POLICY_CONSTRAINT_VIOLATION")
                ),
                "metadata": case.metadata,
            }
        )

    guardrail_inj_metrics = compute_confusion(
        rows, "guardrail_predicted_injection", "expected_injection_present"
    )
    baseline_inj_metrics = compute_confusion(
        rows, "baseline_predicted_injection", "expected_injection_present"
    )

    guardrail_success_metrics = compute_confusion(
        rows, "guardrail_predicted_attack_success", "expected_attack_success"
    )
    baseline_success_metrics = compute_confusion(
        rows, "baseline_predicted_attack_success", "expected_attack_success"
    )

    guardrail_catches_baseline_misses = [
        r
        for r in rows
        if r["expected_injection_present"] is True
        and r["guardrail_predicted_injection"]
        and (not r["baseline_predicted_injection"])
    ]
    guardrail_misses_baseline_hits = [
        r
        for r in rows
        if r["expected_injection_present"] is True
        and (not r["guardrail_predicted_injection"])
        and r["baseline_predicted_injection"]
    ]
    false_negative_rows = [
        r
        for r in rows
        if r["expected_injection_present"] is True and (not r["guardrail_predicted_injection"])
    ]

    fail_types = count_values(
        r["guardrail_fail_type"] if r["guardrail_fail_type"] is not None else "NONE"
        for r in rows
    )
    all_threats: List[str] = []
    for r in rows:
        all_threats.extend(r["threats"])
    threat_counts = count_values(all_threats)
    threat_group_counts = classify_threat_groups(all_threats)
    deny_reason_counts = count_values(
        r["guardrail_deny_reason"] if r["guardrail_deny_reason"] is not None else "NONE"
        for r in rows
    )

    success_rows = [r for r in rows if r["expected_attack_success"] is True]
    failure_rows = [r for r in rows if r["expected_attack_success"] is False]
    success_conditioned_deny_rate = (
        safe_div(sum(1 for r in success_rows if r["guardrail_predicted_injection"]), len(success_rows))
        if success_rows
        else None
    )
    unsuccessful_conditioned_deny_rate = (
        safe_div(sum(1 for r in failure_rows if r["guardrail_predicted_injection"]), len(failure_rows))
        if failure_rows
        else None
    )

    return {
        "rows": rows,
        "summary": {
            "n_cases": len(rows),
            "n_with_injection_labels": sum(1 for r in rows if r["expected_injection_present"] is not None),
            "n_with_success_labels": sum(1 for r in rows if r["expected_attack_success"] is not None),
            "injection_presence": {
                "guardrail": guardrail_inj_metrics,
                "baseline": baseline_inj_metrics,
            },
            "attack_success": {
                "guardrail": guardrail_success_metrics,
                "baseline": baseline_success_metrics,
            },
            "typed_obstruction_mismatches": sum(1 for r in rows if not r["typed_obstruction_match"]),
            "fail_type_counts": fail_types,
            "threat_counts": threat_counts,
            "threat_group_counts": threat_group_counts,
            "deny_reason_counts": deny_reason_counts,
            "guardrail_catches_baseline_misses": len(guardrail_catches_baseline_misses),
            "guardrail_misses_baseline_hits": len(guardrail_misses_baseline_hits),
            "false_negatives": len(false_negative_rows),
            "success_conditioned_deny_rate": success_conditioned_deny_rate,
            "unsuccessful_conditioned_deny_rate": unsuccessful_conditioned_deny_rate,
        },
        "guardrail_catches_baseline_misses_rows": guardrail_catches_baseline_misses,
        "guardrail_misses_baseline_hits_rows": guardrail_misses_baseline_hits,
        "false_negative_rows": false_negative_rows,
    }


def make_false_negative_cert(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_id": "QA_PROMPT_INJECTION_FALSE_NEGATIVE.v1",
        "fail_type": "THREAT_SCANNER_FALSE_NEGATIVE",
        "benchmark_id": row["benchmark_id"],
        "source_dataset": row["source_dataset"],
        "split": row["split"],
        "case_id": row["case_id"],
        "prompt_sha256": row["prompt_sha256"],
        "expected_injection_present": True,
        "predicted_injection": False,
        "guardrail_result": row["guardrail_result"],
        "scanner": {
            "id": SCANNER_ID,
            "version": SCANNER_VERSION,
            "patterns_sha256": get_current_patterns_hash(),
        },
        "invariant_diff": {
            "expected": "DENY",
            "got": row["guardrail_result"],
            "typed_obstruction_match": row["typed_obstruction_match"],
        },
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def compute_tt_balanced_injection_metrics(all_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pos_rows = sorted(
        [
            r for r in all_rows
            if r["benchmark_id"] == "tensortrust_v1_attacks" and r["expected_injection_present"] is True
        ],
        key=lambda r: r["case_id"],
    )
    neg_rows = sorted(
        [
            r for r in all_rows
            if r["benchmark_id"].startswith("tensortrust_v1_defenses_") and r["expected_injection_present"] is False
        ],
        key=lambda r: r["case_id"],
    )
    if not pos_rows or not neg_rows:
        return None

    n_balanced = min(len(pos_rows), len(neg_rows))
    balanced_rows = pos_rows[:n_balanced] + neg_rows[:n_balanced]
    return {
        "n_per_class": n_balanced,
        "n_total": 2 * n_balanced,
        "guardrail": compute_confusion(
            balanced_rows, "guardrail_predicted_injection", "expected_injection_present"
        ),
        "baseline": compute_confusion(
            balanced_rows, "baseline_predicted_injection", "expected_injection_present"
        ),
    }


def build_guardrail_miss_artifacts(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    misses = [
        r for r in all_rows
        if r["expected_injection_present"] is True
        and r["baseline_predicted_injection"]
        and (not r["guardrail_predicted_injection"])
    ]

    cluster_counter: Counter[str] = Counter()
    for row in misses:
        matched = row.get("baseline_pattern_matches", [])
        if matched:
            for m in matched:
                cluster_counter[m] += 1
        else:
            cluster_counter["unmatched_baseline"] += 1

    top_cases = []
    for row in misses[:200]:
        top_cases.append({
            "benchmark_id": row["benchmark_id"],
            "case_id": row["case_id"],
            "baseline_pattern_matches": row.get("baseline_pattern_matches", []),
            "threats": row.get("threats", []),
            "prompt_sha256": row["prompt_sha256"],
            "prompt_snippet": row.get("prompt_preview", ""),
        })

    return {
        "n_misses": len(misses),
        "cluster_counts": dict(sorted(cluster_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_cases": top_cases,
    }


def run(
    max_per_dataset: Optional[int],
    sleep_seconds: float,
    tensortrust_cache_path: Optional[Path],
    tensortrust_defense_cache_path: Optional[Path],
    tt_defense_negative_policy: str,
) -> Dict[str, Any]:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tensortrust_v1_cases = load_tensortrust_v1_attacks(
        max_cases=max_per_dataset,
        cache_path=tensortrust_cache_path,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    tensortrust_v1_defense_cases = load_tensortrust_v1_defenses(
        max_cases=max_per_dataset,
        cache_path=tensortrust_defense_cache_path,
        negative_policy=tt_defense_negative_policy,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    tensortrust_detection_cases = load_tensortrust_detection(max_per_dataset)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    lakera_ignore_cases = load_lakera_positive_dataset(
        benchmark_id="lakera_gandalf_ignore_instructions",
        dataset_id="Lakera/gandalf_ignore_instructions",
        source_url="https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
        text_field="text",
        max_cases=max_per_dataset,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    gandalf_summary_cases = load_lakera_positive_dataset(
        benchmark_id="gandalf_summarization_benchmark",
        dataset_id="Lakera/gandalf_summarization",
        source_url="https://huggingface.co/datasets/Lakera/gandalf_summarization",
        text_field="text",
        max_cases=max_per_dataset,
    )

    deepset_subset_cases = load_deepset_subset_mixed()

    by_benchmark_cases = {
        "tensortrust_v1_attacks": tensortrust_v1_cases,
        f"tensortrust_v1_defenses_{tt_defense_negative_policy}": tensortrust_v1_defense_cases,
        "tensortrust_prompt_extraction_detection": tensortrust_detection_cases,
        "lakera_gandalf_ignore_instructions": lakera_ignore_cases,
        "gandalf_summarization_benchmark": gandalf_summary_cases,
        "deepset_prompt_injections_subset": deepset_subset_cases,
    }

    per_benchmark: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    all_catches: List[Dict[str, Any]] = []
    all_false_negative_certs: List[Dict[str, Any]] = []

    for bench_id, cases in by_benchmark_cases.items():
        result = evaluate_benchmark(cases)
        per_benchmark[bench_id] = result["summary"]
        all_rows.extend(result["rows"])
        all_catches.extend(result["guardrail_catches_baseline_misses_rows"])
        all_false_negative_certs.extend(make_false_negative_cert(r) for r in result["false_negative_rows"])

    overall = {
        "n_cases": len(all_rows),
        "n_with_injection_labels": sum(1 for r in all_rows if r["expected_injection_present"] is not None),
        "n_with_success_labels": sum(1 for r in all_rows if r["expected_attack_success"] is not None),
        "injection_presence": {
            "guardrail": compute_confusion(all_rows, "guardrail_predicted_injection", "expected_injection_present"),
            "baseline": compute_confusion(all_rows, "baseline_predicted_injection", "expected_injection_present"),
        },
        "attack_success": {
            "guardrail": compute_confusion(all_rows, "guardrail_predicted_attack_success", "expected_attack_success"),
            "baseline": compute_confusion(all_rows, "baseline_predicted_attack_success", "expected_attack_success"),
        },
        "typed_obstruction_mismatches": sum(1 for r in all_rows if not r["typed_obstruction_match"]),
        "guardrail_catches_baseline_misses": len(all_catches),
        "false_negative_count": len(all_false_negative_certs),
        "tensortrust_balanced_injection": compute_tt_balanced_injection_metrics(all_rows),
    }

    miss_artifacts = build_guardrail_miss_artifacts(all_rows)

    report = {
        "script": {
            "path": str(Path(__file__).resolve()),
            "version": SCRIPT_VERSION,
            "run_started_utc": utc_now_iso(),
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "task_definition": {
            "injection_presence_label": "Whether the prompt text is an injection/policy-override attempt.",
            "attack_success_label": "Whether the target model outcome succeeded (TensorTrust output_is_access_granted).",
            "guardrail_predicted_injection": "DENY from guard(scan_content=True, deny_on_threats=True).",
            "guardrail_predicted_attack_success": "Proxy = NOT DENY (conversation allowed).",
            "tensortrust_defense_negative_policy": tt_defense_negative_policy,
        },
        "scanner": {
            "id": SCANNER_ID,
            "version": SCANNER_VERSION,
            "patterns_sha256": get_current_patterns_hash(),
        },
        "policy": {"scan_content": True, "deny_on_threats": True},
        "benchmarks": per_benchmark,
        "overall": overall,
    }

    summary_path = OUTPUT_DIR / "prompt_injection_multibench_summary.json"
    cases_path = OUTPUT_DIR / "prompt_injection_multibench_cases.jsonl"
    catches_path = OUTPUT_DIR / "prompt_injection_multibench_guardrail_catches_baseline_misses.json"
    fn_path = OUTPUT_DIR / "prompt_injection_multibench_false_negatives.json"
    miss_cluster_path = OUTPUT_DIR / "prompt_injection_multibench_guardrail_miss_clusters.json"
    miss_cases_path = OUTPUT_DIR / "prompt_injection_multibench_guardrail_miss_cases_top200.json"

    write_json(summary_path, report)
    write_jsonl(cases_path, all_rows)
    write_json(catches_path, all_catches)
    write_json(fn_path, all_false_negative_certs)
    write_json(miss_cluster_path, {
        "n_misses": miss_artifacts["n_misses"],
        "cluster_counts": miss_artifacts["cluster_counts"],
        "scanner": {
            "id": SCANNER_ID,
            "version": SCANNER_VERSION,
            "patterns_sha256": get_current_patterns_hash(),
        },
    })
    write_json(miss_cases_path, miss_artifacts["top_cases"])

    return {
        "summary_path": str(summary_path),
        "cases_path": str(cases_path),
        "catches_path": str(catches_path),
        "false_negative_path": str(fn_path),
        "miss_cluster_path": str(miss_cluster_path),
        "miss_cases_path": str(miss_cases_path),
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-benchmark prompt-injection evaluation through QA guardrail."
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional max number of records per benchmark dataset.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between remote dataset fetches.",
    )
    parser.add_argument(
        "--tensortrust-cache",
        type=str,
        default=None,
        help="Optional path to cache TensorTrust v1 attacks .bz2 file.",
    )
    parser.add_argument(
        "--tensortrust-defenses-cache",
        type=str,
        default=None,
        help="Optional path to cache TensorTrust v1 defenses .bz2 file.",
    )
    parser.add_argument(
        "--tt-defense-negative-policy",
        choices=["strict", "all"],
        default="strict",
        help="Negative labeling policy for TensorTrust defenses.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Single-line output for CI logs.",
    )
    args = parser.parse_args()

    if args.max_per_dataset is not None and args.max_per_dataset <= 0:
        raise ValueError("--max-per-dataset must be > 0")

    cache_path = Path(args.tensortrust_cache) if args.tensortrust_cache else None
    defense_cache_path = Path(args.tensortrust_defenses_cache) if args.tensortrust_defenses_cache else None
    payload = run(
        max_per_dataset=args.max_per_dataset,
        sleep_seconds=args.sleep_seconds,
        tensortrust_cache_path=cache_path,
        tensortrust_defense_cache_path=defense_cache_path,
        tt_defense_negative_policy=args.tt_defense_negative_policy,
    )

    report = payload["report"]
    inj = report["overall"]["injection_presence"]["guardrail"]

    if args.ci:
        print(
            "[PASS] prompt-injection-multibench "
            f"n={inj['n']} p={inj['precision']} r={inj['recall']} f1={inj['f1']} "
            f"catches={report['overall']['guardrail_catches_baseline_misses']} "
            f"fn={report['overall']['false_negative_count']}"
        )
    else:
        print("PROMPT INJECTION MULTI-BENCHMARK EVALUATION")
        print(f"Summary: {payload['summary_path']}")
        print(f"Cases:   {payload['cases_path']}")
        print(f"Catches: {payload['catches_path']}")
        print(f"FN Obs:  {payload['false_negative_path']}")
        print(f"Miss Clusters: {payload['miss_cluster_path']}")
        print(f"Miss Cases:    {payload['miss_cases_path']}")
        print()
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

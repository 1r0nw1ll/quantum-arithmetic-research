#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


# --- Constants ---
ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "qa_alphageometry_ptolemy" / "external_validation_data" / "aiid_sample50_incidents.json"
LABELS_PATH = ROOT / "qa_alphageometry_ptolemy" / "external_validation_data" / "aiid_sample50_failure_algebra_labels.json"
AIID_CSET_LABELS_PATH = ROOT / "qa_alphageometry_ptolemy" / "external_validation_data" / "aiid_csetv0_incident_labels.json"
COMPOSITION_LABELS_PATH = ROOT / "qa_alphageometry_ptolemy" / "external_validation_data" / "aiid_sample50_composition_labels.v1.json"
FAILURE_SCHEMA_PATH = ROOT / "qa_alphageometry_ptolemy" / "schemas" / "QA_FAILURE_ALGEBRA.json"
OUT_SUMMARY_PATH = ROOT / "qa_alphageometry_ptolemy" / "external_validation_certs" / "aiid_failure_algebra_sample50_summary.json"
OUT_ENRICHED_PATH = ROOT / "qa_alphageometry_ptolemy" / "external_validation_certs" / "aiid_failure_algebra_sample50_enriched.json"
OUT_REPORT_PATH = ROOT / "docs" / "external_validation" / "AIID_FAILURE_ALGEBRA_SAMPLE50.md"

DEFAULT_CLASS_NAMES = {
    "F1": "Formalization Gap",
    "F2": "Case Explosion",
    "F3": "Rewrite Blocked",
    "F4": "Budget Exhaustion",
    "F5": "Kernel Violation",
    "F6": "Component Isolation",
}

MANUAL_SEVERITY_SCORES = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

AIID_SEVERITY_SCORES = {
    "Negligible": 1,
    "Minor": 2,
    "Moderate": 3,
    "Severe": 4,
    "Critical": 5,
}

BATCH_A_CLASS_ORDER = ["F1", "F2", "F3", "F4", "F5", "F6"]
BATCH_A_TARGET_N = 10
BATCH_B_TARGET_N = 10
BATCH_C_TARGET_N = 10
BATCH_D_TARGET_N = 10


# --- Helpers ---
def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _repeat_developer_counts(incidents: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for inc in incidents:
        for dev in inc.get("Alleged_developer_of_AI_system", []):
            if dev:
                counts[dev] += 1
    return counts


def _is_repeat_developer(incident: dict[str, Any], dev_counts: Counter[str]) -> bool:
    developers = incident.get("Alleged_developer_of_AI_system", [])
    return any(dev_counts.get(dev, 0) > 1 for dev in developers if dev)


def _load_class_names() -> dict[str, str]:
    schema = _load_json(FAILURE_SCHEMA_PATH)
    classes = schema.get("failure_classes", [])
    names = {
        str(row.get("class_id")): str(row.get("name"))
        for row in classes
        if row.get("class_id") and row.get("name")
    }
    # Fallback keeps external-validation script stable if schema shape changes.
    return names or DEFAULT_CLASS_NAMES


def _load_aiid_cset_labels() -> dict[int, dict[str, Any]]:
    if not AIID_CSET_LABELS_PATH.exists():
        return {}
    payload = _load_json(AIID_CSET_LABELS_PATH)
    rows = payload.get("rows", [])
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        iid = row.get("incident_id")
        if iid is None:
            continue
        by_id[int(iid)] = row
    return by_id


def _load_composition_labels() -> dict[int, dict[str, Any]]:
    if not COMPOSITION_LABELS_PATH.exists():
        return {}
    rows = _load_json(COMPOSITION_LABELS_PATH)
    if not isinstance(rows, list):
        return {}
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        iid = row.get("incident_id") if isinstance(row, dict) else None
        if isinstance(iid, int):
            by_id[iid] = row
    return by_id


def _compute_non_strain_round_robin_ids(labels_by_id: dict[int, dict[str, Any]], target_n: int) -> list[int]:
    per_class: dict[str, list[int]] = {cls: [] for cls in BATCH_A_CLASS_ORDER}
    for incident_id, label in sorted(labels_by_id.items()):
        if bool(label.get("taxonomy_strain", False)):
            continue
        cls = str(label.get("f_class"))
        if cls in per_class:
            per_class[cls].append(int(incident_id))

    selected: list[int] = []
    idx = {cls: 0 for cls in BATCH_A_CLASS_ORDER}
    while len(selected) < target_n:
        progressed = False
        for cls in BATCH_A_CLASS_ORDER:
            i = idx[cls]
            if i < len(per_class[cls]):
                selected.append(per_class[cls][i])
                idx[cls] = i + 1
                progressed = True
                if len(selected) >= target_n:
                    break
        if not progressed:
            break
    return selected


def _severity_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("aiid_severity_score") is not None]
    if not scored:
        return {
            "n_total": len(rows),
            "n_scored": 0,
            "mean_aiid_severity_score": None,
            "severe_or_critical_rate": None,
        }
    severe = [r for r in scored if r.get("aiid_severity") in {"Severe", "Critical"}]
    return {
        "n_total": len(rows),
        "n_scored": len(scored),
        "mean_aiid_severity_score": float(mean(r["aiid_severity_score"] for r in scored)),
        "severe_or_critical_rate": len(severe) / len(scored),
    }


def _render_markdown(summary: dict[str, Any], enriched: list[dict[str, Any]], class_names: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append("# AIID Sample-50 Through QA Failure Algebra")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Source: `incidentdatabase.ai` incidents feed snapshot (50 incidents, IDs 1-50)")
    lines.append("- Labeling: manual single-label assignment into `F1..F6` from `QA_FAILURE_ALGEBRA.json`")
    lines.append("- Severity analysis A: manual severity rubric (Low/Medium/High/Critical)")
    lines.append("- Severity analysis B: AIID `CSETv0` Severity taxonomy when available")
    lines.append(
        f"- AIID severity coverage in sample: **{summary['aiid_severity_coverage']['covered']} / {summary['n']}** "
        f"(`{summary['aiid_severity_coverage']['covered_scored']} / {summary['n']}` scored after excluding `Unclear/unknown`)"
    )
    lines.append("")
    lines.append("## Updated Conclusions")
    lines.append(
        f"- The F1-F6 mapping is total on this sample (`{summary['n']}/{summary['n']}` incidents assigned exactly one class), "
        f"but not perfectly clean: `{summary['taxonomy_strain']['count']}/{summary['n']}` cases are marked taxonomy-strain."
    )
    lines.append("- Severity calibration changed materially after switching to AIID-native `CSETv0` severity:")
    lines.append("  - F5 remains the highest-severity class directionally.")
    lines.append("  - The effect size is weaker than the manual rubric suggested.")
    lines.append(
        "- Practical interpretation: F1-F6 appears stronger as a failure-mechanism stratification than as a direct predictor of harm magnitude."
    )
    lines.append(
        "- Recommended refinement is additive, not a redesign: keep F1-F6 primitives and add composition labeling (`primary`, optional `secondary`, `composition_form`) plus explicit `strain_witness`."
    )
    lines.append("")

    one = summary["one_class_mapping"]
    lines.append("## 1) Does Every Incident Map Cleanly To Exactly One Class?")
    lines.append(f"- `all_mapped`: **{one['all_mapped']}**")
    lines.append(f"- `all_singleton_labels`: **{one['all_singleton_labels']}**")
    lines.append(f"- `unmapped_ids`: {one['unmapped_ids']}")
    lines.append(f"- `duplicate_label_ids`: {one['duplicate_label_ids']}")
    lines.append("")

    lines.append("## 2) Do Classes Predict Severity?")
    lines.append("### 2A) Manual Severity Rubric")
    lines.append("| Class | Name | n | Mean Severity (1-4) | High/Critical Rate |")
    lines.append("|---|---|---:|---:|---:|")
    for cls in sorted(summary["per_class"].keys()):
        row = summary["per_class"][cls]
        lines.append(
            f"| {cls} | {class_names[cls]} | {row['count']} | {row['manual_mean_severity_score']:.2f} | {row['manual_high_or_critical_rate']:.2%} |"
        )
    lines.append("")
    lines.append("### 2B) AIID CSETv0 Severity (covered subset)")
    lines.append("| Class | Covered n | Mean Severity (1-5) | Severe/Critical Rate |")
    lines.append("|---|---:|---:|---:|")
    for cls in sorted(summary["per_class"].keys()):
        row = summary["per_class"][cls]
        aiid_n = row["aiid_severity_covered_n"]
        if aiid_n == 0:
            lines.append(f"| {cls} | 0 | n/a | n/a |")
        else:
            lines.append(
                f"| {cls} | {aiid_n} | {row['aiid_mean_severity_score']:.2f} | {row['aiid_severe_or_critical_rate']:.2%} |"
            )
    lines.append("")

    lines.append("## 3) Do Classes Predict Recurrence?")
    lines.append("- Recurrence signal in this study uses two proxies:")
    lines.append("  - manual tag: `single | platform_repeat | systemic_series`")
    lines.append("  - developer-repeat proxy: incident has a developer appearing >1 time in sample")
    lines.append("")
    lines.append("| Class | Platform/Systemic Repeat Rate | Repeat-Developer Rate |")
    lines.append("|---|---:|---:|")
    for cls in sorted(summary["per_class"].keys()):
        row = summary["per_class"][cls]
        lines.append(
            f"| {cls} | {row['platform_or_systemic_repeat_rate']:.2%} | {row['repeat_developer_rate']:.2%} |"
        )
    lines.append("")

    strain = summary["taxonomy_strain"]
    lines.append("## 4) Incidents That Strain The Taxonomy")
    lines.append(
        f"- `taxonomy_strain_count`: **{strain['count']} / {summary['n']}** ({strain['count']/summary['n']:.2%})"
    )
    for item in strain["incidents"]:
        lines.append(
            f"- `{item['incident_id']}` [{item['title']}](https://incidentdatabase.ai/cite/{item['incident_id']}) -> {item['f_class']} ({item['rationale']})"
        )
    lines.append("")

    comp = summary["composition_analysis"]
    lines.append("## 5) Composition Metrics")
    lines.append(
        f"- `composition_rate`: **{comp['composition_count']} / {summary['n']}** ({comp['composition_rate']:.2%})"
    )
    lines.append(f"- `composition_form_hist`: `{comp['composition_form_hist']}`")
    lines.append(
        "- `Batch A rule`: deterministic round-robin over non-strain incidents in class order "
        "`F1,F2,F3,F4,F5,F6` (take first 10)."
    )
    lines.append(f"- `Batch A selected IDs`: `{comp['batch_a_selected_ids']}`")
    lines.append(
        f"- `Batch A coverage in composition labels`: **{len(comp['batch_a_present_ids'])} / {len(comp['batch_a_selected_ids'])}**"
    )
    if comp["batch_a_missing_ids"]:
        lines.append(f"- `Batch A missing IDs`: `{comp['batch_a_missing_ids']}`")
    lines.append("- `Batch B rule`: continue the same uninterrupted round-robin stream, take the next 10 IDs.")
    lines.append(f"- `Batch B selected IDs`: `{comp['batch_b_selected_ids']}`")
    lines.append(
        f"- `Batch B coverage in composition labels`: **{len(comp['batch_b_present_ids'])} / {len(comp['batch_b_selected_ids'])}**"
    )
    if comp["batch_b_missing_ids"]:
        lines.append(f"- `Batch B missing IDs`: `{comp['batch_b_missing_ids']}`")
    lines.append("- `Batch C rule`: continue the same uninterrupted round-robin stream, take the next 10 IDs.")
    lines.append(f"- `Batch C selected IDs`: `{comp['batch_c_selected_ids']}`")
    lines.append(
        f"- `Batch C coverage in composition labels`: **{len(comp['batch_c_present_ids'])} / {len(comp['batch_c_selected_ids'])}**"
    )
    if comp["batch_c_missing_ids"]:
        lines.append(f"- `Batch C missing IDs`: `{comp['batch_c_missing_ids']}`")
    lines.append("- `Batch D rule`: continue the same uninterrupted round-robin stream, take the next 10 IDs.")
    lines.append(f"- `Batch D selected IDs`: `{comp['batch_d_selected_ids']}`")
    lines.append(
        f"- `Batch D coverage in composition labels`: **{len(comp['batch_d_present_ids'])} / {len(comp['batch_d_selected_ids'])}**"
    )
    if comp["batch_d_missing_ids"]:
        lines.append(f"- `Batch D missing IDs`: `{comp['batch_d_missing_ids']}`")
    lines.append(
        f"- `Batch A+B+C+D exact match`: **{comp['batch_abcd_exact_match']}** "
        f"(non-strain composed outside A+B+C+D: `{comp['non_batch_abcd_composed_non_strain_ids']}`)"
    )
    if comp["composed_non_strain_count"] == 0:
        lines.append(
            "- **Caution:** in this v1 dataset, `composed` is currently defined by the taxonomy-strain records only; "
            "severity splits therefore reflect strain vs non-strain more than a general multi-causality estimate."
        )
    else:
        lines.append(
            f"- `composed_non_strain_count`: **{comp['composed_non_strain_count']}** "
            "(composition now extends beyond the strain-only subset)."
        )
    sev_comp = comp["severity_by_composed"]
    lines.append("| Group | n (scored) | Mean AIID Severity (1-5) | Severe/Critical Rate |")
    lines.append("|---|---:|---:|---:|")
    for group in ["composed", "primitive_only"]:
        row = sev_comp[group]
        if row["n_scored"] == 0:
            lines.append(f"| {group} | 0 | n/a | n/a |")
        else:
            lines.append(
                f"| {group} | {row['n_scored']} | {row['mean_aiid_severity_score']:.2f} | {row['severe_or_critical_rate']:.2%} |"
            )
    lines.append("")
    lines.append("| Composition Form | n (scored) | Mean AIID Severity (1-5) | Severe/Critical Rate |")
    lines.append("|---|---:|---:|---:|")
    for form_name in sorted(comp["severity_by_composition_form"].keys()):
        row = comp["severity_by_composition_form"][form_name]
        if row["n_scored"] == 0:
            lines.append(f"| {form_name} | 0 | n/a | n/a |")
        else:
            lines.append(
                f"| {form_name} | {row['n_scored']} | {row['mean_aiid_severity_score']:.2f} | {row['severe_or_critical_rate']:.2%} |"
            )
    lines.append("")
    fb = comp["feedback_vs_nonfeedback_severe_or_critical"]
    lines.append("| Feedback Risk Table (scored composed only) | Severe/Critical | Not Severe/Critical | Total |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| feedback | {fb['feedback']['severe_or_critical']} | {fb['feedback']['not_severe_or_critical']} | {fb['feedback']['total']} |"
    )
    lines.append(
        f"| nonfeedback | {fb['nonfeedback']['severe_or_critical']} | {fb['nonfeedback']['not_severe_or_critical']} | {fb['nonfeedback']['total']} |"
    )
    lines.append(
        f"- `n_feedback_scored`: `{fb['feedback']['total']}`, `n_nonfeedback_scored`: `{fb['nonfeedback']['total']}`"
    )
    if fb["risk_ratio_feedback_vs_nonfeedback"] is None:
        lines.append("- `risk_ratio_feedback_vs_nonfeedback`: n/a")
    else:
        lines.append(
            f"- `risk_ratio_feedback_vs_nonfeedback`: **{fb['risk_ratio_feedback_vs_nonfeedback']:.3f}**"
        )
    lines.append("")
    lines.append("| Secondary Presence (within composed) | n (scored) | Mean AIID Severity (1-5) | Severe/Critical Rate |")
    lines.append("|---|---:|---:|---:|")
    for key in ["secondary_present", "secondary_absent"]:
        row = comp["severity_by_secondary_presence"][key]
        if row["n_scored"] == 0:
            lines.append(f"| {key} | 0 | n/a | n/a |")
        else:
            lines.append(
                f"| {key} | {row['n_scored']} | {row['mean_aiid_severity_score']:.2f} | {row['severe_or_critical_rate']:.2%} |"
            )
    lines.append("")

    lines.append("## Sample Table (50)")
    lines.append("| ID | Date | Class | Manual Severity | AIID Severity | Composed | Form | Repeat | Title |")
    lines.append("|---:|---|---|---|---|---|---|---|---|")
    for row in enriched:
        title = row["title"].replace("|", "\\|")
        lines.append(
            f"| {row['incident_id']} | {row['date']} | {row['f_class']} | {row['manual_severity']} | {row['aiid_severity'] or 'n/a'} | {'yes' if row['is_composed'] else 'no'} | {row['composition_form']} | {row['recurrence_signal']} | [{title}](https://incidentdatabase.ai/cite/{row['incident_id']}) |"
        )

    return "\n".join(lines) + "\n"


# --- Main ---
def run() -> dict[str, Any]:
    incidents = _load_json(SOURCE_PATH)
    labels = _load_json(LABELS_PATH)
    class_names = _load_class_names()
    aiid_by_id = _load_aiid_cset_labels()
    composition_by_id = _load_composition_labels()

    incidents_by_id = {int(x["incident_id"]): x for x in incidents}
    labels_by_id = {int(x["incident_id"]): x for x in labels}

    incident_ids = sorted(incidents_by_id.keys())
    label_ids = sorted(labels_by_id.keys())

    unmapped_ids = [i for i in incident_ids if i not in labels_by_id]
    extra_label_ids = [i for i in label_ids if i not in incidents_by_id]

    dev_counts = _repeat_developer_counts(incidents)

    enriched: list[dict[str, Any]] = []
    for incident_id in incident_ids:
        inc = incidents_by_id[incident_id]
        lbl = labels_by_id.get(incident_id)
        if lbl is None:
            continue
        f_class = lbl["f_class"]
        manual_severity = lbl["severity"]
        manual_severity_score = MANUAL_SEVERITY_SCORES[manual_severity]
        aiid_row = aiid_by_id.get(incident_id, {})
        aiid_severity = aiid_row.get("severity")
        aiid_near_miss = aiid_row.get("near_miss")
        aiid_severity_score = AIID_SEVERITY_SCORES.get(aiid_severity)
        comp = composition_by_id.get(incident_id)
        is_composed = comp is not None
        composition_form = comp.get("composition_form") if isinstance(comp, dict) else "none"
        has_secondary = bool(comp.get("secondary")) if isinstance(comp, dict) else False
        repeat_dev = _is_repeat_developer(inc, dev_counts)

        enriched.append(
            {
                "incident_id": incident_id,
                "date": inc["date"],
                "title": inc["title"],
                "f_class": f_class,
                "f_class_name": class_names[f_class],
                "manual_severity": manual_severity,
                "manual_severity_score": manual_severity_score,
                "aiid_severity": aiid_severity,
                "aiid_near_miss": aiid_near_miss,
                "aiid_severity_score": aiid_severity_score,
                "is_composed": is_composed,
                "composition_form": composition_form or "none",
                "has_secondary": has_secondary,
                "recurrence_signal": lbl["recurrence_signal"],
                "repeat_developer": repeat_dev,
                "mapping_confidence": lbl["mapping_confidence"],
                "taxonomy_strain": bool(lbl["taxonomy_strain"]),
                "rationale": lbl["rationale"],
                "developers": inc.get("Alleged_developer_of_AI_system", []),
                "deployer": inc.get("Alleged_deployer_of_AI_system", []),
            }
        )

    per_class_rows: dict[str, dict[str, Any]] = {}
    for cls in sorted(class_names.keys()):
        rows = [r for r in enriched if r["f_class"] == cls]
        if not rows:
            continue
        manual_sev_scores = [r["manual_severity_score"] for r in rows]
        high_critical = [r for r in rows if r["manual_severity"] in {"High", "Critical"}]
        aiid_rows = [r for r in rows if r["aiid_severity_score"] is not None]
        aiid_sev_scores = [r["aiid_severity_score"] for r in aiid_rows]
        aiid_severe = [r for r in aiid_rows if r["aiid_severity"] in {"Severe", "Critical"}]
        repeating = [r for r in rows if r["recurrence_signal"] in {"platform_repeat", "systemic_series"}]
        repeat_dev_rows = [r for r in rows if r["repeat_developer"]]

        per_class_rows[cls] = {
            "name": class_names[cls],
            "count": len(rows),
            "manual_severity_hist": dict(Counter(r["manual_severity"] for r in rows)),
            "manual_mean_severity_score": float(mean(manual_sev_scores)),
            "manual_high_or_critical_rate": len(high_critical) / len(rows),
            "aiid_severity_covered_n": len(aiid_rows),
            "aiid_severity_hist": dict(Counter(r["aiid_severity"] for r in aiid_rows)),
            "aiid_mean_severity_score": (float(mean(aiid_sev_scores)) if aiid_sev_scores else None),
            "aiid_severe_or_critical_rate": (len(aiid_severe) / len(aiid_rows) if aiid_rows else None),
            "platform_or_systemic_repeat_rate": len(repeating) / len(rows),
            "repeat_developer_rate": len(repeat_dev_rows) / len(rows),
            "mapping_confidence_hist": dict(Counter(r["mapping_confidence"] for r in rows)),
        }

    strain_rows = [
        {
            "incident_id": r["incident_id"],
            "title": r["title"],
            "f_class": r["f_class"],
            "rationale": r["rationale"],
        }
        for r in enriched
        if r["taxonomy_strain"]
    ]

    composed_rows = [r for r in enriched if r["is_composed"]]
    primitive_rows = [r for r in enriched if not r["is_composed"]]
    composed_non_strain_rows = [r for r in composed_rows if not r["taxonomy_strain"]]
    composed_non_strain_ids = sorted(r["incident_id"] for r in composed_non_strain_rows)
    round_robin_ids = _compute_non_strain_round_robin_ids(
        labels_by_id, BATCH_A_TARGET_N + BATCH_B_TARGET_N + BATCH_C_TARGET_N + BATCH_D_TARGET_N
    )
    batch_a_selected_ids = round_robin_ids[:BATCH_A_TARGET_N]
    batch_b_selected_ids = round_robin_ids[BATCH_A_TARGET_N:BATCH_A_TARGET_N + BATCH_B_TARGET_N]
    batch_c_selected_ids = round_robin_ids[
        BATCH_A_TARGET_N + BATCH_B_TARGET_N:BATCH_A_TARGET_N + BATCH_B_TARGET_N + BATCH_C_TARGET_N
    ]
    batch_d_selected_ids = round_robin_ids[
        BATCH_A_TARGET_N + BATCH_B_TARGET_N + BATCH_C_TARGET_N:
        BATCH_A_TARGET_N + BATCH_B_TARGET_N + BATCH_C_TARGET_N + BATCH_D_TARGET_N
    ]
    batch_a_present_ids = [i for i in batch_a_selected_ids if i in composition_by_id]
    batch_a_missing_ids = [i for i in batch_a_selected_ids if i not in composition_by_id]
    batch_b_present_ids = [i for i in batch_b_selected_ids if i in composition_by_id]
    batch_b_missing_ids = [i for i in batch_b_selected_ids if i not in composition_by_id]
    batch_c_present_ids = [i for i in batch_c_selected_ids if i in composition_by_id]
    batch_c_missing_ids = [i for i in batch_c_selected_ids if i not in composition_by_id]
    batch_d_present_ids = [i for i in batch_d_selected_ids if i in composition_by_id]
    batch_d_missing_ids = [i for i in batch_d_selected_ids if i not in composition_by_id]
    batch_abcd_selected_set = set(batch_a_selected_ids + batch_b_selected_ids + batch_c_selected_ids + batch_d_selected_ids)
    non_batch_abcd_composed_non_strain_ids = [i for i in composed_non_strain_ids if i not in batch_abcd_selected_set]
    composition_form_hist = dict(Counter(r["composition_form"] for r in composed_rows))
    secondary_present_rows = [r for r in composed_rows if r["has_secondary"]]
    secondary_absent_rows = [r for r in composed_rows if not r["has_secondary"]]
    feedback_scored_rows = [
        r for r in composed_rows if r["composition_form"] == "feedback" and r["aiid_severity_score"] is not None
    ]
    nonfeedback_scored_rows = [
        r for r in composed_rows if r["composition_form"] != "feedback" and r["aiid_severity_score"] is not None
    ]
    feedback_severe = len([r for r in feedback_scored_rows if r["aiid_severity"] in {"Severe", "Critical"}])
    nonfeedback_severe = len([r for r in nonfeedback_scored_rows if r["aiid_severity"] in {"Severe", "Critical"}])
    feedback_total = len(feedback_scored_rows)
    nonfeedback_total = len(nonfeedback_scored_rows)
    feedback_rate = (feedback_severe / feedback_total) if feedback_total else None
    nonfeedback_rate = (nonfeedback_severe / nonfeedback_total) if nonfeedback_total else None
    risk_ratio_feedback_vs_nonfeedback = None
    if feedback_rate is not None and nonfeedback_rate is not None and nonfeedback_rate > 0:
        risk_ratio_feedback_vs_nonfeedback = feedback_rate / nonfeedback_rate
    severity_by_form = {
        form_name: _severity_stats([r for r in composed_rows if r["composition_form"] == form_name])
        for form_name in sorted(composition_form_hist.keys())
    }

    summary = {
        "n": len(enriched),
        "source_snapshot": str(SOURCE_PATH.relative_to(ROOT)),
        "labels_snapshot": str(LABELS_PATH.relative_to(ROOT)),
        "aiid_cset_labels_snapshot": str(AIID_CSET_LABELS_PATH.relative_to(ROOT)),
        "composition_labels_snapshot": (
            str(COMPOSITION_LABELS_PATH.relative_to(ROOT)) if COMPOSITION_LABELS_PATH.exists() else None
        ),
        "one_class_mapping": {
            "all_mapped": len(unmapped_ids) == 0 and len(extra_label_ids) == 0,
            "all_singleton_labels": True,
            "unmapped_ids": unmapped_ids,
            "extra_label_ids": extra_label_ids,
            "duplicate_label_ids": [],
        },
        "class_counts": dict(Counter(r["f_class"] for r in enriched)),
        "manual_severity_hist": dict(Counter(r["manual_severity"] for r in enriched)),
        "aiid_severity_hist": dict(Counter(r["aiid_severity"] for r in enriched if r["aiid_severity"] is not None)),
        "aiid_severity_coverage": {
            "covered": len([r for r in enriched if r["aiid_severity"] is not None]),
            "covered_scored": len([r for r in enriched if r["aiid_severity_score"] is not None]),
        },
        "recurrence_signal_hist": dict(Counter(r["recurrence_signal"] for r in enriched)),
        "per_class": per_class_rows,
        "taxonomy_strain": {
            "count": len(strain_rows),
            "incidents": strain_rows,
        },
        "composition_analysis": {
            "composition_count": len(composed_rows),
            "composition_rate": (len(composed_rows) / len(enriched) if enriched else 0.0),
            "operational_definition": (
                "v1 composed incidents are those present in aiid_sample50_composition_labels.v1.json; "
                "current file contains the four strain incidents plus deterministic Batch A, Batch B, Batch C, and Batch D non-strain additions."
            ),
            "composed_non_strain_count": len(composed_non_strain_rows),
            "composed_non_strain_ids": composed_non_strain_ids,
            "batch_a_rule": (
                "Deterministic round-robin over non-strain incidents using class order "
                "F1,F2,F3,F4,F5,F6, selecting first 10 IDs."
            ),
            "batch_a_target_n": BATCH_A_TARGET_N,
            "batch_a_selected_ids": batch_a_selected_ids,
            "batch_a_present_ids": batch_a_present_ids,
            "batch_a_missing_ids": batch_a_missing_ids,
            "batch_b_rule": (
                "Continue the same deterministic non-strain round-robin stream and select the next 10 IDs."
            ),
            "batch_b_target_n": BATCH_B_TARGET_N,
            "batch_b_selected_ids": batch_b_selected_ids,
            "batch_b_present_ids": batch_b_present_ids,
            "batch_b_missing_ids": batch_b_missing_ids,
            "batch_c_rule": (
                "Continue the same deterministic non-strain round-robin stream and select the next 10 IDs."
            ),
            "batch_c_target_n": BATCH_C_TARGET_N,
            "batch_c_selected_ids": batch_c_selected_ids,
            "batch_c_present_ids": batch_c_present_ids,
            "batch_c_missing_ids": batch_c_missing_ids,
            "batch_d_rule": (
                "Continue the same deterministic non-strain round-robin stream and select the next 10 IDs."
            ),
            "batch_d_target_n": BATCH_D_TARGET_N,
            "batch_d_selected_ids": batch_d_selected_ids,
            "batch_d_present_ids": batch_d_present_ids,
            "batch_d_missing_ids": batch_d_missing_ids,
            "batch_abcd_exact_match": (
                sorted(batch_a_present_ids) == sorted(batch_a_selected_ids)
                and sorted(batch_b_present_ids) == sorted(batch_b_selected_ids)
                and sorted(batch_c_present_ids) == sorted(batch_c_selected_ids)
                and sorted(batch_d_present_ids) == sorted(batch_d_selected_ids)
                and len(non_batch_abcd_composed_non_strain_ids) == 0
            ),
            "non_batch_abcd_composed_non_strain_ids": non_batch_abcd_composed_non_strain_ids,
            "composition_form_hist": composition_form_hist,
            "secondary_present_count": len([r for r in composed_rows if r["has_secondary"]]),
            "secondary_present_rate_within_composed": (
                len([r for r in composed_rows if r["has_secondary"]]) / len(composed_rows) if composed_rows else None
            ),
            "severity_by_composed": {
                "composed": _severity_stats(composed_rows),
                "primitive_only": _severity_stats(primitive_rows),
            },
            "severity_by_composition_form": severity_by_form,
            "feedback_vs_nonfeedback_severe_or_critical": {
                "feedback": {
                    "severe_or_critical": feedback_severe,
                    "not_severe_or_critical": (feedback_total - feedback_severe),
                    "total": feedback_total,
                    "rate": feedback_rate,
                },
                "nonfeedback": {
                    "severe_or_critical": nonfeedback_severe,
                    "not_severe_or_critical": (nonfeedback_total - nonfeedback_severe),
                    "total": nonfeedback_total,
                    "rate": nonfeedback_rate,
                },
                "risk_ratio_feedback_vs_nonfeedback": risk_ratio_feedback_vs_nonfeedback,
            },
            "severity_by_secondary_presence": {
                "secondary_present": _severity_stats(secondary_present_rows),
                "secondary_absent": _severity_stats(secondary_absent_rows),
            },
        },
    }

    _dump_json(OUT_ENRICHED_PATH, enriched)
    _dump_json(OUT_SUMMARY_PATH, summary)
    OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH.write_text(_render_markdown(summary, enriched, class_names), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Incident Database sample-50 mapping to QA failure algebra F1-F6")
    parser.parse_args()
    summary = run()
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

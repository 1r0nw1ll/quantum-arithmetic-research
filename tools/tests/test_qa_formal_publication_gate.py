# noqa: DECL-1 (test harness — not empirical QA code)
"""
Executable regression tests for tools/qa_formal_publication_gate.py.

Phase 1 only needs a small set of failure-mode fixtures:
  - tautological invariant
  - missing audience translation
  - semantics/bounds conflation
  - premature publication-ready claim
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GATE = REPO_ROOT / "tools" / "qa_formal_publication_gate.py"
sys.path.insert(0, str(REPO_ROOT / "tools"))
from qa_formal_publication_gate import score_bundle  # noqa: E402
_results: list[tuple[str, bool, str]] = []


def test(name: str):
    def decorator(fn):
        def runner():
            try:
                fn()
                _results.append((name, True, ""))
                print(f"[PASS] {name}")
            except AssertionError as exc:
                _results.append((name, False, str(exc)))
                print(f"[FAIL] {name}: {exc}")
            except Exception as exc:
                _results.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return runner
    return decorator


def _run_gate(repo_root: Path, *paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), "--repo-root", str(repo_root), "--paths", *paths, "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def _seed_bundle(
    repo_root: Path,
    *,
    audience_translation: str | None = None,
    semantics_boundary: str | None = None,
    tla_text: str | None = None,
    extra_markdown: str | None = None,
) -> Path:
    bundle_root = repo_root / "qa_alphageometry_ptolemy"
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "Spec.tla").write_text(
        tla_text or (
            "---- MODULE Spec ----\n"
            "VARIABLE x\n"
            "Init == x = 0\n"
            "Next == x' = x + 1\n"
            "Inv_Semantic == x >= 0\n"
            "====\n"
        ),
        encoding="utf-8",
    )
    if audience_translation is not None:
        (bundle_root / "audience_translation.md").write_text(audience_translation, encoding="utf-8")
    if semantics_boundary is not None:
        (bundle_root / "semantics_boundary.md").write_text(semantics_boundary, encoding="utf-8")
    _write_json(
        bundle_root / "repo_fit_review.json",
        {
            "target_repo": "tlaplus/examples",
            "comparables": ["Counter.tla", "Clock.tla"],
            "why_belongs": "Matches small explanatory examples",
            "maintainer_value": "Readable outsider-facing example",
        },
    )
    _write_json(
        bundle_root / "skeptical_review.json",
        {
            "reviewer": "hostile-maintainer-sim",
            "recommendation": "revise",
            "rejection_arguments": ["Clarify one semantics note"],
        },
    )
    _write_json(
        bundle_root / "source_grounding.json",
        {
            "entries": [
                {
                    "claim": "The module models a simple counter.",
                    "artifact_element": "variable:x",
                    "source_ref": "visible task statement::counter",
                    "source_excerpt": "The task describes a bounded counter value.",
                    "interpretation": "The variable x tracks the current counter value.",
                    "modeled_consequence": "The model state includes the counter value.",
                    "authority_tier": "formalism",
                },
                {
                    "claim": "The module has a transition action.",
                    "artifact_element": "action:Next",
                    "source_ref": "visible task statement::counter",
                    "source_excerpt": "The task describes counter transitions.",
                    "interpretation": "The action Next advances the counter semantics.",
                    "modeled_consequence": "The model transitions update the counter value.",
                    "authority_tier": "formalism",
                },
                {
                    "claim": "The intrinsic semantics come from the task statement.",
                    "artifact_element": "semantics:intrinsic",
                    "source_ref": "visible task statement::counter",
                    "source_excerpt": "The task states the semantics of the counter transitions.",
                    "interpretation": "The semantics are the stated counter transitions.",
                    "modeled_consequence": "The model meaning follows the task transitions rather than an internal note.",
                    "authority_tier": "formalism",
                },
            ]
        },
    )
    _write_json(
        bundle_root / "repo_comparables.json",
        {
            "target_repo": "tlaplus/examples",
            "candidate_scope": "Small outsider-readable counter example",
            "in_scope_rationale": "The example matches the scope and audience of small teaching examples.",
            "comparables": [
                {
                    "artifact_name": "Clock.tla",
                    "artifact_ref": "tlaplus/examples/Clock.tla",
                    "norm_supported": "Shows simple state-machine style and outsider-facing README expectations.",
                    "similarity_axes": ["structure", "audience", "readme"],
                },
                {
                    "artifact_name": "CounterExample.tla",
                    "artifact_ref": "tlaplus/examples/CounterExample.tla",
                    "norm_supported": "Shows bounded-state educational example scope.",
                    "similarity_axes": ["structure", "semantics"],
                },
            ],
        },
    )
    if extra_markdown is not None:
        (bundle_root / "QARM_PROOF_LEDGER.md").write_text(extra_markdown, encoding="utf-8")
    return bundle_root


@test("self-test returns ok=true")
def t_self_test():
    proc = subprocess.run(
        [sys.executable, str(GATE), "--self-test"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True


@test("rejects missing audience translation")
def t_missing_audience_translation():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        _seed_bundle(
            repo_root,
            audience_translation=None,
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("audience_translation.md" in err for err in payload["reports"][0]["errors"])


@test("rejects semantics and bounds conflation")
def t_semantics_bounds_conflation():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary="The semantics are the TLC cap of 24 states and the search bound.",
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("intrinsic semantics" in err.lower() for err in payload["reports"][0]["errors"])


@test("rejects tautological invariants")
def t_tautological_invariant():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
            tla_text=(
                "---- MODULE Spec ----\n"
                "VARIABLE b\n"
                "Init == b = 0\n"
                "Next == b' = b + 1\n"
                "Inv_S1_NoSquareOperator == b * b >= 0\n"
                "====\n"
            ),
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("tautological" in err.lower() for err in payload["reports"][0]["errors"])


@test("rejects premature publication claims")
def t_premature_publication_claim():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
            extra_markdown="This bundle is publication-ready and submittable upstream.\n",
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("publication-ready" in err.lower() or "submittable upstream" in err.lower() for err in payload["reports"][0]["errors"])


@test("rejects content-free repo fit review")
def t_content_free_repo_fit():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        _write_json(
            bundle_root / "repo_fit_review.json",
            {
                "target_repo": "tlaplus/examples",
                "comparables": ["TBD", "N/A"],
                "why_belongs": "tbd",
                "maintainer_value": "placeholder",
            },
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("content-free" in err.lower() or "comparables" in err.lower() for err in payload["reports"][0]["errors"])


@test("rejects templated skeptical review")
def t_templated_skeptical_review():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        _write_json(
            bundle_root / "skeptical_review.json",
            {
                "reviewer": "template-bot",
                "recommendation": "accept",
                "rejection_arguments": ["Looks fine.", "Template: no issues found."],
            },
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("skeptical_review.json" in err for err in payload["reports"][0]["errors"])


@test("rejects mechanically generated human approval")
def t_mechanical_human_approval():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        _write_json(
            bundle_root / "human_approval.json",
            {
                "approved": True,
                "approver": "automation-bot",
                "approved_at": "2026-04-23",
                "scope": "formal publication",
                "justification": "This approval was auto-generated by the pipeline after the JSON files were created automatically.",
            },
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("human_approval.json" in err for err in payload["reports"][0]["errors"])


@test("rejects unsupported source grounding")
def t_unsupported_source_grounding():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        _write_json(
            bundle_root / "source_grounding.json",
            {
                "entries": [
                    {
                        "claim": "The semantics come from an internal theorem note.",
                        "artifact_element": "semantics:intrinsic",
                        "source_ref": "private/qa_internal_note.md",
                        "source_excerpt": "internal note excerpt",
                        "interpretation": "The semantics definitely prove the public model and extra observer claims.",
                        "modeled_consequence": "The public model inherits the private theorem.",
                        "authority_tier": "internal",
                    }
                ]
            },
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("source_grounding.json" in err for err in payload["reports"][0]["errors"])


@test("rejects self-referential source grounding")
def t_self_referential_source_grounding():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        _write_json(
            bundle_root / "source_grounding.json",
            {
                "entries": [
                    {
                        "claim": "The variable is justified by the README.",
                        "artifact_element": "variable:x",
                        "source_ref": "README.md",
                        "source_excerpt": "The README says x is the variable.",
                        "interpretation": "The README fully grounds the semantics.",
                        "modeled_consequence": "The model is grounded by itself.",
                        "authority_tier": "self",
                    },
                    {
                        "claim": "The action is justified by the README.",
                        "artifact_element": "action:Next",
                        "source_ref": "README.md",
                        "source_excerpt": "The README says Next updates the state.",
                        "interpretation": "The README fully grounds the action.",
                        "modeled_consequence": "The model is grounded by itself.",
                        "authority_tier": "self",
                    },
                    {
                        "claim": "The semantics are justified by the README.",
                        "artifact_element": "semantics:intrinsic",
                        "source_ref": "README.md",
                        "source_excerpt": "The README states the semantics.",
                        "interpretation": "The README fully grounds the semantics.",
                        "modeled_consequence": "The model is grounded by itself.",
                        "authority_tier": "self",
                    },
                ]
            },
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("self-referential" in err.lower() for err in payload["reports"][0]["errors"])


@test("rejects weak repo comparables")
def t_weak_repo_comparables():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        _write_json(
            bundle_root / "repo_comparables.json",
            {
                "target_repo": "tlaplus/examples",
                "candidate_scope": "small example",
                "in_scope_rationale": "fits",
                "comparables": [
                    {
                        "artifact_name": "Example",
                        "artifact_ref": "somewhere",
                        "norm_supported": "generic",
                        "similarity_axes": ["misc"],
                    },
                    {
                        "artifact_name": "Another",
                        "artifact_ref": "somewhere-else",
                        "norm_supported": "generic",
                        "similarity_axes": ["misc"],
                    },
                ],
            },
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("repo_comparables.json" in err for err in payload["reports"][0]["errors"])


@test("adversarial checker lowers admissibility when evidence is weak")
def t_adversarial_checker_lowers_admissibility():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a theorem-facing observer projection firewall.\n"
                "Why useful: useful for public review.\n"
                "This uses formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the observer projection firewall is maintained.\n"
                "TLC bounds: the model checking cap is separate.\n"
            ),
        )
        _write_json(
            bundle_root / "source_grounding.json",
            {
                "entries": [
                    {
                        "claim": "The semantics come from a private theorem note.",
                        "artifact_element": "semantics:intrinsic",
                        "source_ref": "private/theorem_note.md",
                        "source_excerpt": "private theorem note excerpt supporting semantics",
                        "interpretation": "The excerpt proves the public semantics and extra observer obligations.",
                        "modeled_consequence": "The public model inherits the observer theory.",
                        "authority_tier": "internal",
                    }
                ]
            },
        )
        report = score_bundle(bundle_root, require_artifacts=True)
        assert any("Adversarial check" in err for err in report["errors"])
        assert report["scores"]["external_admissibility_score"] <= 1


@test("rejects publication claims in unexpected text files")
def t_publication_claim_in_unexpected_text_file():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        (bundle_root / "submission.txt").write_text(
            "This contribution is publication-ready and submittable upstream.\n",
            encoding="utf-8",
        )
        proc = _run_gate(repo_root, "qa_alphageometry_ptolemy/Spec.tla")
        assert proc.returncode == 1
        payload = json.loads(proc.stdout)
        assert any("submission.txt" in err for err in payload["reports"][0]["errors"])


@test("review scoring catches visible semantics bounds conflation")
def t_review_scoring_semantics_bounds_conflation():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        (bundle_root / "README.md").write_text(
            "# Counter Example\n\n"
            "The semantics of this model are the TLC cap of 3 and the bounded search depth.\n",
            encoding="utf-8",
        )
        report = score_bundle(bundle_root, require_artifacts=False)
        assert any("conflates intrinsic semantics with TLC bounds" in err for err in report["errors"])
        assert report["scores"]["semantics_vs_bounds_clarity_score"] <= 1


@test("stuttering check does not false-positive real transitions")
def t_stuttering_check_avoids_real_transitions():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
            tla_text=(
                "---- MODULE Counter ----\n"
                "EXTENDS Naturals\n"
                "VARIABLE counter\n"
                "Init == counter = 0\n"
                "Next == \\/ /\\ counter < 3\n"
                "           /\\ counter' = counter + 1\n"
                "        \\/ /\\ counter' = 0\n"
                "====\n"
            ),
        )
        report = score_bundle(bundle_root, require_artifacts=False)
        assert not any("stuttering-only" in err for err in report["errors"])


@test("scoring rewards source grounding and comparables evidence")
def t_scoring_rewards_source_grounding_and_comparables():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        (bundle_root / "README.md").write_text(
            "# Counter Example\n\n"
            "This module models a single bounded counter.\n\n"
            "## Source grounding\n"
            "The semantics come directly from the visible task statement.\n"
            "The chosen variable `x` tracks the current value and the chosen action names are justified by the task transitions.\n\n"
            "## Repository fit\n"
            "This belongs in tlaplus/examples, similar to Clock.tla and other small counter-style examples.\n",
            encoding="utf-8",
        )
        report = score_bundle(bundle_root, require_artifacts=False)
        assert report["scores"]["source_grounding_score"] >= 2
        assert report["scores"]["repo_comparables_evidence_score"] >= 2


@test("scoring penalizes bare repository fit claims without comparables")
def t_scoring_penalizes_bare_repo_fit_claim():
    with tempfile.TemporaryDirectory(prefix="qa_formal_gate_") as tmp:
        repo_root = Path(tmp)
        bundle_root = _seed_bundle(
            repo_root,
            audience_translation=(
                "What is modeled: a simple counter in TLA+.\n"
                "Why useful: it explains the state machine and why it matters to formal readers.\n"
                "This uses outsider-facing formal vocabulary.\n"
            ),
            semantics_boundary=(
                "Intrinsic semantics: the counter evolves by Next.\n"
                "TLC bounds: the model checking bound is only a search cap.\n"
            ),
        )
        (bundle_root / "README.md").write_text(
            "# Counter Example\n\n"
            "This module models a single bounded counter.\n\n"
            "## Repository fit\n"
            "This belongs in tlaplus/examples.\n",
            encoding="utf-8",
        )
        report = score_bundle(bundle_root, require_artifacts=False)
        assert any("comparable evidence" in err.lower() for err in report["errors"])
        assert report["scores"]["repo_comparables_evidence_score"] <= 1


def main() -> int:
    for name, fn in list(globals().items()):
        if name.startswith("t_") and callable(fn):
            fn()
    failures = [entry for entry in _results if not entry[1]]
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

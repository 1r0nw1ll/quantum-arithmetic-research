"""Beta-A agent-task specifications (spec-only; execution is Beta-B).

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Five tasks split into two classes:

HEAD-TO-HEAD (content-graded; both QA-MEM and A-RAG attempt):
  T1 — primary-source grounding for a Keely triune claim
  T2 — surface both sides of a documented dispute

CAPABILITY-ONLY (QA-MEM internal consistency):
  T3 — provenance traversal via kg.why()
  T4 — domain filter purity
  T5 — min_authority=internal excludes agent nodes (vacuous today at
       0 agent nodes; auto-activates as regression when first agent
       node lands)

This module is a spec, not a runner. Beta-B imports TASKS and executes.
The head-to-head grading rubric is documented per task; graders are
instructed to not see system-of-origin labels when scoring T1/T2.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTask:
    id: str
    kind: str  # "head_to_head" | "capability"
    prompt: str
    pass_criterion: str
    systems_under_test: tuple[str, ...]
    grading_rubric: str


T1 = AgentTask(
    id="T1",
    kind="head_to_head",
    prompt=(
        "Find the specific primary-source text that grounds the Keely "
        "triune-force claim. Return the quote, the source, and the "
        "cert family that formalizes it."
    ),
    pass_criterion=(
        "Returned output lets an uninformed reader identify the Keely "
        "Law 11 text (sc:keely_law_11_triune_force) AND the cert "
        "cert:fs:qa_keely_triune_cert_v1 WITHOUT the reader needing to "
        "run further queries."
    ),
    systems_under_test=("qa_mem", "a_rag"),
    grading_rubric=(
        "Grader receives the system output labeled only as 'System A' "
        "or 'System B'. Pass = both the primary quote AND the cert ID "
        "present in the output. Partial credit is NOT given — both "
        "must be present for pass."
    ),
)

T2 = AgentTask(
    id="T2",
    kind="head_to_head",
    prompt=(
        "Is there a documented dispute about Keely Law 17's classification? "
        "Summarize both sides and cite the source of each."
    ),
    pass_criterion=(
        "Returned output surfaces BOTH (i) the original Keely Law 17 text "
        "(sc:keely_law_17_transformation_21_octaves) AND (ii) the Vibes "
        "2026-04-05 structural-reclassification observation "
        "(obs:keely_law_17_vibes_structural_reclassification). Reader "
        "must be able to tell that the dispute exists and what each side "
        "claims."
    ),
    systems_under_test=("qa_mem", "a_rag"),
    grading_rubric=(
        "Double-blind: grader receives System A/B output without the "
        "system label. Pass = both sides of the dispute are identifiable "
        "in the output text. Missing either side = fail."
    ),
)

T3 = AgentTask(
    id="T3",
    kind="capability",
    prompt=(
        "Trace the provenance chain of cert:fs:qa_keely_triune_cert_v1 "
        "back to the axiom layer."
    ),
    pass_criterion=(
        "kg.why('cert:fs:qa_keely_triune_cert_v1', max_depth=3) returns "
        "at least one chain segment whose destination has "
        "epistemic_status='axiom'. Expected: axiom:A1 at depth 1."
    ),
    systems_under_test=("qa_mem",),
    grading_rubric=(
        "Machine-checked: assert axiom:A1 appears as a dst_id with "
        "epistemic_status='axiom' in the why() output chain."
    ),
)

T4 = AgentTask(
    id="T4",
    kind="capability",
    prompt=(
        "Retrieve Parker-related primary material, filtered to "
        "geometry domain only."
    ),
    pass_criterion=(
        "KG.search_authority_ranked('Parker', domain='geometry', k=10) "
        "returns results where EVERY hit has domain='geometry'. No "
        "domain='' or other-domain leaks."
    ),
    systems_under_test=("qa_mem",),
    grading_rubric=(
        "Machine-checked: "
        "all(hit.node['domain'] == 'geometry' for hit in results)."
    ),
)

T5 = AgentTask(
    id="T5",
    kind="capability",
    prompt=(
        "Retrieve material at min_authority='internal'; verify no agent "
        "authority leaks into the result."
    ),
    pass_criterion=(
        "KG.search_authority_ranked('keely', min_authority='internal', "
        "k=10) returns results where NO hit has authority='agent'. "
        "Vacuous-PASS today at 0 agent nodes; auto-activates as "
        "regression check when the first agent node lands."
    ),
    systems_under_test=("qa_mem",),
    grading_rubric=(
        "Machine-checked: "
        "not any(hit.authority == 'agent' for hit in results). "
        "Log count of agent nodes in DB at execution time so a future "
        "run can distinguish vacuous-pass from real-exclusion-pass."
    ),
)


TASKS: tuple[AgentTask, ...] = (T1, T2, T3, T4, T5)


def as_dict_list() -> list[dict]:
    """JSON-serializable view for report generation in Beta-B."""
    return [
        {
            "id": t.id,
            "kind": t.kind,
            "prompt": t.prompt,
            "pass_criterion": t.pass_criterion,
            "systems_under_test": list(t.systems_under_test),
            "grading_rubric": t.grading_rubric,
        }
        for t in TASKS
    ]


if __name__ == "__main__":
    import json
    print(json.dumps(as_dict_list(), indent=2))

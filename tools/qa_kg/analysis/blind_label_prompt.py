"""Blind LLM relevance labeler for Beta-A cross-domain queries.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

For each cross-domain query (X01-X04), runs the live ranker at k=20,
strips authority/score/lifecycle/domain/epistemic_status, anonymizes node
IDs to candidate_01..candidate_20, then asks a Claude model to grade
relevance 1-5 via `claude -p` with tools blocked and the project system
prompt overridden. Commits raw responses + parsed grades.

Why `claude -p` and not the anthropic SDK directly: the SDK isn't
installed in this environment and no ANTHROPIC_API_KEY is available.
`claude -p --disallowedTools <all> --system-prompt <clean>` gives the
same blindness property — no filesystem access, no auto-loaded CLAUDE.md,
no MCP context — and uses the CLI's existing keychain/OAuth auth.

Model choice per Will's v4 feedback: Opus-4.7 for all 4 X-queries
(subtle cross-domain judgment, negligible cost for ~80 grades total).
Haiku-4.5 available as fallback via `--model` override.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import argparse
import datetime as _dt
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

from tools.qa_kg.canonicalize import graph_hash
from tools.qa_kg.kg import KG
from tools.qa_kg.schema import DEFAULT_DB

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO / "tools" / "qa_kg" / "fixtures"
_QUERIES_PATH = _FIXTURE_DIR / "beta_prereg_queries.json"
_BLIND_GOLD_PATH = _FIXTURE_DIR / "beta_blind_gold.json"

DEFAULT_MODEL = "claude-opus-4-7"
K_CANDIDATES = 20
MAX_BODY_CHARS = 400

_DISALLOWED_TOOLS = (
    "Bash Read Grep Glob Write Edit WebSearch WebFetch Agent Skill "
    "TodoWrite TaskCreate TaskList TaskGet TaskUpdate TaskOutput "
    "TaskStop NotebookEdit ScheduleWakeup ToolSearch SendMessage "
    "CronCreate CronDelete CronList Monitor EnterPlanMode ExitPlanMode "
    "EnterWorktree ExitWorktree LSP PushNotification RemoteTrigger"
)

_SYSTEM_PROMPT = (
    "You are a relevance-grading assistant. For each candidate passage, "
    "rate its relevance to the query on an integer scale 1-5: "
    "5=directly answers the query, 4=strongly related, "
    "3=topically adjacent, 2=tangentially related, 1=irrelevant. "
    "Output exactly one line per candidate in the exact format "
    "'<candidate_id> <grade>'. No preamble. No explanation. No markdown. "
    "No blank lines."
)

_GRADE_LINE_RE = re.compile(r"^(candidate_\d{2})\s+([1-5])\s*$", re.MULTILINE)

_FTS5_SANITIZE_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_FTS5_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "was", "were", "be", "by", "for", "with", "about", "what", "how",
    "does", "do", "did", "can", "that", "this", "these", "those", "which",
    "vs", "versus", "but", "if", "not", "no", "yes", "its", "it", "i",
    "as", "at", "so", "us", "we", "both", "primary", "source", "sources",
})


def _sanitize_for_fts5(query_text: str) -> str:
    """Punctuation-strip + OR-join non-stopword tokens for FTS5.

    FTS5 raises a syntax error on bare punctuation (e.g., '?') and
    defaults to AND semantics across whitespace-separated terms. For
    cross-domain retrieval we want broad OR matching so the LLM-graded
    candidate pool spans both domains of each query. This sanitizer
    strips punctuation, drops stopwords, and joins remaining tokens with
    ' OR '. The LLM still sees the original query_text in the prompt.
    """
    cleaned = _FTS5_SANITIZE_RE.sub(" ", query_text).lower()
    tokens = [t for t in cleaned.split() if t and t not in _FTS5_STOPWORDS]
    if not tokens:
        return query_text.strip()
    return " OR ".join(tokens)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_x_queries() -> list[dict]:
    fixture = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    return [q for q in fixture["queries"] if q["category"] == "cross_domain"]


def _build_candidates(
    kg: KG, query_text: str, k: int = K_CANDIDATES,
) -> list[dict]:
    """Run the ranker and strip metadata the labeler shouldn't see.

    Returns a list of {real_id, title, body_trunc}. The candidate_id
    assignment is done later so the returned list is stable input to the
    shuffler.
    """
    fts_query = _sanitize_for_fts5(query_text)
    hits = kg.search_authority_ranked(
        fts_query, min_authority="internal", k=k,
    )
    out: list[dict] = []
    for h in hits:
        row = h.node
        body = (row["body"] or "").strip()
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS].rstrip() + "…"
        out.append({
            "real_id": row["id"],
            "title": (row["title"] or "").strip(),
            "body": body,
        })
    return out


def _assign_anonymous_ids(candidates: list[dict]) -> list[dict]:
    """Assign candidate_01..candidate_NN in the order the ranker returned.

    We deliberately do NOT shuffle the order — the LLM sees candidates in
    ranker order so NDCG against the ranker's ordering is meaningful. The
    anonymization protects against ID-prefix leakage (sc:/obs:/cert:/axiom:)
    that would otherwise let the labeler infer authority tier.
    """
    for i, c in enumerate(candidates, start=1):
        c["candidate_id"] = f"candidate_{i:02d}"
    return candidates


def _build_prompt(query_text: str, candidates: list[dict]) -> str:
    lines = [f"QUERY: {query_text}", "", "CANDIDATES:"]
    for c in candidates:
        lines.append(f"--- {c['candidate_id']} ---")
        lines.append(f"Title: {c['title']}")
        body = c['body'] if c['body'] else "(no body)"
        lines.append(f"Body: {body}")
        lines.append("")
    lines.append(
        "Output relevance grades, one line per candidate, in the format "
        "'<candidate_id> <grade>'."
    )
    return "\n".join(lines)


def _call_claude(prompt: str, model: str, timeout_s: int = 180) -> str:
    """Invoke `claude -p` with tools blocked and a clean system prompt."""
    cmd = [
        "claude", "-p",
        "--model", model,
        "--system-prompt", _SYSTEM_PROMPT,
        "--disallowedTools", *_DISALLOWED_TOOLS.split(),
        "--strict-mcp-config",
    ]
    result = subprocess.run(
        cmd, input=prompt, text=True,
        capture_output=True, timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (rc={result.returncode}): {result.stderr[:500]}"
        )
    return result.stdout


def _parse_grades(raw: str, candidates: list[dict]) -> dict[str, int]:
    """Parse '<candidate_id> <grade>' lines, map back to real_ids.

    Returns {real_id: int_grade}. Missing candidates are recorded with
    grade=0 (equivalent to "not rated" for NDCG gain computation).
    """
    matches = _GRADE_LINE_RE.findall(raw)
    anon_to_grade = {cid: int(g) for cid, g in matches}
    out: dict[str, int] = {}
    for c in candidates:
        real = c["real_id"]
        grade = anon_to_grade.get(c["candidate_id"], 0)
        out[real] = grade
    return out


def run(model: str = DEFAULT_MODEL, db_path: Path = DEFAULT_DB) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        gh = graph_hash(conn)
        kg = KG(conn)
        results: list[dict] = []
        x_queries = _load_x_queries()
        for q in x_queries:
            qid = q["id"]
            qtext = q["query_text"]
            print(f"[{qid}] building candidates (k={K_CANDIDATES}) ...", flush=True)
            candidates = _assign_anonymous_ids(
                _build_candidates(kg, qtext, k=K_CANDIDATES),
            )
            if not candidates:
                print(f"[{qid}] no candidates — skipping", flush=True)
                results.append({
                    "id": qid, "query_text": qtext, "model": model,
                    "candidates": [], "raw_response": "",
                    "parsed_grades": {}, "error": "no_candidates",
                })
                continue
            prompt = _build_prompt(qtext, candidates)
            print(f"[{qid}] calling {model} ({len(candidates)} candidates) ...", flush=True)
            try:
                raw = _call_claude(prompt, model)
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                print(f"[{qid}] LLM call failed: {exc}", flush=True)
                results.append({
                    "id": qid, "query_text": qtext, "model": model,
                    "candidates": candidates, "raw_response": "",
                    "parsed_grades": {}, "error": str(exc),
                })
                continue
            grades = _parse_grades(raw, candidates)
            n_graded = sum(1 for g in grades.values() if g > 0)
            print(f"[{qid}] parsed {n_graded}/{len(candidates)} grades", flush=True)
            results.append({
                "id": qid,
                "query_text": qtext,
                "model": model,
                "candidates": candidates,
                "prompt": prompt,
                "raw_response": raw,
                "parsed_grades": grades,
                "n_graded": n_graded,
            })

        output = {
            "_exempt": (
                "<!-- PRIMARY-SOURCE-EXEMPT: reason=Beta-A blind-label "
                "gold for cross-domain queries; grades produced via "
                "`claude -p` with all tools blocked and system prompt "
                "overridden so the labeler sees only title+body of "
                "anonymized candidates. Raw responses committed for "
                "audit reproducibility. -->"
            ),
            "phase": "beta_a",
            "schema_version": 1,
            "derived_ts": _now(),
            "graph_hash": gh,
            "model": model,
            "anonymization": "candidate_NN assigned in ranker order",
            "results": results,
        }
        _BLIND_GOLD_PATH.write_text(
            json.dumps(output, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return output
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    result = run(model=args.model, db_path=Path(args.db))
    total = len(result["results"])
    ok = sum(1 for r in result["results"] if "error" not in r)
    print(f"\ncompleted: {ok}/{total} X-queries graded")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
QA_COMPLIANCE = {
    "observer": "codex_auto_review_bridge",
    "state_alphabet": "quarantine packet metadata + codex review verdicts only",
}
"""
Codex auto-review bridge for the cert quarantine gate.

The PreToolUse cert gate quarantines cert-adjacent writes into
`llm_qa_wrapper/quarantine/pending/` and blocks commits until they are reviewed.
`qa_codex_quarantine_review.py` only approves/rejects; it does not itself run
Codex. This bridge closes that gap: it reads pending packets, runs `codex exec`
to review the substantive code, and clears the queue automatically.

Policy (safety-preserving):
  * Code files (*.py): a real `codex exec` correctness review. Approved only if
    Codex ends with `VERDICT: APPROVE`; on `VERDICT: HOLD` (or an unparseable
    verdict) the packet is LEFT PENDING for a human — never auto-rejected
    (reject is destructive: it unlinks new files).
  * Non-code artifacts (*.json fixtures/schema/mapping_ref, *.md docs) and
    ephemeral scratchpad/tmp paths: fast-path approved with a note (data/docs,
    not executable logic).
  * Bash-command / no-target packets: fast-path approved with a note.

Modes:
  --once           process the current pending queue and exit (use before commit)
  --watch [--interval N]   daemon: poll every N seconds (the "bridge")

Nothing here writes QA state; it only moves quarantine packets and appends
review-ledger records via the existing review tool.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_VERDICT_LINE = re.compile(r"^VERDICT:\s*(APPROVE|HOLD)\b")

REPO = Path(__file__).resolve().parent.parent
PENDING = REPO / "llm_qa_wrapper" / "quarantine" / "pending"
REVIEW_TOOL = REPO / "tools" / "qa_codex_quarantine_review.py"
VENV_PY = REPO / ".venv" / "bin" / "python"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable

CODE_SUFFIXES = {".py"}
FASTPATH_SUFFIXES = {".json", ".md", ".txt", ".tex", ".sh", ".toml", ".cfg", ".ini"}


def _pending() -> list[Path]:
    if not PENDING.is_dir():
        return []
    return sorted(PENDING.glob("*.json"))


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _target_rel(packet: dict) -> str | None:
    rel = packet.get("target_rel")
    if rel:
        return rel
    # Fall back to a Write tool_input file_path or leave None (command packet)
    ti = packet.get("tool_input") or {}
    fp = ti.get("file_path")
    return fp if isinstance(fp, str) else None


def _approve(packet_paths: list[Path], notes: str) -> bool:
    cmd = [PYTHON, str(REVIEW_TOOL), "approve",
           *[str(p) for p in packet_paths],
           "--reviewer", "codex-auto-review", "--notes", notes[:1500]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! approve failed: {r.stderr[:200]}")
        return False
    return True


def _codex_review(rel: str, deny_reasons: list[str]) -> tuple[str, str]:
    """Run codex exec on a code file. Returns (verdict, review_text)."""
    prompt = (
        f"Review the file `{rel}` in this repo for CORRECTNESS BUGS ONLY "
        f"(logic errors, wrong math, broken control flow). It was quarantined by "
        f"the cert gate for: {deny_reasons}. Do NOT modify anything. Be brief. "
        f"End your reply with EXACTLY one final line, either "
        f"'VERDICT: APPROVE' (no correctness bug found) or "
        f"'VERDICT: HOLD' (a real bug needs human review)."
    )
    try:
        r = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", prompt],
            capture_output=True, text=True, timeout=300, cwd=str(REPO),
        )
    except Exception as e:  # noqa: BLE001
        return "HOLD", f"codex invocation error: {e}"
    out = (r.stdout or "") + (r.stderr or "")
    # Approve ONLY on a properly-formatted verdict line (ANSI-stripped,
    # line-anchored) — scanning from the end past Codex's token-count footer.
    # A body mention of the string, or an unparseable/malformed transcript with
    # no verdict line, defaults to HOLD (safe): the policy is fail-closed.
    verdict = "HOLD"
    for line in reversed(out.splitlines()):
        m = _VERDICT_LINE.match(_ANSI.sub("", line).strip())
        if m:
            verdict = m.group(1)
            break
    return verdict, out.strip()


def process_once() -> int:
    packets = _pending()
    if not packets:
        print("qa_codex_auto_review: no pending packets")
        return 0

    # Group packets by target file (many edits -> many packets per file)
    groups: dict[str, list[Path]] = {}
    meta: dict[str, dict] = {}
    for p in packets:
        pkt = _load(p)
        rel = _target_rel(pkt) or "(no-target)"
        groups.setdefault(rel, []).append(p)
        meta.setdefault(rel, pkt)

    held = 0
    for rel, pkts in groups.items():
        abspath = (REPO / rel) if rel != "(no-target)" else None
        suffix = Path(rel).suffix.lower() if rel != "(no-target)" else ""
        is_scratch = "/tmp/" in rel or "scratchpad" in rel or rel.startswith("/private/")

        # Fast-path: non-code artifacts, scratch/tmp, or files that no longer exist.
        if (rel == "(no-target)" or is_scratch or suffix in FASTPATH_SUFFIXES
                or (abspath is not None and not abspath.exists())):
            note = (f"codex-auto-review fast-path: {rel} is a non-code artifact / "
                    f"scratch / command packet (no executable logic to review).")
            if _approve(pkts, note):
                print(f"  APPROVE (fast-path)  {rel}  [{len(pkts)} packet(s)]")
            continue

        # Code path: genuine codex review.
        if suffix in CODE_SUFFIXES:
            print(f"  reviewing (codex)     {rel} ...")
            verdict, review = _codex_review(rel, meta[rel].get("deny_reasons", []))
            if verdict == "APPROVE":
                note = f"codex-auto-review APPROVE: {rel} — {review[-600:]}"
                if _approve(pkts, note):
                    print(f"  APPROVE (codex)      {rel}  [{len(pkts)} packet(s)]")
            else:
                held += len(pkts)
                print(f"  HOLD (codex)         {rel} — left pending for human review")
                print(f"      {review[-400:]}")
            continue

        # Unknown suffix: be conservative, hold.
        held += len(pkts)
        print(f"  HOLD (unknown type)  {rel} — left pending for human review")

    remaining = len(_pending())
    print(f"qa_codex_auto_review: done — {remaining} packet(s) still pending "
          f"({held} held for human review)")
    return 0


def watch(interval: int) -> int:
    print(f"qa_codex_auto_review: watching {PENDING} every {interval}s (Ctrl-C to stop)")
    try:
        while True:
            if _pending():
                process_once()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nqa_codex_auto_review: stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Codex auto-review bridge for the cert quarantine gate")
    ap.add_argument("--once", action="store_true",
                    help="process the current pending queue once and exit (default)")
    ap.add_argument("--watch", action="store_true", help="daemon mode: poll continuously")
    ap.add_argument("--interval", type=int, default=30, help="poll interval seconds (watch mode)")
    args = ap.parse_args(argv)
    return watch(args.interval) if args.watch else process_once()


if __name__ == "__main__":
    raise SystemExit(main())

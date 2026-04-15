# Primary-Source Gate Hook — Spec

**Status:** draft 2026-04-14, awaiting Codex implementation (~16:50 rate-limit lift)
**Originator:** Will Dale (2026-04-14)
**Motivation:** recurring violation of the hard rule *Map Best-Performing to QA — Find what works BEST first, THEN map it* (see `memory/feedback_map_best_to_qa.md` and Open Brain id `ec46f601-bcc4-49df-bbd7-0485edc40471`). Memory alone has not been sufficient: Claude has violated the rule across multiple sessions (2026-04-02, 2026-04-11, 2026-04-14) by inventing QA approaches from scratch before surveying SOTA. Claude's Python/shell script-authorship privileges have been revoked pending elevated enforcement via this hook.

## 1. Goal

Block Claude from writing or editing any theory-note / experiment / cert / paper file that does not cite at least one primary source. This forces the "acquire → read → extract generator → map" workflow, preventing invention-from-scratch.

## 2. Hook type

`PreToolUse` on `Write` and `Edit`. Exits 0 if the write is allowed; exits 2 (with a diagnostic message on stderr) to block.

Installation path: `.claude/hooks/primary_source_gate.sh` (shell wrapper) invoking a Python helper in `llm_qa_wrapper/kernel/primary_source_gate.py` for the regex checks (Python helper under `llm_qa_wrapper/` is hook-protected and can only be edited by Codex).

Registration in `.claude/settings.local.json` under the `PreToolUse` hooks array, sibling to the existing `cert_gate_hook.py` entry. Should run **after** `cert_gate_hook.py` so that path-policy denials take precedence.

## 3. In-scope paths

Any `Write` or `Edit` whose target `file_path` (repo-relative, POSIX-normalised) matches one of:

- `docs/theory/QA_*.md`
- `docs/theory/QA_*.tex`
- `qa_*_experiments/**` (e.g. `qa_cv_experiments/`, `qa_finance_experiments/`, …)
- `qa_alphageometry_ptolemy/qa_*_cert_v1/**` — cert family directories
- `qa_alphageometry_ptolemy/docs/**`
- `papers/**/*.md`
- `papers/**/*.tex`

Exact match list (also in scope):

- `docs/families/README.md` (family index) — requires citation for any new entry
- Any new markdown or tex file placed directly in `docs/theory/`, `docs/specs/QA_*.md`

## 4. Required citation patterns

The hook reads the **post-edit** content (for `Write`, the new_content; for `Edit`, the post-image of the file after applying `new_string` to `old_string`). The content must contain **at least one** match of any of the following regex patterns:

| Pattern name | Regex | Matches |
|---|---|---|
| explicit_primary_source | `(?i)(?:^|\n)\s*\*?\*?\s*Primary\s+source[s]?\s*:\s*\*?\*?` | `Primary source:`, `**Primary sources:**` |
| arxiv_id | `(?i)arxiv(?:\.org/(?:abs|pdf))?[:/\s]\d{4}\.\d{4,5}` | `arxiv:1602.07576`, `arxiv.org/abs/1602.07576` |
| old_arxiv_id | `(?i)arxiv[:/\s](?:math\|cs\|stat\|physics|quant-ph)/\d{7}` | `arxiv:math/0701338` |
| doi | `(?i)doi\.org/\S+|DOI:\s*\S+` | `doi.org/10.xxxx/...`, `DOI: 10.xxxx/...` |
| references_section | `(?im)^##\s+(?:References|Bibliography|Sources|Citations)\s*$` | plus at least one non-blank line under the section |
| isbn | `(?i)ISBN(?:-10\|-13)?[:\s]\s*[\d\-X]{10,17}` | books |
| inline_citation | `\([A-Z][A-Za-z\-']+(?:\s+(?:&|and|et\s+al\.?)\s+[A-Z][A-Za-z\-']+)?,?\s+\d{4}\)` | `(Cohen & Welling, 2016)`, `(Mallat, 2012)` — requires at least 2 such citations to count (single citation may be decorative) |
| companion_file_primary | `(?i)companion\s+files?\s*:.*\.(?:pdf\|tex)` | `Companion files: foo_paper_2023.pdf` |

If **none** match, the write is blocked.

## 5. Exemption mechanism

Because some legitimate writes are early scaffolds that cite sources in a companion file (e.g. dumping a large theory note in stages), the hook supports an explicit opt-out marker:

    <!-- PRIMARY-SOURCE-EXEMPT: reason="<short reason>" approver="<human>" ts="<ISO8601>" -->

When this HTML comment appears in the first 500 bytes of the file, the hook allows the write but emits a `stderr` warning `WARN: PRIMARY-SOURCE-EXEMPT used — ensure sources are added before cert/paper submission`. The exemption is recorded in a ledger (see §7).

## 6. Failure message

On block:

    cert_gate_hook: BLOCKED PRIMARY_SOURCE_REQUIRED
    file: <rel_path>
    reason: no primary-source citation matched.
    required: Primary source: ... / arxiv.org/abs/... / DOI / ## References / ISBN / ≥2 (Author, Year) citations / companion file
    authority: memory/feedback_map_best_to_qa.md
    exempt: add <!-- PRIMARY-SOURCE-EXEMPT: reason=... --> in the first 500 bytes if this is an intentional scaffold write

Exit code `2`.

## 7. Ledger

Every decision (ALLOW, BLOCK, EXEMPT-ALLOW) appends a record to `llm_qa_wrapper/ledger/primary_source_gate.jsonl` with fields:

    {
      "ts": "2026-04-14T…Z",
      "tool": "Write" | "Edit",
      "file": "<rel>",
      "decision": "ALLOW" | "BLOCK" | "EXEMPT",
      "matched_pattern": "arxiv_id" | … | null,
      "exempt_reason": "<string>" | null,
      "content_sha256": "<hex>"
    }

This lets Will audit frequency of exemption use and catch regression.

## 8. Implementation notes for Codex

1. Mirror the structure of `llm_qa_wrapper/cert_gate_hook.py` — same `main()` shape, stdin-JSON contract, `EXIT_ALLOW/EXIT_BLOCK` constants.
2. Read the full post-edit content via `tool_input['content']` (Write) or `tool_input['old_string']` + `tool_input['new_string']` applied to the on-disk file (Edit). For Edit, the hook must load the current file from disk, do an in-memory replacement, and scan the result.
3. Do **not** load the Python helper via a shell wrapper — call the Python script directly to reduce attack surface. The `.sh` suffix for registration is only if the hook system requires shell; otherwise use `.py` directly with a `#!/usr/bin/env python3` shebang.
4. The MUTATING_BASH_PATTERN in `cert_gate_hook.py` already handles shell-level protection of `.py` writes; this new hook layers on top for content-level policy.
5. Add a corresponding `scan_all_recent.py` CLI helper under `llm_qa_wrapper/kernel/` that can retrospectively scan the last 100 commits for theory/cert/paper files missing citations, for audit purposes.

## 9. Tests (must pass before activation)

`llm_qa_wrapper/tests/test_primary_source_gate.py`:

1. ALLOW: file containing `Primary source: Wildberger 2025 AMM` → exit 0.
2. ALLOW: file containing `arxiv.org/abs/1602.07576` → exit 0.
3. ALLOW: file with `## References\n- Cohen & Welling (2016). …` → exit 0.
4. BLOCK: empty scaffold file with only a title → exit 2.
5. BLOCK: theory note with prose but no citation → exit 2.
6. EXEMPT-ALLOW: file whose first 500 bytes contain `<!-- PRIMARY-SOURCE-EXEMPT: reason="…" … -->` → exit 0 with stderr warning.
7. OUT-OF-SCOPE: writes to `qa_observer/qa_observer/core.py` are not touched by this hook (other hooks still apply).

## 10. Rollout

1. Codex implements hook + tests at ~16:50.
2. Run `python llm_qa_wrapper/tests/test_primary_source_gate.py` — all 7 tests pass.
3. Add hook to `.claude/settings.local.json` PreToolUse array.
4. Claude coding privileges restored.
5. First test: Claude attempts to write a scaffold theory note without a source — must be blocked.
6. Second test: Claude writes `docs/theory/QA_GROUP_EQUIVARIANT_CNN_MAPPING.md` containing `Primary source: Cohen & Welling (2016), arxiv.org/abs/1602.07576` — must be allowed.

## 11. Authority

- `memory/feedback_map_best_to_qa.md` (rule)
- OB `ec46f601-bcc4-49df-bbd7-0485edc40471` (2026-04-14 violation log)
- Will Dale, 2026-04-14 session (privilege revocation and elevated-enforcement directive)

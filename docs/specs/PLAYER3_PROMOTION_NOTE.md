# Player3 Promotion Note

**Date:** 2026-04-29
**Action:** player3 (Mac mini M4) promoted to canonical working node.
**Author:** Player3

## Status

| Role | Machine | Notes |
|---|---|---|
| Canonical | player3 (Mac mini M4, Darwin 25.3.0 arm64, Python 3.13) | Active Claude repo work happens here. |
| Backup / reference | player2 (Linux Kali) | Stop active repo-editing sessions; pull from GitHub when needed. |

## Why player3 is canonical now

Subnet split prevents Mac → player2 reachability:

- player2: `192.168.2.100/24`
- player3: `192.168.0.49/24`
- Mac → player2 ICMP and SSH both time out (player2 → Mac works fine; that path was used for the rsync transfer).

A shared live `qa-collab` bus would require either reverse SSH tunneling or routing changes. Both add fragility and aren't justified tonight. Cleanest tonight is one canonical machine, with GitHub as the cross-machine sync bridge.

## Operational rule

- Run only one active Claude repo-editing session at a time.
- That session lives on player3.
- player2 may be used for read-only reference, backup, or commands that don't mutate the repo.
- Cross-machine sync = `git push` / `git pull` on GitHub.

This avoids split-brain without needing shared file locks.

## Validation baseline (player3)

- Repo path: `/Users/player3/signal_experiments`
- HEAD pre-promotion: `ffa511f` (evals: pass 22)
- `python tools/qa_axiom_linter.py --all` → CLEAN
- `python qa_alphageometry_ptolemy/qa_meta_validator.py` → parity with player2 (only [228] D4 differs, expected per its `platform=Darwin; cross-platform parity deferred` note)
- `python -m tools.qa_kg.cli build` → 379 certs, 186 claims, 0 firewall violations
- `.claude` hooks parse cleanly; statusline path = `/Users/player3/signal_experiments`; zero `/home/player2` refs in `.claude/`

## Excluded data (still deferred)

These were not migrated and remain on player2 only:

- `archive/` (do-not-touch, ~13 GB)
- `qa_lab/data/` (~3.4 GB)
- `Documents/wildberger_corpus/`, `Documents/haramein_rsf/` (large PDF corpora)
- `llm_qa_wrapper/spec/states/` (~3.3 GB TLC traces; sources preserved)
- `qa_lab/qa_venv/`, `qa_lab/target/` (rebuildable)

Transfer subpaths only when a specific run needs them.

## Known follow-ups (not blocking)

- `llm_qa_wrapper/cert_gate_hook.py` uses PEP 604 `T | None` syntax. macOS system `python3` is 3.9 → TypeError → hook exits 0. Hook is therefore a no-op on Mac until either the shebang pins to `python3.10+` (or the venv interpreter) or the syntax is downgraded. Doesn't block this commit because the failure mode is open-by-default, but the gate isn't actually enforcing.
- `qa-collab` bus not running on Mac (port 5555 down). Hooks gracefully skip when bus is absent. Standing up a Mac-local bus is a separate task.
- 1367 `/home/player2` path references remain outside `.claude/` (in domain Python, YAML task records, docs). These are source semantics — leave as-is until a concrete script needs to run on Mac.

## Promotion checklist (per ChatGPT plan)

| # | Item | Status |
|---|---|---|
| 1 | Confirm repo state | ✓ |
| 2 | Confirm validation baseline (parity, not clean) | ✓ |
| 3 | Create promotion marker | ✓ (this file) |
| 4 | Commit the marker | (this commit) |
| 5 | Push to GitHub | (next step) |
| 6 | player2 stops active repo work | (manual, on player2) |
| 7 | player2 pulls when ready | (manual, on player2) |

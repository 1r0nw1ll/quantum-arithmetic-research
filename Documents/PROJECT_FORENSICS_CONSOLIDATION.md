# Project Forensics + Consolidation Notes

Generated: 2026-02-14

Primary artifacts (latest run):
- `_forensics/forensics_20260214_154253/REPORT.md`
- `_forensics/forensics_20260214_154253/keyword_hotspots.tsv`
- `_forensics/forensics_20260214_154253/chat_path_mentions.tsv`
- `_forensics/forensics_20260214_154253/chat_python_targets.tsv`
- `_forensics/forensics_20260214_154253/script_artifacts.tsv`

## 1) Where the project actually “lives” (bytes)

From `_forensics/forensics_20260214_154253/REPORT.md`:
- Total: ~23.5 GiB
- Dominant subtree: `qa_lab/` (~15.5 GiB), mostly `qa_lab/logs/` (~7.2 GiB) + `qa_lab/data/` (~3.4 GiB) + build/git internals.
- Chat export bulk: `chat_data_extracted/` (~1.0 GiB) and `chat_data/` (~371 MiB).
- Build artifacts: `qa_alphageometry/target/` (~377 MiB), plus multiple venvs and git object stores.

Practical implication: most “mass” is logs/data/builds, while most *interpretable* research structure is in the smaller code+docs areas.

## 2) Highest-signal “open these first” files (claims + verification language)

From `_forensics/forensics_20260214_154253/keyword_hotspots.tsv` (top hits):
- `qa_alphageometry_ptolemy/qa_meta_validator.py`
- `qa_alphageometry_ptolemy/qa_certificate.py`
- `qa_alphageometry_ptolemy/test_understanding_certificate.py`
- `qa_alphageometry_ptolemy/QA_MAP_CANONICAL.md`
- `qa_alphageometry_ptolemy/qa_verify.py`
- `qa_alphageometry_ptolemy/qa_generalization_validator_v3.py`
- `qa_alphageometry_ptolemy/qa_sparse_attention_validator_v3.py`
- `qa_alphageometry_ptolemy/qa_neuralgcm_validator_v3.py`
- `qa_alphageometry_ptolemy/qa_guardrail/qa_guardrail.py`

These are “spine candidates”: they (a) contain strong verification-oriented language, and (b) are referenced heavily by chats (see next section).

## 3) Chat ↔ repo reality check (what you actually kept coming back to)

From `_forensics/forensics_20260214_154253/REPORT.md`:
- Most-mentioned repo path in chat: `qa_alphageometry_ptolemy/qa_meta_validator.py`
- Other frequently mentioned: `qa_competency/qa_competency_validator.py`, `docs/families/README.md`, and multiple validators in `qa_alphageometry_ptolemy/`.
- Chat command mix is heavy on `python`/`pip`/`git`/`cargo`, which matches a “research + validation + tooling” workflow.

From `_forensics/forensics_20260214_154253/chat_python_targets.tsv`:
- Many chat-run python targets do **not** exist in this folder anymore (or were from other repos/old paths).
  This is a key forensic finding: it explains “we did the work, but can’t find it.”

Actionable follow-up:
- For the top missing python targets, search inside your *archives* (`*.zip`) or other workspace folders for those filenames before assuming the work is lost.

## 4) Experiments ↔ artifacts (what scripts appear to generate real outputs)

From `_forensics/forensics_20260214_154253/REPORT.md` and `_forensics/forensics_20260214_154253/script_artifacts.tsv`:
- `qa_alphageometry_ptolemy/qa_meta_validator.py` references many `qa_alphageometry_ptolemy/certs/*.json` artifacts (strong “real work” signal).
- `qa_theorem_discovery_orchestrator_rust.py` references dataset/obstruction outputs.
- `hi_2_0_visualization.py` and `hi_2_0_ablation_study.py` reference CSV/JSON/PNG outputs that exist.
- `geometric_autopsy.py` references PNGs at repo root (`1_angular_spectrum.png`, `2_tda_persistence_diagram.png`, etc.), linking a script to visible artifacts.

This gives you a concrete “what ran and produced something” trail without having to trust memory.

## 5) Redundancy clusters worth consolidating (high ROI)

From `_forensics/forensics_20260214_154253/version_families.tsv` + the size census:

### A) Snapshot archives at repo root
- Large zip snapshots exist (`archive(7).zip`, `signal_experiments.zip`, `signal_experiments1.zip`, `workspaces2.zip`…`workspaces7.zip`, `grokking_qa_overlay*.zip`, etc.).

Recommendation:
- Create a single `archives/` directory and move these zips there.
- Add a lightweight `archives/ARCHIVES_INDEX.md` with one line per zip: date, purpose, and what subproject it belongs to.

### B) “Final/corrected” script families
- Example family: `run_signal_experiments.py`, `run_signal_experiments_corrected.py`, `run_signal_experiments_final.py`.

Recommendation:
- Pick a canonical entrypoint and turn variants into flags (or archive the non-canonical ones).
- Add a short “entrypoints” table somewhere stable (e.g. `QUICKSTART.md` or a new `Documents/ENTRYPOINTS.md`).

### C) Duplicate / near-duplicate text dumps in `docs/Google AI Studio/`
- Multiple files differ only by suffixes like `(1)`, `(2)` and appear to be duplicated content.

Recommendation:
- Consolidate to one canonical file per topic and move the rest to `docs/Google AI Studio/_duplicates/` (or archive).

## 6) Suggested “results registry” structure (so nothing stays buried)

Create a single file (new): `Documents/RESULTS_REGISTRY.md` with rows like:
- Result id
- Claim (1–3 sentences)
- Evidence (validator run, cert path, plot path, or log path)
- Reproduction command(s)
- Source files
- Date / status (draft, verified, superseded)

Start by adding entries for the top 5 hotspots in section (2) + their artifacts from section (4).

Auto-generated first pass:
- `Documents/RESULTS_REGISTRY.md`

## 7) How to re-run / extend the forensics scan

- Core scan (fast, default):
  - `python tools/project_forensics.py`
- Full scan (slow; walks everything not excluded by directory name):
  - `python tools/project_forensics.py --scope full`
- Add extra scope (example: include private notes):
  - `python tools/project_forensics.py --include private/QAnotes`

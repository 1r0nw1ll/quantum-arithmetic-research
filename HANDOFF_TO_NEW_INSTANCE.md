Handoff: Move QA Fast-Path Workflow to New Instance

What’s ready
- Fast-path pipeline (gates → QE → E8) wired into QA Lab
- Automatic evaluation + daily summary + trend plotting
- Rust batch kernels built and used by default

Minimal move steps
1) Copy the repo directory to the new instance (e.g., `signal_experiments/`).
2) Create and activate a virtual environment:
   - `python -m venv .venv && source .venv/bin/activate`
3) Install packages and QA Lab in editable mode:
   - `pip install --upgrade pip wheel`
   - `pip install -e .`
4) Build the Rust extension:
   - `make -C qa_lab rust-py-build`
5) Ensure E8 roots exist (optional but recommended):
   - `python qa_lab/scripts/generate_e8_roots.py`

Run it
- One-shot evaluation: `make -C qa_lab fast-eval`
- Full meta pass: `make -C qa_lab meta`
- Agent loop (runs eval + daily summary each cycle):
  - `make -C qa_lab agent_loop`
  - Or daemon: `bash qa_lab/scripts/agent_daemon.sh`

Where to look for results
- `artifacts/evals/fastpath_eval.txt/.json` (pipeline vs baseline)
- `artifacts/evals/daily_summary_latest.txt` (human-readable daily)
- `artifacts/evals/fastpath_trends.json/.png` (trend snapshots/plot)

Useful toggles (optional)
- Gate policy: `QA_FP_ENABLE_WHEEL`, `QA_FP_ENABLE_FAMILY`, `QA_FP_FAMILY_TOL`, `QA_FP_POS_MIN`
- QE weights: `QA_QE_CURV_WEIGHT`, `QA_QE_FAMILY_WEIGHT`, `QA_QE_PHI_WEIGHT`, `QA_QE_PHI_EB_WEIGHT`, `QA_QE_IDEAL_WEIGHT`
- E8 strategy: `QA_E8_PREFER_NUMPY`, `QA_E8_DISABLE_RUST`, `QA_E8_VEC_CHUNK`, `QA_E8_ROOT_CHUNK`

Notes
- Defaults are safe; no toggles required to get value.
- If you want daily summaries broadcast to a channel, set `QA_COLLAB_LIVE=1`.


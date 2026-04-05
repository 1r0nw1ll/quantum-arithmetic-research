# QA Paper Pipeline — Reproducing the Evidence Bundle

**Status**: operational as of 2026-04-04 (cwd fix + make target chain verified).
**Authority**: 2026-04-02 strategic pivot — "GitHub-first publication. No journals,
no arXiv. Papers still written, published to this repo."
**Outputs**: training-compute benchmark evidence (SGD vs QA-HGD) + Raman
spectral classification evidence, both as Overleaf-ready LaTeX sections with
figure PNGs, plus a bundled tarball for archival.

---

## What this pipeline produces

Running the `qa-paper` target generates a complete, reproducible evidence
bundle landing under `qa_lab/artifacts/overleaf/`:

| Artifact | Source | Content |
|---|---|---|
| `qa_training_compute_section.tex` | `qa_lab/src/bin/qa_theory_export.rs` | LaTeX section: training compute benchmark claims + tables |
| `qa_raman_section.tex` | Makefile copy from `qa_lab/docs/` | LaTeX section: Raman classifier SGD-vs-HGD results |
| `qa_benchmarks.png` | `qa_lab/scripts/qa_plot_benchmarks.py` (over summary.csv) | SGD/HGD convergence + compute scaling |
| `qa_pcn_theta_compare.png` | `qa_lab/scripts/qa_plot_pcn.py` | Predictive coding network theta=0 vs theta=π |
| `qa_jepa_convergence.png` | `qa_lab/scripts/qa_plot_jepa.py` | JEPA SGD vs HGD convergence curves |
| `qa_raman_accuracy.png` | `qa_lab/scripts/qa_plot_raman.py` | Raman classification accuracy vs epoch |
| `qa_overleaf_bundle.tar.gz` | `make qa-paper-bundle` | tarball of all of the above + README_PAPER.md |
| `qa_paper.json` (optional) | `qa_lab/qa_agents/cli/qa_paper.py --json-out` | Structured summary with metrics for agent consumption |

Full provenance chain on every run: Rust binary SHA at build time → CSV/JSON
summary → Python plotting script → PNG → LaTeX section → bundle.

---

## How to run it

### One-shot (the canonical path)

```bash
cd /home/player2/signal_experiments
python qa_lab/qa_agents/cli/qa_paper.py --run \
    --json-out artifacts/overleaf/qa_paper.json \
    --min-step-speedup 1.2 --min-compute-ratio 1.2
```

The script computes `_QA_LAB_DIR` from `__file__` and runs `make` from
`qa_lab/` regardless of caller cwd. The Makefile target chain is
`qa-paper: qa-theory qa-plots-bench qa-plots-pcn qa-plots-jepa qa-plots-raman
qa-paper-bundle` at `qa_lab/Makefile:488`.

### Through the kernel

```python
from qa_lab.kernel.loop import QALabKernel, Task, TaskType
from qa_lab.agents.experiment_agent import ExperimentAgent

kernel = QALabKernel(modulus=9, verbose=False)
kernel.register_agent(TaskType.EXPERIMENT, ExperimentAgent(timeout=1200))

task = Task(
    task_type=TaskType.EXPERIMENT,
    description="QA paper evidence pipeline",
    inputs={
        "script_path": "qa_lab/qa_agents/cli/qa_paper.py",
        "args": ["--run", "--json-out", "artifacts/overleaf/qa_paper.json",
                 "--min-step-speedup", "1.2", "--min-compute-ratio", "1.2"],
    },
)
result = kernel.run_cycle(task)
```

`ExperimentAgent` auto-injects `PYTHONPATH=qa_lab/` when the script lives
under `qa_lab/` (F5 fix, 2026-04-04), so the subprocess can import
`qa_agents.cli.*` by top-level name.

### Guardrails

The `--min-step-speedup` and `--min-compute-ratio` flags are hard
acceptance gates. If the measured metrics fall below them, `qa_paper.py`
raises `SystemExit(f"Regression: ...")` and the cycle verdict becomes
INCONCLUSIVE. This is the canonical way to pin paper claims against
future regressions.

---

## Runtime characteristics

- **Cold build** (first run): ~5–10 min. Compiles 5 Rust binaries
  (`qa_speed_benchmark`, `qa_theory_export`, `qa_pcn_sheaf_demo`,
  `qa_jepa_demo`, `qa_raman_demo`) plus dependencies.
- **Warm run** (cached Rust): ~15 s. All binaries re-run, plot scripts
  re-render, bundle re-tars.
- **Dependencies**: Rust toolchain + cargo, Python with matplotlib + numpy
  (both in `qa_lab/qa_venv`), `tar`. No network calls.

---

## GitHub-first publication workflow

1. Run the pipeline to regenerate the evidence bundle locally.
2. The artifacts under `qa_lab/artifacts/overleaf/` are untracked by
   design — they are build outputs, not source. Commit the bundle
   manually when you want to pin a specific version: `git add -f
   qa_lab/artifacts/qa_overleaf_bundle.tar.gz` and commit with a message
   that references the Rust binary SHAs and input args used.
3. Paper text lives in `papers/in-progress/<topic>/paper.tex` and is
   written by hand against the generated figures.
4. Repo-published: the full reproducible chain (source + runner +
   frozen bundle snapshot) lives in this repo; readers can either
   re-run from source or read the pinned snapshot.
5. **No arXiv uploads**. Per MEMORY.md hard rule: paper venue is
   Will's decision alone.

---

## Verification history

- **2026-04-04**: cwd fix + `make qa-paper` target chain verified end-to-end.
  Full pipeline cold-built `qa_speed_benchmark` in 1m20s, subsequent runs
  complete in ~15s. All 5 acceptance artifacts land cleanly.
- Pipeline is now routable through `QALabKernel` as a TaskType.EXPERIMENT
  and appears in the heterogeneous lab-status runner as a regression check.

## Known gaps

- `qa_raman_experiment.yaml` is NOT this pipeline — it was rewritten
  2026-04-04 to invoke `qa_raman_effect.py` demo (QA tuple Stokes/Anti-Stokes
  analysis). The "SGD vs HGD Raman classifier" task was aspirational and
  had no backing script.
- `artifacts/qa_overleaf_bundle.tar.gz` is gitignored by default; pin
  snapshots explicitly with `-f` when publishing.

---

*Filed by: `cert-batch-empirical` session, 2026-04-04, as follow-up to
the F2 qa-paper pipeline unblocking.*

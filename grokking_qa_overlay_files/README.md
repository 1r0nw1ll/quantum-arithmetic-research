# Grokking × QA Overlay Files

This directory contains **our QA-specific overlay work** built on top of the
external grokking paper replication:

> LucasPrietoAl/grokking-at-the-edge-of-numerical-stability
> https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability

## What's here

38 files we authored or produced while overlaying QA instrumentation on the
upstream grokking experiments (modular-addition cross-entropy task, with
a QA logger that records the training trajectory as QA orbit features):

- `grokking_experiments_qa.py` — QA-instrumented runner
- `qa_logger.py`, `qa_logger_fixed.py` — trajectory logger
- `qa_analysis_notebook.py`, `qa_plot_fixed.py` — post-run analysis
- `run_qa_experiment.sh`, `setup_dependencies.sh` — reproduction scripts
- `test_qa_logger.py`, `verify_no_perturbation.py` — sanity tests
- `modular_addition_cross_entropy_seed0.jsonl` — training trajectory data
- `qa_analysis_modular_addition_cross_entropy_seed0.png` — analysis plot
- `FINAL_WORKING_QA_NOTEBOOK.ipynb`, `WORKING_NOTEBOOK.ipynb`,
  `ploutos_qa_overlay_demo.ipynb` — Jupyter notebooks
- ~20 status / README / checklist markdown files from the execution thread

## How to reassemble a runnable workspace

1. Clone the upstream repo into a sibling directory (or anywhere else):

   ```bash
   git clone https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability.git
   ```

2. Copy or symlink the files from this directory into the upstream clone's
   working tree. These files are overlays, not a fork — they live alongside
   the upstream code.

3. Run `setup_dependencies.sh` to install the grokking paper's requirements,
   then `run_qa_experiment.sh` for the QA-instrumented training run.

## Why this layout

The original working directory (`grokking_qa_overlay/` in this repo) contained
a full clone of the upstream repo with our overlay files layered on top, as an
untracked working tree inside an embedded `.git`. That layout cannot be cleanly
committed to the outer repo — embedded repos either need to be proper
submodules or be unpacked. This directory is the "unpack only our files"
resolution, keeping the upstream repo separate and clonable on demand.

`grokking_qa_overlay/` itself is now gitignored as a local scratch clone.

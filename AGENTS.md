# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python scripts (`run_signal_experiments*.py`, `dynamic_coprocessor_test.py`, `statistical_validation_gauntlet.py`) hold runnable experiments; keep new modules adjacent and note entry points in header comments.
- `data/` stores MNIST and CIFAR downloads; regenerated plots or CSVs should stay beside their scripts and replace prior artifacts only when sources change.
- Archive long-form notes in `files/` or `QAnotes/` and surface publication-ready summaries through `Documents/` and `PAPER_SUBMISSION_README.md`.

## Build, Test, and Development Commands
- Work inside a virtual environment (`python -m venv .venv && source .venv/bin/activate`) to isolate dependencies.
- Install the common stack with `pip install numpy matplotlib torch torchvision tqdm seaborn scipy scikit-learn`, adding extras noted in individual scripts before committing.
- Run the harmonic simulator via `python run_signal_experiments.py`; tweak the constant block up top for new scenarios and capture the emitted PNGs/logs in the repo root.
- Reproduce the MNIST and CIFAR hybrids with `python dynamic_coprocessor_test.py` and `python statistical_validation_gauntlet.py`; both populate `data/` automatically and emit comparison plots.

## Coding Style & Naming Conventions
- Target Python 3.10+, enforce 4-space indentation, snake_case functions, UpperCamelCase classes, and uppercase module constants to match existing files.
- Maintain the `# --- Section ---` dividers to signal pipeline phases and extend them when inserting preprocessing, training, or visualization steps.
- Group imports by origin (stdlib, third-party, local), favor vectorized NumPy/Torch logic over loops, and export any notebook prototypes to `.py` modules that run headlessly.

## Testing Guidelines
- Script-based diagnostics (`dynamic_coprocessor_test.py`, `symmetry_generative_test.py`, `final_generative_test.py`) double as the regression suite; run them before proposing changes and log headline metrics in PRs.
- Prefix new diagnostic runners with their focus (e.g., `qa_entropy_test.py`), ensure they print summary stats, and save plots to the working directory.
- Seed randomness near `__main__` (`np.random.seed`, `torch.manual_seed`) so experiment diffs stay reproducible, and stage future pytest suites under `tests/` with notes in `QAnotes/`.

## Commit & Pull Request Guidelines
- With no bundled Git log, default to short, imperative subjects (“Add CIFAR stress overlay”) and describe parameter or data shifts in wrapped body text when relevant.
- Mention affected scripts, regenerated artifacts, and required reruns (“Re-run `statistical_validation_gauntlet.py` for updated plots”) inside each commit or PR.
- PRs should summarize intent, list touched modules, share before/after accuracy or loss deltas, embed paths for new figures, and link supporting notes in `files/` or `QAnotes/`.

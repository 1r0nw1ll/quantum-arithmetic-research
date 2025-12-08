codex_on_QA workspace
=====================

Purpose
-------
- Offline, Rust-first utilities to work with QA ingestion candidates without Python deps.
- All outputs and notes are saved under this folder.

Layout
------
- `candidates/` — snapshot(s) of the latest candidate file pulled from `ingestion candidates/`.
- `out/` — run manifests, logs, and verification artifacts.
- `rust_ingestion_pull/` — standalone Rust CLI to pull the latest candidate.
- `outline_to_plan/` — Rust CLI that parses the ODT text dump and emits a JSON experiment plan.
- `qa_graph_stats/` — Rust CLI that ingests a real GraphML in-repo dataset and computes QA invariants.
- `qa_graph_experiments/` — Rust CLI for spectral clustering baselines and QA-weighted variants (X, J, Mix, Full-20 invariant kernel).
 - `feature_maps/qa_feature_map_v3.py` — Python QA feature map (qa21/qa27/qa83) with 83 invariants.
 - `spec/qa_invariants.yaml` — YAML rulebase (canonical + expanded) for QA invariants.
 - `scripts/` — helper scripts (ingestion, manifold demos, benchmarks).

Usage (from repo root)
----------------------
1) Build:
   - `cd codex_on_QA/rust_ingestion_pull && cargo build --release`
2) Run (defaults use `ingestion candidates/` as source):
   - `./target/release/rust_ingestion_pull \
        --source "../../ingestion candidates" \
        --dest   "../candidates" \
        --manifest "../out/latest_candidate.json"`

Notes
-----
- No external crates required; standard library only. Safe to run with network disabled.
- The tool selects the most recently modified `.odt|.pdf|.doc|.docx` file.

Outline → Plan
--------------
- Build: `cd codex_on_QA/outline_to_plan && cargo build --release`
- Run: `./target/release/outline_to_plan --input ../out/qastructure_features.txt --output ../out/experiment_plan.json`

Real Data (Graph) Stats
-----------------------
- Build: `cd codex_on_QA/qa_graph_stats && cargo build --release`
- Run on included dataset: `./target/release/qa_graph_stats --graph ../../qa_knowledge_graph.graphml --out ../out/graph_stats.json`
- Output shows node/edge counts, degree stats, and top nodes by QA invariants J, X, G.

Spectral Experiments (Baseline + QA)
------------------------------------
- Build: `cd codex_on_QA/qa_graph_experiments && cargo build --release`
- Run: `./target/release/qa_graph_experiments --graph ../../qa_knowledge_graph.graphml --out ../out/graph_spectral_summary.json`
- Modes emitted: baseline, qa_weight_x, qa_weight_j, qa_weight_mix, qa_weight_full (20-invariant kernel)
- Tuning the full kernel scale: `--tau_scale 0.1` often improves Q on dense graphs
- New: choose QA feature block for the full kernel with `--qa-mode qa21|qa27` (default: qa21)
- Full list of knobs: `--phase-mode none|raw|sincos`, `--scale-mode none|zscore`, `--k 2,4,6,8,10`
- Outputs under `codex_on_QA/out/`: summary JSON, per-mode labels CSVs, and per-mode cluster stats JSONs.

Raman as a Graph (single- and multi‑tuple)
------------------------------------------
- Single‑tuple GraphML (from CSV id,b,e,label):
  `PYTHONPATH=. python codex_on_QA/scripts/raman_to_graph.py --csv codex_on_QA/out/raman_qa_fundovt_bcwin_v2.csv --qa-mode qa21 --k 8 --out codex_on_QA/data/raman_qa21.graphml`
- Multi‑tuple GraphML (from CSV id,b1,e1,b2,e2,...,label):
  `PYTHONPATH=. python codex_on_QA/scripts/raman_to_graph_multi.py --csv codex_on_QA/out/raman_multi_fundovt_fingerprint_multiseg.csv --qa-mode qa21 --k 8 --out codex_on_QA/data/raman_multi_qa21.graphml`
- Then run the Rust spectral binary on the produced GraphML as usual (see Spectral Experiments above).

QA Feature Map (Python)
-----------------------
- Use: `from codex_on_QA.feature_maps.qa_feature_map_v3 import qa_feature_vector`
- Modes: `qa21` (canonical), `qa27` (canonical-expanded), `qa83` (full stack)
- Example: `vec, names = qa_feature_vector(b, e, mode="qa27")`

One-Shot Bench (Moons/Circles/Swiss)
----------------------
- Use the project venv: `source codex_on_QA/.venv/bin/activate && PYTHONPATH=. ...`
- Run (no UI): `PYTHONPATH=. python codex_on_QA/scripts/qa_one_shot_bench.py --dataset moons --encoding first2 --no-show`
- Datasets: `--dataset moons|circles|swiss`
- Encodings for (b,e): `--encoding first2|pca2|swiss_radangle|swiss_rady`
- Saves JSON summaries to `codex_on_QA/out/qa_one_shot_<dataset>[_<encoding>].json` and prints a small table.

CSV Bench (id,b,e,label)
------------------------
- Run: `PYTHONPATH=. python codex_on_QA/scripts/qa_csv_bench.py --csv path/to/data.csv`
- Input schema: `id,b,e,label` (header required)
- Compares modes: raw, qa21, qa27, qa83; saves `<stem>_csv_bench.json` to `codex_on_QA/out/`.

Raman Export (build CSV from repo spectra)
-----------------------------------------
- Export: `PYTHONPATH=. python codex_on_QA/scripts/raman_export_csv.py`
- Output: `codex_on_QA/out/raman_qa.csv` (id,b,e,label), derived from `qa_lab/qa_data/raman/*`.
- Then bench: `PYTHONPATH=. python codex_on_QA/scripts/qa_csv_bench.py --csv codex_on_QA/out/raman_qa.csv`

Sample Efficiency (few-shot)
----------------------------
- Run: `PYTHONPATH=. python codex_on_QA/scripts/qa_one_shot_efficiency.py --dataset moons --encoding first2 --thresholds 0.9,0.95`
- Outputs `codex_on_QA/out/qa_one_shot_efficiency_<dataset>_<encoding>.json` reporting minimal train sizes to hit targets for LogReg and MLP across raw/qa21/qa27/qa83.

Louvain Baseline (Python)
-------------------------
- Partition: `PYTHONPATH=. python codex_on_QA/scripts/louvain_partition.py --graph codex_on_QA/data/football.graphml --outdir codex_on_QA/out`
- Evaluate vs ground truth (GraphML node attribute `value`):
  `PYTHONPATH=. python codex_on_QA/scripts/eval_partition.py --graph codex_on_QA/data/football.graphml --labels codex_on_QA/out/labels_louvain.csv --out codex_on_QA/out/louvain_metrics.json`
- Merge comparison: `PYTHONPATH=. python codex_on_QA/scripts/build_comparison.py --rust-json codex_on_QA/out/football_spectral_with_metrics.json --out codex_on_QA/out/football_comparison.json`

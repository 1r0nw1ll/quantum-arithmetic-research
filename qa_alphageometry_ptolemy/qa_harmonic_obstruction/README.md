# QA Harmonic Obstruction

Deterministic generator for `QA_HARMONIC_OBSTRUCTION.v1` benchmark episodes and
its corresponding discovery pipeline batch plan.

## Sweep Configuration (v1)

- `alpha` (20): `1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 2/3, 2/5, 3/5, 3/7, 4/7, 4/9, 5/8, 5/12, 7/10, 7/12, 11/18`
- `window` (3): `1024, 4096, 16384`
- generator sets (3):
  - `gA = ["sigma", "mu"]`
  - `gB = ["sigma", "lambda"]`
  - `gC = ["sigma", "mu", "lambda", "nu"]`

Total runs (single-k plan): `20 * 3 * 3 = 180`.

## Generate Episodes + Plan

```bash
python3 qa_alphageometry_ptolemy/qa_harmonic_obstruction/generate_episodes.py
```

This writes:

- episodes: `qa_alphageometry_ptolemy/qa_harmonic_obstruction/episodes/EP_HO_*.json`
- plan: `qa_alphageometry_ptolemy/qa_discovery_pipeline/plans/plan_harmonic_sweep_v1.json`

## Generate Plan Only

```bash
python3 qa_alphageometry_ptolemy/qa_harmonic_obstruction/generate_episodes.py \
  --skip_episode_write
```

Use this mode when you only want to refresh the batch plan artifact.

## Generate k-Sweep Plan (Example: 16,64,256)

```bash
python3 qa_alphageometry_ptolemy/qa_harmonic_obstruction/generate_episodes.py \
  --skip_episode_write \
  --k_values 16,64,256 \
  --plan_out qa_alphageometry_ptolemy/qa_discovery_pipeline/plans/plan_harmonic_sweep_k3_v1.json \
  --created_utc 2026-02-13T17:00:00Z
```

This emits a 540-run plan (`180` cases x `3` k-values).

## Chunk a k-Sweep Plan (Preserve k-Groups)

When you split a multi-k plan for overnight runs, chunk by **episode/case** so
each chunk contains all k-levels for a case (otherwise the sweep summarizer will
flag missing k-levels / phase-law violations).

```bash
python3 qa_alphageometry_ptolemy/qa_discovery_pipeline/split_plan_kgrouped.py \
  --plan qa_alphageometry_ptolemy/qa_discovery_pipeline/plans/plan_harmonic_sweep_k3_v1.json \
  --out_dir /tmp/ho_k3_chunks \
  --cases-per-chunk 30
```

## Run a Chunk (Batch + CI + Summary)

```bash
qa_alphageometry_ptolemy/qa_discovery_pipeline/run_overnight.sh \
  --plan /tmp/ho_k3_chunks/plan_harmonic_sweep_k3_v1__chunk_001_of_006.json \
  --allow-fail \
  --summarize-harmonic
```

## Summarize Sweep Outputs

```bash
python3 qa_alphageometry_ptolemy/qa_harmonic_obstruction/summarize_sweep.py \
  --out_dir /tmp/ho_sweep_v1
```

This writes deterministic report artifacts in `out_dir`:

- `harmonic_report_<bundle8>.md`
- `harmonic_report_<bundle8>.csv`
- `harmonic_report_<bundle8>.json`

Report rows include `planned_k`, and the markdown report contains a
`k x window x generator` status table.

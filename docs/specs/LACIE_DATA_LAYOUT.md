# LaCie External Drive — Data Layout

**Drive**: `/Volumes/lacie` (APFS, 2TB)  
**Offload root**: `/Volumes/lacie/signal_experiments_offload/`  
**Date migrated**: 2026-06-11

## Why This Exists

The internal drive was 97% full (184GB used of 228GB). Large corpus datasets,
experiment caches, and raw EEG data were moved to LaCie to free space.
All moved paths have **symlinks** left at their original locations — existing
code paths work transparently as long as LaCie is mounted.

## Moved Paths (symlinks active in repo)

| Original path (symlink) | LaCie destination | Size |
|---|---|---|
| `corpus/pepe_pose/` | `…/corpus/pepe_pose/` | 55 GB |
| `corpus/cmu_mocap_zhou2019/` | `…/corpus/cmu_mocap_zhou2019/` | 4.1 GB |
| `corpus/modelnet40/` | `…/corpus/modelnet40/` | 2.3 GB |
| `corpus/cmu_mocap_asfamc/` | `…/corpus/cmu_mocap_asfamc/` | 1.0 GB |
| `experiments/qa_ml/cache_pepe_ch4_pose1_table_4_2_finetuned_full13/` | `…/experiments/qa_ml/cache_pepe_…/` | 8.8 GB |
| `experiments/qa_ml/pepe_ch2_rot3_rebuilt/` | `…/experiments/qa_ml/pepe_ch2_rot3_rebuilt/` | 3.3 GB |
| `experiments/qa_ml/cache_full_psp/` | `…/experiments/qa_ml/cache_full_psp/` | 1.6 GB |
| `experiments/qa_ml/gptq_awq_env/` | `…/experiments/qa_ml/gptq_awq_env/` | 1.1 GB |
| `qa_lab/data/rruff_raman/` | `…/qa_lab/data/rruff_raman/` | 1.6 GB |
| `qa_lab/data/rruff_zips/` | `…/qa_lab/data/rruff_zips/` | 677 MB |
| `qa_lab/data/houston2013_raw/` | `…/qa_lab/data/houston2013_raw/` | 186 MB |
| `qa_lab/data/cifar-10-batches-py/` | `…/qa_lab/data/cifar-10-batches-py/` | 178 MB |
| `archive/phase_artifacts/phase2_data/eeg/` | `…/archive/…/eeg/` | 25 GB |
| `results/qa_exact_orbit_theorem_demo_2026_06_09.db` | `…/results/` | 375 MB |
| `.venv/` | `…/venv/signal_experiments_venv/` | 1.2 GB |
| `~/.cache/torch/` | `…/home_cache/torch/` | 2.5 GB |

## Moved Without Symlinks (Downloads)

These are installer/download files with no code references:
- `~/Downloads/Wolfram Player 14.3/` (2.6 GB)
- `~/Downloads/Ring35_Dataset_Txt 2/` (639 MB — duplicate)
- `~/Downloads/Claude.dmg` (302 MB)
- `~/Downloads/Codex.dmg` (294 MB)
- `~/Downloads/Ring35_Dataset_Txt.zip` (255 MB)

## Rules for Future Large Files

**If you are generating output that exceeds ~500 MB, write it directly to LaCie:**

```python
LACIE = Path("/Volumes/lacie/signal_experiments_offload")
# Example: large EEG output
output_path = LACIE / "results" / "my_experiment_output.h5"
```

**Categories that belong on LaCie (not internal):**
- Raw EEG / EDF files > 100 MB
- Dataset zip archives > 200 MB
- Model checkpoint files > 500 MB (`.pt`, `.pth`, `.ckpt`, `.h5`)
- Experiment result databases > 100 MB (`.db`, `.sqlite`)
- Numpy array dumps > 500 MB (`.npy`, `.npz`)
- Downloaded corpus zips > 200 MB
- Conda/Python environments (use `.venv` symlink)
- HuggingFace model cache (`~/.cache/huggingface/hub/`)

**Internal drive** (keep here): source code, cert validators, JSON results,
small plot PNGs, JSONL logs, markdown docs, git objects.

## ⚠️ Dependency

All symlinked paths **require LaCie to be mounted** at `/Volumes/lacie`.
- Check: `ls /Volumes/lacie/signal_experiments_offload/` before running experiments
- If LaCie is not connected, Python venv (.venv symlink) will be broken —
  use `python3 -m venv .venv && pip install -r requirements.txt` to recreate locally

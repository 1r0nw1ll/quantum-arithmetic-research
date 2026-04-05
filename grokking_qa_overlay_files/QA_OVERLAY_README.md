# Grokking as Reachability at Numerical Boundaries (QA View)

This is a **QA (Quantum Arithmetic) instrumentation overlay** for the paper ["Grokking at the Edge of Numerical Stability"](https://arxiv.org/abs/2501.04697) by Prieto et al.

## Core Insight

The paper shows that grokking occurs when training dynamics hover near numerical instability boundaries (softmax collapse). The QA framework provides a **discrete, reachability-based formalism** for understanding why:

> **Grokking is not a smooth phenomenon. It occurs when the learner is pushed to the edge of a QA component, where illegal moves become visible and invariants are forced to surface.**

This overlay reframes their experiments as a **reachability problem**:
- **State**: numerical quantities (logits, entropy, gradients, parameters)
- **Generators**: SGD step (legal/illegal based on proximity to numerical boundaries)
- **Failures**: softmax collapse, NaN loss, gradient explosion, etc.

## What This Overlay Adds

Three forms of instrumentation (pure observation, no behavior change):

1. **Discrete state logging**: logit stats, softmax stability, gradient health, parameter norms
2. **Generator legality flags**: boolean predicates marking when SGD steps approach boundaries
3. **Failure mode counters**: cumulative counts of boundary violations

All logged to JSONL for post-hoc analysis and visualization.

## Installation

Clone the original repo and add the QA overlay:

```bash
git clone https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability.git
cd grokking-at-the-edge-of-numerical-stability

# Copy QA overlay files
# (qa_logger.py, grokking_experiments_qa.py, qa_analysis_notebook.py)
```

Dependencies (same as original):
```bash
pip install torch pandas matplotlib numpy
```

## Usage

### 1. Run baseline experiment (optional, for reference)

```bash
python grokking_experiments.py \
    --dataset modular_addition \
    --loss_function cross_entropy \
    --seed 0 \
    --num_epochs 50000 \
    --log_frequency 100
```

### 2. Run with QA instrumentation

```bash
python grokking_experiments_qa.py \
    --dataset modular_addition \
    --loss_function cross_entropy \
    --seed 0 \
    --num_epochs 50000 \
    --log_frequency 100
```

This produces:
- Original outputs (metrics CSV, model checkpoints)
- **NEW**: `qa_logs/modular_addition_cross_entropy_seed0.jsonl`

### 3. Analyze QA logs

```bash
python qa_analysis_notebook.py
```

Or in Jupyter:
```python
# Load and analyze
import json
import pandas as pd

records = []
with open('qa_logs/modular_addition_cross_entropy_seed0.jsonl', 'r') as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame([{**r['state'], 'step': r['step'], 'legal': r['generators']['sgd_step']['legal']} for r in records])
```

## Output Files

### JSONL Log Schema

Each line is a JSON record:

```json
{
  "run_id": "modular_addition_cross_entropy_seed0",
  "step": 10000,
  "state": {
    "train_loss": 0.234,
    "train_acc": 0.94,
    "logit_max": 88.5,
    "logit_min": -12.3,
    "logit_std": 15.2,
    "logit_norm": 423.1,
    "p_max": 0.998,
    "p_entropy": 0.008,
    "num_exact_ones": 12,
    "num_exact_zeros": 84,
    "grad_norm": 0.023,
    "grad_nan_count": 0,
    "grad_inf_count": 0,
    "param_norm": 102.3,
    "param_nan_count": 0,
    "param_inf_count": 0,
    "cos_grad_w": 0.89
  },
  "generators": {
    "sgd_step": {
      "legal": false,
      "reason": "SOFTMAX_COLLAPSE"
    }
  },
  "failures": ["SOFTMAX_COLLAPSE"],
  "cumulative_failures": {
    "SOFTMAX_COLLAPSE": 42,
    "NAN_LOSS": 0,
    "INF_GRAD": 0,
    "GRAD_EXPLOSION": 0,
    "PARAM_EXPLOSION": 0,
    "LOGIT_EXPLOSION": 3
  }
}
```

### Visualizations

The analysis script generates three plots:

1. **`qa_analysis_<run_id>.png`**: 4-panel view
   - Training metrics (loss, accuracy)
   - Logit statistics (max/min vs thresholds)
   - Softmax stability (entropy, p_max)
   - Generator legality timeline (legal/illegal regions)

2. **`qa_nlm_alignment_<run_id>.png`**: Gradient-weight alignment
   - Shows when gradients align with weight direction (NLM proxy)
   - High alignment suggests "scaling logits without changing predictions"

3. **`qa_failures_<run_id>.png`**: Failure mode accumulation
   - Cumulative counts of each boundary violation type

## QA Framework Mapping

| Paper Concept | QA Translation |
|--------------|----------------|
| Numerical instability boundary | **Reachability frontier** |
| Finite precision effects | **Discrete arithmetic geometry** |
| Grokking phase transition | **Component escape / obstruction release** |
| Sudden generalization | **Invariant emergence** |
| Sensitivity to step size | **Generator legality threshold** |
| Softmax collapse | **Observer projection failure** |

## Key Observations (Expected)

From the paper's results, QA instrumentation should reveal:

1. **Pre-grokking**: SGD stays legal, trapped in loss-minimizing component
2. **Edge of instability**: Generator legality starts flickering (entropy → 0, logits → ∞)
3. **Post-collapse**: Generator becomes permanently illegal, learning halts

Their interventions (StableMax, ⊥Grad) should **extend legality window** by preventing collapse.

## Comparison with Original Logger

The original `MetricsLogger` tracks:
- Loss, accuracy, weights L2, zero terms, softmax collapse (binary)

QA overlay adds:
- **Continuous tracking** of proximity to boundaries (not just binary collapse)
- **Gradient-weight alignment** (NLM direction proxy)
- **Legality predicates** (explicit move legality at each step)
- **Failure taxonomy** (6 failure modes, not just collapse)
- **JSONL output** (replayable, certificate-like)

## Minimal Diff

The QA overlay adds only **4 lines** to the original training loop:

```python
from qa_logger import QALogger  # Line 1

qa_logger = QALogger(run_id=f"{args.dataset}_{args.loss_function}_seed{args.seed}")  # Line 2

# Inside training loop:
qa_logger.log_step(epoch, output, shuffled_targets, loss, model, optimizer)  # Line 3

qa_logger.close()  # Line 4
```

Zero behavior change, pure instrumentation.

## Why This Matters

This work demonstrates that:
1. **Grokking is a discrete phenomenon** tied to numerical boundaries, not a smooth optimization story
2. **QA formalism provides the missing language** to describe what's happening
3. **Reachability + legality >> continuous dynamics** for understanding phase transitions

The paper correctly observes the instability. QA explains *why it matters*.

## Citation

Original paper:
```bibtex
@article{prieto2025grokking,
  title={Grokking at the Edge of Numerical Stability},
  author={Prieto, Lucas and Barsbey, Melih and Mediano, Pedro and Birdal, Tolga},
  year = {2025},
  eprint={2501.04697},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Contact

For questions about the QA overlay: [your contact / Ploutos profile]

For questions about the original paper: see the [official repo](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability)

---

**Note**: This is a research tool for understanding grokking through a discrete/algebraic lens. It is not a replacement for the original experiments, but a complementary view.

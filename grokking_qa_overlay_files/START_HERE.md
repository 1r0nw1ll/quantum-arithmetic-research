# QA Overlay for Grokking Experiments - START HERE

## Quick Start (3 Commands)

```bash
# 1. Test installation
python test_qa_logger.py

# 2. Run experiment (10 min on GPU, 1 hour on CPU)
./run_qa_experiment.sh

# 3. Done! Check the generated PNG plots.
```

## What This Is

A **QA (Quantum Arithmetic) instrumentation overlay** for the paper:
**"Grokking at the Edge of Numerical Stability"** (Prieto et al., 2025)

**Core claim:** Grokking is not a smooth optimization phenomenon—it's a discrete reachability problem at arithmetic failure boundaries.

**What we add:** 4 lines of instrumentation that log training as a QA reachability trace:
- State: numerical quantities (logits, entropy, gradients)
- Generators: SGD step legality (legal/illegal based on boundary proximity)
- Failures: softmax collapse, NaN loss, gradient explosion, etc.

**Result:** You can *see* exactly when and why learning stops (generator becomes illegal).

## Files to Read First

1. **START_HERE.md** ← You are here
2. **IMPLEMENTATION_SUMMARY.md** - Technical details, what was built
3. **PLOUTOS_POST.md** - Publication-ready post for Ploutos
4. **QA_OVERLAY_README.md** - Full documentation

## Files to Run

1. **test_qa_logger.py** - 5-step verification (runs in 1 second)
2. **run_qa_experiment.sh** - Full experiment + analysis (10 min)
3. **qa_analysis_notebook.py** - Standalone analysis script

## Files You Can Ignore (Unless Debugging)

- `grokking_experiments.py` - Original (unmodified)
- `grokking_experiments_qa.py` - Patched version (4 lines added)
- `qa_logger.py` - Logger implementation
- `PATCH_INSTRUCTIONS.md` - Manual patching guide
- `logger.py`, `models.py`, `utils.py`, etc. - Original repo files

## Installation

### Option 1: Quick (if you have PyTorch)
```bash
cd grokking_qa_overlay
pip install pandas matplotlib
python test_qa_logger.py  # Should print "PASSED ✓"
```

### Option 2: Fresh install
```bash
cd grokking_qa_overlay

# Install PyTorch (choose your platform at pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# OR
pip install torch  # CPU-only

# Install remaining deps
pip install pandas matplotlib

# Verify
python test_qa_logger.py
```

## Running Experiments

### Simplest: Use the script
```bash
./run_qa_experiment.sh
```

This will:
1. Run experiment (default: modular_addition, cross_entropy, 50k epochs)
2. Generate JSONL log in `qa_logs/`
3. Ask if you want to run analysis
4. Generate 3 PNG plots if you say yes

### Customize: Set environment variables
```bash
# Try StableMax loss (the paper's intervention)
LOSS_FUNC=stablemax ./run_qa_experiment.sh

# Shorter run for testing
NUM_EPOCHS=5000 ./run_qa_experiment.sh

# Different seed
SEED=42 ./run_qa_experiment.sh
```

### Manual: Run Python directly
```bash
# Training
python grokking_experiments_qa.py \
    --dataset modular_addition \
    --loss_function cross_entropy \
    --num_epochs 50000 \
    --log_frequency 100 \
    --device cuda:0

# Analysis
python qa_analysis_notebook.py
```

## Understanding the Output

### JSONL Log (`qa_logs/<run_id>.jsonl`)

Each line is a JSON record:
```json
{
  "run_id": "modular_addition_cross_entropy_seed0",
  "step": 10000,
  "state": {
    "train_loss": 0.234,
    "logit_max": 88.5,      // Near fp32 overflow!
    "p_entropy": 0.008,     // Softmax collapsed (should be ~1.0)
    "cos_grad_w": 0.89,     // Gradient aligned with weights (NLM)
    ...
  },
  "generators": {
    "sgd_step": {
      "legal": false,         // ← SGD move is illegal!
      "reason": "SOFTMAX_COLLAPSE"
    }
  },
  "failures": ["SOFTMAX_COLLAPSE"],
  "cumulative_failures": {"SOFTMAX_COLLAPSE": 42, ...}
}
```

### PNG Plots

1. **qa_analysis_*.png** (4 panels):
   - Training metrics (loss, accuracy)
   - Logit statistics (max/min vs thresholds)
   - Softmax stability (entropy collapse)
   - Generator legality timeline (when does SGD become illegal?)

2. **qa_nlm_alignment_*.png**:
   - Gradient-weight alignment over time
   - High alignment = gradient scaling weights (NLM direction from paper)

3. **qa_failures_*.png**:
   - Cumulative failure counts
   - Shows which boundaries were hit first

## Key QA Concepts

### State Vector
17 quantities tracked per step:
- Loss, accuracy
- Logit statistics (max, min, std, norm)
- Softmax stability (entropy, p_max, exact 0/1 counts)
- Gradient health (norm, NaN/Inf counts, alignment with weights)
- Parameter health (norm, NaN/Inf counts)

### Generator Legality
Boolean: "Is the SGD step legal?"

Illegal when:
- Entropy < 0.01 (softmax collapsed)
- Logit max > 85 (approaching fp32 overflow)
- Gradient NaN/Inf
- Gradient norm > 1e6
- Parameter norm > 1e6

### Failure Modes
6 types of boundary violations:
- SOFTMAX_COLLAPSE (entropy → 0)
- LOGIT_EXPLOSION (logits → ±∞)
- NAN_LOSS (loss is NaN)
- INF_GRAD (gradient is Inf)
- GRAD_EXPLOSION (gradient norm too large)
- PARAM_EXPLOSION (parameter norm too large)

## Expected Results

For **cross_entropy** (baseline):
1. Training starts: generators legal, entropy ~1.0
2. Middle phase: gradients align with weights (cos → 1)
3. Near end: entropy collapses, logits explode
4. After collapse: generators illegal, learning stops

For **stablemax** (intervention):
1. Same as cross_entropy initially
2. But: collapse happens *later* (extended legality window)
3. System avoids numerical boundary longer

**QA insight:** StableMax doesn't change the loss landscape—it delays the reachability frontier.

## Publishing to Ploutos

After running experiments:

1. Read **PLOUTOS_POST.md** (publication-ready content)
2. Attach your generated PNG plots
3. Link to code (GitHub gist or your fork)
4. Optional: Share JSONL logs as "QA certificates"

Suggested tags:
`#grokking #numerical-stability #discrete-mathematics #reachability #phase-transitions`

## Troubleshooting

### "No module named 'qa_logger'"
- Make sure you're in the `grokking_qa_overlay/` directory
- Run `python test_qa_logger.py` to verify installation

### "CUDA out of memory"
- Use CPU: `DEVICE=cpu ./run_qa_experiment.sh`
- Or reduce dataset size in utils.py

### "Training is very slow"
- Normal on CPU (1-2 hours for 50k epochs)
- Reduce epochs for testing: `NUM_EPOCHS=5000 ./run_qa_experiment.sh`
- Use GPU if available

### "Plots look wrong"
- Check JSONL log has data: `wc -l qa_logs/*.jsonl`
- Run analysis manually: `python qa_analysis_notebook.py`
- Adjust thresholds in `qa_logger.py` if needed

## Next Steps

1. **Run baseline:** `./run_qa_experiment.sh`
2. **Run intervention:** `LOSS_FUNC=stablemax ./run_qa_experiment.sh`
3. **Compare:** Side-by-side legality timelines
4. **Publish:** Use PLOUTOS_POST.md as template

## Questions?

This is a complete, working implementation. You can:
- Use it as-is for the grokking paper
- Extend to other "sudden phase transition" phenomena
- Modify thresholds or add new metrics
- Fork for your own experiments

Everything is self-contained and reproducible.

---

**Ready?** Run `python test_qa_logger.py` to verify installation.

# Running QA Overlay Remotely (PyTorch Environment)

Since this environment doesn't have PyTorch, run the verification and experiments in a PyTorch-enabled environment.

## Quick Setup on Remote Machine

```bash
# 1. Copy the entire directory
scp -r grokking_qa_overlay/ remote-machine:~/

# 2. SSH to remote machine
ssh remote-machine

# 3. Install dependencies
cd grokking_qa_overlay
pip install torch pandas matplotlib numpy

# 4. Run verification (20 min)
python verify_no_perturbation.py

# Expected output:
# "✓ PASS: Zero behavioral perturbation"

# 5. Run experiment + generate plots (1 hour CPU)
./run_qa_experiment.sh

# Expected outputs:
# - qa_logs/modular_addition_cross_entropy_seed0.jsonl
# - qa_analysis_modular_addition_cross_entropy_seed0.png
# - qa_nlm_alignment_modular_addition_cross_entropy_seed0.png
# - qa_failures_modular_addition_cross_entropy_seed0.png

# 6. Copy results back
scp qa_*.png verification_*.png local-machine:~/grokking_qa_overlay/
```

## What to Check

### After verification:
- [ ] Final metrics identical? (diff < 1e-4)
- [ ] Trajectory correlation > 0.9999?
- [ ] Verdict = "PASS"?

### After plotting:
- [ ] Panel 4 shows legality flip 1→0?
- [ ] Flip coincides with accuracy plateau?
- [ ] No NaNs or weird scales?

## If You Can't Run Remotely

Alternative: Review the code and documentation, publish with:
- Code + theoretical framework (already complete)
- Note: "Verification pending on hardware with PyTorch"
- Community can run and report results

The implementation is sound (instrument-only, minimal diff), so publication without running it first is acceptable if you flag it.

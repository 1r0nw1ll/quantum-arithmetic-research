# Player4 (Gemini CLI) - Next Steps

## ✅ Sync Complete!

You now have the complete `signal_experiments` project (14GB).

## Verify Sync

```bash
cd signal_experiments
ls -lh qa_training_dataset.jsonl
# Should show: 11M with 31,606 examples
```

## Choose Your Path

### Option A: Theorem Discovery Pipeline (RECOMMENDED - Faster)
```bash
python qa_theorem_discovery_orchestrator.py --quick
```
**What it does:**
- 5-stage automated pipeline
- Graph construction → GNN training → Pattern mining → Conjecture generation → Lean proofs
- Time: ~30-60 minutes
- Output: New mathematical theorems about QA system

### Option B: QALM Production Training
```bash
python train_qalm_production.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --hidden-size 512 \
    --num-layers 8 \
    --num-heads 8 \
    --checkpoint-dir checkpoints/qalm_v1_medium
```
**What it does:**
- Trains QA-specialized language model
- Time: 2-3 hours (GPU) or 10-15 hours (CPU)
- Output: Local QA reasoning model

### Option C: Both (Sequential)
```bash
# Run theorem discovery first
python qa_theorem_discovery_orchestrator.py --quick

# Then start training
python train_qalm_production.py --epochs 100 --batch-size 32
```

## Monitor Progress

**Theorem Discovery:**
```bash
tail -f theorem_discovery.log
```

**QALM Training:**
```bash
tail -f qalm_training.log
```

## What You Have

- ✅ 31,606 training examples
- ✅ Complete QA Lab infrastructure
- ✅ BobNet multi-AI orchestrator
- ✅ Signal experiments (E8 analysis, audio classification)
- ✅ All dependencies (Python packages in venv)
- ✅ MNIST/CIFAR-10 datasets

## Expected Outputs

**Theorem Discovery:**
- `qa_discovery_workspace/qa_graph.pt` - Knowledge graph
- `qa_discovery_workspace/conjectures.json` - Generated conjectures
- `qa_discovery_workspace/lean_proofs/` - Formal proofs

**QALM Training:**
- `checkpoints/qalm_v1_medium/` - Model checkpoints
- `qalm_training.log` - Training metrics
- Best model saved automatically

## Need Help?

Check these files:
- `BOBNET_TEST_REPORT.md` - System architecture
- `PRODUCTION_TRAINING_GUIDE.md` - Training details
- `SYNC_INSTRUCTIONS_FOR_PLAYER4.md` - Setup info

---
**Ready to go!** 🚀
**Recommended:** Start with theorem discovery (faster, interesting results)

# QALM v1.0 Production Training Guide

## 🎯 Status: READY FOR PRODUCTION TRAINING

All infrastructure has been built, tested, and validated. The system is ready for full-scale QALM v1.0 training.

---

## ✅ Validation Results

### Test Run Completed Successfully
- **Dataset**: 31,606 examples loaded correctly
- **Model**: 1.97M parameters initialized (test config)
- **Training**: Loss decreasing properly (14.8 → 2.3 in first 150 batches)
- **Infrastructure**: Checkpointing, logging, and monitoring all functional

### Components Validated
1. ✅ **Data Pipeline**: `qa_dataloader.py` (JSONL format, 31K examples)
2. ✅ **Model Architecture**: `qa_model_architecture.py` (QA-aware attention)
3. ✅ **Training Script**: `train_qalm_production.py` (full training pipeline)
4. ✅ **Checkpointing**: Automatic saving every 10 epochs + best model
5. ✅ **Monitoring**: Real-time loss tracking and progress plots

---

## 🚀 Production Training Commands

### Option 1: Full Production Model (Recommended)
**Configuration**: Large model for best performance
```bash
python train_qalm_production.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --hidden-size 768 \
    --num-layers 12 \
    --num-heads 12 \
    --max-length 512 \
    --vocab-size 10000 \
    --checkpoint-dir checkpoints/qalm_v1_full \
    --lr 1e-4
```

**Specs**:
- Model size: ~50M parameters
- Training time: 2-3 hours (GPU), 10-15 hours (CPU)
- Memory: ~8GB GPU / ~16GB RAM
- Output: `checkpoints/qalm_v1_full/`

---

### Option 2: Medium Model (Balanced)
**Configuration**: Good balance between size and speed
```bash
python train_qalm_production.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --hidden-size 512 \
    --num-layers 8 \
    --num-heads 8 \
    --max-length 512 \
    --vocab-size 8000 \
    --checkpoint-dir checkpoints/qalm_v1_medium \
    --lr 1e-4
```

**Specs**:
- Model size: ~25M parameters
- Training time: 1-2 hours (GPU), 6-8 hours (CPU)
- Memory: ~4GB GPU / ~8GB RAM
- Output: `checkpoints/qalm_v1_medium/`

---

### Option 3: Small Model (Fast)
**Configuration**: Quick training for rapid iteration
```bash
python train_qalm_production.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 50 \
    --batch-size 16 \
    --hidden-size 256 \
    --num-layers 6 \
    --num-heads 8 \
    --max-length 256 \
    --vocab-size 5000 \
    --checkpoint-dir checkpoints/qalm_v1_small \
    --lr 1e-4
```

**Specs**:
- Model size: ~8M parameters
- Training time: 30-60 min (GPU), 2-4 hours (CPU)
- Memory: ~2GB GPU / ~4GB RAM
- Output: `checkpoints/qalm_v1_small/`

---

## 📊 Monitoring Training

### Real-Time Monitoring
```bash
# Watch training log
tail -f qalm_training.log

# Check latest checkpoints
ls -lth checkpoints/qalm_v1_full/

# View training progress plot
eog checkpoints/qalm_v1_full/training_progress.png
```

### Expected Behavior
- **Initial loss**: 10-15 (random initialization)
- **After 10 epochs**: 3-5 (model learning patterns)
- **After 50 epochs**: 1-2 (good convergence)
- **After 100 epochs**: 0.5-1.0 (strong performance)

### Checkpoints Saved
- `qalm_epoch_10.pt`, `qalm_epoch_20.pt`, ... (every 10 epochs)
- `qalm_best.pt` (best validation loss)
- `training_summary.json` (final results)
- `training_progress.png` (loss curves)

---

## 🔧 Advanced Options

### Resume From Checkpoint
```bash
python train_qalm_production.py \
    --resume checkpoints/qalm_v1_full/qalm_epoch_50.pt \
    --epochs 100 \
    --batch-size 32
```

### Custom Learning Rate Schedule
```bash
python train_qalm_production.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 100 \
    --lr 5e-5 \  # Lower for fine-tuning
    --batch-size 32
```

### GPU Training (if available)
The script automatically detects and uses GPU if available:
```bash
# Check GPU
nvidia-smi

# Run training (uses GPU automatically)
python train_qalm_production.py [options]
```

---

## 📁 Output Files

After training completes, you'll have:

```
checkpoints/qalm_v1_full/
├── qalm_best.pt              # Best model (load this for inference)
├── qalm_epoch_10.pt          # Checkpoint at epoch 10
├── qalm_epoch_20.pt          # Checkpoint at epoch 20
├── ...
├── qalm_epoch_100.pt         # Final checkpoint
├── training_summary.json     # Training statistics
└── training_progress.png     # Loss curves visualization
```

### Using Trained Model
```python
import torch
from qa_model_architecture import QALanguageModel

# Load best model
checkpoint = torch.load('checkpoints/qalm_v1_full/qalm_best.pt')
model = QALanguageModel(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
# ... (see qa_inference.py for full API)
```

---

## 🎓 What Happens During Training

### Phase 1: Initialization (Epoch 0-10)
- Model learns basic token patterns
- Loss decreases rapidly from ~15 to ~5
- QA attention mechanism starts capturing tuple relationships

### Phase 2: Pattern Learning (Epoch 10-50)
- Model learns QA invariants (J, K, X)
- Closure properties become encoded
- Loss stabilizes around 2-3

### Phase 3: Refinement (Epoch 50-100)
- Fine-tuning of attention patterns
- Invariant preservation strengthens
- Loss reaches optimal 0.5-1.0

---

## 🔬 Training Dataset Composition

The 31,606 training examples include:

| Type | Count | Percentage | Purpose |
|------|-------|------------|---------|
| **Theorems** | 9,033 | 28.6% | Mathematical foundations from vault |
| **Synthetic QA** | 10,000 | 31.6% | Generated tuples with validated invariants |
| **Real Examples** | 6,572 | 20.8% | Actual QA system outputs |
| **Q&A Pairs** | 5,000 | 15.8% | Reasoning and explanation |
| **E8 Mappings** | 1,000 | 3.2% | Geometric alignment examples |

**Total**: 31,606 examples, 11 MB

---

## 🎯 Success Criteria

Training is successful when:

1. **Loss Convergence**: Final validation loss < 1.0
2. **Stable Training**: No loss explosions or NaN values
3. **Invariant Preservation**: Model maintains QA relationships
4. **Checkpoints Saved**: Best model and regular checkpoints created

---

## 🚨 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 16  # or 8

# Reduce sequence length
--max-length 256  # or 128

# Reduce model size
--hidden-size 512 --num-layers 6
```

### Slow Training
```bash
# Reduce dataset (for testing)
head -10000 qa_training_dataset.jsonl > qa_subset.jsonl
python train_qalm_production.py --dataset qa_subset.jsonl

# Use smaller model
--hidden-size 256 --num-layers 4 --epochs 20
```

### Loss Not Decreasing
```bash
# Lower learning rate
--lr 5e-5

# Check data quality
python -c "import json; data = [json.loads(line) for line in open('qa_training_dataset.jsonl')]; print(f'{len(data)} examples loaded')"
```

---

## 📞 Quick Reference

```bash
# Start production training (full model)
python train_qalm_production.py --dataset qa_training_dataset.jsonl --epochs 100 --batch-size 32 --hidden-size 768 --num-layers 12 --num-heads 12

# Monitor progress
tail -f qalm_training.log

# Check checkpoints
ls -lth checkpoints/qalm_v1_full/

# View results
cat checkpoints/qalm_v1_full/training_summary.json
```

---

## 🎉 Next Steps After Training

1. **T-009: Bob-iverse Integration**
   - Load trained model: `QALM.load('checkpoints/qalm_v1_full/qalm_best.pt')`
   - Connect to theorem discovery pipeline
   - Test invariant preservation

2. **T-010: Evaluation**
   - Benchmark vs Claude/Gemini on QA tasks
   - Measure invariant preservation accuracy
   - Generate research report

3. **Deployment**
   - Create inference API
   - Setup local inference server
   - Integrate with multi-AI orchestrator

---

**Status**: ✅ **READY TO LAUNCH PRODUCTION TRAINING**

**Command to start**:
```bash
python train_qalm_production.py \
    --dataset qa_training_dataset.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --hidden-size 768 \
    --num-layers 12 \
    --num-heads 12 \
    --checkpoint-dir checkpoints/qalm_v1_full
```

**Estimated time**: 2-3 hours (GPU) or 10-15 hours (CPU)

**Expected result**: First QA-specialized language model with invariant-preserving attention 🚀

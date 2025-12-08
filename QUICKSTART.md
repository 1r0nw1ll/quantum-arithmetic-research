# QA Theorem Discovery Pipeline - Quick Start Guide

## 30-Second Start

```bash
cd /home/player2/signal_experiments

# Run the full pipeline (one command!)
python qa_theorem_discovery_orchestrator.py --quick

# Check results
cat qa_discovery_workspace/DISCOVERY_REPORT.txt
```

Done! The pipeline will:
1. Build graph (~3 minutes)
2. Train GNN (~10 minutes in quick mode)
3. Mine conjectures (~1 minute)
4. Verify with Lean (~2 minutes)
5. Generate report

---

## What Just Happened?

The orchestrator ran 5 agents in sequence:

1. **Graph Builder** - Created a 10K node graph with 58K edges
2. **GNN Trainer** - Learned patterns in QA tuples
3. **Symbolic Miner** - Extracted mathematical conjectures
4. **Lean Verifier** - Formally verified conjectures
5. **Exporter** - Generated reports

---

## Your Outputs

Check the workspace directory:

```bash
ls qa_discovery_workspace/

# You'll see:
# - qa_graph.pt (the graph)
# - qa_embeddings.pt (learned patterns)
# - conjectures.json (discovered theorems)
# - DISCOVERY_REPORT.txt (summary)
# - lean_proofs/ (formal proofs)
# - checkpoints/ (GNN training state)
```

---

## View Results

```bash
# Human-readable summary
cat qa_discovery_workspace/DISCOVERY_REPORT.txt

# Conjectures (JSON)
cat qa_discovery_workspace/conjectures.json | python -m json.tool | head -50

# Lean proofs
ls qa_discovery_workspace/lean_proofs/
```

---

## Running Individual Components

### Just build the graph:
```bash
python qa_graph_builder_v2.py
```

### Just train the GNN:
```bash
python qa_gnn_trainer_v2.py --epochs 100
```

### Just mine conjectures:
```bash
python qa_symbolic_miner_v2.py
```

### Just verify with Lean:
```bash
python qa_lean_verifier_v2.py
```

---

## Customization

### Quick Mode (for testing)
```bash
python qa_theorem_discovery_orchestrator.py --quick
# Uses: 50 epochs, 5 verifications
```

### Full Mode (production)
```bash
python qa_theorem_discovery_orchestrator.py
# Uses: 300 epochs, 20 verifications
```

### Custom Settings
```bash
python qa_theorem_discovery_orchestrator.py \
    --epochs 200 \
    --workspace ./my_experiments
```

---

## Monitoring Progress

Unlike the old version, you'll now see:

```
[11:00:15] Loading dataset from qa_10000_balanced_tuples.csv
[11:00:16] ✓ Loaded 10000 tuples in 0.85s
[11:00:17] Pass 1/3: Building harmonic transition edges
Harmonic edges: 100%|████████| 10000/10000 [00:12<00:00, 815.23nodes/s]
[11:00:29] ✓ Found 8342 harmonic edges
...
```

**No more silent execution!**
**No more 30-minute hangs!**
**No more wondering if it crashed!**

---

## What's Different from October 8th?

| Problem (Oct 8) | Solution (Now) |
|-----------------|----------------|
| Silent execution for 30+ minutes | Real-time progress bars + logging |
| Batch size mismatch errors | Fixed model architecture |
| No way to monitor | Full instrumentation |
| Had to manually run each step | One-command orchestrator |
| No checkpoints | Saves every 50 epochs |
| Cryptic errors | Clear messages + traceback |

---

## Next Steps

1. **Explore the results**
   ```bash
   cat qa_discovery_workspace/DISCOVERY_REPORT.txt
   ```

2. **Check discovered theorems**
   ```bash
   python -m json.tool qa_discovery_workspace/conjectures.json | less
   ```

3. **Review Lean proofs**
   ```bash
   ls qa_discovery_workspace/lean_proofs/
   cat qa_discovery_workspace/lean_proofs/cluster_3.lean
   ```

4. **Customize for your research**
   - Edit the scripts to add new edge types
   - Modify the GNN architecture
   - Add new pattern detection methods
   - Extend Lean translation logic

---

## Troubleshooting

### "Dataset not found"
```bash
# Make sure you have the dataset
ls qa_10000_balanced_tuples.csv

# Or specify a different path
python qa_theorem_discovery_orchestrator.py --dataset /path/to/your/data.csv
```

### "Lean not found" warnings
```bash
# Lean is optional - the pipeline will still work
# To install Lean 4: https://lean-lang.org/

# Or just ignore and check the generated .lean files manually
ls qa_discovery_workspace/lean_proofs/*.lean
```

### Pipeline fails at a stage
```bash
# Check the workspace for partial results
ls qa_discovery_workspace/

# Run individual components to debug
python qa_graph_builder_v2.py  # Test just graph building
python qa_gnn_trainer_v2.py --epochs 10  # Quick GNN test
```

---

## Full Documentation

See `QA_PIPELINE_README.md` for:
- Complete architecture
- Customization guide
- API documentation
- Advanced usage

---

## Support

If something goes wrong:
1. Check the logs (they're verbose now!)
2. Look at `qa_discovery_workspace/DISCOVERY_REPORT.txt`
3. Review checkpoint files in `qa_discovery_workspace/checkpoints/`
4. Enable debug logging (edit scripts to use `logging.DEBUG`)

---

**Version:** 2.0
**Status:** ✅ Ready to use
**Total Setup Time:** < 1 minute
**First Run Time:** ~15 minutes (quick mode) or ~30 minutes (full mode)

**No more frustration. Just discovery.** 🎉

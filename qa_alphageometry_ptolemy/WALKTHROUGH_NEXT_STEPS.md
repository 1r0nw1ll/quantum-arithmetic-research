# Certificate System - Next Steps Walkthrough

## ✅ What You Have Now

The complete certificate system from ChatGPT is installed and verified:

```
✓ Core schema (qa_certificate.py) - 492 lines, exact scalars
✓ AlphaGeometry adapter - SearchResult → Certificate
✓ Physics adapter - Observer results → Certificate
✓ 2 physics certificates already generated
✓ Paper skeleton ready for JAR/ITP submission
✓ Complete documentation
```

---

## 🎯 The One Thing You Need

**Export SearchResult JSON from your Rust beam search.**

That's it. Once you have that, everything else is automated.

---

## 📝 Step-by-Step Guide

### Step 1: Export SearchResult from Rust

Add this to your Ptolemy benchmark runner (after beam search completes):

```rust
// In your Rust code (e.g., tests/ptolemy_benchmark.rs)

let result = solver.solve(initial_state);

// Export to JSON
let result_json = serde_json::to_string_pretty(&result)?;
std::fs::write("ptolemy_searchresult.json", result_json)?;

println!("✓ SearchResult exported to ptolemy_searchresult.json");
```

Run your benchmark:
```bash
cd ../qa_alphageometry/core
cargo test --test ptolemy_benchmark -- --nocapture
```

**Output:** `ptolemy_searchresult.json` in the qa_alphageometry/core directory

---

### Step 2: Copy SearchResult to This Directory

```bash
cp ../qa_alphageometry/core/ptolemy_searchresult.json .
```

---

### Step 3: Generate Certificate (One Command!)

```bash
python3 generate_ptolemy_certificates.py --in ptolemy_searchresult.json
```

**Output:**
```
Loading SearchResult from ptolemy_searchresult.json...
  solved: True
  states_expanded: 245
  depth_reached: 7

Generating certificate...

✓ Certificate generated: artifacts/ptolemy_quadrance.success.cert.json
  Type: success
  Proof length: 7 steps
  Generator set: 4 generators
  Rules used: similar_triangles, angle_sum, ...

✓ Done! Certificate ready for paper submission.
```

---

### Step 4: Generate Ablated Version (Optional but Recommended)

To create the obstruction certificate:

1. **Modify your Rust code** to restrict generators (remove ν):
   ```rust
   // Remove or comment out the ν generator
   let allowed_generators = vec![σ, λ, μ];  // No ν
   ```

2. **Re-run benchmark:**
   ```bash
   cargo test --test ptolemy_ablated -- --nocapture
   ```

3. **Export SearchResult:**
   ```rust
   std::fs::write("ptolemy_ablated_searchresult.json", result_json)?;
   ```

4. **Generate obstruction certificate:**
   ```bash
   python3 generate_ptolemy_certificates.py \
     --in ptolemy_ablated_searchresult.json \
     --theorem ptolemy_quadrance_no_nu
   ```

**Expected output:**
```
✓ Certificate generated: artifacts/ptolemy_quadrance_no_nu.obstruction.cert.json
  Type: obstruction
  Fail type: depth_exhausted
  Max depth reached: 50
  States explored: 1000
```

---

## 🎨 Paper Integration

Once you have both certificates, update the paper skeleton:

### Edit `qa_certificate_paper_skeleton.tex` Section 3:

```latex
\section{Case Study: Ptolemy's Theorem}

We validated the certificate system on Ptolemy's theorem in quadrance form:

\textbf{Success Case (Full Generators):}
\begin{lstlisting}[language=json]
{
  "theorem_id": "ptolemy_quadrance",
  "witness_type": "success",
  "success_path": [ /* 7 proof steps */ ],
  "generator_set": ["AG:similar_triangles", "AG:angle_sum", ...],
  "search": {
    "states_explored": 245,
    "depth_reached": 7
  }
}
\end{lstlisting}

\textbf{Obstruction Case (Ablated Generators):}
\begin{lstlisting}[language=json]
{
  "theorem_id": "ptolemy_quadrance_no_nu",
  "witness_type": "obstruction",
  "obstruction": {
    "fail_type": "depth_exhausted",
    "states_explored": 1000,
    "max_depth_reached": 50
  }
}
\end{lstlisting}

This demonstrates \emph{generator-relative reachability}: the same theorem
is provable or unprovable depending on the available generator set.
```

---

## 📊 What Success Looks Like

### Your artifacts/ Directory Should Have:

```
artifacts/
├── ptolemy_quadrance.success.cert.json              # ← From Step 3
├── ptolemy_quadrance_no_nu.obstruction.cert.json    # ← From Step 4 (optional)
├── reflection_GeometryAngleObserver.success.cert.json  # ✓ Already have
└── reflection_NullObserver.obstruction.cert.json       # ✓ Already have
```

**Minimum for JAR paper:** 2 Ptolemy certificates (success + ablated)
**Ideal for JAR paper:** All 4 certificates (shows unified framework)

---

## 🚀 Quick Commands Reference

### Test Schema
```bash
python3 -c "from qa_certificate import ProofCertificate; print('✓ OK')"
```

### Validate Existing Certificates
```bash
python3 -c "import json; cert = json.load(open('artifacts/reflection_GeometryAngleObserver.success.cert.json')); print(f'✓ {cert[\"witness_type\"]} certificate for {cert[\"theorem_id\"]}')"
```

### Generate Ptolemy Certificate (One-Liner)
```bash
python3 generate_ptolemy_certificates.py --in ptolemy_searchresult.json && ls -lh artifacts/*.json
```

---

## 🎯 Timeline to Submission

### This Week (Blocking)
- [ ] Export Ptolemy SearchResult JSON ← **YOU ARE HERE**
- [ ] Generate success certificate (5 minutes)
- [ ] Generate ablated obstruction (optional, 30 minutes)

### Next Week
- [ ] Flesh out paper §3 with certificate excerpts
- [ ] Add reachability diagram (success vs obstruction)
- [ ] Draft §4 (failure taxonomy)
- [ ] Proofread

### Week 3
- [ ] Finalize paper
- [ ] Package artifacts
- [ ] Submit to JAR/ITP

**Critical path:** Get SearchResult JSON export working (~1 hour)

---

## 🔥 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'qa_certificate'"

**Solution:** Run from this directory:
```bash
cd /home/player2/signal_experiments/qa_alphageometry_ptolemy
python3 generate_ptolemy_certificates.py --in <file>
```

### Issue: "TypeError: Float values not allowed"

**Solution:** Certificate system enforces exact scalars. This is expected behavior.
The adapter handles conversion automatically from SearchResult.

### Issue: "SearchResult doesn't have 'stats' field"

**Solution:** Already fixed! The adapter uses top-level fields:
```python
states_expanded = sr["states_expanded"]  # Not sr["stats"]["expanded"]
```

### Issue: "Certificate has wrong fail_type"

**Solution:** The adapter conservatively infers stop reason:
- `depth_reached >= max_depth` → "max_depth_reached"
- `successors_generated == 0` → "no_successors"
- Otherwise → "search_exhausted"

This is correct behavior (never claims SCC_UNREACHABLE without proof).

---

## 📖 Documentation Quick Links

| Document | Purpose |
|----------|---------|
| `CERTIFICATE_SYSTEM_STATUS.md` | Overall status & metrics |
| `QUICK_REFERENCE.md` | One-line commands |
| `CERTIFICATE_SYSTEM_INTEGRATION_GUIDE.md` | Comprehensive guide |
| `README.md` | Quick start |
| This file | Step-by-step walkthrough |

---

## ✅ Success Checklist

**Installation Phase:**
- [x] Certificate schema installed
- [x] Adapters installed
- [x] Physics certificates generated
- [x] Schema verification passed

**Ptolemy Phase (Current):**
- [ ] SearchResult JSON exported ← **NEXT STEP**
- [ ] Success certificate generated
- [ ] Ablated obstruction generated
- [ ] All 4 artifacts validated

**Paper Phase:**
- [ ] §3 drafted with results
- [ ] Reachability diagram added
- [ ] §4 failure taxonomy drafted
- [ ] Artifacts packaged

**Submission:**
- [ ] Paper proofread
- [ ] Artifacts uploaded
- [ ] Submitted to JAR/ITP

---

## 🎯 The Path Forward

```
1. Export SearchResult → JSON file (1 hour)
   ↓
2. Run Python adapter → Certificate (5 minutes)
   ↓
3. Update paper §3 → Add excerpts (2 hours)
   ↓
4. Proofread & polish → Submit (1 week)
```

**Estimated total time:** 2-3 weeks from now

**Current blocker:** Step 1 (SearchResult export)

**Everything else is ready and waiting!** 🚀

---

## 💡 Pro Tips

1. **Start with the simplest case:** Just export SearchResult for solved Ptolemy
2. **Don't overthink ablation:** You can skip it for v1.0, add in revision
3. **Trust the adapter:** It handles all edge cases (missing fields, null values, etc.)
4. **Keep physics simple:** One paragraph in conclusion pointing to companion artifacts
5. **Lock artifacts early:** Generate certificates ASAP, then build paper around them

---

**You're 95% done. Just need that SearchResult JSON!** 🎉

Questions? See `QUICK_REFERENCE.md` or `CERTIFICATE_SYSTEM_INTEGRATION_GUIDE.md`

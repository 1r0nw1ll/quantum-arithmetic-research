# QA FST Module: Field Structure Theory Completion Layer

QA formalization of Don Briddell's **Field Structure Theory (FST) / Structural Physics (SP)** as a certificate-backed completion layer. Companion artifact set for submission to *Frontiers of Physics*.

## What gets validated

| Check | Type | Description |
|-------|------|-------------|
| delta_sym | Hard | Proton symmetry defect <= 0.01 (hexagon side imbalance witness) |
| u/d ratio | Hard | MeV and loop quark ratios agree within 0.001 |
| Lambda bookkeeping | Hard | 2187 - 243 - 81 - 27 = 1836 (exact integer arithmetic) |
| Proton MeV drift | Warning | Bookkeeping MeV (936.322) vs PDG proton mass (938.272): logged as `SOURCE_NUMERIC_DRIFT` |

## Commands

```bash
# Run all self-tests (8 checks)
python qa_fst_validate.py

# Validate spine + bundle + manifest (JSON output)
python qa_fst_validate.py --validate

# Regenerate SHA256 manifest over all artifacts
python qa_fst_validate.py --generate-manifest
```

## Warning semantics

`SOURCE_NUMERIC_DRIFT` = the source text's arithmetic is internally consistent, but a downstream comparison to an independent reference (PDG proton mass) shows drift. This is a **warning, not a hard fail** — the completion layer verifies what is checkable and logs the rest.

## Files

| File | Role |
|------|------|
| `qa_fst_module_spine.json` | Module spine: Plenum, 5 generators, 4 invariants, 4 obstructions |
| `qa_fst_cert_bundle.json` | Worked certificates: proton stability + loop-to-MeV homomorphism |
| `qa_fst_submission_packet_spine.json` | Companion paper posture lock (Frontiers of Physics) |
| `qa_fst_manifest.json` | SHA256 manifest over all replayable artifacts |
| `qa_fst_validate.py` | Deterministic validator + self-test |
| `schemas/*.schema.json` | Draft-07 JSON schemas (5 types) |

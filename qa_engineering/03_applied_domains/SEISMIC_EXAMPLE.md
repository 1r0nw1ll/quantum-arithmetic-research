# Seismic Example: Wave Propagation as QA Orbit Traversal

Seismology is the second certified domain instance in the QA control stack. It was chosen specifically because it has **no obvious connection to cymatics** — proving the compiler law is structural, not a consequence of shared physical substrate.

Cert family [110] `QA_SEISMIC_CONTROL_CERT.v1`.

---

## The State Mapping

| Seismic state | QA orbit family | Physical meaning |
|--------------|----------------|-----------------|
| Quiet | Singularity | No seismic activity; ground at rest |
| P-wave (primary wave) | Satellite | Compressional wave; transitional |
| Surface wave (Rayleigh/Love) | Cosmos | Full propagation pattern; stable, reproducible |

Seismic waves follow a progression: a seismic event first creates P-waves (fast, compressional) which then develop into surface waves (slow, large-amplitude, most destructive). This progression — quiet → P-wave → surface wave — is exactly `singularity → satellite → cosmos` with k=2.

---

## The Generator Mapping

| Physical operation | QA generator | Physical meaning |
|------------------|-------------|-----------------|
| Increase gain | maps to σ or domain-specific | Amplify the seismic signal |
| Apply lowpass filter | maps to ν-like | Reduce high-frequency components, expose wave structure |

The certified path:
```
initial: quiet (singularity)
  → apply increase_gain
intermediate: p_wave (satellite)
  → apply apply_lowpass
final: surface_wave (cosmos)
path_length_k = 2
```

Same orbit trajectory as cymatics. Same path length. Different physics.

---

## The Cross-Domain Comparison

This is the key table from cert family [117]:

| Domain | Initial | Intermediate | Target | Orbit path | k | Moves |
|--------|---------|--------------|--------|-----------|---|-------|
| Cymatics | flat | stripes | hexagons | singularity→satellite→cosmos | 2 | increase_amplitude, set_frequency |
| Seismology | quiet | p_wave | surface_wave | singularity→satellite→cosmos | 2 | increase_gain, apply_lowpass |

The orbit path and path length are identical. The moves and state labels are completely different. This is what "the compiler law is structural" means.

---

## Running the Seismic Validator

```bash
cd qa_alphageometry_ptolemy/qa_seismic_control

python qa_seismic_control_validate.py --self-test
python qa_seismic_control_validate.py --demo
```

The PASS fixture certifies the quiet → p_wave → surface_wave path.
The FAIL fixture certifies that an illegal transition (e.g., jumping from quiet directly to surface_wave, skipping satellite) produces `ILLEGAL_TRANSITION`.

---

## What This Means for SVP Practice

Seismic phenomena are relevant to SVP because they involve **resonance propagation through a medium** — the same principle as acoustic resonance, but in solid/fluid earth materials.

The QA certification shows:
1. The same abstract control law governs radically different physical media
2. If you understand resonance in one medium (air, water), you have a structural template for others
3. "Sympathetic vibration" is not limited to identical materials — the orbit structure is the invariant

For a practitioner: if you can map your medium's states to QA orbit families, the control design from any other certified domain transfers immediately.

---

## Seismic-Specific Failure Modes

The seismic domain adds these to the standard QA failure taxonomy:

| Fail type | Trigger |
|-----------|---------|
| `ILLEGAL_TRANSITION` | Generator applied outside its legal precondition for seismic states |
| `GOAL_NOT_REACHED` | Final wave type ≠ target wave type |
| `PATH_LENGTH_EXCEEDED` | Generator sequence longer than max_path_length_k |

---

## Source References

- Cert family [110]: `docs/families/110_qa_seismic_control.md`
- Implementation: `qa_alphageometry_ptolemy/qa_seismic_control/`
- Control Stack (cross-domain proof): cert family [117], `qa_alphageometry_ptolemy/qa_control_stack/`
- Cross-domain principle: `CROSS_DOMAIN_PRINCIPLE.md`

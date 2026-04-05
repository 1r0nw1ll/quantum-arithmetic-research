# QA Cert Family Roadmap

A guide to which certificate families to study, in what order, and why. The full ecosystem has 126 certified families — this document helps you navigate to the ones that matter for your use case.

---

## The Essential 8 (Start Here)

These eight families form the backbone. Read them in this order:

| Order | Family | What it is | Why it matters |
|-------|--------|-----------|---------------|
| 1 | **[120]** QA_PUBLIC_OVERVIEW_DOC | Presentation-grade overview | Fastest way to understand the whole system |
| 2 | **[119]** QA_DUAL_SPINE_UNIFICATION | Formal top-level map | Places both spines in context; the synthesis |
| 3 | **[107]** QA_CORE_SPEC | Kernel axioms | The root of everything; all other certs inherit from this |
| 4 | **[105]** QA_CYMATIC_CONTROL | Cymatics 4-tier | Most intuitive applied example for SVP practitioners |
| 5 | **[106]** QA_PLAN_CONTROL_COMPILER | Compiler law | The formal bridge between planning and execution |
| 6 | **[110]** QA_SEISMIC_CONTROL | Seismology instance | Proves [105] isn't domain-specific |
| 7 | **[117]** QA_CONTROL_STACK | Cross-domain proof | The Control Stack Theorem in cert form |
| 8 | **[116]** QA_OBSTRUCTION_STACK_REPORT | Obstruction spine | What's impossible and why |

---

## By Use Case

### "I want to apply QA to a new physical domain"

1. Read [120] for the big picture
2. Read [105] (cymatics) as your template
3. Read [106] to understand the compiler you'll need to satisfy
4. Map your domain using the checklist in `03_applied_domains/CROSS_DOMAIN_PRINCIPLE.md`
5. Build your Tier 1 cert (recognition/mode cert) first
6. Work up to Tier 4 (planner cert) before submitting to [106]

Key families: [107], [105], [106], [110], [117]

### "I want to understand what's impossible before I search"

1. Read [111] → [112] → [113] → [114] → [115] → [116] in sequence
2. The obstruction spine: how arithmetic impossibility propagates up through the cert stack

| Family | What it certifies |
|--------|-----------------|
| [111] QA_AREA_QUANTIZATION_PK | v_p(r)=1 → forbidden quadrea |
| [112] QA_OBSTRUCTION_COMPILER_BRIDGE | Forbidden quadrea → unreachable control state |
| [113] QA_OBSTRUCTION_AWARE_PLANNER | Planner prunes before BFS |
| [114] QA_OBSTRUCTION_EFFICIENCY | nodes_expanded=0 for obstructed targets |
| [115] QA_OBSTRUCTION_STACK | Synthesis cert |
| [116] QA_OBSTRUCTION_STACK_REPORT | Reader-ready validated report |

Canonical example: r=6, p=3, k=2 → v₃(6)=1 → UNREACHABLE → 0 nodes expanded → pruning_ratio=1.0

### "I want to do neural network work with QA"

Key theory: `memory/curvature_theory.md` and the Finite-Orbit Descent Theorem
Scripts: `intelligent_coprocessor_v2.py`, `statistical_validation_gauntlet.py`
Paper: `papers/in-progress/unified-curvature/paper.tex` (14pp, arXiv-ready)

Related cert families: [98] GNN spectral gain, [99] attention spectral gain, [101] gradient L2 norm

### "I want to work with Pythagorean geometry"

Paper: `papers/in-progress/pythagorean-families/paper.tex` (14pp, submission-ready)
Verify script: `papers/in-progress/pythagorean-families/verify_classification.py` → 10/10 PASS

Key result: Five Pythagorean families = orbits of F=[[0,1],[1,1]] on (Z/9Z)²
Barning-Berggren Intertwining Theorem (Theorem 6): τ(M_X·u) = R_X·τ(u)
Venue: International Journal of Number Theory (IJNT)

### "I want to work with signal processing / audio"

Scripts: `run_signal_experiments_final.py`, `geometric_autopsy.py`
Tests: pure tones, major/minor chords, tritones, white noise
Metrics: E8 alignment, Harmonic Index, noise annealing

Note: E8 hypothesis was FALSIFIED [cert [100]]: QA median 0.911 < random baseline 0.930. E8 alignment is incidental, not structural.

### "I want to understand the full QA ecosystem (all 126 families)"

See: `docs/families/README.md` — the full family index with two-tract checklist
Range: families [18]–[120], documented in `docs/families/[NN]_*.md`
Meta-validator: `qa_alphageometry_ptolemy/qa_meta_validator.py`

---

## The Inheritance Tree

```
[107] QA_CORE_SPEC.v1 (kernel — root of everything)
  │
  ├── [108] QA_AREA_QUANTIZATION_CERT (family_extension)
  │
  ├── [106] QA_PLAN_CONTROL_COMPILER (family_extension)
  │     ├── [105] QA_CYMATIC_CONTROL (domain_instance)
  │     └── [110] QA_SEISMIC_CONTROL (domain_instance)
  │
  ├── [109] QA_INHERITANCE_COMPAT (certifies inheritance edges)
  │
  ├── [111]–[116] Obstruction spine
  │     [111]→[112]→[113]→[114]→[115]→[116]
  │
  ├── [117] QA_CONTROL_STACK (synthesizes [105]+[110]+[106])
  │     └── [118] QA_CONTROL_STACK_REPORT
  │
  └── [119] QA_DUAL_SPINE_UNIFICATION
        └── [120] QA_PUBLIC_OVERVIEW_DOC
```

**Scope rules** (what can inherit from what):
- `kernel` → `family_extension`: ✓
- `family_extension` → `family_extension`: ✓
- `family_extension` → `domain_instance`: ✓
- `domain_instance` → any further inheritance: ✗ (must go through family_extension)

---

## Family Tags

| Tag | Meaning | Examples |
|-----|---------|---------|
| `kernel` | Root spec; no parents | [107] |
| `family_extension` | Extends kernel; adds domain-specific generators | [106], [108] |
| `domain_instance` | Instantiates a family_extension in a specific domain | [105], [110] |
| `synthesis` | Certifies cross-family properties | [117], [119] |
| `report` | Reader-facing validated report | [118], [120] |
| `obstruction` | Documents impossibility and pruning | [111]–[116] |

---

## Running the Full Ecosystem

```bash
# Verify all 126 families in ~10 seconds:
cd qa_alphageometry_ptolemy
python qa_meta_validator.py

# Verify a specific family:
python qa_core_spec/qa_core_spec_validate.py --self-test
python qa_cymatics/qa_cymatics_validate.py --self-test
python qa_plan_control_compiler/qa_plan_control_compiler_validate.py --self-test
python qa_control_stack/qa_control_stack_validate.py --self-test

# Run Gate 0 protocol checks:
python ../qa_mapping_protocol/validator.py --self-test
python ../qa_mapping_protocol_ref/validator.py --self-test
```

---

## Next Families Being Developed

Based on SVP domains of interest for Tier 4:

| Planned family | Domain | Notes |
|---------------|--------|-------|
| Sound healing / tuning forks | Acoustic resonance | Map Hz frequencies to (b,e) pairs |
| Crystal bowl harmonics | Standing waves in solid | Faraday-like cert structure |
| Water cymatics | Fluid surface patterns | Extension of [105] Faraday cert |
| Plant growth geometry | Fibonacci / golden ratio | Connect to Pythagorean families |
| EEG brain state control | Neural oscillations | Seizure vs. baseline orbit families |

If you want to work on one of these: start with the domain mapping checklist in `03_applied_domains/CROSS_DOMAIN_PRINCIPLE.md`.

# Family [193] QA_LEVIN_COGNITIVE_LIGHTCONE_CERT.v1

## One-line summary

Michael Levin's cognitive light cone (CLC) — the spatiotemporal scale of the largest goal an agent can pursue — mapped to QA orbit radius. Singularity=0 (no goals), Satellite=8 (local), Cosmos=24 (far-reaching). Cancer = CLC shrinkage = Cosmos→Satellite transition. Tiered Reachability [191] gives the structural CLC ceiling: 26% L1-reachable.

## Mathematical content

### CLC ↔ Orbit radius mapping

| Orbit | Cycle length | CLC radius | Cognitive regime |
|-------|-------------|------------|------------------|
| Singularity | 1 | 0 | No goals, fixed point |
| Satellite | 4-8 | 8 | Local/unicellular goals |
| Cosmos | 12-24 | 24 | Organism-level, far-reaching |

### Cancer as CLC shrinkage

Cancer = cells whose CLC drops from Cosmos (24-cycle, organism) to Satellite (8-cycle, unicellular). Cells revert to ancient single-cell goals. NOT Singularity (which is stasis) — cancer cells are ACTIVE (proliferating, migrating = cycling behavior).

### Structural ceiling

[191] Tiered Reachability: only 26% of S_9 is Level-I reachable. This constrains the CLC — an agent at Level I cannot reach 74% of configurations without higher-order operators.

## Dependencies

- [191] QA_BATESON_LEARNING_LEVELS_CERT.v1
- Beyond Neurons certificate

## Sources

- Levin & Resnik, "Mind Everywhere" (Biological Theory, 2026)
- Lyons, Pio-Lopez & Levin, "From Cancer to AI Alignment" (Preprints, 2026)

## Verification Note (2026-07-04)

Independently checked both citations rather than trusting them at face
value. **Levin & Resnik, "Mind Everywhere: A Framework for
Conceptualizing Goal-Directedness in Biology and Other Domains"**
(Biological Theory, two-part, published 2026-02-25) confirmed real —
the "cognitive light cone" concept and its exact framing ("an effective
explanation of what [agential systems] are doing must involve events in
a larger cognitive light cone around them than the local here and now")
appear in this exact paper. **Lyons, Pio-Lopez & Levin, "From Cancer to
AI Alignment: Tackling Externalities Through Homeostatic Principles"**
(Preprints.org, DOI 10.20944/preprints202604.0056.v1, 2026) confirmed
real — its own stated thesis is "cancer is what happens when cells
disconnect from the collective's cognitive light cone and revert to
local optimization," matching this cert's "Cancer as CLC shrinkage"
claim almost verbatim.

Validator (`qa_levin_cognitive_lightcone_cert_validate.py`) imports the
shared QA primitives from `tools/qa_kg/orbit_failure_enumeration.py`
(cert [263]'s utility module) and performs genuine orbit-family
recomputation, not fixture-trusting. No bugs found; both source papers
are accurately represented.

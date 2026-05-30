<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert documentation; primary sources cited in mapping_protocol_ref.json and validator -->

# [280] QA Nuclear Magic Orbit

**Cert family**: `qa_nuclear_magic_orbit_cert_v1`
**Primary sources**:
- Mayer, M. G. (1949). On Closed Shells in Nuclei. *Physical Review*, 75(12), 1969–1970. DOI: [10.1103/PhysRev.75.1969](https://doi.org/10.1103/PhysRev.75.1969)
- Haxel, O., Jensen, J. H. D., & Suess, H. E. (1949). On the "Magic Numbers" in Nuclear Structure. *Physical Review*, 75(11), 1766. DOI: [10.1103/PhysRev.75.1766.2](https://doi.org/10.1103/PhysRev.75.1766.2)
- Mechanism: cert [279] QA Orbit Access Theorem (Wall 1960)

## Claim

The canonical nuclear magic numbers {2, 8, 20, 28, 50, 82, 126, 184} partition under mod-9 route enumeration into exactly three classes:

| Class | Value(s) | Routes | Cosmos | Satellite | Singularity |
|---|---|---|---|---|---|
| `no_routes` | 2 | 0 | — | — | — |
| `coprime_to_3` | 8, 20, 28, 50, 82, 184 | 3–91 each | all | 0 | 0 |
| `mul_9` | **126** | 62 | 42 | **14** | **6** |

**a=126 is the unique nuclear magic number divisible by 3** (and by 9). All others in {8, 20, 28, 50, 82, 184} are coprime to 3.

## Structural Significance

- Pb-208 (Z=82, N=126) is the heaviest known doubly-magic stable nucleus. Under this cert: Z=82 is pure Cosmos; N=126 has full orbit access (Cosmos+Satellite+Singularity). The two magic numbers composing the most stable nucleus are structurally distinct.
- The Satellite orbit is 3D (period 8, vs Cosmos period 24). a=126 being the only magic number with Satellite access is consistent with 126 being the "hardest" shell to account for in simple nuclear shell models.
- a=2 has no QA routes at all — the simplest nuclear closure (helium) sits outside the mod-9 route framework entirely.

## Scope Boundaries

- Does **not** claim QA causes nuclear shell closure
- Does **not** predict binding energy, angular momentum, or shell occupancy
- Does **not** assert a=184 is a confirmed stable shell (it is a predicted island-of-stability candidate)
- The orbit classification is purely arithmetic; no physical mechanism is claimed

## Gates

- **NMO_1**: `no_routes` fixtures: total_routes=0
- **NMO_2**: `coprime_to_3` fixtures: satellite=0 AND singularity=0
- **NMO_3**: `mul_9` fixtures: exact counts match (total=62, cosmos=42, satellite=14, singularity=6)
- **NMO_4**: declared `magic_number_class` matches actual gcd(a,3) classification
- **SRC**: `mapping_protocol_ref.json` present and well-formed
- **F**: every FAIL fixture declares `expected_fail_type` and fires

6 PASS fixtures, 4 FAIL fixtures. Validator: `qa_nuclear_magic_orbit_cert_validate.py --self-test`.

## Lineage

Discovered 2026-05-30 during domain sweep across cert [279] (Orbit Access Theorem). Cert [279] provides the algebraic mechanism (Satellite iff 3\|a, Singularity iff 9\|a); cert [280] applies it to the nuclear magic number set from Mayer (1949) and Haxel/Jensen/Suess (1949). Companion cert [281] (pending) addresses the Pisano-Orbit Correspondence.

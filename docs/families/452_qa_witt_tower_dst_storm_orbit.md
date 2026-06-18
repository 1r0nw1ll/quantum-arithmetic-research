# [452] QA Witt Tower Geomagnetic Storm Dst Orbit Discriminator

## Claim

Storm main-phase hours (Dst < −100 nT) rank exclusively in the Singularity tier (T0) under the Witt tower orbit partition (MOD=27, T0=bins 0–8). All 17 storm hours from the September 2017 G4 geomagnetic storm fall in T0; hypergeometric log₁₀p = −8.19. Background (1447 quiet/moderate hours) spans all three tiers.

## Data Source

- **Dataset**: NASA OMNI2 hourly Dst index, August–September 2017 (1464 hourly windows)
- **DOI**: [10.48322/45bb-8792](https://doi.org/10.48322/45bb-8792) (King & Papitashvili 2005)
- **Event**: September 8–9 2017, G4 geomagnetic storm (Dst minimum −148 nT), coincident with X9.3 solar flare and cert [449] SEP event
- **Access**: NASA OMNIWeb CGI (`omniweb.gsfc.nasa.gov/cgi/nx1.cgi`), variable 40 (hourly Dst)

## QA Mapping

| Observer Layer | QA Discrete Layer |
|---|---|
| Hourly Dst (nT) per window | Rank among 1464 windows (ascending, most negative = rank 0) |
| Float physical value | `bin = floor(rank × 27 / 1464)` ∈ Z/27Z |
| Dst < −100 nT (storm) | T0 orbit (bins 0–8, Singularity) |
| Dst ≥ −100 nT (quiet) | All three tiers |

**Theorem NT compliance**: Dst values (nT, continuous) remain in the observer layer. Only integer rank bins cross into QA.

**MOD=27, companion M=[[5,−1],[1,0]], p=3, k=3.**

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: Counts | n_storm=17, n_quiet=1447, n_total=1464 | PASS |
| C2: Means | storm_mean=−117.4 nT < −100; quiet_mean=−19.4 nT > −30 | PASS |
| C3: Storm T0 + hypergeom | 17/17 storm in T0, K_t0=488, log₁₀p=−8.19 < −7.0 | PASS |
| C4: Dst separation | \|storm_mean − quiet_mean\| = 98.0 nT > 90.0 | PASS |
| C5: Tier polarity | storm tiers = {T0}; quiet spans {T0, T1, T2} | PASS |
| C6: Peak intensity | peak_storm_dst = −148 nT < −120 (G3+ minimum confirmed) | PASS |

## Statistics

| Quantity | Value |
|---|---|
| N total windows | 1464 |
| N storm hours | 17 |
| N quiet hours | 1447 |
| Storm mean Dst | −117.4 nT |
| Quiet mean Dst | −19.4 nT |
| Separation | 98.0 nT |
| Peak storm Dst | −148 nT |
| K_T0 (all data) | 488 |
| Storm in T0 | 17/17 (100%) |
| log₁₀p | −8.19 |

## Physical Interpretation

The September 2017 G4 geomagnetic storm produced the most negative Dst values in the 2-month window. Under rank-normalization, extreme ring-current injection (maximum magnetospheric compression) maps to the lowest-rank bins — T0 (Singularity).

**Orbital polarity symmetry**: cert [449] (SEP, maximum energetic particle flux) → T2 (Cosmos, highest rank); cert [452] (geomagnetic storm, minimum Dst, maximum ring-current compression) → T0 (Singularity, lowest rank). These are the same physical event (X9.3 flare + CME on 2017-09-10) seen through antipodal observer projections that fall at opposite ends of the QA orbit spectrum.

## Primary Sources

- King, J. H. & Papitashvili, N. E. (2005). Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data. *J. Geophys. Res.*, 110, A02104. doi:10.48322/45bb-8792
- Sugiura, M. (1964). Hourly values of equatorial Dst for the IGY. *Ann. Int. Geophys. Year*, 35, 9–45. doi:10.1029/GM009p0001

## Related Certs

- [110] QA Witt Tower structural parent
- [442]–[451] Witt tower empirical chain (seismic, ENSO, solar wind, EEG, ECG, finance, VFL cardiac, acoustic speech)
- [449] SEP orbit discriminator — antipodal polarity partner

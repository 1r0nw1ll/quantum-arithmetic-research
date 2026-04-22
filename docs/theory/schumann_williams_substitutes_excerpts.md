<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.8 item 6 follow-up; Williams 1992 Science AAAS-paywalled, substitutes landed here -->

# Schumann Resonance — Williams-1992 Substitutes — Primary-Source Excerpts

**Corpus root:** `Documents/schumann_resonance/`
**Scope:** QA-MEM Phase 4.8 item 6 Track C follow-up. Williams, E. R. (1992). *The Schumann resonance: A global tropical thermometer*. Science 256(5060):1184–1187. DOI: 10.1126/science.256.5060.1184 is AAAS-paywalled; the 2026-04-20 acquisition session (commit `3bde4e4`) captured it as DEFERRED. This file grounds three open-access primary-source substitutes that together cover the two numerical quantities Williams 1992 brought to the literature: (i) the Earth–ionosphere cavity **quality factor Q**, and (ii) the **solar-cycle / solar-activity modulation** of Schumann resonance parameters.
**Domain:** `physics` (QA-MEM taxonomy).

**QA-research grounding:** the eventual `qa_heartmath_mapping_cert_v1` numerical cert — per `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_ACQUISITION.md` §C — must declare which harmonic family (Schumann-theoretical `√(n(n+1))` vs NOAA-observed) it certifies against, and must carry a bounded-error claim grounded in primary data. Williams 1992 was originally scoped as the modern measured-harmonic-values source. With Williams deferred, these three substitutes provide: Price 2016's canonical `Q = 4–6` Earth value (scalar target), Bozóki et al. 2021's long-term SR intensity solar-cycle modulation (up to 60 %) and Sátori 2005 12–15 % Q-factor modulation, and Ikeda et al. 2018's direct 2003 solar X-ray ↔ SR-frequency event data. Authoring the mapping cert remains out of scope for the current acquisition session — spec is explicit.

**Firewall note (Theorem NT):** all quantities below are continuous-valued observer projections. They enter the QA discrete layer only through the integer harmonic index `n ∈ {1, 2, 3, …}`, integer-ratio approximations `ω_n / ω_1`, and bounded-error residuals — never as causal inputs. Any future mapping cert must preserve that boundary.

---

## 1. Price 2016 — canonical Earth Q-factor + observed-harmonic framework

### Publication

C. Price, *ELF Electromagnetic Waves from Lightning: The Schumann Resonances*, **Atmosphere** 7(9):116 (2016). DOI: 10.3390/atmos7090116. MDPI, CC-BY. Retrieved 2026-04-22 via Semantic Scholar PDF mirror (`pdfs.semanticscholar.org/765e/0527e9d0f264efb1fd530cf5fe79b191f692.pdf`). On disk: `Documents/schumann_resonance/price_2016_atmosphere_elf_schumann.pdf` (4.0 MB, 20 pages).

### #price-2016-q-factor-earth-four-to-six (p.3, after Eq. 4)

> "The dimensionless quality factor Q of the resonant cavity may be determined as a ratio between the stored energy and the energy loss per cycle. Considering only the electrically stored energy [34]: Q = Re S / (2 Im S) (4). On Earth, the resonance is characterized by a quality factor Q ranging from 4 to 6 [35]."

**Target metric:** canonical scalar Q value for the Earth–ionosphere cavity. Citation [35] in the paper is Jones, D.L. (1974), *J. Geophys. Res.* 69:4037–4046. This is the load-bearing scalar for any mapping-cert bounded-error claim against QA `d`-index integer ratios.

### #price-2016-observed-harmonics-8-14-20-26 (p.5)

> "SR modes can usually be observed in the frequency domain at 8 Hz, 14 Hz, 20 Hz, 26 Hz, etc. (Figure 4b). For studies of global lightning activity, the SR spectra are normally fitted to a set of Lorentzian curves [54,57,58] where the curve for each mode is described by three parameters: peak amplitude, peak frequency and the quality factor."

**Target metric:** observed-harmonic series (integer-rounded). Slightly coarser than the NOAA 7.83 / 14.3 / 20.8 / 27.3 / 33.8 Hz values quoted in `mccraty_item6_excerpts.md`, reflecting Lorentzian-fit averaging across geographic stations.

### #price-2016-solar-cycle-and-flare-modulation (p.14)

> "SR parameters have been shown to vary during cosmic gamma-ray bursts [177,178], solar flares and solar proton events [62,178–180], as well as over the 11-year solar cycle [181–183]."

**Target metric:** canonical textbook statement of the solar-cycle-modulation phenomenon. Bibliographic anchor for the numerical substrate in Bozóki 2021 (§2 below) and Ikeda 2018 (§3 below).

---

## 2. Bozóki, Sátori, Williams et al. 2021 — Williams-lineage solar-cycle cavity deformation

### Publication

T. Bozóki, G. Sátori, E. R. Williams, I. Mironova, P. Steinbach, E. C. Bland, A. Koloskov, Y. M. Yampolski, O. V. Budanov, M. Neska, A. K. Sinha, R. Rawat, M. Sato, C. D. Beggan, S. Toledo-Redondo, Y. Liu, R. Boldi, *Solar Cycle-Modulated Deformation of the Earth–Ionosphere Cavity*, **Frontiers in Earth Science** 9:689127 (2021). DOI: 10.3389/feart.2021.689127. Frontiers, CC-BY (GOLD). Retrieved 2026-04-22 via Frontiers direct PDF endpoint. On disk: `Documents/schumann_resonance/bozoki_satori_williams_2021_frontiers_solar_cycle_cavity.pdf` (5.5 MB, 20 pages).

**Williams-substitute status:** Earle R. Williams (Parsons Laboratory, MIT) is a co-author — this paper is Williams-lineage successor work to the 1992 *Science* paper and inherits its authorial intent. Gabriella Sátori (EPSS Sopron) is the long-term Schumann-resonance PI whose 2005 Q-factor modulation value is the quantitative core cited throughout.

### #bozoki-2021-abstract-solar-cycle-cavity-deformation (p.1, abstract)

> "SR intensity is an excellent indicator of lightning activity and its distribution on global scales. However, long-term measurements from high latitude SR stations revealed a pronounced in-phase solar cycle modulation of SR intensity seemingly contradicting optical observations of lightning from satellite, which do not show any significant solar cycle variation in the intensity and spatial distribution of lightning activity on the global scale. The solar cycle-modulated local deformation of the Earth–ionosphere cavity by the ionization of energetic electron precipitation (EEP) has been suggested as a [candidate mechanism…]"

**Target metric:** the framing contradiction (SR intensity solar-modulated; lightning activity not solar-modulated) that motivates the cavity-deformation mechanism. Sets up the numerical claims that follow.

### #bozoki-2021-satori-2005-q-factor-12-15-percent (p.13, Discussion §4.1)

> "The height and scale height of the lower (electric) layer were considered to be invariant over the solar cycle while for the upper (magnetic) layer the authors inferred a height decrease of 4.7 km and a scale height decrease from 6.9 to 4.2 km for the first SR mode from solar maximum to solar minimum. These changes result in an increase of the resonator's Q-factor by 12–15 % (Sátori et al., 2005). The intensity of a damped simple harmonic oscillator is known to be proportional to the square of its Q-factor (Feynman et al., 1963), in agreement with the theoretical description of SR (Nelson, 1967, Eqs 2–24; Williams et al., 2006). It follows that a 12–15 % change in the Q-factor of the Earth–ionosphere cavity can result in a 25–30 % increase of the first SR mode's intensity."

**Target metric:** `ΔQ / Q ≈ 0.12–0.15` over the 11-year solar cycle, with the causal chain `height decrease 4.7 km → scale-height 6.9→4.2 km → Q×(1.12–1.15) → intensity×(1.25–1.30)`. This is the sharpest Q-modulation numerical in the primary-source literature.

### #bozoki-2021-vernadsky-sr-intensity-60-percent (p.2, Introduction)

> "In 2015 long-term SR intensity records from the Ukrainian Antarctic station 'Akademik Vernadsky' were published which clearly showed a pronounced solar cycle modulation (up to 60 %) of SR intensity (Nickolaenko et al., 2015), just as in an earlier paper by Füllekrug et al. (2002) using data from the Antarctic station Arrival Heights."

**Target metric:** `ΔI_SR / I_SR ≤ 0.60` solar-cycle peak-to-trough at high-latitude stations. Corroborated at Vernadsky (65°S), Arrival Heights (78°S), and a new Svalbard (78°N) station.

### #bozoki-2021-eep-q-factor-10-30-percent (p.12, §3.5)

> "SR intensity clearly increases during the event not only in the H NS component at Syowa (so near the radar) but in the Northern Hemisphere under daytime conditions at the Hornsund (HRN) station as well where increased Q-factor values (by ∼10–30 %) can be observed during the event. The relative increase of SR intensity is as large as 50–100 % at Syowa during this event."

**Target metric:** event-scale `ΔQ / Q ≈ 0.10–0.30` from a single EEP (energetic electron precipitation) event, distinct from the long-term solar-cycle 12–15 % envelope. Grounds a higher-frequency modulation term.

---

## 3. Ikeda, Uozumi, Yoshikawa, Fujimoto, Abe 2018 — direct 2003 solar-flare SR-frequency event data

### Publication

A. Ikeda, T. Uozumi, A. Yoshikawa, A. Fujimoto, S. Abe, *Schumann resonance parameters at Kuju station during solar flares*, **E3S Web of Conferences** 62:01012 (2018), Solar-Terrestrial Relations and Physics of Earthquake Precursors. DOI: 10.1051/e3sconf/20186201012. EDP Sciences, CC-BY. Retrieved 2026-04-22 via e3s-conferences.org direct PDF endpoint. On disk: `Documents/schumann_resonance/ikeda_2018_e3s_kuju_solar_flares_sr.pdf` (1.9 MB, 4 pages).

**Williams-substitute status:** this paper explicitly cites and extends Roldugin et al. (Ann. Geophys. 17:1293 (1999) reference [6]) — Roldugin's 2004 JGR paper (also cited in Bozóki 2021) is the canonical single-event solar-X-ray SR-frequency-shift study that Williams 1992 motivates but does not itself carry. Ikeda 2018 provides the same class of measurement at a different (low-latitude) station, CC-BY-licensed.

### #ikeda-2018-abstract-x-ray-spe-response (p.1, abstract)

> "We examined the Schumann resonance (SR) at low-latitude station KUJ by comparing with solar X-ray flux and solar proton flux at a geostationary orbit. For intense solar activity in October-November 2003, the reaction of the SR frequency to X-ray enhancement and SPEs was different. The SR frequency in H component increased at the time of the X-ray enhancement. The response of SR seems to be caused by the increase of the electron density in the ionospheric D region which ionized by the enhanced solar X-ray flux. In the case of the SPEs, the SR frequency in D component decreased with enhancement of solar proton flux."

**Target metric:** direction-of-response asymmetry — X-ray events increase `f_1` in H, SPEs decrease `f_1` in D. Grounds the sign of any solar-activity term in a cert mapping.

### #ikeda-2018-methodology-goes10-50hz-fft (p.2, §2 Data Set)

> "We examined the fundamental mode of SR observed by an induction magnetometer at Kuju, Japan (M.Lat. = 23.4 degree, M. Lon. = 201.0 degree). The observation is a part of activities by International Center for Space Weather Science and Education, Kyushu University and the observation of the induction magnetometer started in 2003. The components of ground magnetic field used in this study are horizontal northward component (H) and horizontal eastward component (D). Sampling rate of the observation data is 50 Hz. We identified frequency of the SR by calculating PSD (power spectral density) every 10 seconds segment by using FFT (First Fourier Transformation). Generally, frequency of the fundamental mode of SR is about 7.8 Hz [5], we therefore found a peak frequency ranging 6.0 Hz to 9.0 Hz. This peak frequency is denoted by SR frequency. To compare with SR frequency, solar X-ray (0.05-0.3 nm) and proton flux (40-80 MeV) were also analyzed. These data were obtained by the GOES-10 satellite on a geostationary orbit."

**Target metric:** measurement protocol — 50 Hz sampling, 10-s PSD windows, FFT, 6.0–9.0 Hz `f_1` peak-search range, GOES-10 X-ray/proton reference. Satisfies the `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_ACQUISITION.md` §C "measurement protocol (sampling rate for HRV papers, measurement apparatus for Schumann, statistical test for Radin)" requirement for the Schumann track.

### #ikeda-2018-19-day-xray-enhancement-event (p.2, §3 Results)

> "The enhancement of solar X-ray flux occurred on 18 October, 2003 and lasted for about 19 days before it recovered to a previous level. The SR frequency in H component followed the variation of the X-ray flux (see black arrows in Figure 1). The effect on SR frequency is not related to the local time at the observatory, since a high degree of the SR frequency in H component continued during the enhancement of solar X-ray flux. The effect on SR seems to be the global character."

**Target metric:** event identification (18 Oct – 6 Nov 2003, 19 days) + the global-character claim (low-latitude Kuju response correlates with global SR intensity, not local ionospheric state). This pairs with Roldugin 1999 high-latitude Lovozero observations for a two-station consistency check.

---

## Phase 4.8 item 6 — Williams-substitute acquisition follow-up

1. **Williams 1992 Science 256:1184** remains DEFERRED. AAAS-paywalled across Semantic Scholar / Unpaywall / NASA ADS as of 2026-04-22. Not a blocker for the mapping cert because Price 2016 carries the canonical `Q = 4–6` scalar, Bozóki 2021 carries the long-term solar-cycle `ΔQ/Q = 0.12–0.15` and `ΔI/I ≤ 0.60` primary data, and Ikeda 2018 carries a direct 2003 single-event SR-frequency response at a CC-BY venue.
2. **Roldugin 2004 JGR (doi:10.1029/2003JA010019)** — Wiley-hosted BRONZE OA but Cloudflare-blocks unauthenticated curl; retrievable via Playwright MCP or institutional access session. Unpaywall confirms `is_oa=True`, `oa_status=bronze`, `url_for_pdf=https://onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2003JA010019`. Not acquired in this session because Ikeda 2018 cites Roldugin 1999 directly and carries the equivalent single-event structure, but future session may add for quantitative triangulation.
3. **Nickolaenko, Koloskov, Hayakawa, Yampolski, Budanov, Korepanov 2015** *Sun Geosph.* 10:39–49 (11-year solar cycle at Vernadsky Antarctic) — open but not acquired in this session; cited inside Bozóki 2021 quote at `#bozoki-2021-vernadsky-sr-intensity-60-percent`, so the numerical claim flows through. Future session may add if the cert requires a direct-source anchor.
4. **Harmonic-family declaration** remains the load-bearing decision for the eventual `qa_heartmath_mapping_cert_v1` session (per OB capture 2026-04-21T03:40Z and the 2026-04-20 ChatGPT-pushback exchange). Price 2016 integer-rounded 8/14/20/26 Hz series, Bozóki 2021 `√(n(n+1))`-compatible modelling, and Ikeda 2018 `f_1 ≈ 7.8 Hz` all remain on the observed side — declare observed-canonical unless new primary data justifies theoretical-canonical.
5. **Cert authoring is out-of-scope for this session.** Scope-bound acquisition discipline mirrors the 2026-04-20 predecessor session (commit `3bde4e4`).

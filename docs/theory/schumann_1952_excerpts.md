<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.8 item 6 primary-source excerpt snapshot. Verbatim quotes from Schumann 1952 Z. Naturforschg. 7a:149-154. Page locators are PDF page (/journal page). Text is in German; each anchor carries the verbatim German quote followed by an English gloss marked EN:. Do not edit the German quote fields — they are the primary record. Authored 2026-04-20 under Phase 4.8 item 6 acquisition. -->

# Schumann 1952 — Primary-Source Excerpts

**Corpus root:** `Documents/schumann_resonance/`
**Scope:** QA-MEM Phase 4.8 item 6 Track C (Schumann resonance) — derivation of the closed-form Earth-ionosphere cavity eigenfrequency and harmonic series. Corresponds to target #10 in `docs/specs/QA_MEM_PHASE_4_8_ITEM_6_ACQUISITION.md`.
**Domain:** `physics` (QA-MEM taxonomy).

**Publication:** W. O. Schumann, *Über die strahlungslosen Eigenschwingungen einer leitenden Kugel, die von einer Luftschicht und einer Ionosphärenhülle umgeben ist*, Zeitschrift für Naturforschung 7a:149–154 (1952). Received 1 October 1951. DOI: 10.1515/zna-1952-0202.
**Author:** Winfried Otto Schumann, Elektrophysikalisches Institut der Technischen Hochschule München.
**On-disk:** `Documents/schumann_resonance/schumann_1952_zfn_7a_149.pdf` (5.1 MB, 6 PDF pages / 6 journal pages; retrieved 2026-04-20 via `degruyterbrill.com/document/doi/10.1515/zna-1952-0202/pdf`).

**QA-research grounding:** this is the canonical original derivation for the 7.83 Hz fundamental invoked throughout the HeartMath / McCraty / Oschman literature on heart-Earth resonance coupling. The theoretical harmonic series `m_n = √(n(n+1))` derived here (see `#schumann-1952-harmonic-series-sqrt-nnp1`) is the integer-rational substrate the Phase 4.8 item 6 spec flags as the target of a future `qa_heartmath_mapping_cert_v1` numerical claim (Schumann harmonics ↔ mod-9/mod-24 orbit-harmonic ratios). The cert cannot be authored in this session — spec is explicit that acquisition is boundary-only.

**Firewall note:** Schumann's eigenfrequency solutions are continuous-valued and cannot enter the QA discrete layer as causal inputs (Theorem NT — Observer Projection Firewall). They enter only through the discrete harmonic index `n ∈ {1, 2, 3, …}` and the closed-form algebraic relation `ω_n / ω_1 = √(n(n+1)/2)`. Any future mapping cert must preserve that boundary.

---

## Schumann (1952) — Cavity eigenfrequency of the Earth-ionosphere system

### #schumann-1952-title-and-author (PDF p.1 / journal p.149, title block)

> "Über die strahlungslosen Eigenschwingungen einer leitenden Kugel, die von einer Luftschicht und einer Ionosphärenhülle umgeben ist. Von W. 0. SCHUMANN. AVIS dem Elektrophysikalischen Institut der Technischen Hochschule München. (Z. Naturforschg. 7a, 149-154 [1952]; eingegangen am 1. Oktober 1951)"

**EN:** "On the non-radiating eigenoscillations of a conducting sphere surrounded by an air layer and an ionospheric shell. By W. O. Schumann, Electrophysical Institute of the Technische Hochschule Munich. (Z. Naturforschg. 7a, 149–154 [1952]; received 1 October 1951.)"

### #schumann-1952-abstract-eleven-cps (PDF p.1 / journal p.149, abstract)

> "Die tiefste Eigenfrequenz ungedämpfter Schwingungen ωei liegt zwischen Null und ωei = √2 c/r je nach der relativen Dicke der Luftschicht und der Plasmaeigenfrequenz ω₀. Für irdische Dimensionen liegt ωei nahe dem oberen Wert und ergibt eine Frequenz von ≈ 11 Per/sec."

**EN:** "The lowest eigenfrequency of undamped oscillations ωei lies between zero and ωei = √2 c/r depending on the relative thickness of the air layer and the plasma eigenfrequency ω₀. For terrestrial dimensions ωei lies near the upper value and yields a frequency of approximately 11 cycles per second."

### #schumann-1952-earth-numerical-calc (PDF p.4 / journal p.152, numerical evaluation)

> "Setzen wir für die Erde R = 6000 km und c = 300 000 km/sec, so ist ωei = 50 √2 ≈ 70 sec⁻¹ und die sekundliche Frequenz f_ei max ≈ 11 sec⁻¹. Durch eine stoßartige Erregung dieses 'Hohlraumes', z. B. durch eine Blitzentladung, sind als tiefste erregte Schwingungen Frequenzen dieser Größe zu erwarten."

**EN:** "Setting for the Earth R = 6000 km and c = 300 000 km/sec, one has ωei = 50 √2 ≈ 70 sec⁻¹ and the ordinary-units frequency f_ei_max ≈ 11 sec⁻¹. Under impulsive excitation of this 'cavity', e.g. by a lightning discharge, the lowest excited oscillations should be at frequencies of this magnitude."

### #schumann-1952-harmonic-series-sqrt-nnp1 (PDF p.5 / journal p.153, overtone series)

> "Für ma' → ∞ ist m₁ = √(n(n+1)) = ωei R/c. Es sind dies die zonalen Oberschwingungen geringer Ordnung für 'sehr große Plasmadichte' R/c · ω₀ » 1, aber auch zugleich die Oberschwingungen einer von zwei unendlich guten Leitern begrenzten Kugelschale aus Luft."

**EN:** "For ma' → ∞ one has m₁ = √(n(n+1)) = ωei R/c. These are the low-order zonal overtones for 'very high plasma density' R/c · ω₀ » 1, and simultaneously the overtones of a spherical air shell bounded by two infinitely conducting surfaces."

### #schumann-1952-vacuum-wavelength-27300km (PDF p.5 / journal p.153, wavelength note)

> "Der berechneten Frequenz von f_ei = 11 sec⁻¹ entspricht eine Vakuumwellenlänge freier Wellen von λ_ei = 27 300 km."

**EN:** "The computed frequency of f_ei = 11 sec⁻¹ corresponds to a free-space vacuum wavelength of λ_ei = 27 300 km."

### #schumann-1952-phase-velocity-sqrt2c (PDF p.5 / journal p.153, phase velocity)

> "Die Winkelgeschwindigkeit dieser Wellen längs des Kugelumfangs ist dθ/dt = ω und die Lineargeschwindigkeit v = R dθ/dt = ω R. Für ωei = √2 c/R wird v = √2 c."

**EN:** "The angular velocity of these waves along the sphere's circumference is dθ/dt = ω and the linear velocity is v = R dθ/dt = ωR. For ωei = √2 c/R one gets v = √2 c."

---

## Phase 4.8 item 6 follow-up actions

1. **Williams 1992 (target #11)** — AAAS-paywalled across Semantic Scholar, Unpaywall, NASA ADS; author's MIT Lincoln Lab and EAPS pages unreachable from this session. Captured as DEFERRED row in `tools/qa_kg/CORPUS_INDEX.md` pending a browser-context fetch or institutional-access session. The Schumann 1952 closed-form √(n(n+1)) harmonic law is the primary structural substrate for any future mapping cert, so Williams 1992 is a cross-check rather than a blocker.
2. **Numerical-mapping cert authoring is out-of-scope for this session.** The cert authoring session must operate on at least this Schumann excerpt file + McCraty numerical tables + an explicit decision about which harmonic series (Schumann-theoretical √(n(n+1)) with f₁ ≈ 11 Hz vs NOAA-observed with f₁ ≈ 7.83 Hz and integer-approximating harmonics 14.3/20.8/27.3/33.8 Hz) is canonical for the QA mapping.
3. **Harmonic-ratio pre-computation** (optional, for cert-authoring session): Schumann's theoretical ratios ω_n/ω_1 = √(n(n+1)/2) give {√1, √3, √6, √10, √15, …} ≈ {1.000, 1.732, 2.449, 3.162, 3.873, …}. NOAA-observed ratios f_n/f_1 with f₁ = 7.83 Hz: 14.3/7.83 ≈ 1.827, 20.8/7.83 ≈ 2.657, 27.3/7.83 ≈ 3.487, 33.8/7.83 ≈ 4.317. The observed ratios track √(n+1/2)·√n rather than the lossless √(n(n+1)) — the ≈5-15 % offset is the Earth-cavity damping correction flagged in §II of this paper. The mapping cert must declare which ratio family it certifies against.

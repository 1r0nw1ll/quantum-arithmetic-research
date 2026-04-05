# Engineering Domains Quick Map

For practitioners from specific engineering backgrounds. Find your field, see the QA translation immediately.

---

## Mechanical Engineering

**Classical**: ODEs governing motion (Newton's second law), springs, dampers, masses
**QA translation**:
- Displacement/velocity state → `(b, e)` in Caps(N, N)
- Oscillation modes → orbit families (singularity=still, satellite=transient, cosmos=steady oscillation)
- Natural frequency → orbit period (24 for cosmos in mod-24)
- Damping → orbit contraction factor ρ(O) < 1 (Finite-Orbit Descent Theorem)
- Resonance → orbit family = cosmos

**First cert to build**: Map a spring-mass system. State: (displacement_class, velocity_class). Generator: σ increments velocity. Orbit: cosmos = sustained oscillation. See `QA_ENGINEERING_CORE_CERT_SPEC.md` fixture 1.

---

## Electrical Engineering / Circuits

**Classical**: Kirchhoff's laws, impedance, RLC circuits, signal processing
**QA translation**:
- Voltage/current state → `(b, e)` pair
- Impedance resonance (LC at ω₀ = 1/√LC) → cosmos orbit
- Transient response → satellite orbit traversal (k=1)
- Steady-state → cosmos orbit (periodic, stable)
- Low-pass filter → ν generator (halves coordinates = halves frequency content)
- Amplifier → λ generator (scales by k)
- Fourier frequency bins → mod-24 orbit families

**Key insight**: The Fourier series `a = m + 2n` (for mode numbers m, n) is identical to the QA tuple formula `a = b + 2e`. Frequency analysis IS QA orbit analysis.

---

## Signal Processing / Audio

**Classical**: FFT, filtering, convolution, power spectral density
**QA translation**:
- Signal amplitude → b coordinate
- Signal frequency → e coordinate (in quantized bins)
- Filter application → generator application (ν for lowpass, λ for gain)
- Harmonic series → Fibonacci cosmos family (norm {1,8})
- Fundamental + octave → F and lambda_2(F) orbit members
- White noise → uniform distribution over Caps(N,N) — no orbit preference

**Applied in this repo**:
- `run_signal_experiments_final.py` tests QA on pure tones, major/minor chords, tritones, white noise
- Harmonic Index HI = E8_alignment × exp(-0.1 × loss) as signal quality metric
- Signal injection: external audio injects into `b` state variable

---

## Aerospace / Control Systems

**Classical**: State-space (ẋ = Ax + Bu), LQR, PID, Kalman filter
**QA translation**:
- State vector → (b, e, d, a) tuple
- Dynamics matrix A → QA generator set {σ, μ, λ, ν}
- Control input u → generator selection
- Kalman rank condition → BFS reachability (with arithmetic obstruction override)
- LQR optimal gain K → BFS planner with minimality_witness
- Kalman filter estimate → orbit family classification from partial observations

**Critical difference from classical**: The Kalman rank condition is necessary but not sufficient. QA adds EC11: arithmetic obstructions (v_p(r) = 1 for inert prime) make some targets unreachable regardless of rank. This catches failure modes invisible to classical analysis.

---

## Computer Engineering / Algorithms

**Classical**: Graphs, BFS/DFS, complexity, automata theory
**QA translation** (most natural — QA is already discrete):
- State machine → Caps(N,N) with generators
- BFS → QA planner cert (already using BFS with minimality_witness)
- Regular language → generator word over {σ, μ, λ, ν}*
- Automaton states → orbit families
- Halting condition → reaching target orbit family
- Time complexity → path_length_k
- Space complexity → |Caps(N,N)| = N²

**QA as formal verification**: The cert ecosystem is essentially a formal verification system. Every claim is machine-checkable. This is familiar territory for computer engineers.

---

## Civil / Structural Engineering

**Classical**: Structural loads, resonance frequencies, modal analysis
**QA translation**:
- Structural modes → orbit families (mode 1 = singularity, transitional modes = satellite, full resonance = cosmos)
- Load path → generator path (sequence of stress transitions)
- Resonance disaster (Tacoma Narrows) → reaching cosmos orbit without control = unbounded cosmos traversal
- Damping → orbit contraction ρ(O) < 1

**Seismic connection**: cert family [110] QA_SEISMIC_CONTROL already maps seismic wave propagation (quiet → p_wave → surface_wave) to orbit traversal. This is directly applicable to structural seismic analysis.

---

## Acoustic / Cymatics (SVP-Native)

This is the most natural mapping — fully certified in the repo.

**Classical**: Chladni figures, Faraday waves, standing wave patterns
**QA translation**: *exact* (not approximate)
- Chladni formula `a = m + 2n` = QA tuple derivation `a = b + 2e`
- Mode (m, n) → state (b, e)
- No pattern → singularity orbit
- Stripe pattern → satellite orbit
- Hexagonal pattern → cosmos orbit
- Frequency increase → σ generator
- Amplitude tuning → λ generator

**Certified in**: cert family [105] QA_CYMATIC_CONTROL (4 tiers: mode, reachability, control, planner)
**Run it**: `python qa_alphageometry_ptolemy/qa_cymatics/qa_cymatics_validate.py --demo`

---

## Biology / Morphogenesis

**Classical**: Reaction-diffusion (Turing patterns), Fibonacci phyllotaxis, growth sequences
**QA translation**:
- Turing spot patterns → cosmos orbit (Fibonacci family, norm {1,8})
- Stripe patterns → satellite orbit (transitional)
- No pattern → singularity
- Fibonacci phyllotaxis (137.5° golden angle) → Z[φ] structure directly
- Cell division → generator application on (b, e) pair
- Growth law → T = F² (QA Fibonacci matrix) acts as ×φ² scaling in Z[φ]

**Natural connection**: Fibonacci sequences in biology ARE QA orbit traversal in the Fibonacci cosmos family. This is not metaphor — the number field Q(√5) underlying QA is the same field containing φ.

---

## Finance / Market Systems

**Classical**: Stochastic processes, Black-Scholes, regime-switching models
**QA translation**:
- Market regime (bull/bear/neutral) → orbit family
- Price return → generator application
- Volatility regime → path length in cosmos orbit
- Market crash → reaching satellite or singularity from cosmos

**Applied in this repo**:
- `backtest_advanced_strategy.py` applies Harmonic Index to S&P 500 regime detection
- S&P 500 data from yfinance, combined with 200-day SMA

---

## Neuroscience / EEG

**Classical**: Frequency bands (delta/theta/alpha/beta/gamma), oscillation patterns
**QA translation**:
- Brain quiescence → singularity
- Transitional oscillation → satellite
- Full gamma synchrony / seizure onset → cosmos (or specific cosmos sub-family)
- Seizure = unwanted cosmos traversal
- Treatment = steer back to satellite/singularity

**Applied in this repo**:
- `eeg_hi2_0_experiment.py` tests QA features on CHB-MIT EEG seizure dataset
- `compare_seizure_vs_baseline.py` compares orbit families across brain states

---

## Your Domain

Use this checklist to map any new domain:

1. What are the **distinct states** my system passes through? → These are your Caps(N,N) elements
2. What are the **lawful transitions** between them? → These are your generators
3. What are the **failure modes** (illegal transitions)? → These map to OUT_OF_BOUNDS, PARITY, etc.
4. What is **quiescence** (no activity)? → This is singularity
5. What is **full activation** (target behavior)? → This is cosmos
6. What is the **transitional phase**? → This is satellite
7. Does your system progress through a **3-phase sequence** (quiescence → transitional → active)? → Path length k=2

If yes to #7: the Control Stack Theorem guarantees your system follows `singularity → satellite → cosmos` with k=2, same as cymatics and seismology.

See `03_applied_domains/CROSS_DOMAIN_PRINCIPLE.md` for the formal argument.


QA_COMPLIANCE = "empirical_observer — audio/signal as observer input; QA coupling is discrete state"

import numpy as np
import matplotlib.pyplot as plt
from qa_core import QASystem, build_open_brain_capture, ensure_run_dir, write_json

# --- 1. CONFIGURATION PARAMETERS (FINAL & STABLE) ---
NUM_NODES = 16
MODULUS = 24
COUPLING_STRENGTH = 0.2
NOISE_BASE = 0.2
NOISE_ANNEALING = 0.995
TIMESTEPS = 150
SIGNAL_INJECTION_STRENGTH = 0.2

HYPOTHESIS = (
    "Tonal inputs should end with higher harmonic coherence than white noise in the final QA signal system."
)
SUCCESS_CRITERIA = [
    "Major Chord final HI > White Noise final HI",
    "Minor Chord final HI > White Noise final HI",
    "At least three tonal inputs end with HI above White Noise",
]

# Signal Generation Parameters
SAMPLE_RATE = 44100
DURATION = 2.0
BASE_FREQ = 261.63

# --- 2. SIGNAL GENERATION FUNCTIONS (Unchanged) ---
def generate_signal(timesteps):
    t = np.linspace(0, DURATION, timesteps, endpoint=False)
    return t

def generate_pure_tone(timesteps):
    t = generate_signal(timesteps)
    return np.sin(2 * np.pi * BASE_FREQ * t)

def generate_major_chord(timesteps):
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * 1.0 * t); note2 = np.sin(2 * np.pi * BASE_FREQ * (5/4) * t); note3 = np.sin(2 * np.pi * BASE_FREQ * (6/4) * t)
    return (note1 + note2 + note3) / 3.0

def generate_minor_chord(timesteps):
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * 1.0 * t); note2 = np.sin(2 * np.pi * BASE_FREQ * (12/10) * t); note3 = np.sin(2 * np.pi * BASE_FREQ * (15/10) * t)
    return (note1 + note2 + note3) / 3.0

def generate_tritone(timesteps):
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * 1.0 * t); note2 = np.sin(2 * np.pi * BASE_FREQ * np.sqrt(2) * t)
    return (note1 + note2) / 2.0

def generate_white_noise(timesteps):
    return np.random.uniform(-1, 1, size=timesteps)


def _history_to_jsonable(history):
    return {key: [float(value) for value in values] for key, values in history.items()}

# --- 3. QA-CPLEARN SYSTEM IMPLEMENTATION (FINAL CORRECTED LOGIC) ---
# Moved to qa_core.QASystem with signal_mode="final".

# --- 4. MAIN EXPERIMENT EXECUTION ---
if __name__ == "__main__":
    # Seed randomness for reproducibility
    np.random.seed(42)
    signals_to_test = {'Pure Tone': generate_pure_tone, 'Major Chord': generate_major_chord, 'Minor Chord': generate_minor_chord, 'Tritone': generate_tritone, 'White Noise': generate_white_noise}
    results = {}; full_history = {}
    run_dir = ensure_run_dir("signal", "run_signal_experiments_final")

    prereg_capture = build_open_brain_capture(
        "task",
        ["signal", "run_signal_experiments_final", "pre-registration"],
        (
            "Pre-registered signal coherence run. "
            f"Hypothesis: {HYPOTHESIS} "
            f"Success criteria: {'; '.join(SUCCESS_CRITERIA)}"
        ),
        metadata={
            "config": {
                "num_nodes": NUM_NODES,
                "modulus": MODULUS,
                "coupling_strength": COUPLING_STRENGTH,
                "noise_base": NOISE_BASE,
                "noise_annealing": NOISE_ANNEALING,
                "timesteps": TIMESTEPS,
                "signal_injection_strength": SIGNAL_INJECTION_STRENGTH,
            },
            "signals": list(signals_to_test.keys()),
        },
    )
    write_json(run_dir / "open_brain_task_capture.json", prereg_capture)

    print("--- Starting Phase 3: Real-World Signal Classification (FINAL CORRECTED LOGIC) ---")
    for name, generator_func in signals_to_test.items():
        print(f"\n[INFO] Testing signal: {name}")
        signal_data = generator_func(TIMESTEPS)
        system = QASystem(
            num_nodes=NUM_NODES,
            modulus=MODULUS,
            coupling=COUPLING_STRENGTH,
            noise_base=NOISE_BASE,
            noise_annealing=NOISE_ANNEALING,
            signal_injection_strength=SIGNAL_INJECTION_STRENGTH,
            signal_mode="final",
        )
        system.run_simulation(TIMESTEPS, signal_data)
        
        final_hi = system.history['hi'][-1]
        final_align = system.history['e8_alignment'][-1]
        final_loss = system.history['loss'][-1]
        converged_at = next((i for i, h in enumerate(system.history['hi']) if h > 0.3 and system.history['loss'][i] < 1.0), TIMESTEPS)
        results[name] = {
            'Harmonic Index (HI)': float(final_hi),
            'E8 Alignment': float(final_align),
            'Harmonic Loss': float(final_loss),
            'Convergence Time': int(converged_at) if converged_at < TIMESTEPS else None,
        }
        full_history[name] = system.history

    white_noise_hi = results['White Noise']['Harmonic Index (HI)']
    tonal_names = ['Pure Tone', 'Major Chord', 'Minor Chord', 'Tritone']
    tonal_above_noise = [name for name in tonal_names if results[name]['Harmonic Index (HI)'] > white_noise_hi]
    criteria_results = {
        'major_above_noise': results['Major Chord']['Harmonic Index (HI)'] > white_noise_hi,
        'minor_above_noise': results['Minor Chord']['Harmonic Index (HI)'] > white_noise_hi,
        'three_tonal_above_noise': len(tonal_above_noise) >= 3,
    }
    passed = sum(1 for ok in criteria_results.values() if ok)
    verdict = 'PASS' if passed == len(criteria_results) else ('PARTIAL' if passed > 0 else 'FAIL')

    summary_payload = {
        'script': 'run_signal_experiments_final.py',
        'domain': 'signal',
        'hypothesis': HYPOTHESIS,
        'success_criteria': SUCCESS_CRITERIA,
        'criteria_results': criteria_results,
        'verdict': verdict,
        'results': results,
        'ranked_by_hi': [
            {
                'signal': name,
                'final_hi': results[name]['Harmonic Index (HI)'],
            }
            for name in sorted(results.keys(), key=lambda item: results[item]['Harmonic Index (HI)'], reverse=True)
        ],
    }
    write_json(run_dir / 'summary.json', summary_payload)
    write_json(
        run_dir / 'histories.json',
        {name: _history_to_jsonable(history) for name, history in full_history.items()},
    )

    observation_capture = build_open_brain_capture(
        'observation',
        ['signal', 'run_signal_experiments_final', verdict.lower()],
        (
            "Signal coherence run completed. "
            f"Verdict: {verdict}. "
            f"White Noise final HI={white_noise_hi:.4f}. "
            f"Major Chord final HI={results['Major Chord']['Harmonic Index (HI)']:.4f}. "
            f"Minor Chord final HI={results['Minor Chord']['Harmonic Index (HI)']:.4f}. "
            f"Tonal signals above White Noise by HI: {', '.join(tonal_above_noise) if tonal_above_noise else 'none'}."
        ),
        metadata=summary_payload,
    )
    write_json(run_dir / 'open_brain_observation_capture.json', observation_capture)

# --- 5. REPORTING & VISUALIZATION ---
    print("\n--- Final Results Summary (Corrected Logic) ---")
    print("-" * 85); print(f"{'Signal Type':<20} | {'Harmonic Index (HI)':<22} | {'E8 Alignment':<15} | {'Convergence Time'}"); print("-" * 85)
    for name, data in results.items():
        conv_time_str = f"~{data['Convergence Time']} steps" if isinstance(data['Convergence Time'], int) else data['Convergence Time']
        print(f"{name:<20} | {data['Harmonic Index (HI)']:<22.4f} | {data['E8 Alignment']:<15.4f} | {conv_time_str}")
    print("-" * 85)

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True); fig.suptitle('QA-CPLearn: Signal Classification (Corrected Logic)', fontsize=16)
    ax = axes[0]; names = list(results.keys()); hi_scores = [r['Harmonic Index (HI)'] for r in results.values()]
    colors = plt.cm.viridis(np.array(hi_scores) / (np.max(hi_scores) + 1e-9)); ax.bar(names, hi_scores, color=colors)
    ax.set_title('Final Harmonic Index (HI) by Signal Type'); ax.set_ylabel('Harmonic Index'); ax.set_ylim(0, 1); ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, name in enumerate(['Major Chord', 'White Noise']):
        ax = axes[i+1]; history = full_history[name]
        ax.plot(history['e8_alignment'], 'b-', label='E8 Alignment'); ax.plot(history['hi'], 'g-', label='Harmonic Index (HI)', linewidth=2)
        ax.set_title(f'Evolution for: {name}'); ax.set_xlabel('Timestep'); ax.set_ylabel('Score')
        ax.legend(loc='upper left'); ax.grid(True, linestyle='--', alpha=0.6); ax.set_ylim(0, 1)
        ax2 = ax.twinx(); ax2.plot(history['loss'], 'r:', label='Harmonic Loss'); ax2.set_ylabel('Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red'); ax2.legend(loc='upper right')
    plot_path = run_dir / 'signal_classification_results.png'
    plt.savefig(plot_path, dpi=200)
    print(f"Artifacts saved under: {run_dir}")
    print(f"Pre-registration capture: {run_dir / 'open_brain_task_capture.json'}")
    print(f"Observation capture: {run_dir / 'open_brain_observation_capture.json'}")
    print(f"Plot: {plot_path}")
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()

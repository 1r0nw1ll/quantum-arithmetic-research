import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qa_core import QASystem

# --- 1. CONFIGURATION PARAMETERS (FINAL & STABLE) ---
NUM_NODES = 16
MODULUS = 24
COUPLING_STRENGTH = 0.2
NOISE_BASE = 0.2
NOISE_ANNEALING = 0.995
TIMESTEPS = 150
SIGNAL_INJECTION_STRENGTH = 0.2

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

# --- 3. QA-CPLEARN SYSTEM IMPLEMENTATION (FINAL CORRECTED LOGIC) ---
# Moved to qa_core.QASystem with signal_mode="final".

# --- 4. MAIN EXPERIMENT EXECUTION ---
if __name__ == "__main__":
    # Seed randomness for reproducibility
    np.random.seed(42)
    signals_to_test = {'Pure Tone': generate_pure_tone, 'Major Chord': generate_major_chord, 'Minor Chord': generate_minor_chord, 'Tritone': generate_tritone, 'White Noise': generate_white_noise}
    results = {}; full_history = {}

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
        results[name] = {'Harmonic Index (HI)': final_hi, 'E8 Alignment': final_align, 'Harmonic Loss': final_loss, 'Convergence Time': converged_at if converged_at < TIMESTEPS else 'N/A'}
        full_history[name] = system.history

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
    plt.show()

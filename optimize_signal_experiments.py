import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from run_signal_experiments_final import QASystem, generate_pure_tone, generate_major_chord, generate_minor_chord, generate_tritone, generate_white_noise

# --- 1. CONFIGURATION PARAMETERS (from run_signal_experiments_final.py) ---
# These will be overridden by the optimization loop
NUM_NODES = 24
MODULUS = 24
COUPLING_STRENGTH = 0.1
NOISE_BASE = 0.4
NOISE_ANNEALING = 0.995
TIMESTEPS = 150
SIGNAL_INJECTION_STRENGTH = 0.2

# --- 2. HYPERPARAMETER OPTIMIZATION SETUP ---
# Define the search space for hyperparameters
param_space = {
    'NUM_NODES': [16, 24, 32],
    'MODULUS': [16, 24, 32],
    'COUPLING_STRENGTH': [0.05, 0.1, 0.2],
    'NOISE_BASE': [0.2, 0.4, 0.6],
    'NOISE_ANNEALING': [0.99, 0.995, 0.999],
    'SIGNAL_INJECTION_STRENGTH': [0.1, 0.2, 0.3]
}

# Signals to test
signals_to_test = {
    'Pure Tone': generate_pure_tone,
    'Major Chord': generate_major_chord,
    'Minor Chord': generate_minor_chord,
    'Tritone': generate_tritone,
    'White Noise': generate_white_noise
}

best_hi_score = -1
best_params = {}
all_results = []

print("--- Starting Hyperparameter Optimization for QASystem ---")

# Simple Grid Search
keys = list(param_space.keys())
for p_values in tqdm(itertools.product(*param_space.values()), total=np.prod([len(v) for v in param_space.values()]), desc="Optimizing"):
    current_params = dict(zip(keys, p_values))

    # Update QASystem parameters
    NUM_NODES = current_params['NUM_NODES']
    MODULUS = current_params['MODULUS']
    COUPLING_STRENGTH = current_params['COUPLING_STRENGTH']
    NOISE_BASE = current_params['NOISE_BASE']
    NOISE_ANNEALING = current_params['NOISE_ANNEALING']
    SIGNAL_INJECTION_STRENGTH = current_params['SIGNAL_INJECTION_STRENGTH']

    # Run simulations for each signal type
    hi_scores_for_params = []
    for name, generator_func in signals_to_test.items():
        signal_data = generator_func(TIMESTEPS)
        system = QASystem(NUM_NODES, MODULUS, COUPLING_STRENGTH, NOISE_BASE, NOISE_ANNEALING, SIGNAL_INJECTION_STRENGTH)
        system.run_simulation(TIMESTEPS, signal_data)
        hi_scores_for_params.append(system.history['hi'][-1])
    
    avg_hi = np.mean(hi_scores_for_params)
    all_results.append({'params': current_params, 'avg_hi': avg_hi})

    if avg_hi > best_hi_score:
        best_hi_score = avg_hi
        best_params = current_params

print("\n--- Optimization Complete ---")
print(f"Best Average Harmonic Index: {best_hi_score:.4f}")
print("Best Parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Optional: Save results or plot
# results_df = pd.DataFrame(all_results)
# results_df.to_csv('hyperparameter_optimization_results.csv', index=False)

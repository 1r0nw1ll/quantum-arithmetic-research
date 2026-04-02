#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
Signal Classification with PAC-Bayesian Bounds + Pisano Period Analysis

Enhancements:
1. INFORMED PRIOR from initial QA state
2. LARGER DATASET (1000 timesteps)
3. OPTIMAL TRANSPORT for exact D_QA
4. PISANO PERIOD CLASSIFICATION for hypothesis complexity
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
from qa_core import QASystem
from qa_pac_bayes import (
    dqa_divergence,
    compute_pac_constants,
    pac_generalization_bound
)
from pisano_analysis import PisanoClassifier, add_pisano_analysis_to_results

# Configuration
NUM_NODES = 16
MODULUS = 24
TIMESTEPS = 1000
COUPLING_STRENGTH = 0.2
NOISE_BASE = 0.2
NOISE_ANNEALING = 0.998
SIGNAL_INJECTION_STRENGTH = 0.2

SAMPLE_RATE = 44100
DURATION = 2.0
BASE_FREQ = 261.63

N_SAMPLES_FOR_DQA = 200
CONFIDENCE_DELTA = 0.05

# Signal generation functions (same as before)
def generate_signal(timesteps):
    return np.linspace(0, DURATION, timesteps, endpoint=False)

def generate_pure_tone(timesteps):
    t = generate_signal(timesteps)
    return np.sin(2 * np.pi * BASE_FREQ * t)

def generate_major_chord(timesteps):
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * 1.0 * t)
    note2 = np.sin(2 * np.pi * BASE_FREQ * (5/4) * t)
    note3 = np.sin(2 * np.pi * BASE_FREQ * (6/4) * t)
    return (note1 + note2 + note3) / 3.0

def generate_minor_chord(timesteps):
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * 1.0 * t)
    note2 = np.sin(2 * np.pi * BASE_FREQ * (12/10) * t)
    note3 = np.sin(2 * np.pi * BASE_FREQ * (15/10) * t)
    return (note1 + note2 + note3) / 3.0

def generate_tritone(timesteps):
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * 1.0 * t)
    note2 = np.sin(2 * np.pi * BASE_FREQ * np.sqrt(2) * t)
    return (note1 + note2) / 2.0

def generate_white_noise(timesteps):
    return np.random.uniform(-1, 1, size=timesteps)

def create_informed_prior(initial_system, n_samples=200):
    """Create informed prior from initial QA state"""
    mean_b = np.mean(initial_system.b)
    mean_e = np.mean(initial_system.e)
    std_b = np.std(initial_system.b)
    std_e = np.std(initial_system.e)
    
    samples = []
    for _ in range(n_samples):
        b_sample = (np.random.randn() * std_b + mean_b) % initial_system.modulus
        e_sample = (np.random.randn() * std_e + mean_e) % initial_system.modulus
        samples.append([b_sample, e_sample])
    
    return np.array(samples)

def extract_learned_distribution(system, n_samples):
    """Extract learned QA distribution samples"""
    indices = np.random.choice(len(system.b), size=n_samples, replace=True)
    return np.column_stack([system.b[indices], system.e[indices]])

def compute_classification_risk(results, name):
    """Compute 0-1 loss for harmonic classification"""
    hi = results[name]['Harmonic Index (HI)']
    is_harmonic = (hi > 0.5)
    true_harmonic = (name in ['Pure Tone', 'Major Chord', 'Minor Chord'])
    return 0.0 if (is_harmonic == true_harmonic) else 1.0

if __name__ == "__main__":
    print("="*80)
    print("SIGNAL CLASSIFICATION WITH PAC BOUNDS + PISANO ANALYSIS")
    print("="*80)
    print(f"  Nodes: {NUM_NODES}")
    print(f"  Modulus: {MODULUS}")
    print(f"  Timesteps: {TIMESTEPS}")
    print(f"  Prior: INFORMED")
    print(f"  D_QA: OPTIMAL TRANSPORT")
    print(f"  Pisano: MOD-9 PERIOD CLASSIFICATION")
    print("="*80)
    print()

    workspace = Path("phase1_workspace")
    workspace.mkdir(exist_ok=True)

    # Compute PAC constants
    pac_constants = compute_pac_constants(N=NUM_NODES, modulus=MODULUS, lipschitz_C=1.0)
    
    # Initialize Pisano classifier
    pisano_classifier = PisanoClassifier(modulus=9)
    print(f"✓ Pisano classifier initialized (mod-9)")
    print(f"✓ PAC constants: K₁={pac_constants.K1:.1f}, K₂={pac_constants.K2:.3f}")
    print()

    np.random.seed(42)

    signals_to_test = {
        'Pure Tone': generate_pure_tone,
        'Major Chord': generate_major_chord,
        'Minor Chord': generate_minor_chord,
        'Tritone': generate_tritone,
        'White Noise': generate_white_noise
    }

    results = {}
    pac_results = {}
    pisano_results = {}
    full_history = {}

    print("Running experiments...")
    print("-" * 80)

    for name, generator_func in signals_to_test.items():
        print(f"\\n[{name}]")

        # Generate signal and create system
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

        # Create informed prior
        prior_samples = create_informed_prior(system, N_SAMPLES_FOR_DQA)

        # Run simulation
        print(f"  Running {TIMESTEPS}-step simulation...")
        system.run_simulation(TIMESTEPS, signal_data)

        # Extract metrics
        final_hi = system.history['hi'][-1]
        final_align = system.history['e8_alignment'][-1]
        final_loss = system.history['loss'][-1]
        converged_at = next((i for i, h in enumerate(system.history['hi'])
                            if h > 0.3 and system.history['loss'][i] < 1.0), TIMESTEPS)

        results[name] = {
            'Harmonic Index (HI)': final_hi,
            'E8 Alignment': final_align,
            'Harmonic Loss': final_loss,
            'Convergence Time': converged_at if converged_at < TIMESTEPS else 'N/A'
        }
        full_history[name] = system.history

        # PAC-Bayes analysis
        print(f"  Computing PAC bounds...")
        learned_samples = extract_learned_distribution(system, N_SAMPLES_FOR_DQA)
        dqa = dqa_divergence(learned_samples, prior_samples, MODULUS, method='optimal')
        empirical_risk = compute_classification_risk(results, name)
        generalization_bound = pac_generalization_bound(
            empirical_risk=empirical_risk,
            dqa=dqa,
            m=TIMESTEPS,
            constants=pac_constants,
            delta=CONFIDENCE_DELTA
        )

        pac_results[name] = {
            'D_QA': dqa,
            'Empirical Risk': empirical_risk,
            'Generalization Bound': generalization_bound,
            'Generalization Gap': generalization_bound - empirical_risk
        }

        # PISANO PERIOD ANALYSIS
        print(f"  Analyzing Pisano periods...")
        pisano_analysis = add_pisano_analysis_to_results(system, name, pisano_classifier)
        pisano_results[name] = pisano_analysis

        print(f"  ✓ HI: {final_hi:.3f} | D_QA: {dqa:.2f} | "
              f"PAC Bound: {generalization_bound:.1%} | "
              f"Pisano Family: {pisano_analysis['dominant_family']}")

    # Print results table with Pisano info
    print("\\n" + "="*100)
    print("RESULTS WITH PISANO PERIOD ANALYSIS")
    print("="*100)
    print(f"{'Signal':<15} | {'HI':>6} | {'D_QA':>8} | {'Emp Risk':>9} | "
          f"{'PAC Bound':>10} | {'Gap':>8} | {'Pisano Family':>15} | {'Period':>6}")
    print("-"*100)

    for name in results.keys():
        data = pac_results[name]
        pisano = pisano_results[name]
        print(f"{name:<15} | {results[name]['Harmonic Index (HI)']:>6.3f} | "
              f"{data['D_QA']:>8.2f} | {data['Empirical Risk']:>9.1%} | "
              f"{data['Generalization Bound']:>10.1%} | {data['Generalization Gap']:>7.1%} | "
              f"{pisano['dominant_family']:>15} | {pisano['avg_period']:>6.1f}")
    print("-"*100)

    # Save results
    output_data = {
        'configuration': {
            'num_nodes': NUM_NODES,
            'modulus': MODULUS,
            'timesteps': TIMESTEPS,
            'prior_type': 'informed',
            'dqa_method': 'optimal',
            'pisano_enabled': True
        },
        'results': {
            name: {
                'classification': results[name],
                'pac_bayes': pac_results[name],
                'pisano_analysis': pisano_results[name]
            }
            for name in results.keys()
        }
    }

    json_path = workspace / "signal_pac_pisano_results.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\\nResults saved to: {json_path}")

    print("\\n✓ Analysis complete with Pisano period classification!")

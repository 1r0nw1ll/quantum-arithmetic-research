#!/usr/bin/env python3
"""Multiseed robustness check for a recovered QA signal configuration."""

QA_COMPLIANCE = "empirical_observer — audio/signal as observer input; QA coupling is discrete state"


from __future__ import annotations

import numpy as np

from qa_core import QASystem, build_open_brain_capture, ensure_run_dir, write_json


TIMESTEPS = 150
DURATION = 2.0
BASE_FREQ = 261.63
SEEDS = list(range(8))
CONFIG = {
    "num_nodes": 16,
    "modulus": 24,
    "coupling": 0.12,
    "noise_base": 0.1,
    "noise_annealing": 0.995,
    "signal_injection_strength": 0.2,
    "signal_mode": "final",
}


def generate_signal(timesteps: int) -> np.ndarray:
    return np.linspace(0, DURATION, timesteps, endpoint=False)


def generate_pure_tone(timesteps: int) -> np.ndarray:
    t = generate_signal(timesteps)
    return np.sin(2 * np.pi * BASE_FREQ * t)


def generate_major_chord(timesteps: int) -> np.ndarray:
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * t)
    note2 = np.sin(2 * np.pi * BASE_FREQ * (5 / 4) * t)
    note3 = np.sin(2 * np.pi * BASE_FREQ * (6 / 4) * t)
    return (note1 + note2 + note3) / 3.0


def generate_minor_chord(timesteps: int) -> np.ndarray:
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * t)
    note2 = np.sin(2 * np.pi * BASE_FREQ * (12 / 10) * t)
    note3 = np.sin(2 * np.pi * BASE_FREQ * (15 / 10) * t)
    return (note1 + note2 + note3) / 3.0


def generate_tritone(timesteps: int) -> np.ndarray:
    t = generate_signal(timesteps)
    note1 = np.sin(2 * np.pi * BASE_FREQ * t)
    note2 = np.sin(2 * np.pi * BASE_FREQ * np.sqrt(2) * t)
    return (note1 + note2) / 2.0


def generate_white_noise(timesteps: int) -> np.ndarray:
    return np.random.uniform(-1, 1, size=timesteps)


SIGNALS = {
    "Pure Tone": generate_pure_tone,
    "Major Chord": generate_major_chord,
    "Minor Chord": generate_minor_chord,
    "Tritone": generate_tritone,
    "White Noise": generate_white_noise,
}


def main() -> None:
    run_dir = ensure_run_dir("signal", "signal_experiment_multiseed_eval")
    observations = {name: [] for name in SIGNALS}

    for seed in SEEDS:
        for signal_index, (name, generator_func) in enumerate(SIGNALS.items()):
            np.random.seed(1000 * seed + signal_index)
            system = QASystem(**CONFIG)
            system.run_simulation(TIMESTEPS, generator_func(TIMESTEPS), progress=False)
            observations[name].append(float(system.history["hi"][-1]))

    means = {name: float(np.mean(values)) for name, values in observations.items()}
    stds = {name: float(np.std(values)) for name, values in observations.items()}
    white_noise_mean = means["White Noise"]
    win_counts = {
        name: sum(1 for value in values if value > white_noise_mean)
        for name, values in observations.items()
        if name != "White Noise"
    }
    summary = {
        "script": "signal_experiment_multiseed_eval.py",
        "domain": "signal",
        "config": CONFIG,
        "seeds": SEEDS,
        "means": means,
        "stds": stds,
        "raw_hi": observations,
        "white_noise_mean": white_noise_mean,
        "wins_over_white_noise_mean": win_counts,
    }
    write_json(run_dir / "summary.json", summary)

    observation_capture = build_open_brain_capture(
        "observation",
        ["signal", "signal_experiment_multiseed_eval", "instability-check"],
        (
            "Multiseed robustness check completed for the recovered signal configuration. "
            f"White Noise mean HI={white_noise_mean:.4f}. "
            f"Pure Tone mean HI={means['Pure Tone']:.4f}, Major Chord mean HI={means['Major Chord']:.4f}, "
            f"Minor Chord mean HI={means['Minor Chord']:.4f}, Tritone mean HI={means['Tritone']:.4f}. "
            f"Win counts over the White Noise mean: {win_counts}."
        ),
        metadata=summary,
    )
    write_json(run_dir / "open_brain_observation_capture.json", observation_capture)

    print(f"Means: {means}")
    print(f"Stds: {stds}")
    print(f"Win counts over White Noise mean: {win_counts}")
    print(f"Artifacts saved under: {run_dir}")


if __name__ == "__main__":
    main()

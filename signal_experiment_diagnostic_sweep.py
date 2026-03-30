#!/usr/bin/env python3
"""Focused diagnostic sweep for the QA signal coherence experiments.

This script searches the nearby parameter region around the current signal
experiment settings and ranks configurations by the same tonal-vs-noise
criteria used in `run_signal_experiments_final.py`.
"""

QA_COMPLIANCE = "empirical_observer — audio/signal as observer input; QA coupling is discrete state"


from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from tqdm import tqdm

from qa_core import QASystem, build_open_brain_capture, ensure_run_dir, write_json


TIMESTEPS = 150
DURATION = 2.0
BASE_FREQ = 261.63

HYPOTHESIS = (
    "There exists a nearby parameter region where tonal inputs outrank white noise by final harmonic index."
)
SUCCESS_CRITERIA = [
    "Major Chord final HI > White Noise final HI",
    "Minor Chord final HI > White Noise final HI",
    "At least three tonal inputs end with HI above White Noise",
]

PARAM_GRID = {
    "num_nodes": [16, 24],
    "modulus": [24],
    "coupling": [0.02, 0.04, 0.08, 0.12],
    "noise_base": [0.1, 0.2, 0.4],
    "noise_annealing": [0.99, 0.995],
    "signal_injection_strength": [0.05, 0.1, 0.2, 0.4],
    "signal_mode": ["original", "corrected", "final"],
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


def evaluate_config(config: dict[str, Any], seed_base: int) -> dict[str, Any]:
    results: dict[str, dict[str, float | int | None]] = {}
    for signal_index, (name, generator_func) in enumerate(SIGNALS.items()):
        np.random.seed(seed_base + signal_index)
        signal_data = generator_func(TIMESTEPS)
        system = QASystem(
            num_nodes=int(config["num_nodes"]),
            modulus=int(config["modulus"]),
            coupling=float(config["coupling"]),
            noise_base=float(config["noise_base"]),
            noise_annealing=float(config["noise_annealing"]),
            signal_injection_strength=float(config["signal_injection_strength"]),
            signal_mode=str(config["signal_mode"]),
        )
        system.run_simulation(TIMESTEPS, signal_data, progress=False)
        converged_at = next(
            (
                i
                for i, hi in enumerate(system.history["hi"])
                if hi > 0.3 and system.history["loss"][i] < 1.0
            ),
            TIMESTEPS,
        )
        results[name] = {
            "final_hi": float(system.history["hi"][-1]),
            "final_e8_alignment": float(system.history["e8_alignment"][-1]),
            "final_loss": float(system.history["loss"][-1]),
            "convergence_time": int(converged_at) if converged_at < TIMESTEPS else None,
        }

    white_noise_hi = float(results["White Noise"]["final_hi"])
    tonal_names = ["Pure Tone", "Major Chord", "Minor Chord", "Tritone"]
    tonal_above_noise = [name for name in tonal_names if float(results[name]["final_hi"]) > white_noise_hi]
    criteria_results = {
        "major_above_noise": float(results["Major Chord"]["final_hi"]) > white_noise_hi,
        "minor_above_noise": float(results["Minor Chord"]["final_hi"]) > white_noise_hi,
        "three_tonal_above_noise": len(tonal_above_noise) >= 3,
    }
    pass_count = sum(1 for ok in criteria_results.values() if ok)
    verdict = "PASS" if pass_count == len(criteria_results) else ("PARTIAL" if pass_count > 0 else "FAIL")
    score = (
        100.0 * pass_count
        + 10.0 * len(tonal_above_noise)
        + 5.0 * (float(results["Major Chord"]["final_hi"]) - white_noise_hi)
        + 5.0 * (float(results["Minor Chord"]["final_hi"]) - white_noise_hi)
        + 2.0 * (float(results["Pure Tone"]["final_hi"]) - white_noise_hi)
    )

    return {
        "config": config,
        "results": results,
        "criteria_results": criteria_results,
        "tonal_above_noise": tonal_above_noise,
        "pass_count": pass_count,
        "score": float(score),
        "verdict": verdict,
    }


def main() -> None:
    run_dir = ensure_run_dir("signal", "signal_experiment_diagnostic_sweep")
    prereg_capture = build_open_brain_capture(
        "task",
        ["signal", "signal_experiment_diagnostic_sweep", "pre-registration"],
        (
            "Focused parameter sweep for the QA signal experiment. "
            f"Hypothesis: {HYPOTHESIS} "
            f"Success criteria: {'; '.join(SUCCESS_CRITERIA)}"
        ),
        metadata={"parameter_grid": PARAM_GRID, "timesteps": TIMESTEPS},
    )
    write_json(run_dir / "open_brain_task_capture.json", prereg_capture)

    keys = list(PARAM_GRID.keys())
    param_product = list(itertools.product(*(PARAM_GRID[key] for key in keys)))
    all_results = []

    print("--- Starting signal experiment diagnostic sweep ---")
    for index, values in enumerate(tqdm(param_product, desc="Sweeping")):
        config = dict(zip(keys, values))
        all_results.append(evaluate_config(config, seed_base=4242 + index * 100))

    ranked = sorted(
        all_results,
        key=lambda item: (item["pass_count"], item["score"]),
        reverse=True,
    )
    top_results = ranked[:20]
    passing = [item for item in ranked if item["verdict"] == "PASS"]
    best = ranked[0]

    summary = {
        "script": "signal_experiment_diagnostic_sweep.py",
        "domain": "signal",
        "hypothesis": HYPOTHESIS,
        "success_criteria": SUCCESS_CRITERIA,
        "parameter_grid": PARAM_GRID,
        "total_configs": len(all_results),
        "passing_configs": len(passing),
        "best_result": best,
        "top_results": top_results,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "all_results.json", ranked)

    observation_capture = build_open_brain_capture(
        "observation",
        ["signal", "signal_experiment_diagnostic_sweep", "pass" if passing else "fail"],
        (
            "Signal diagnostic sweep completed. "
            f"Checked {len(all_results)} configurations. "
            f"Passing configurations: {len(passing)}. "
            f"Best verdict: {best['verdict']}. "
            f"Best config: {best['config']}. "
            f"Best tonal signals above White Noise: {', '.join(best['tonal_above_noise']) if best['tonal_above_noise'] else 'none'}."
        ),
        metadata=summary,
    )
    write_json(run_dir / "open_brain_observation_capture.json", observation_capture)

    print(f"Total configs checked: {len(all_results)}")
    print(f"Passing configs: {len(passing)}")
    print(f"Best verdict: {best['verdict']}")
    print(f"Best config: {best['config']}")
    print(f"Tonal signals above White Noise: {best['tonal_above_noise']}")
    print(f"Artifacts saved under: {run_dir}")


if __name__ == "__main__":
    main()

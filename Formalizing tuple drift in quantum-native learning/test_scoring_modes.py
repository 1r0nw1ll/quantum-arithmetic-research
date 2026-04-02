"""
Quick test: Compare QAWM-Greedy scoring function variants
Tests 4 scoring modes to find optimal configuration.
"""

import numpy as np
import joblib

from qa_oracle import QAOracle, construct_qa_state
from rml_policy import QAWMGreedyPolicy, RMLTask, run_episode


def evaluate_scoring_mode(mode: str, num_episodes: int = 50):
    """Evaluate QAWM-Greedy with specific scoring mode"""
    print(f"\n{'='*70}")
    print(f"Testing Scoring Mode: {mode}")
    print(f"{'='*70}")

    # Setup
    oracle = QAOracle(N=30, q_def="none")
    task = RMLTask.diagonal_task(N=30, k=20)
    qawm_model = joblib.load('qawm_model.pkl')

    # Initialize policy with scoring mode
    policy = QAWMGreedyPolicy(qawm_model, oracle, task.generators, N=30,
                              scoring_mode=mode)

    successes = []
    oracle_calls_list = []

    # Run episodes
    for episode_idx in range(num_episodes):
        # Random off-diagonal start
        b = np.random.randint(1, oracle.N + 1)
        e = np.random.randint(1, oracle.N + 1)
        while b == e:
            e = np.random.randint(1, oracle.N + 1)

        start_state = construct_qa_state(b, e)

        # Run episode
        result = run_episode(policy, start_state, task, oracle)

        successes.append(result['success'])
        oracle_calls_list.append(result['oracle_calls'])

    # Results
    success_rate = np.mean(successes)
    avg_oracle_calls = np.mean([c for c, s in zip(oracle_calls_list, successes) if s])
    if np.isnan(avg_oracle_calls):
        avg_oracle_calls = 0.0

    print(f"\nResults ({num_episodes} episodes):")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Avg oracle calls (successful): {avg_oracle_calls:.1f}")

    return {
        'mode': mode,
        'success_rate': success_rate,
        'avg_oracle_calls': avg_oracle_calls,
        'num_successes': sum(successes)
    }


def main():
    print("=" * 70)
    print("SCORING MODE COMPARISON - QAWM-GREEDY")
    print("Testing 4 scoring function variants")
    print("=" * 70)

    # Test all modes
    modes = [
        'product',         # Original: p_legal × p_return
        'weighted_sum',    # 0.3·p_legal + 0.7·p_return
        'return_only',     # p_return only
        'legal_threshold'  # p_return if p_legal > 0.5 else 0
    ]

    results = []
    for mode in modes:
        result = evaluate_scoring_mode(mode, num_episodes=50)
        results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY - SCORING MODE COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Mode':<20} {'Success Rate':<15} {'Oracle Calls':<15} {'# Successes':<15}")
    print(f"{'-'*70}")

    for r in results:
        print(f"{r['mode']:<20} "
              f"{r['success_rate']:<15.1%} "
              f"{r['avg_oracle_calls']:<15.1f} "
              f"{r['num_successes']:<15d}")

    # Find best mode
    best_result = max(results, key=lambda x: x['success_rate'])

    print(f"\n{'='*70}")
    print(f"BEST SCORING MODE: {best_result['mode']}")
    print(f"{'='*70}")
    print(f"  Success rate: {best_result['success_rate']:.1%}")
    print(f"  Oracle calls: {best_result['avg_oracle_calls']:.1f}")

    if best_result['success_rate'] > 0.19:
        print(f"\n✅ SUCCESS: Better than Random-Legal (19%)")
        print(f"   Recommend using '{best_result['mode']}' mode for Paper 3")
    else:
        print(f"\n⚠ Still below Random-Legal (19%)")
        print(f"   Scoring function tuning insufficient, consider Option A (retraining)")

    print(f"\n✅ Scoring mode test complete!")


if __name__ == "__main__":
    main()

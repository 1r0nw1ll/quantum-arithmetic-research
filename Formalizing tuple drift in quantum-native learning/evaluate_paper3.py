"""
Paper 3 Evaluation Script
Compares all RML baselines and generates results.

Baselines:
  1. Random-Legal (lower bound)
  2. Oracle-Greedy (upper bound)
  3. QAWM-Greedy (key result - no learning)
  4. RML Policy (meta-learning)

Primary Metric: Oracle calls to success
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm

from qa_oracle import QAOracle, construct_qa_state
from rml_policy import (
    RandomLegalPolicy,
    OracleGreedyPolicy,
    QAWMGreedyPolicy,
    RMLPolicy,
    RMLTask,
    run_episode
)


# =============================================================================
# Evaluation Runner
# =============================================================================

def evaluate_policy(policy, policy_name: str, oracle: QAOracle, task: RMLTask,
                   num_episodes: int = 100, verbose: bool = True):
    """
    Evaluate a policy over multiple episodes.

    Returns:
        results: dict with aggregated metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {policy_name}")
        print(f"{'='*70}")

    successes = []
    steps_list = []
    oracle_calls_list = []

    # Run episodes
    iterator = tqdm(range(num_episodes), desc=policy_name) if verbose else range(num_episodes)

    for episode_idx in iterator:
        # Random start state (off-diagonal)
        b = np.random.randint(1, oracle.N + 1)
        e = np.random.randint(1, oracle.N + 1)

        # Ensure off-diagonal (not already at target)
        while b == e:
            e = np.random.randint(1, oracle.N + 1)

        start_state = construct_qa_state(b, e)

        # Run episode
        result = run_episode(policy, start_state, task, oracle)

        successes.append(result['success'])
        steps_list.append(result['steps'])
        oracle_calls_list.append(result['oracle_calls'])

    # Aggregate results
    success_rate = np.mean(successes)
    avg_steps = np.mean([s for s, success in zip(steps_list, successes) if success])
    avg_oracle_calls = np.mean([c for c, success in zip(oracle_calls_list, successes) if success])

    if verbose:
        print(f"\nResults ({num_episodes} episodes):")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Avg steps (successful): {avg_steps:.1f}")
        print(f"  Avg oracle calls (successful): {avg_oracle_calls:.1f}")

    return {
        'policy_name': policy_name,
        'success_rate': success_rate,
        'avg_steps': avg_steps if not np.isnan(avg_steps) else 0,
        'avg_oracle_calls': avg_oracle_calls if not np.isnan(avg_oracle_calls) else 0,
        'successes': successes,
        'steps': steps_list,
        'oracle_calls': oracle_calls_list
    }


# =============================================================================
# Paper 3 Main Evaluation
# =============================================================================

def main():
    print("=" * 70)
    print("PAPER 3: RML BASELINE COMPARISON")
    print("Meta-Policy Learning Over QA Manifolds")
    print("=" * 70)

    # Setup
    oracle = QAOracle(N=30, q_def="none")
    task = RMLTask.diagonal_task(N=30, k=20)  # Increased from k=10 to k=20
    num_episodes = 100

    print(f"\nTask Configuration:")
    print(f"  Manifold: Caps(30,30)")
    print(f"  Target: Diagonal {{(b,b)}}, {len(task.target_class)} states")
    print(f"  Horizon: k = {task.horizon} steps")
    print(f"  Generators: {task.generators}")
    print(f"  Episodes: {num_episodes}")

    # Load QAWM model
    print(f"\n{'='*70}")
    print(f"LOADING QAWM MODEL (from Paper 2)")
    print(f"{'='*70}")

    try:
        qawm_model = joblib.load('qawm_model.pkl')
        print(f"  ✓ QAWM model loaded successfully")
    except:
        print(f"  ✗ QAWM model not found!")
        print(f"  Run train_qawm_sklearn.py first")
        return

    # Initialize policies
    policies = []

    # Baseline 1: Random-Legal
    print(f"\nInitializing Random-Legal policy...")
    random_policy = RandomLegalPolicy(oracle, task.generators)
    policies.append(('Random-Legal', random_policy))

    # Baseline 2: Oracle-Greedy
    print(f"Initializing Oracle-Greedy policy...")
    oracle_greedy = OracleGreedyPolicy(oracle, task.generators)
    policies.append(('Oracle-Greedy', oracle_greedy))

    # Baseline 3: QAWM-Greedy (KEY RESULT)
    print(f"Initializing QAWM-Greedy policy...")
    qawm_greedy = QAWMGreedyPolicy(qawm_model, oracle, task.generators, N=30,
                                   scoring_mode='return_only')  # Best mode from ablation
    policies.append(('QAWM-Greedy', qawm_greedy))

    # Run evaluations
    results = []

    for policy_name, policy in policies:
        result = evaluate_policy(
            policy,
            policy_name,
            oracle,
            task,
            num_episodes=num_episodes,
            verbose=True
        )
        results.append(result)

    # =========================================================================
    # Results Summary
    # =========================================================================
    print(f"\n\n{'='*70}")
    print(f"PAPER 3 RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Policy':<20} {'Success Rate':<15} {'Avg Steps':<12} {'Oracle Calls':<15}")
    print(f"{'-'*70}")

    for r in results:
        print(f"{r['policy_name']:<20} "
              f"{r['success_rate']:<15.1%} "
              f"{r['avg_steps']:<12.1f} "
              f"{r['avg_oracle_calls']:<15.1f}")

    # =========================================================================
    # Key Comparisons
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"KEY COMPARISONS (Primary Metric: Oracle Efficiency)")
    print(f"{'='*70}")

    # Find QAWM-Greedy and Oracle-Greedy results
    qawm_result = next(r for r in results if r['policy_name'] == 'QAWM-Greedy')
    oracle_result = next(r for r in results if r['policy_name'] == 'Oracle-Greedy')
    random_result = next(r for r in results if r['policy_name'] == 'Random-Legal')

    # Oracle efficiency comparison
    qawm_calls = qawm_result['avg_oracle_calls']
    oracle_calls = oracle_result['avg_oracle_calls']

    if oracle_calls > 0:
        efficiency_ratio = qawm_calls / oracle_calls
        print(f"\nOracle Efficiency:")
        print(f"  QAWM-Greedy:   {qawm_calls:.1f} calls/episode")
        print(f"  Oracle-Greedy: {oracle_calls:.1f} calls/episode")
        print(f"  Ratio: {efficiency_ratio:.2f}× (lower is better)")

        if efficiency_ratio < 0.5:
            print(f"  ✅ STRONG: QAWM uses <50% oracle calls of Oracle-Greedy")
            verdict = "PUBLICATION-READY"
        elif efficiency_ratio < 0.75:
            print(f"  ✓ MODERATE: QAWM reduces oracle calls")
            verdict = "DEFENSIBLE"
        else:
            print(f"  ⚠ WEAK: Oracle savings minimal")
            verdict = "NEEDS WORK"
    else:
        verdict = "N/A"

    # Success rate comparison
    print(f"\nSuccess Rate:")
    print(f"  QAWM-Greedy:   {qawm_result['success_rate']:.1%}")
    print(f"  Oracle-Greedy: {oracle_result['success_rate']:.1%}")
    print(f"  Random-Legal:  {random_result['success_rate']:.1%}")

    success_gap = qawm_result['success_rate'] - random_result['success_rate']
    print(f"  Improvement over random: +{success_gap:.1%}")

    # Oracle-call normalized success (efficiency per success)
    print(f"\nOracle-Call Normalized Success (successes / avg_oracle_calls):")
    random_norm = (random_result['success_rate'] * 100) / random_result['avg_oracle_calls'] if random_result['avg_oracle_calls'] > 0 else 0
    oracle_norm = (oracle_result['success_rate'] * 100) / oracle_result['avg_oracle_calls'] if oracle_result['avg_oracle_calls'] > 0 else 0
    qawm_norm = (qawm_result['success_rate'] * 100) / qawm_result['avg_oracle_calls'] if qawm_result['avg_oracle_calls'] > 0 else 0

    print(f"  Random-Legal:  {random_norm:.2f} (successes per call)")
    print(f"  Oracle-Greedy: {oracle_norm:.2f} (successes per call)")
    print(f"  QAWM-Greedy:   {qawm_norm:.2f} (successes per call)")

    if qawm_norm >= oracle_norm * 0.95:
        print(f"  ✅ STRONG: QAWM matches Oracle-Greedy's efficiency per success!")
    elif qawm_norm > random_norm:
        print(f"  ✓ QAWM more efficient per success than random")

    # =========================================================================
    # Visualization
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Success Rate
    policy_names = [r['policy_name'] for r in results]
    success_rates = [r['success_rate'] * 100 for r in results]

    axes[0].bar(policy_names, success_rates, alpha=0.7,
                color=['red', 'green', 'blue'])
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('Success Rate Comparison', fontsize=14)
    axes[0].set_ylim([0, 100])
    axes[0].grid(alpha=0.3, axis='y')

    for i, (name, rate) in enumerate(zip(policy_names, success_rates)):
        axes[0].text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=11)

    # Plot 2: Oracle Efficiency (PRIMARY METRIC)
    oracle_calls = [r['avg_oracle_calls'] for r in results]

    axes[1].bar(policy_names, oracle_calls, alpha=0.7,
                color=['red', 'green', 'blue'])
    axes[1].set_ylabel('Avg Oracle Calls (Successful Episodes)', fontsize=12)
    axes[1].set_title('Oracle Efficiency (PRIMARY METRIC)', fontsize=14)
    axes[1].grid(alpha=0.3, axis='y')

    for i, (name, calls) in enumerate(zip(policy_names, oracle_calls)):
        axes[1].text(i, calls + 1, f'{calls:.1f}', ha='center', fontsize=11)

    # Plot 3: Normalized Success (Efficiency per Success)
    normalized_success = []
    for r in results:
        if r['avg_oracle_calls'] > 0:
            norm = (r['success_rate'] * 100) / r['avg_oracle_calls']
        else:
            norm = 0
        normalized_success.append(norm)

    axes[2].bar(policy_names, normalized_success, alpha=0.7,
                color=['red', 'green', 'blue'])
    axes[2].set_ylabel('Successes per Oracle Call', fontsize=12)
    axes[2].set_title('Normalized Success (Efficiency per Success)', fontsize=14)
    axes[2].grid(alpha=0.3, axis='y')

    for i, (name, norm) in enumerate(zip(policy_names, normalized_success)):
        axes[2].text(i, norm + 0.05, f'{norm:.2f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('paper3_results.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Figure saved: paper3_results.png")

    # =========================================================================
    # Final Verdict
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT - PAPER 3")
    print(f"{'='*70}")

    print(f"\nCore Thesis: \"Learning enables control via structural queries\"")
    print(f"Status: {verdict}")

    if verdict == "PUBLICATION-READY":
        print(f"\n✅ QAWM-Greedy demonstrates strong oracle efficiency")
        print(f"   Achieves {qawm_result['success_rate']:.1%} success with {efficiency_ratio:.1f}× fewer calls")
        print(f"   Paper 3 is ready to write (RML learning optional)")
    elif verdict == "DEFENSIBLE":
        print(f"\n✓ QAWM-Greedy shows improvement over baselines")
        print(f"   Results are defensible but not showstopping")
        print(f"   RML learning may provide additional gains")
    else:
        print(f"\n⚠ QAWM-Greedy results weak")
        print(f"   Consider: (1) Adjust scoring function (2) More QAWM training")

    # =========================================================================
    # LaTeX Table Output
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"LATEX TABLE (ready to paste)")
    print(f"{'='*70}")

    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{RML baseline comparison: Control via structural predictions.}}
\\label{{tab:rml_baselines}}
\\begin{{tabular}}{{lccc}}
\\toprule
Policy & Success Rate & Avg Steps & Oracle Calls \\\\
\\midrule
Random-Legal & {random_result['success_rate']:.1%} & {random_result['avg_steps']:.1f} & {random_result['avg_oracle_calls']:.1f} \\\\
Oracle-Greedy & {oracle_result['success_rate']:.1%} & {oracle_result['avg_steps']:.1f} & {oracle_result['avg_oracle_calls']:.1f} \\\\
\\textbf{{QAWM-Greedy}} & \\textbf{{{qawm_result['success_rate']:.1%}}} & \\textbf{{{qawm_result['avg_steps']:.1f}}} & \\textbf{{{qawm_result['avg_oracle_calls']:.1f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
    """)

    print(f"\n✅ Paper 3 baseline evaluation complete!")
    print(f"   Next: (Optional) Implement RML learning for additional gains")


if __name__ == "__main__":
    main()

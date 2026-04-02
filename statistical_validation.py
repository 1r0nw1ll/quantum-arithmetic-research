#!/usr/bin/env python3
"""
Statistical Validation for Phase 2 Results

Performs rigorous statistical tests:
- Paired t-tests (compare methods on same data)
- Wilcoxon signed-rank test (non-parametric alternative)
- McNemar's test (classification comparisons)
- Cross-validation with multiple random seeds
- Effect size calculations (Cohen's d)
"""

import numpy as np
from scipy import stats
from typing import Any, Dict, List, Tuple
import json
from pathlib import Path


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def paired_ttest(method1_scores: np.ndarray, method2_scores: np.ndarray,
                 method1_name: str = "Method 1", method2_name: str = "Method 2") -> Dict:
    """
    Perform paired t-test comparing two methods.

    Args:
        method1_scores: Accuracy scores for method 1 (across CV folds)
        method2_scores: Accuracy scores for method 2 (across CV folds)
        method1_name: Name of first method
        method2_name: Name of second method

    Returns:
        Dictionary with test results
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)

    # Cohen's d (effect size)
    diff = method1_scores - method2_scores
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)

    # Interpretation
    if abs(cohen_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size = "small"
    elif abs(cohen_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    # Significance
    alpha = 0.05
    significant = p_value < alpha

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohen_d': float(cohen_d),
        'effect_size': effect_size,
        'significant': significant,
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff, ddof=1)),
        'comparison': f"{method1_name} vs {method2_name}",
        'conclusion': (
            f"{method1_name} {'significantly' if significant else 'not significantly'} "
            f"{'better' if np.mean(diff) > 0 else 'worse'} than {method2_name} "
            f"(p={p_value:.4f}, d={cohen_d:.3f}, {effect_size} effect)"
        )
    }


def wilcoxon_test(method1_scores: np.ndarray, method2_scores: np.ndarray,
                  method1_name: str = "Method 1", method2_name: str = "Method 2") -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        method1_scores: Accuracy scores for method 1
        method2_scores: Accuracy scores for method 2
        method1_name: Name of first method
        method2_name: Name of second method

    Returns:
        Dictionary with test results
    """
    # Wilcoxon test
    stat, p_value = stats.wilcoxon(method1_scores, method2_scores)

    # Significance
    alpha = 0.05
    significant = p_value < alpha

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': significant,
        'comparison': f"{method1_name} vs {method2_name}",
        'conclusion': (
            f"{method1_name} {'significantly' if significant else 'not significantly'} "
            f"different from {method2_name} (Wilcoxon p={p_value:.4f})"
        )
    }


def mcnemar_test(method1_preds: np.ndarray, method2_preds: np.ndarray,
                 true_labels: np.ndarray,
                 method1_name: str = "Method 1", method2_name: str = "Method 2") -> Dict:
    """
    Perform McNemar's test for paired nominal data (classification).

    Tests if two methods disagree in a systematic way.

    Args:
        method1_preds: Predictions from method 1
        method2_preds: Predictions from method 2
        true_labels: Ground truth labels
        method1_name: Name of first method
        method2_name: Name of second method

    Returns:
        Dictionary with test results
    """
    # Create contingency table
    #             Method 2 correct | Method 2 wrong
    # Method 1 correct    a        |      b
    # Method 1 wrong      c        |      d

    method1_correct = (method1_preds == true_labels)
    method2_correct = (method2_preds == true_labels)

    a = np.sum(method1_correct & method2_correct)
    b = np.sum(method1_correct & ~method2_correct)
    c = np.sum(~method1_correct & method2_correct)
    d = np.sum(~method1_correct & ~method2_correct)

    # McNemar's test statistic (with continuity correction)
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0.0
        p_value = 1.0

    # Significance
    alpha = 0.05
    significant = p_value < alpha

    return {
        'chi2_statistic': float(chi2),
        'p_value': float(p_value),
        'contingency_table': {
            'both_correct': int(a),
            'method1_only': int(b),
            'method2_only': int(c),
            'both_wrong': int(d)
        },
        'significant': significant,
        'comparison': f"{method1_name} vs {method2_name}",
        'conclusion': (
            f"{method1_name} and {method2_name} "
            f"{'have significantly different error patterns' if significant else 'do not differ significantly'} "
            f"(McNemar p={p_value:.4f})"
        )
    }


def cross_validation_comparison(accuracies: Dict[str, List[float]]) -> Dict:
    """
    Comprehensive statistical comparison across cross-validation folds.

    Args:
        accuracies: Dict mapping method names to lists of CV fold accuracies

    Returns:
        Dictionary with all statistical test results
    """
    results = {
        'summary': {},
        'pairwise_ttests': [],
        'pairwise_wilcoxon': [],
        'friedman_test': {}
    }

    methods = list(accuracies.keys())

    # Summary statistics
    for method in methods:
        scores = np.array(accuracies[method])
        results['summary'][method] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores, ddof=1)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }

    # Pairwise t-tests
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            scores1 = np.array(accuracies[method1])
            scores2 = np.array(accuracies[method2])

            ttest_result = paired_ttest(scores1, scores2, method1, method2)
            results['pairwise_ttests'].append(ttest_result)

            wilcoxon_result = wilcoxon_test(scores1, scores2, method1, method2)
            results['pairwise_wilcoxon'].append(wilcoxon_result)

    # Friedman test (non-parametric equivalent of repeated measures ANOVA)
    if len(methods) > 2:
        scores_matrix = np.array([accuracies[m] for m in methods])
        stat, p_value = stats.friedmanchisquare(*scores_matrix)

        results['friedman_test'] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'conclusion': (
                f"Methods {'do' if p_value < 0.05 else 'do not'} differ significantly "
                f"(Friedman p={p_value:.4f})"
            )
        }

    return results


def run_validation_example():
    """
    Example validation on simulated data.

    Replace with actual validation results.
    """
    print("="*80)
    print("STATISTICAL VALIDATION - EXAMPLE")
    print("="*80)
    print()

    # Simulate CV fold accuracies (replace with real data)
    np.random.seed(42)
    n_folds = 5

    accuracies = {
        'QA': [0.85, 0.87, 0.83, 0.86, 0.84],
        'CNN': [0.92, 0.93, 0.91, 0.94, 0.92],
        'LSTM': [0.90, 0.91, 0.89, 0.92, 0.90]
    }

    # Run comprehensive comparison
    results = cross_validation_comparison(accuracies)

    # Print summary
    print("SUMMARY STATISTICS")
    print("-" * 80)
    for method, stats in results['summary'].items():
        print(f"{method}:")
        print(f"  Mean ± Std:  {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Range:       [{stats['min']:.3f}, {stats['max']:.3f}]")
        print()

    # Print pairwise comparisons
    print("PAIRWISE T-TESTS")
    print("-" * 80)
    for test in results['pairwise_ttests']:
        print(f"{test['comparison']}:")
        print(f"  {test['conclusion']}")
        print()

    # Print Friedman test
    if results['friedman_test']:
        print("FRIEDMAN TEST (Overall)")
        print("-" * 80)
        print(results['friedman_test']['conclusion'])
        print()

    # Save results
    output_dir = Path("phase2_workspace")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "statistical_validation.json"

    with open(output_path, 'w') as f:
        json.dump(_to_jsonable(results), f, indent=2)

    print(f"✓ Results saved to {output_path}")
    print()

    return results


def generate_statistical_report(results: Dict, output_path: Path):
    """
    Generate human-readable statistical report.

    Args:
        results: Statistical validation results
        output_path: Where to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")

        # Summary
        f.write("1. SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        for method, stats in results['summary'].items():
            f.write(f"\n{method}:\n")
            f.write(f"  Mean Accuracy: {stats['mean']:.3f} (±{stats['std']:.3f})\n")
            f.write(f"  95% CI: [{stats['mean'] - 1.96*stats['std']:.3f}, "
                   f"{stats['mean'] + 1.96*stats['std']:.3f}]\n")

        # Pairwise tests
        f.write("\n\n2. PAIRWISE COMPARISONS (T-tests)\n")
        f.write("-" * 80 + "\n")
        for test in results['pairwise_ttests']:
            f.write(f"\n{test['comparison']}:\n")
            f.write(f"  t-statistic: {test['t_statistic']:.4f}\n")
            f.write(f"  p-value: {test['p_value']:.4f}\n")
            f.write(f"  Cohen's d: {test['cohen_d']:.3f} ({test['effect_size']} effect)\n")
            f.write(f"  Conclusion: {test['conclusion']}\n")

        # Friedman test
        if results.get('friedman_test'):
            f.write("\n\n3. OVERALL COMPARISON (Friedman Test)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Statistic: {results['friedman_test']['statistic']:.4f}\n")
            f.write(f"  p-value: {results['friedman_test']['p_value']:.4f}\n")
            f.write(f"  Conclusion: {results['friedman_test']['conclusion']}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✓ Statistical report saved to {output_path}")


if __name__ == "__main__":
    results = run_validation_example()

    # Generate report
    report_path = Path("phase2_workspace/statistical_report.txt")
    generate_statistical_report(results, report_path)

    print("="*80)
    print("✓ STATISTICAL VALIDATION COMPLETE")
    print("="*80)

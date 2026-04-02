#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
HI 2.0 Ablation Study: Comprehensive Weight Configuration Analysis

Tests Harmonicity Index 2.0 across different weight configurations to identify
optimal settings for different signal classification domains.

Reference: Phase 2 Signal Classification Paper, Section 6.2.3
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
from qa_harmonicity_v2 import (
    qa_tuple, compute_hi_1_0, compute_hi_2_0,
    classify_gender, pythagorean_triple
)
from typing import List, Dict, Tuple
import json

# ============================================================================
# EXPERIMENTAL CONFIGURATIONS
# ============================================================================

# Weight configurations to test
WEIGHT_CONFIGS = {
    'HI_1.0_baseline': (1.0, 0.0, 0.0),  # E8-only (angular component)
    'Balanced_default': (0.4, 0.3, 0.3),  # Default HI 2.0
    'High_angular': (0.6, 0.2, 0.2),      # Pisano period emphasis
    'High_radial': (0.3, 0.5, 0.2),       # Primitivity emphasis
    'High_family': (0.3, 0.2, 0.5),       # Classical subfamily emphasis
    'Angular_radial': (0.5, 0.5, 0.0),    # Geometry-focused (no family)
    'Radial_family': (0.0, 0.6, 0.4),     # Number theory (no angular)
    'Equal_weights': (0.33, 0.34, 0.33),  # Fully balanced
}

# Test dataset: Representative QA tuples from different categories
TEST_TUPLES = [
    # Fibonacci family
    ((1, 1), 'Fibonacci_primitive_golden', 'Primitive male, all 3 families'),
    ((2, 3), 'Fibonacci_female', 'Female (gcd=2)'),
    ((5, 8), 'Fibonacci_primitive_large', 'Primitive male, large values'),

    # Lucas family
    ((2, 1), 'Lucas_female_plato', 'Female (gcd=2), Plato family'),
    ((1, 2), 'Lucas_primitive_pythagoras', 'Primitive male, Pythagoras'),
    ((7, 4), 'Lucas_primitive', 'Primitive male'),

    # Phibonacci family
    ((3, 1), 'Phibonacci_primitive_plato', 'Primitive male, Plato'),
    ((4, 5), 'Phibonacci_primitive', 'Primitive male'),

    # Tribonacci family (8-cycle)
    ((3, 3), 'Tribonacci_composite', 'Composite male (gcd=9)'),
    ((6, 9), 'Tribonacci_composite_large', 'Composite male (gcd=3)'),

    # Ninbonacci family (fixed point)
    ((9, 9), 'Ninbonacci_fixed_point', 'Composite male (gcd=27)'),

    # Edge cases
    ((1, 3), 'Edge_lucas', 'Lucas primitive'),
    ((4, 7), 'Edge_lucas_2', 'Lucas primitive'),
]

# ============================================================================
# ABLATION STUDY FUNCTIONS
# ============================================================================

def run_ablation_study() -> pd.DataFrame:
    """
    Run comprehensive ablation study across all weight configurations and test tuples.

    Returns:
        DataFrame with results for all (tuple, config) combinations
    """
    results = []

    for (b, e), label, description in TEST_TUPLES:
        q = qa_tuple(b, e)
        C, F, G = pythagorean_triple(q)
        gender = classify_gender(q)

        # Baseline HI 1.0
        hi_1_0 = compute_hi_1_0(q)

        for config_name, (w_ang, w_rad, w_fam) in WEIGHT_CONFIGS.items():
            result = compute_hi_2_0(q, w_ang=w_ang, w_rad=w_rad, w_fam=w_fam)

            results.append({
                'tuple_label': label,
                'b': b,
                'e': e,
                'C': C,
                'F': F,
                'G': G,
                'gcd': result['gcd'],
                'gender': gender,
                'families': ', '.join(result['families']),
                'description': description,
                'config': config_name,
                'w_ang': w_ang,
                'w_rad': w_rad,
                'w_fam': w_fam,
                'HI_1.0': hi_1_0,
                'HI_2.0': result['HI_2.0'],
                'H_angular': result['H_angular'],
                'H_radial': result['H_radial'],
                'H_family': result['H_family'],
                'improvement_vs_HI1': result['HI_2.0'] - hi_1_0,
                'improvement_pct': ((result['HI_2.0'] - hi_1_0) / hi_1_0 * 100) if hi_1_0 > 0 else 0,
            })

    return pd.DataFrame(results)


def analyze_by_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze HI 2.0 performance by gender category.

    Returns:
        Summary DataFrame grouped by (gender, config)
    """
    summary = df.groupby(['gender', 'config']).agg({
        'HI_2.0': ['mean', 'std', 'min', 'max'],
        'improvement_vs_HI1': ['mean', 'std'],
        'improvement_pct': 'mean',
    }).round(4)

    return summary


def analyze_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze HI 2.0 performance by classical subfamily membership.

    Returns:
        Summary DataFrame grouped by family presence
    """
    # Create binary features for family membership
    df['has_fermat'] = df['families'].str.contains('Fermat')
    df['has_pythagoras'] = df['families'].str.contains('Pythagoras')
    df['has_plato'] = df['families'].str.contains('Plato')
    df['family_count'] = df['has_fermat'].astype(int) + df['has_pythagoras'].astype(int) + df['has_plato'].astype(int)

    summary = df.groupby(['family_count', 'config']).agg({
        'HI_2.0': ['mean', 'std'],
        'improvement_pct': 'mean',
    }).round(4)

    return summary


def find_best_configs_by_domain() -> Dict[str, str]:
    """
    Identify best weight configurations for different application domains.

    Returns:
        Dictionary mapping domain → best config name
    """
    df = run_ablation_study()

    # Domain 1: Seismic (primitive vs composite discrimination)
    primitive_mask = df['gender'] == 'Male (Primitive)'
    composite_mask = df['gender'].str.contains('Composite')
    seismic_sep = df[primitive_mask].groupby('config')['HI_2.0'].mean() - \
                  df[composite_mask].groupby('config')['HI_2.0'].mean()
    best_seismic = seismic_sep.idxmax()

    # Domain 2: EEG (family membership for transitional states)
    has_family_mask = df['families'] != 'None'
    eeg_score = df[has_family_mask].groupby('config')['H_family'].mean()
    best_eeg = eeg_score.idxmax()

    # Domain 3: General classification (overall HI 2.0 mean)
    general_score = df.groupby('config')['HI_2.0'].mean()
    best_general = general_score.idxmax()

    # Domain 4: Low-data regime (high radial for strong inductive bias)
    low_data_score = df.groupby('config')['H_radial'].mean()
    best_low_data = low_data_score.idxmax()

    return {
        'seismic': best_seismic,
        'eeg': best_eeg,
        'general': best_general,
        'low_data': best_low_data,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*100)
    print("HI 2.0 ABLATION STUDY - Comprehensive Weight Configuration Analysis")
    print("="*100)
    print()

    # Run full ablation study
    print("Running ablation study across", len(TEST_TUPLES), "test tuples and",
          len(WEIGHT_CONFIGS), "weight configurations...")
    df_results = run_ablation_study()
    print(f"Generated {len(df_results)} result combinations")
    print()

    # Save full results to CSV
    csv_path = 'hi_2_0_ablation_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Saved full results to {csv_path}")
    print()

    # === ANALYSIS 1: Overall Performance Comparison ===
    print("="*100)
    print("ANALYSIS 1: Overall Performance by Configuration")
    print("="*100)
    print()

    summary_by_config = df_results.groupby('config').agg({
        'HI_2.0': ['mean', 'std'],
        'improvement_vs_HI1': 'mean',
        'improvement_pct': 'mean',
    }).round(4)
    summary_by_config = summary_by_config.sort_values(('HI_2.0', 'mean'), ascending=False)
    print(summary_by_config)
    print()

    # === ANALYSIS 2: Performance by Gender Category ===
    print("="*100)
    print("ANALYSIS 2: Performance by Gender Category")
    print("="*100)
    print()

    gender_summary = analyze_by_gender(df_results)
    print(gender_summary)
    print()

    # === ANALYSIS 3: Performance by Family Membership ===
    print("="*100)
    print("ANALYSIS 3: Performance by Classical Family Membership")
    print("="*100)
    print()

    family_summary = analyze_by_family(df_results)
    print(family_summary)
    print()

    # === ANALYSIS 4: Best Configurations by Domain ===
    print("="*100)
    print("ANALYSIS 4: Recommended Configurations by Application Domain")
    print("="*100)
    print()

    best_configs = find_best_configs_by_domain()
    print("Domain-specific recommendations:")
    print()
    for domain, config in best_configs.items():
        w_ang, w_rad, w_fam = WEIGHT_CONFIGS[config]
        print(f"  {domain.upper():12s}: {config:20s} (w_ang={w_ang:.2f}, w_rad={w_rad:.2f}, w_fam={w_fam:.2f})")
    print()

    # === ANALYSIS 5: Key Findings ===
    print("="*100)
    print("ANALYSIS 5: Key Findings")
    print("="*100)
    print()

    # Compare HI 1.0 vs HI 2.0 (balanced)
    hi_1_0_mean = df_results[df_results['config'] == 'HI_1.0_baseline']['HI_2.0'].mean()
    hi_2_0_mean = df_results[df_results['config'] == 'Balanced_default']['HI_2.0'].mean()
    improvement = ((hi_2_0_mean - hi_1_0_mean) / hi_1_0_mean) * 100

    print(f"1. HI 1.0 (E8-only) mean: {hi_1_0_mean:.4f}")
    print(f"   HI 2.0 (balanced) mean: {hi_2_0_mean:.4f}")
    print(f"   Overall improvement: {improvement:+.2f}%")
    print()

    # Identify tuples with biggest improvement
    baseline_df = df_results[df_results['config'] == 'HI_1.0_baseline'][['tuple_label', 'HI_2.0']].rename(columns={'HI_2.0': 'HI_1.0_value'})
    balanced_df = df_results[df_results['config'] == 'Balanced_default'][['tuple_label', 'HI_2.0']].rename(columns={'HI_2.0': 'HI_2.0_value'})
    comparison = baseline_df.merge(balanced_df, on='tuple_label')
    comparison['improvement'] = comparison['HI_2.0_value'] - comparison['HI_1.0_value']
    comparison = comparison.sort_values('improvement', ascending=False)

    print("2. Top 5 tuples with biggest HI 2.0 improvement:")
    for i, row in comparison.head(5).iterrows():
        print(f"   {row['tuple_label']:30s}: HI 1.0 = {row['HI_1.0_value']:.4f} → HI 2.0 = {row['HI_2.0_value']:.4f} ({row['improvement']:+.4f})")
    print()

    print("3. Bottom 5 tuples (least improvement or decrease):")
    for i, row in comparison.tail(5).iterrows():
        print(f"   {row['tuple_label']:30s}: HI 1.0 = {row['HI_1.0_value']:.4f} → HI 2.0 = {row['HI_2.0_value']:.4f} ({row['improvement']:+.4f})")
    print()

    # Primitive vs Female separation
    primitive_hi2 = df_results[(df_results['gender'] == 'Male (Primitive)') &
                               (df_results['config'] == 'High_radial')]['HI_2.0'].mean()
    female_hi2 = df_results[(df_results['gender'] == 'Female') &
                            (df_results['config'] == 'High_radial')]['HI_2.0'].mean()
    separation = primitive_hi2 - female_hi2

    print(f"4. Primitive vs Female separation (High Radial config):")
    print(f"   Primitive mean HI 2.0: {primitive_hi2:.4f}")
    print(f"   Female mean HI 2.0: {female_hi2:.4f}")
    print(f"   Separation: {separation:.4f} ({separation/primitive_hi2*100:.1f}% relative)")
    print()

    # Save summary JSON
    summary_json = {
        'overall_improvement_pct': float(improvement),
        'hi_1_0_mean': float(hi_1_0_mean),
        'hi_2_0_balanced_mean': float(hi_2_0_mean),
        'primitive_female_separation': float(separation),
        'best_configs_by_domain': best_configs,
        'weight_configurations': {k: list(v) for k, v in WEIGHT_CONFIGS.items()},
        'num_test_tuples': len(TEST_TUPLES),
        'num_configurations': len(WEIGHT_CONFIGS),
    }

    with open('hi_2_0_ablation_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    print("✓ Saved summary to hi_2_0_ablation_summary.json")
    print()
    print("="*100)
    print("ABLATION STUDY COMPLETE")
    print("="*100)
    print()
    print("Generated files:")
    print("  - hi_2_0_ablation_results.csv (full results)")
    print("  - hi_2_0_ablation_summary.json (key findings)")
    print()

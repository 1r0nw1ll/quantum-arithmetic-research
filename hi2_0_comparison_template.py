#!/usr/bin/env python3
"""
HI 2.0 vs HI 1.0 Comparison Table Generator

Compiles results from seismic and EEG experiments into publication-ready tables.
"""

import json
import pandas as pd

def load_results():
    """Load experimental results from JSON files."""
    results = {}

    try:
        with open('seismic_hi2_0_results.json', 'r') as f:
            results['seismic'] = json.load(f)
    except FileNotFoundError:
        print("⚠ Seismic results not yet available")
        results['seismic'] = None

    try:
        with open('eeg_hi2_0_results.json', 'r') as f:
            results['eeg'] = json.load(f)
    except FileNotFoundError:
        print("⚠ EEG results not yet available")
        results['eeg'] = None

    return results

def generate_comparison_table(results):
    """Generate LaTeX/Markdown comparison table."""

    table_data = []

    # Seismic results
    if results['seismic']:
        s = results['seismic']
        table_data.append({
            'Domain': 'Seismic',
            'Task': 'Earthquake vs Explosion',
            'HI_1.0_Acc': f"{s.get('hi1_accuracy', 0)*100:.1f}%",
            'HI_2.0_Acc': f"{s.get('hi2_accuracy', 0)*100:.1f}%",
            'Improvement': f"{(s.get('hi2_accuracy', 0) - s.get('hi1_accuracy', 0))*100:+.1f}%",
            'Config': 'Radial_family'
        })

    # EEG results
    if results['eeg']:
        e = results['eeg']
        table_data.append({
            'Domain': 'EEG',
            'Task': 'Seizure State Detection',
            'HI_1.0_Acc': f"{e.get('hi1_f1', 0)*100:.1f}%",
            'HI_2.0_Acc': f"{e.get('hi2_f1', 0)*100:.1f}%",
            'Improvement': f"{(e.get('hi2_f1', 0) - e.get('hi1_f1', 0))*100:+.1f}%",
            'Config': 'Angular_radial'
        })

    df = pd.DataFrame(table_data)

    # Markdown table
    print("\n" + "="*80)
    print("HI 1.0 vs HI 2.0 COMPARISON TABLE")
    print("="*80)
    print(df.to_markdown(index=False))

    # LaTeX table
    latex = df.to_latex(index=False, caption='HI 2.0 Performance Comparison',
                        label='tab:hi2_comparison')

    with open('hi2_0_comparison_table.tex', 'w') as f:
        f.write(latex)

    print("\n✓ Saved LaTeX table to: hi2_0_comparison_table.tex")

if __name__ == '__main__':
    results = load_results()
    if results['seismic'] or results['eeg']:
        generate_comparison_table(results)
    else:
        print("\n⏳ Waiting for experimental results...")

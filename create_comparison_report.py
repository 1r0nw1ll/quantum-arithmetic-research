#!/usr/bin/env python3
"""
Create comparison table and visualizations for QA vs baseline methods
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_baseline_results(dataset_name):
    """Load baseline results for a dataset"""
    results_dir = Path(f"results/{dataset_name.lower().replace(' ', '_')}")

    metrics_file = results_dir / "baseline_metrics.json"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)

def load_qa_results(dataset_name):
    """Load QA results for a dataset"""
    results_dir = Path(f"results/{dataset_name.lower().replace(' ', '_')}_qa")

    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)

def create_comparison_dataframe():
    """Create comprehensive comparison DataFrame"""
    data = []

    datasets = ['indian_pines', 'pavia_u']

    for dataset in datasets:
        # Load baseline results
        baseline = load_baseline_results(dataset)
        if baseline:
            # K-means raw
            if 'kmeans_raw' in baseline and 'error' not in baseline['kmeans_raw']:
                data.append({
                    'Dataset': dataset.replace('_', ' ').title(),
                    'Method': 'K-Means Raw',
                    'ARI': baseline['kmeans_raw']['ARI'],
                    'NMI': baseline['kmeans_raw']['NMI'],
                    'Runtime': baseline['kmeans_raw']['runtime'],
                    'Type': 'Baseline'
                })

            # K-means PCA
            if 'kmeans_pca' in baseline and 'error' not in baseline['kmeans_pca']:
                data.append({
                    'Dataset': dataset.replace('_', ' ').title(),
                    'Method': 'K-Means PCA',
                    'ARI': baseline['kmeans_pca']['ARI'],
                    'NMI': baseline['kmeans_pca']['NMI'],
                    'Runtime': baseline['kmeans_pca']['runtime'],
                    'Type': 'Baseline'
                })

        # Load QA results
        qa = load_qa_results(dataset)
        if qa:
            # QA K-means
            if 'kmeans_ari' in qa:
                data.append({
                    'Dataset': dataset.replace('_', ' ').title(),
                    'Method': 'QA Phase-Aware',
                    'ARI': qa['kmeans_ari'],
                    'NMI': qa['kmeans_nmi'],
                    'Runtime': None,  # Not tracked
                    'Type': 'QA Pipeline'
                })

    return pd.DataFrame(data)

def create_comparison_visualization(df):
    """Create bar chart comparison of methods"""
    datasets = df['Dataset'].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5))

    if len(datasets) == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        subset = df[df['Dataset'] == dataset]

        methods = subset['Method']
        ari_scores = subset['ARI']

        bars = axes[i].bar(methods, ari_scores, color=['skyblue', 'lightgreen', 'coral'])
        axes[i].set_title(f'{dataset} - ARI Scores')
        axes[i].set_ylabel('Adjusted Rand Index')
        axes[i].set_ylim(0, max(ari_scores) * 1.2)

        # Add value labels on bars
        for bar, score in zip(bars, ari_scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        '.3f', ha='center', va='bottom')

        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('results/comparison_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Comparison visualization saved to results/comparison_visualization.png")

def print_summary_table(df):
    """Print formatted summary table"""
    print("\n" + "="*80)
    print("HYPERSPECTRAL CLUSTERING COMPARISON SUMMARY")
    print("="*80)

    for dataset in df['Dataset'].unique():
        print(f"\n{dataset}:")
        print("-" * 40)
        subset = df[df['Dataset'] == dataset]
        for _, row in subset.iterrows():
            ari_str = ".3f" if pd.notna(row['ARI']) else "N/A"
            nmi_str = ".3f" if pd.notna(row['NMI']) else "N/A"
            runtime_str = ".2f" if pd.notna(row['Runtime']) else "N/A"
            print(f"  {row['Method']:<15} ARI: {ari_str:<6} NMI: {nmi_str:<6} Time: {runtime_str}s")

if __name__ == "__main__":
    # Create comparison DataFrame
    df = create_comparison_dataframe()

    if df.empty:
        print("No results found to compare")
        exit(1)

    # Save to CSV
    df.to_csv('results/comparison_table.csv', index=False)

    # Create visualization
    create_comparison_visualization(df)

    # Print summary
    print_summary_table(df)

    print("\nResults saved to results/comparison_table.csv")
    print("Visualization saved to results/comparison_visualization.png")
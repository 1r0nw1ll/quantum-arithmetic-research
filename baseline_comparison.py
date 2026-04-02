#!/usr/bin/env python3
"""
Baseline comparison for hyperspectral clustering
Implements K-means on raw spectra and K-means on PCA-reduced spectra
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from load_hyperspectral_dataset import load_dataset
import time
import json

def evaluate_clustering(true_labels, predicted_labels):
    """Compute clustering evaluation metrics"""
    # Only evaluate on labeled pixels
    mask = true_labels > 0
    if np.sum(mask) == 0:
        return {'ARI': 0.0, 'NMI': 0.0}

    true_labeled = true_labels[mask]
    pred_labeled = predicted_labels[mask]

    return {
        'ARI': adjusted_rand_score(true_labeled, pred_labeled),
        'NMI': normalized_mutual_info_score(true_labeled, pred_labeled)
    }

def baseline_kmeans_raw(spectra, n_clusters):
    """K-means clustering on raw spectra"""
    print(f"Running K-means on raw spectra (k={n_clusters})...")

    # Flatten spatial dimensions
    h, w, b = spectra.shape
    X = spectra.reshape(h * w, b)

    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    runtime = time.time() - start_time

    # Reshape back to spatial
    labels_2d = labels.reshape(h, w)

    return labels_2d, runtime

def baseline_kmeans_pca(spectra, n_clusters, n_components=10):
    """K-means clustering on PCA-reduced spectra"""
    print(f"Running K-means on PCA-reduced spectra (k={n_clusters}, components={n_components})...")

    # Flatten spatial dimensions
    h, w, b = spectra.shape
    X = spectra.reshape(h * w, b)

    start_time = time.time()

    # PCA reduction
    pca = PCA(n_components=min(n_components, b, X.shape[0]))
    X_pca = pca.fit_transform(X)

    # K-means on PCA features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    runtime = time.time() - start_time

    # Reshape back to spatial
    labels_2d = labels.reshape(h, w)

    return labels_2d, runtime, pca.explained_variance_ratio_

def run_baselines_on_dataset(dataset_name, output_dir, n_clusters=None, subsample_factor=4):
    """Run baseline comparisons on a dataset"""
    print(f"\n{'='*60}")
    print(f"Running baselines on {dataset_name}")
    print(f"{'='*60}")

    # Load data
    data, gt = load_dataset(dataset_name)

    # Subsample for computational efficiency
    if subsample_factor > 1:
        print(f"Subsampling data by factor {subsample_factor} for computational efficiency")
        data = data[::subsample_factor, ::subsample_factor, :]
        gt = gt[::subsample_factor, ::subsample_factor]
        print(f"Subsampled shape: {data.shape}")

    # Determine number of clusters
    if n_clusters is None:
        n_clusters = len(np.unique(gt)) - 1  # -1 for background
    print(f"Using {n_clusters} clusters")

    results = {}

    # Baseline 1: K-means on raw spectra
    try:
        labels_raw, time_raw = baseline_kmeans_raw(data, n_clusters)
        metrics_raw = evaluate_clustering(gt, labels_raw)
        results['kmeans_raw'] = {
            'labels': labels_raw,
            'runtime': time_raw,
            **metrics_raw
        }
        print(".3f")
    except Exception as e:
        print(f"K-means raw failed: {e}")
        results['kmeans_raw'] = {'error': str(e)}

    # Baseline 2: K-means on PCA-reduced spectra
    try:
        labels_pca, time_pca, explained_var = baseline_kmeans_pca(data, n_clusters)
        metrics_pca = evaluate_clustering(gt, labels_pca)
        results['kmeans_pca'] = {
            'labels': labels_pca,
            'runtime': time_pca,
            'explained_variance': explained_var.tolist(),
            **metrics_pca
        }
        print(".3f")
    except Exception as e:
        print(f"K-means PCA failed: {e}")
        results['kmeans_pca'] = {'error': str(e)}

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = {
        'dataset': dataset_name,
        'n_clusters': n_clusters,
        'kmeans_raw': {k: v for k, v in results['kmeans_raw'].items() if k != 'labels'},
        'kmeans_pca': {k: v for k, v in results['kmeans_pca'].items() if k != 'labels'}
    }

    with open(output_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(metrics_file, f, indent=2)

    # Save labels as numpy arrays
    if 'labels' in results['kmeans_raw']:
        np.save(output_dir / 'kmeans_raw_labels.npy', results['kmeans_raw']['labels'])
    if 'labels' in results['kmeans_pca']:
        np.save(output_dir / 'kmeans_pca_labels.npy', results['kmeans_pca']['labels'])

    print(f"Results saved to {output_dir}")

    return results

def create_comparison_table(results_indian, results_pavia):
    """Create comparison table across datasets and methods"""
    methods = ['kmeans_raw', 'kmeans_pca']

    table_data = []
    for dataset_name, results in [('Indian Pines', results_indian), ('PaviaU', results_pavia)]:
        for method in methods:
            if method in results and 'error' not in results[method]:
                row = {
                    'Dataset': dataset_name,
                    'Method': method.replace('_', ' ').title(),
                    'ARI': results[method]['ARI'],
                    'NMI': results[method]['NMI'],
                    'Runtime (s)': results[method]['runtime']
                }
                table_data.append(row)

    df = pd.DataFrame(table_data)
    return df

if __name__ == "__main__":
    # Run baselines on priority datasets
    print("Running baseline comparisons...")

    results_indian = run_baselines_on_dataset(
        'Indian_pines_corrected',
        'results/indian_pines',
        n_clusters=16,  # From inspection
        subsample_factor=4
    )

    results_pavia = run_baselines_on_dataset(
        'PaviaU',
        'results/pavia_u',
        n_clusters=9,  # From inspection
        subsample_factor=4
    )

    # Create comparison table
    comparison_df = create_comparison_table(results_indian, results_pavia)
    comparison_df.to_csv('results/comparison_table.csv', index=False)

    print("\nComparison table saved to results/comparison_table.csv")
    print("\n" + str(comparison_df))
#!/usr/bin/env python3
"""
Hyperspectral Dataset Loader
Helper script for loading MATLAB .mat hyperspectral datasets and ground truth.

Supports the standard hyperspectral benchmark datasets:
- Indian Pines
- Pavia University
- Kennedy Space Center
- Salinas Valley
"""

import scipy.io as sio
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

def load_dataset(name: str, data_dir: str = "hyperspectral_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load hyperspectral dataset and ground truth.

    Args:
        name: Dataset name (e.g., 'Indian_pines_corrected', 'PaviaU', 'KSC', 'Salinas_corrected')
        data_dir: Directory containing the .mat files

    Returns:
        Tuple of (data, ground_truth) arrays
    """
    data_file = Path(data_dir) / f"{name}.mat"

    # Handle different ground truth file naming conventions
    if name == 'Indian_pines_corrected':
        gt_file = Path(data_dir) / "Indian_pines_gt.mat"
    elif name == 'Salinas_corrected':
        gt_file = Path(data_dir) / "Salinas_gt.mat"
    else:
        gt_file = Path(data_dir) / f"{name}_gt.mat"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    # Load data (handle different key names)
    data_mat = sio.loadmat(str(data_file))
    gt_mat = sio.loadmat(str(gt_file))

    # Extract actual data (skip MATLAB metadata)
    data_keys = [k for k in data_mat.keys() if not k.startswith('__')]
    gt_keys = [k for k in gt_mat.keys() if not k.startswith('__')]

    if len(data_keys) != 1 or len(gt_keys) != 1:
        raise ValueError(f"Unexpected number of data keys in {name}: data={data_keys}, gt={gt_keys}")

    data = data_mat[data_keys[0]]
    gt = gt_mat[gt_keys[0]]

    return data, gt

def dataset_summary(name: str, data: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    """
    Generate comprehensive dataset statistics.

    Args:
        name: Dataset name
        data: Hyperspectral data array (H×W×B)
        gt: Ground truth array (H×W)

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'name': name,
        'shape': data.shape,
        'height': data.shape[0],
        'width': data.shape[1],
        'bands': data.shape[2],
        'total_pixels': data.shape[0] * data.shape[1],
        'ground_truth_shape': gt.shape,
        'classes': len(np.unique(gt)) - 1,  # -1 for background (usually 0)
        'labeled_pixels': np.sum(gt > 0),
        'unlabeled_pixels': np.sum(gt == 0),
        'labeling_ratio': np.sum(gt > 0) / (data.shape[0] * data.shape[1]),
        'spectral_range': [float(data.min()), float(data.max())],
        'data_type': str(data.dtype),
        'memory_mb': data.nbytes / (1024 * 1024)
    }

    # Class distribution (excluding background)
    unique_labels = np.unique(gt)
    class_counts = {}
    for label in unique_labels:
        if label > 0:  # Skip background
            count = np.sum(gt == label)
            class_counts[int(label)] = int(count)

    stats['class_distribution'] = class_counts

    return stats

def print_dataset_summary(name: str, data: np.ndarray, gt: np.ndarray):
    """
    Print formatted dataset statistics to console.
    """
    stats = dataset_summary(name, data, gt)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Dimensions: {stats['shape']} (H×W×Bands)")
    print(f"Ground truth: {stats['ground_truth_shape']}")
    print(f"Total pixels: {stats['total_pixels']:,}")
    print(f"Spectral bands: {stats['bands']}")
    print(f"Classes: {stats['classes']}")
    print(f"Labeled pixels: {stats['labeled_pixels']:,} ({stats['labeling_ratio']:.1%})")
    print(f"Unlabeled pixels: {stats['unlabeled_pixels']:,}")
    print(f"Spectral range: [{stats['spectral_range'][0]:.2f}, {stats['spectral_range'][1]:.2f}]")
    print(f"Data type: {stats['data_type']}")
    print(f"Memory usage: {stats['memory_mb']:.1f} MB")

    print(f"\nClass distribution:")
    for class_id, count in stats['class_distribution'].items():
        print(f"  Class {class_id}: {count:,} pixels")

def inspect_all_datasets(data_dir: str = "hyperspectral_data") -> Dict[str, Dict[str, Any]]:
    """
    Load and inspect all available hyperspectral datasets.

    Returns:
        Dictionary mapping dataset names to their statistics
    """
    datasets = ['Indian_pines_corrected', 'PaviaU', 'KSC', 'Salinas_corrected']
    results = {}

    print("Inspecting hyperspectral datasets...")
    print(f"Data directory: {data_dir}")

    for name in datasets:
        try:
            print(f"\nLoading {name}...")
            data, gt = load_dataset(name, data_dir)
            stats = dataset_summary(name, data, gt)
            results[name] = stats
            print_dataset_summary(name, data, gt)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            results[name] = {'error': str(e)}

    return results

if __name__ == "__main__":
    # Run inspection when script is executed directly
    results = inspect_all_datasets()

    # Save results to file
    import json

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj

    with open('dataset_inspection_report.json', 'w') as f:
        serializable_results = {name: make_serializable(stats) for name, stats in results.items()}
        json.dump(serializable_results, f, indent=2)

    print("\nDataset inspection complete. Results saved to dataset_inspection_report.json")
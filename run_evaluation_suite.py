#!/usr/bin/env python3
"""
Evaluation Suite Runner

Runs automated seizure detection evaluation suite based on YAML configuration.
This script orchestrates the entire evaluation pipeline:
1. Load configuration
2. Load and preprocess dataset
3. Run all configured methods
4. Generate comparison reports
5. Save results with timestamps and git info

Usage:
    python run_evaluation_suite.py [config_file]

    Default config: test_config.yaml

Example:
    python run_evaluation_suite.py
    python run_evaluation_suite.py custom_config.yaml

Author: Research Framework
Date: 2025-11-13
"""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import our framework
from seizure_detection_evaluator import SeizureDetectionEvaluator
from process_real_chbmit_data import RealEEGProcessor


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded")
    return config


def create_classifier(config: Dict[str, Any]):
    """Create classifier from configuration."""
    clf_type = config.get('type', 'RandomForest')
    params = config.get('params', {})

    if clf_type == 'RandomForest':
        return RandomForestClassifier(**params)
    elif clf_type == 'SVM':
        return SVC(**params)
    elif clf_type == 'LogisticRegression':
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


def main(config_path: str = "test_config.yaml"):
    """
    Main evaluation pipeline.

    Args:
        config_path: Path to YAML configuration file
    """
    print("="*80)
    print("SEIZURE DETECTION EVALUATION SUITE")
    print("="*80)
    print()

    # Load configuration
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print("Expected: test_config.yaml")
        return 1

    config = load_config(config_path)

    # Initialize evaluator
    output_dir = Path(config['output']['output_dir'])
    evaluator = SeizureDetectionEvaluator(
        data_dir=Path(config['dataset']['data_dir']),
        output_dir=output_dir,
        random_state=config['evaluation']['random_state']
    )

    # Initialize data processor
    processor = RealEEGProcessor(
        data_dir=Path(config['dataset']['data_dir'])
    )

    # Construct file paths
    data_dir = Path(config['dataset']['data_dir'])
    baseline_files = [
        data_dir / f for f in config['dataset']['baseline_files']
    ]
    seizure_files = [
        data_dir / f for f in config['dataset']['seizure_files']
    ]

    # Format seizure annotations
    seizure_annotations = config['dataset']['seizure_annotations']

    # Check if files exist
    print("\nValidating dataset files...")
    missing_files = []
    for f in baseline_files + seizure_files:
        if not f.exists():
            missing_files.append(f)
            print(f"  ✗ Missing: {f}")
        else:
            print(f"  ✓ Found: {f.name}")

    if missing_files:
        print(f"\n❌ ERROR: {len(missing_files)} file(s) not found!")
        print("Please check configuration and data directory.")
        return 1

    print("\n✓ All dataset files found")

    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    try:
        X, y = evaluator.load_dataset(
            baseline_files=baseline_files,
            seizure_files=seizure_files,
            seizure_annotations=seizure_annotations,
            feature_extractor=processor
        )

        # Set feature names
        evaluator.feature_names = config['features']['feature_names']

    except Exception as e:
        print(f"\n❌ ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Prepare methods for comparison
    print("\n" + "="*80)
    print("PREPARING METHODS")
    print("="*80)

    methods = []
    for method_config in config['methods']:
        print(f"  - {method_config['name']}: {method_config['description']}")

        # Create classifier
        clf = create_classifier(method_config['classifier'])

        methods.append({
            'name': method_config['name'],
            'description': method_config['description'],
            'classifier': clf,
            'balancing': method_config.get('balancing'),
            'balance_params': method_config.get('balance_params'),
            'metadata': {
                'classifier_type': method_config['classifier']['type'],
                'classifier_params': method_config['classifier']['params']
            }
        })

    print(f"\n✓ Prepared {len(methods)} methods for evaluation")

    # Run comparison
    print("\n" + "="*80)
    print("RUNNING EVALUATION SUITE")
    print("="*80)

    try:
        results = evaluator.compare_methods(
            methods=methods,
            test_size=config['evaluation']['test_size']
        )

    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    try:
        json_path = evaluator.save_results(
            include_git_info=config['output']['include_git_info']
        )

        # Generate report
        if config['output']['generate_report']:
            report = evaluator.generate_report(
                include_plots=config['output']['generate_plots']
            )

    except Exception as e:
        print(f"\n❌ ERROR saving results: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print()
    print("Results Summary:")
    print()

    # Find best method by F1
    best_result = max(results, key=lambda r: r.metrics.f1_score)
    print(f"Best Method (by F1): {best_result.method_name}")
    print(f"  F1 Score: {best_result.metrics.f1_score:.3f}")
    print(f"  Accuracy: {best_result.metrics.accuracy:.1%}")
    print(f"  Recall:   {best_result.metrics.recall:.1%}")
    print(f"  Precision: {best_result.metrics.precision:.1%}")
    print()

    print("All methods ranked by F1:")
    ranked = sorted(results, key=lambda r: r.metrics.f1_score, reverse=True)
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r.method_name:<25} F1={r.metrics.f1_score:.3f}")
    print()

    print(f"Output directory: {output_dir}")
    print(f"  - JSON results: {json_path.name}")
    if config['output']['generate_report']:
        report_files = list(output_dir.glob("evaluation_report_*.md"))
        if report_files:
            print(f"  - Report: {report_files[-1].name}")
    if config['output']['generate_plots']:
        plot_files = list(output_dir.glob("comparison_plots_*.png"))
        if plot_files:
            print(f"  - Plots: {plot_files[-1].name}")
    print()

    print("="*80)
    print("✓ EVALUATION SUITE COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    # Get config file from command line or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else "test_config.yaml"

    sys.exit(main(config_file))

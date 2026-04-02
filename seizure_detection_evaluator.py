#!/usr/bin/env python3
"""
Automated Testing and Evaluation Framework for Seizure Detection

This framework provides standardized, reproducible testing infrastructure for
comparing different seizure detection approaches on the CHB-MIT dataset.

Key Features:
- Consistent data loading and preprocessing
- Pluggable feature extractors and classifiers
- Comprehensive metric computation
- Multi-method comparison with statistical analysis
- Automated report generation with timestamps
- JSON export for reproducibility

Author: Research Framework
Date: 2025-11-13
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
import subprocess
import sys


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    npv: float  # Negative Predictive Value
    confusion_matrix: np.ndarray
    roc_auc: Optional[float] = None
    avg_precision: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['confusion_matrix'] = self.confusion_matrix.tolist()
        return d

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            "Evaluation Metrics:",
            f"  Accuracy:    {self.accuracy:.1%}",
            f"  Precision:   {self.precision:.1%}",
            f"  Recall:      {self.recall:.1%} (Sensitivity)",
            f"  F1-Score:    {self.f1_score:.3f}",
            f"  Specificity: {self.specificity:.1%}",
            f"  NPV:         {self.npv:.1%}",
        ]
        if self.roc_auc is not None:
            lines.append(f"  ROC AUC:     {self.roc_auc:.3f}")
        if self.avg_precision is not None:
            lines.append(f"  Avg Prec:    {self.avg_precision:.3f}")

        lines.append("\nConfusion Matrix:")
        lines.append("              Predicted")
        lines.append("            Baseline | Seizure")
        cm = self.confusion_matrix
        lines.append(f"  Baseline:  {cm[0,0]:6d}   | {cm[0,1]:6d}")
        lines.append(f"  Seizure:   {cm[1,0]:6d}   | {cm[1,1]:6d}")

        return "\n".join(lines)


@dataclass
class ExperimentResult:
    """Container for complete experiment results."""
    method_name: str
    description: str
    metrics: EvaluationMetrics
    feature_importance: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_pred_proba: Optional[np.ndarray] = None
    runtime_seconds: float = 0.0
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'method_name': self.method_name,
            'description': self.description,
            'metrics': self.metrics.to_dict(),
            'runtime_seconds': self.runtime_seconds,
        }
        if self.feature_importance is not None:
            d['feature_importance'] = self.feature_importance.tolist()
        if self.feature_names is not None:
            d['feature_names'] = self.feature_names
        if self.metadata is not None:
            d['metadata'] = self.metadata
        return d


class SeizureDetectionEvaluator:
    """
    Main evaluation framework for seizure detection experiments.

    Provides consistent pipeline for:
    1. Loading CHB-MIT EEG data
    2. Extracting features
    3. Training classifiers
    4. Computing comprehensive metrics
    5. Comparing multiple methods
    6. Generating reports
    """

    def __init__(self,
                 data_dir: Path = Path("phase2_data/eeg/chbmit"),
                 output_dir: Path = Path("phase2_workspace/evaluations"),
                 random_state: int = 42):
        """
        Initialize evaluator.

        Args:
            data_dir: Directory containing CHB-MIT data
            output_dir: Directory for saving results
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Loaded data
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.dataset_info: Dict = {}

        # Results storage
        self.results: List[ExperimentResult] = []

    def load_dataset(self,
                    baseline_files: List[Path],
                    seizure_files: List[Path],
                    seizure_annotations: Dict[str, List[Tuple[float, float]]],
                    feature_extractor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess EEG dataset.

        Args:
            baseline_files: List of EDF files without seizures
            seizure_files: List of EDF files containing seizures
            seizure_annotations: Dict mapping filename to list of (start, end) times
            feature_extractor: Object with .process_file() method

        Returns:
            (X, y) - Feature matrix and labels
        """
        import time
        start_time = time.time()

        print("="*80)
        print("LOADING DATASET")
        print("="*80)

        X_list = []
        y_list = []

        # Process baseline files
        print(f"\nProcessing {len(baseline_files)} baseline files...")
        for i, filepath in enumerate(baseline_files):
            print(f"  [{i+1}/{len(baseline_files)}] {filepath.name}")
            results = feature_extractor.process_file(filepath, seizure_times=[])
            X_list.append(results['features_7d'])
            y_list.append(results['labels'])

        # Process seizure files
        print(f"\nProcessing {len(seizure_files)} seizure files...")
        for i, filepath in enumerate(seizure_files):
            print(f"  [{i+1}/{len(seizure_files)}] {filepath.name}")
            filename = filepath.name
            seizure_times = seizure_annotations.get(filename, [])
            print(f"    Seizure annotations: {seizure_times}")
            results = feature_extractor.process_file(filepath, seizure_times=seizure_times)
            X_list.append(results['features_7d'])
            y_list.append(results['labels'])

        # Combine
        self.X = np.vstack(X_list)
        self.y = np.hstack(y_list)

        # Store dataset info
        self.dataset_info = {
            'baseline_files': [str(f) for f in baseline_files],
            'seizure_files': [str(f) for f in seizure_files],
            'seizure_annotations': {k: v for k, v in seizure_annotations.items()},
            'total_samples': int(len(self.y)),
            'baseline_samples': int(np.sum(self.y == 0)),
            'seizure_samples': int(np.sum(self.y == 1)),
            'class_balance': float(np.sum(self.y == 1) / len(self.y)),
            'feature_dim': int(self.X.shape[1]),
            'load_time_seconds': time.time() - start_time
        }

        print(f"\n" + "="*80)
        print("DATASET LOADED")
        print("="*80)
        print(f"Total samples:    {self.dataset_info['total_samples']}")
        print(f"  Baseline:       {self.dataset_info['baseline_samples']}")
        print(f"  Seizure:        {self.dataset_info['seizure_samples']}")
        print(f"  Class balance:  {self.dataset_info['class_balance']:.1%}")
        print(f"Feature dim:      {self.dataset_info['feature_dim']}")
        print(f"Load time:        {self.dataset_info['load_time_seconds']:.1f}s")

        if self.dataset_info['seizure_samples'] < 5:
            print("\n⚠️  WARNING: Very few seizure samples!")
            print("   Results may not be reliable.")

        return self.X, self.y

    def extract_features(self,
                        feature_extractor_fn: Callable,
                        feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply additional feature extraction/transformation.

        Args:
            feature_extractor_fn: Function that takes X and returns transformed X
            feature_names: Names for the features

        Returns:
            Transformed feature matrix
        """
        print("\nApplying feature transformation...")
        self.X = feature_extractor_fn(self.X)
        if feature_names is not None:
            self.feature_names = feature_names
        print(f"New feature shape: {self.X.shape}")
        return self.X

    def train_classifier(self,
                        clf,
                        test_size: float = 0.2,
                        balancing_method: Optional[str] = None,
                        balance_params: Optional[Dict] = None) -> Tuple:
        """
        Train classifier with optional class balancing.

        Args:
            clf: Classifier object (sklearn-compatible)
            test_size: Fraction of data for testing
            balancing_method: 'oversample', 'undersample', 'smote', or None
            balance_params: Parameters for balancing method

        Returns:
            (clf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba)
        """
        import time
        start_time = time.time()

        if self.X is None or self.y is None:
            raise ValueError("Must load dataset first!")

        print("\n" + "="*80)
        print("TRAINING CLASSIFIER")
        print("="*80)
        print(f"Classifier: {clf.__class__.__name__}")
        print(f"Test size: {test_size}")
        print(f"Balancing: {balancing_method or 'None'}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y if len(np.unique(self.y)) > 1 else None
        )

        print(f"\nTrain: {len(y_train)} samples (seizure: {np.sum(y_train==1)})")
        print(f"Test:  {len(y_test)} samples (seizure: {np.sum(y_test==1)})")

        # Apply balancing if requested
        if balancing_method is not None:
            X_train, y_train = self._apply_balancing(
                X_train, y_train, balancing_method, balance_params
            )

        # Train
        print(f"\nTraining {clf.__class__.__name__}...")
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Get probabilities if available
        y_pred_proba = None
        if hasattr(clf, 'predict_proba'):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

        runtime = time.time() - start_time
        print(f"Training complete in {runtime:.2f}s")

        return clf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, runtime

    def _apply_balancing(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        method: str,
                        params: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply class balancing method."""
        print(f"\nApplying {method} balancing...")
        print(f"Before: {len(y)} samples (seizure: {np.sum(y==1)})")

        if method == 'oversample':
            # Simple random oversampling
            from sklearn.utils import resample

            X_minority = X[y == 1]
            y_minority = y[y == 1]
            X_majority = X[y == 0]
            y_majority = y[y == 0]

            X_minority_upsampled, y_minority_upsampled = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=len(y_majority),
                random_state=self.random_state
            )

            X = np.vstack([X_majority, X_minority_upsampled])
            y = np.hstack([y_majority, y_minority_upsampled])

        elif method == 'undersample':
            # Simple random undersampling
            from sklearn.utils import resample

            X_minority = X[y == 1]
            y_minority = y[y == 1]
            X_majority = X[y == 0]
            y_majority = y[y == 0]

            X_majority_downsampled, y_majority_downsampled = resample(
                X_majority, y_majority,
                replace=False,
                n_samples=len(y_minority),
                random_state=self.random_state
            )

            X = np.vstack([X_majority_downsampled, X_minority])
            y = np.hstack([y_majority_downsampled, y_minority])

        elif method == 'smote':
            # SMOTE requires imblearn
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state, **(params or {}))
                X, y = smote.fit_resample(X, y)
            except ImportError:
                print("⚠️  SMOTE requires imbalanced-learn. Falling back to oversample.")
                return self._apply_balancing(X, y, 'oversample', None)
        else:
            raise ValueError(f"Unknown balancing method: {method}")

        print(f"After:  {len(y)} samples (seizure: {np.sum(y==1)})")
        return X, y

    def evaluate(self,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                y_pred_proba: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            EvaluationMetrics object
        """
        # Basic metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Specificity and NPV
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # ROC AUC and Average Precision (if probabilities available)
        roc_auc_score = None
        avg_prec = None
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            roc_auc_score = auc(*roc_curve(y_true, y_pred_proba)[:2][::-1])
            avg_prec = average_precision_score(y_true, y_pred_proba)

        return EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            specificity=specificity,
            npv=npv,
            confusion_matrix=cm,
            roc_auc=roc_auc_score,
            avg_precision=avg_prec
        )

    def run_experiment(self,
                      method_name: str,
                      description: str,
                      clf,
                      test_size: float = 0.2,
                      balancing_method: Optional[str] = None,
                      balance_params: Optional[Dict] = None,
                      metadata: Optional[Dict] = None) -> ExperimentResult:
        """
        Run complete experiment and store results.

        Args:
            method_name: Name of the method (e.g., "RF_baseline")
            description: Human-readable description
            clf: Classifier object
            test_size: Test set fraction
            balancing_method: Optional balancing method
            balance_params: Parameters for balancing
            metadata: Additional metadata to store

        Returns:
            ExperimentResult object
        """
        print("\n" + "="*80)
        print(f"EXPERIMENT: {method_name}")
        print("="*80)
        print(f"Description: {description}")

        # Train
        clf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, runtime = \
            self.train_classifier(clf, test_size, balancing_method, balance_params)

        # Evaluate
        metrics = self.evaluate(y_test, y_pred, y_pred_proba)

        # Feature importance
        feature_importance = None
        if hasattr(clf, 'feature_importances_'):
            feature_importance = clf.feature_importances_

        # Create result
        result = ExperimentResult(
            method_name=method_name,
            description=description,
            metrics=metrics,
            feature_importance=feature_importance,
            feature_names=self.feature_names,
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            runtime_seconds=runtime,
            metadata=metadata or {}
        )

        # Store
        self.results.append(result)

        # Print metrics
        print("\n" + str(metrics))

        if feature_importance is not None and self.feature_names is not None:
            print("\nFeature Importance:")
            for name, imp in zip(self.feature_names, feature_importance):
                print(f"  {name:10s}: {imp:.3f}")

        return result

    def compare_methods(self,
                       methods: List[Dict[str, Any]],
                       test_size: float = 0.2) -> List[ExperimentResult]:
        """
        Run and compare multiple methods.

        Args:
            methods: List of method specifications, each with:
                - name: Method name
                - description: Description
                - classifier: Classifier object
                - balancing: Optional balancing method
                - balance_params: Optional balancing parameters
                - metadata: Optional metadata
            test_size: Test set fraction

        Returns:
            List of ExperimentResult objects
        """
        print("\n" + "="*80)
        print(f"COMPARING {len(methods)} METHODS")
        print("="*80)

        results = []
        for i, method_spec in enumerate(methods):
            print(f"\n[{i+1}/{len(methods)}]")

            result = self.run_experiment(
                method_name=method_spec['name'],
                description=method_spec['description'],
                clf=method_spec['classifier'],
                test_size=test_size,
                balancing_method=method_spec.get('balancing'),
                balance_params=method_spec.get('balance_params'),
                metadata=method_spec.get('metadata')
            )
            results.append(result)

        # Print comparison table
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Method':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Time':<8}")
        print("-"*80)
        for r in results:
            print(f"{r.method_name:<20} "
                  f"{r.metrics.accuracy:<8.1%} "
                  f"{r.metrics.precision:<8.1%} "
                  f"{r.metrics.recall:<8.1%} "
                  f"{r.metrics.f1_score:<8.3f} "
                  f"{r.runtime_seconds:<8.2f}s")

        return results

    def save_results(self,
                    filename: Optional[str] = None,
                    include_git_info: bool = True) -> Path:
        """
        Save all results to JSON file.

        Args:
            filename: Output filename (auto-generated if None)
            include_git_info: Include git commit hash if available

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

        output_path = self.output_dir / filename

        # Prepare output
        output = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.dataset_info,
            'random_state': self.random_state,
            'results': [r.to_dict() for r in self.results]
        }

        # Add git info if requested
        if include_git_info:
            try:
                git_hash = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=Path(__file__).parent,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                output['git_commit'] = git_hash
            except:
                output['git_commit'] = None

        # Save
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✓ Results saved to {output_path}")
        return output_path

    def generate_report(self,
                       output_path: Optional[Path] = None,
                       include_plots: bool = True) -> str:
        """
        Generate markdown report summarizing all results.

        Args:
            output_path: Path for markdown file (auto-generated if None)
            include_plots: Whether to generate and include plots

        Returns:
            Markdown report as string
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"evaluation_report_{timestamp}.md"

        lines = []
        lines.append("# Seizure Detection Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Dataset info
        lines.append("## Dataset")
        lines.append("")
        lines.append(f"- **Total samples:** {self.dataset_info['total_samples']}")
        lines.append(f"- **Baseline samples:** {self.dataset_info['baseline_samples']}")
        lines.append(f"- **Seizure samples:** {self.dataset_info['seizure_samples']}")
        lines.append(f"- **Class balance:** {self.dataset_info['class_balance']:.1%}")
        lines.append(f"- **Feature dimension:** {self.dataset_info['feature_dim']}")
        lines.append("")
        lines.append("**Files:**")
        for f in self.dataset_info['baseline_files']:
            lines.append(f"- {Path(f).name} (baseline)")
        for f in self.dataset_info['seizure_files']:
            lines.append(f"- {Path(f).name} (seizure)")
        lines.append("")

        # Results
        lines.append("## Results")
        lines.append("")
        lines.append("| Method | Accuracy | Precision | Recall | F1 | Specificity | NPV | Time (s) |")
        lines.append("|--------|----------|-----------|--------|-------|-------------|-----|----------|")

        for r in self.results:
            m = r.metrics
            lines.append(
                f"| {r.method_name} | "
                f"{m.accuracy:.1%} | "
                f"{m.precision:.1%} | "
                f"{m.recall:.1%} | "
                f"{m.f1_score:.3f} | "
                f"{m.specificity:.1%} | "
                f"{m.npv:.1%} | "
                f"{r.runtime_seconds:.2f} |"
            )
        lines.append("")

        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")

        for r in self.results:
            lines.append(f"### {r.method_name}")
            lines.append("")
            lines.append(f"**Description:** {r.description}")
            lines.append("")
            lines.append("**Metrics:**")
            lines.append("")
            lines.append(f"- Accuracy: {r.metrics.accuracy:.1%}")
            lines.append(f"- Precision: {r.metrics.precision:.1%}")
            lines.append(f"- Recall: {r.metrics.recall:.1%} (Sensitivity)")
            lines.append(f"- F1-Score: {r.metrics.f1_score:.3f}")
            lines.append(f"- Specificity: {r.metrics.specificity:.1%}")
            lines.append(f"- NPV: {r.metrics.npv:.1%}")
            if r.metrics.roc_auc:
                lines.append(f"- ROC AUC: {r.metrics.roc_auc:.3f}")
            if r.metrics.avg_precision:
                lines.append(f"- Avg Precision: {r.metrics.avg_precision:.3f}")
            lines.append("")

            lines.append("**Confusion Matrix:**")
            lines.append("")
            cm = r.metrics.confusion_matrix
            lines.append("|  | Pred: Baseline | Pred: Seizure |")
            lines.append("|--|----------------|---------------|")
            lines.append(f"| **True: Baseline** | {cm[0,0]} | {cm[0,1]} |")
            if cm.shape[0] > 1:
                lines.append(f"| **True: Seizure** | {cm[1,0]} | {cm[1,1]} |")
            lines.append("")

            if r.feature_importance is not None and r.feature_names is not None:
                lines.append("**Feature Importance:**")
                lines.append("")
                for name, imp in zip(r.feature_names, r.feature_importance):
                    lines.append(f"- {name}: {imp:.3f}")
                lines.append("")

        # Generate plots if requested
        if include_plots:
            plot_path = self._generate_comparison_plots()
            if plot_path:
                lines.append("## Visualizations")
                lines.append("")
                lines.append(f"![Comparison Plots]({plot_path.name})")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by SeizureDetectionEvaluator*")

        report = "\n".join(lines)

        # Save
        with open(output_path, 'w') as f:
            f.write(report)

        print(f"\n✓ Report saved to {output_path}")

        return report

    def _generate_comparison_plots(self) -> Optional[Path]:
        """Generate comparison plots for all results."""
        if len(self.results) == 0:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"comparison_plots_{timestamp}.png"

        n_methods = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: Metrics comparison
        ax = axes[0, 0]
        methods = [r.method_name for r in self.results]
        metrics_data = {
            'Accuracy': [r.metrics.accuracy for r in self.results],
            'Precision': [r.metrics.precision for r in self.results],
            'Recall': [r.metrics.recall for r in self.results],
            'F1': [r.metrics.f1_score for r in self.results]
        }

        x = np.arange(len(methods))
        width = 0.2
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax.bar(x + i*width, values, width, label=metric)

        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])

        # Panel B: Confusion matrices
        ax = axes[0, 1]
        for i, r in enumerate(self.results):
            cm = r.metrics.confusion_matrix
            if cm.shape == (2, 2):
                # Normalize
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                # Plot only the first method's CM in detail
                if i == 0:
                    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                              xticklabels=['Baseline', 'Seizure'],
                              yticklabels=['Baseline', 'Seizure'],
                              ax=ax, cbar_kws={'label': 'Proportion'})
                    ax.set_title(f'Confusion Matrix: {r.method_name}')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')

        # Panel C: ROC curves (if available)
        ax = axes[1, 0]
        has_roc = False
        for r in self.results:
            if r.y_true is not None and r.y_pred_proba is not None:
                if len(np.unique(r.y_true)) > 1:
                    fpr, tpr, _ = roc_curve(r.y_true, r.y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{r.method_name} (AUC={roc_auc:.3f})')
                    has_roc = True

        if has_roc:
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No probability predictions available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ROC Curves (N/A)')

        # Panel D: Feature importance (first method)
        ax = axes[1, 1]
        if self.results[0].feature_importance is not None:
            imp = self.results[0].feature_importance
            names = self.results[0].feature_names or [f'F{i}' for i in range(len(imp))]
            sorted_idx = np.argsort(imp)
            ax.barh(range(len(imp)), imp[sorted_idx])
            ax.set_yticks(range(len(imp)))
            ax.set_yticklabels([names[i] for i in sorted_idx])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance: {self.results[0].method_name}')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'No feature importance available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance (N/A)')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Comparison plots saved to {plot_path}")
        return plot_path


if __name__ == "__main__":
    print(__doc__)
    print("\nThis is a framework module. See run_evaluation_suite.py for usage examples.")

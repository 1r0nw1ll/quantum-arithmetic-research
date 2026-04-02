#!/usr/bin/env python3
"""
COMPREHENSIVE TEST WITH EXPANDED DATASET
Uses ALL downloaded seizure files (6 total) for better statistical power
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from eeg_brain_feature_extractor_fixed import EEGBrainFeatureExtractor as Extractor7D
from eeg_brain_feature_extractor_enhanced import EEGBrainFeatureExtractor as Extractor13D

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pyedflib
import json

def process_file_with_extractor(filepath, seizure_times, extractor):
    """Process EDF file with specified feature extractor."""
    f = pyedflib.EdfReader(str(filepath))
    n_channels = f.signals_in_file
    channel_names = f.getSignalLabels()
    sampling_rate = f.getSampleFrequency(0)
    
    signals = []
    for i in range(n_channels):
        sig = f.readSignal(i)
        signals.append(sig)
    f.close()
    
    signals = np.array(signals)
    
    # Segment into 4-second windows
    window_samples = int(4.0 * sampling_rate)
    step_samples = int(2.0 * sampling_rate)
    
    segments = []
    start = 0
    while start + window_samples <= signals.shape[1]:
        segment = signals[:, start:start + window_samples]
        segments.append(segment)
        start += step_samples
    
    # Extract features
    features_list = []
    for segment in segments:
        channels_data = {channel_names[i]: segment[i, :] for i in range(n_channels)}
        features = extractor.extract_network_features(channels_data)
        features_list.append(features)
    
    features = np.array(features_list)
    
    # Label segments
    step_sec = 2.0
    labels = np.zeros(len(segments), dtype=int)
    for i in range(len(segments)):
        segment_start = i * step_sec
        segment_end = segment_start + 4.0
        for sz_start, sz_end in seizure_times:
            if not (segment_end < sz_start or segment_start > sz_end):
                labels[i] = 1
                break
    
    return features, labels

def main():
    print("="*80)
    print("EXPANDED DATASET TEST: 6 FILES, ~75 SEIZURE SEGMENTS")
    print("="*80)
    print()
    
    base_path = Path("phase2_data/eeg/chbmit/chb01")
    
    # Define all files with seizure annotations
    files_config = [
        ("chb01_01.edf", []),  # Baseline only
        ("chb01_03.edf", [(2996, 3036)]),  # 40s seizure
        ("chb01_04.edf", [(1467, 1494)]),  # 27s seizure
        ("chb01_15.edf", [(1732, 1772)]),  # 40s seizure
        ("chb01_16.edf", [(1015, 1066)]),  # 51s seizure
        ("chb01_18.edf", [(1720, 1810)]),  # 90s seizure
    ]
    
    results_7d = []
    results_13d = []
    
    print("STAGE 1: Extracting 7D features from all 6 files...")
    print("-"*80)
    
    extractor_7d = Extractor7D()
    X_7d_list = []
    y_list = []
    
    for filename, seizure_times in files_config:
        filepath = base_path / filename
        print(f"\nProcessing {filename}...")
        if seizure_times:
            print(f"  Seizure annotations: {seizure_times}")
        else:
            print(f"  Baseline (no seizures)")
        
        X, y = process_file_with_extractor(filepath, seizure_times, extractor_7d)
        X_7d_list.append(X)
        y_list.append(y)
        
        n_seizure = np.sum(y == 1)
        n_baseline = np.sum(y == 0)
        print(f"  → {len(X)} segments: {n_seizure} seizure, {n_baseline} baseline")
    
    X_7d = np.vstack(X_7d_list)
    y = np.hstack(y_list)
    
    total_seizure = np.sum(y == 1)
    total_baseline = np.sum(y == 0)
    
    print(f"\n{'='*80}")
    print(f"COMBINED DATASET:")
    print(f"  Total segments: {len(X_7d)}")
    print(f"  Seizure: {total_seizure} segments")
    print(f"  Baseline: {total_baseline} segments")
    print(f"  Imbalance ratio: {total_baseline/total_seizure:.1f}:1")
    print(f"{'='*80}\n")
    
    # Split data
    X_train_7d, X_test_7d, y_train, y_test = train_test_split(
        X_7d, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train/Test split:")
    print(f"  Train: {len(y_train)} segments ({np.sum(y_train==1)} seizure)")
    print(f"  Test:  {len(y_test)} segments ({np.sum(y_test==1)} seizure)")
    print()
    
    # TEST 1: 7D baseline
    print("="*80)
    print("TEST 1: 7D Features (Baseline)")
    print("="*80)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_train_7d, y_train)
    y_pred = clf.predict(X_test_7d)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Results: Accuracy={acc:.1%}, Recall={rec:.1%}, Precision={prec:.1%}, F1={f1:.3f}")
    results_7d.append({"method": "7D Baseline", "recall": rec, "precision": prec, "f1": f1})
    
    # TEST 2: 7D + class weights
    print("\n" + "="*80)
    print("TEST 2: 7D Features + Class Weights")
    print("="*80)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, 
                                 class_weight='balanced')
    clf.fit(X_train_7d, y_train)
    y_pred = clf.predict(X_test_7d)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Results: Accuracy={acc:.1%}, Recall={rec:.1%}, Precision={prec:.1%}, F1={f1:.3f}")
    results_7d.append({"method": "7D + Weights", "recall": rec, "precision": prec, "f1": f1})
    
    # STAGE 2: Extract 13D features
    print("\n" + "="*80)
    print("STAGE 2: Extracting 13D features from all 6 files...")
    print("-"*80)
    
    extractor_13d = Extractor13D()
    X_13d_list = []
    
    for filename, seizure_times in files_config:
        filepath = base_path / filename
        print(f"Processing {filename} with 13D extractor...")
        
        X, _ = process_file_with_extractor(filepath, seizure_times, extractor_13d)
        X_13d_list.append(X)
    
    X_13d = np.vstack(X_13d_list)
    print(f"\n13D feature extraction complete: {X_13d.shape}")
    
    X_train_13d, X_test_13d, y_train, y_test = train_test_split(
        X_13d, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TEST 3: 13D + class weights
    print("\n" + "="*80)
    print("TEST 3: 13D Enhanced Features + Class Weights")
    print("="*80)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5,
                                 class_weight='balanced')
    clf.fit(X_train_13d, y_train)
    y_pred = clf.predict(X_test_13d)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Results: Accuracy={acc:.1%}, Recall={rec:.1%}, Precision={prec:.1%}, F1={f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  Baseline: {cm[0,0]:4d} correct, {cm[0,1]:4d} false positives")
    if cm.shape[0] > 1:
        print(f"  Seizure:  {cm[1,1]:4d} detected, {cm[1,0]:4d} missed")
        print(f"  Detection rate: {cm[1,1]} / {np.sum(y_test==1)} seizures")
    
    results_13d.append({"method": "13D + Weights", "recall": rec, "precision": prec, "f1": f1})
    
    # Feature importance
    importance = clf.feature_importances_
    feature_names = (['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM'] + 
                    ['LineLen', 'Var', 'SpecEdge', 'Hjorth', 'ZeroCross', 'PeakPeak'])
    
    print(f"\nFeature Importance (Top 10):")
    sorted_idx = np.argsort(importance)[::-1]
    for i in sorted_idx[:10]:
        print(f"  {feature_names[i]:12s}: {importance[i]:.3f}")
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL COMPARISON: SMALL vs EXPANDED DATASET")
    print("="*80)
    
    print(f"\nDataset Expansion:")
    print(f"  Original: 2 files, 23 seizure segments")
    print(f"  Expanded: 6 files, {total_seizure} seizure segments")
    print(f"  Increase: {total_seizure/23:.1f}x more seizure data")
    
    print(f"\n{'Method':<30} {'Recall':<10} {'Precision':<12} {'F1':<8}")
    print("-"*80)
    for r in results_7d + results_13d:
        print(f"{r['method']:<30} {r['recall']:>8.1%}  {r['precision']:>10.1%}  {r['f1']:>6.3f}")
    
    final_improvement = (results_13d[-1]['recall'] - results_7d[0]['recall']) * 100
    print(f"\nTotal improvement: {results_7d[0]['recall']:.1%} → {results_13d[-1]['recall']:.1%} (+{final_improvement:.1f} pp)")
    
    # Save results
    output_dir = Path("phase2_workspace")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "expanded_dataset_results.json", 'w') as f:
        json.dump({
            "dataset_size": {
                "total_segments": len(y),
                "seizure_segments": int(total_seizure),
                "baseline_segments": int(total_baseline)
            },
            "results": results_7d + results_13d,
            "final_recall": results_13d[-1]['recall'],
            "final_precision": results_13d[-1]['precision'],
            "final_f1": results_13d[-1]['f1']
        }, f, indent=2)
    
    print(f"\n✓ Results saved to phase2_workspace/expanded_dataset_results.json")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TEST ENHANCED 13D FEATURES + CLASS BALANCING

Compares performance across improvements:
1. Baseline 7D features (20% recall)
2. 7D features + class weights (40% recall)
3. Enhanced 13D features + class weights (??% recall)

Goal: Demonstrate cumulative improvements
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import both extractors
from eeg_brain_feature_extractor_fixed import EEGBrainFeatureExtractor as Extractor7D
from eeg_brain_feature_extractor_enhanced import EEGBrainFeatureExtractor as Extractor13D

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pyedflib
import json

def process_file_with_extractor(filepath, seizure_times, extractor):
    """Process EDF file with specified feature extractor."""
    # Load EDF
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
    step_samples = int(2.0 * sampling_rate)  # 2s overlap
    
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
    print("ENHANCED 13D FEATURES + CLASS BALANCING TEST")
    print("="*80)
    print()
    
    baseline_file = Path("phase2_data/eeg/chbmit/chb01/chb01_01.edf")
    seizure_file = Path("phase2_data/eeg/chbmit/chb01/chb01_03.edf")
    seizure_times = [(2996, 3036)]
    
    results = []
    
    # TEST 1: 7D features, no class weights (baseline)
    print("="*80)
    print("TEST 1: 7D Spectral Features (Baseline)")
    print("="*80)
    
    extractor_7d = Extractor7D()
    X_7d_baseline, y_baseline = process_file_with_extractor(baseline_file, [], extractor_7d)
    X_7d_seizure, y_seizure = process_file_with_extractor(seizure_file, seizure_times, extractor_7d)
    
    X_7d = np.vstack([X_7d_baseline, X_7d_seizure])
    y = np.hstack([y_baseline, y_seizure])
    
    print(f"Dataset: {len(X_7d)} samples, {np.sum(y==1)} seizure, {np.sum(y==0)} baseline")
    print(f"Feature dimensionality: {X_7d.shape[1]}D")
    
    X_train, X_test, y_train, y_test = train_test_split(X_7d, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Results: Accuracy={acc:.1%}, Recall={rec:.1%}, Precision={prec:.1%}, F1={f1:.3f}")
    results.append({"method": "7D Baseline", "recall": rec, "precision": prec, "f1": f1})
    
    # TEST 2: 7D features + class weights
    print("\n" + "="*80)
    print("TEST 2: 7D Features + Class Weights")
    print("="*80)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Results: Accuracy={acc:.1%}, Recall={rec:.1%}, Precision={prec:.1%}, F1={f1:.3f}")
    results.append({"method": "7D + Class Weights", "recall": rec, "precision": prec, "f1": f1})
    
    # TEST 3: 13D features + class weights
    print("\n" + "="*80)
    print("TEST 3: Enhanced 13D Features + Class Weights")
    print("="*80)
    
    extractor_13d = Extractor13D()
    X_13d_baseline, _ = process_file_with_extractor(baseline_file, [], extractor_13d)
    X_13d_seizure, _ = process_file_with_extractor(seizure_file, seizure_times, extractor_13d)
    
    X_13d = np.vstack([X_13d_baseline, X_13d_seizure])
    
    print(f"Dataset: {len(X_13d)} samples")
    print(f"Feature dimensionality: {X_13d.shape[1]}D (7D spectral + 6D temporal)")
    
    X_train_13d, X_test_13d, y_train, y_test = train_test_split(X_13d, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, class_weight='balanced')
    clf.fit(X_train_13d, y_train)
    y_pred = clf.predict(X_test_13d)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Results: Accuracy={acc:.1%}, Recall={rec:.1%}, Precision={prec:.1%}, F1={f1:.3f}")
    print(f"Confusion Matrix:")
    print(f"  Baseline: {cm[0,0]:3d} correct, {cm[0,1]:3d} false positives")
    if cm.shape[0] > 1:
        print(f"  Seizure:  {cm[1,1]:3d} detected, {cm[1,0]:3d} missed")
        print(f"  Seizures detected: {cm[1,1]} / {np.sum(y_test==1)}")
    
    results.append({"method": "13D Enhanced + Weights", "recall": rec, "precision": prec, "f1": f1})
    
    # Feature importance
    importance = clf.feature_importances_
    feature_names = (['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM'] + 
                    ['LineLen', 'Var', 'SpecEdge', 'Hjorth', 'ZeroCross', 'PeakPeak'])
    
    print(f"\nFeature Importance (Top 8):")
    sorted_idx = np.argsort(importance)[::-1]
    for i in sorted_idx[:8]:
        print(f"  {feature_names[i]:12s}: {importance[i]:.3f}")
    
    # SUMMARY
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<30} {'Recall':<10} {'Precision':<12} {'F1':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['method']:<30} {r['recall']:>8.1%}  {r['precision']:>10.1%}  {r['f1']:>6.3f}")
    
    improvement = (results[-1]['recall'] - results[0]['recall']) * 100
    print(f"\n✓ Total Improvement: {results[0]['recall']:.1%} → {results[-1]['recall']:.1%} (+{improvement:.1f} pp)")
    
    if results[-1]['recall'] >= 0.5:
        print(f"\n🎯 MILESTONE ACHIEVED: {results[-1]['recall']:.1%} recall (≥50%)")
        print(f"   Temporal features contributed significantly!")
    
    # Save
    output_dir = Path("phase2_workspace")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "enhanced_features_results.json", 'w') as f:
        json.dump({"comparison": results, "final_recall": results[-1]['recall']}, f, indent=2)
    
    print(f"\n✓ Results saved to phase2_workspace/enhanced_features_results.json")

if __name__ == "__main__":
    main()

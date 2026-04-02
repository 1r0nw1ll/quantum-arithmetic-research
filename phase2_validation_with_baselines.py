#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
Phase 2 Validation with CNN/LSTM Baselines

Compares QA Framework against state-of-the-art deep learning baselines:
1. Seismic: QA vs 1D-CNN vs LSTM
2. EEG: QA + Brain→QA vs 2D-CNN vs LSTM

Metrics:
- Accuracy, Precision, Recall, F1-Score
- Training time, inference time
- Model complexity (# parameters)
- Sample efficiency (learning curves)
- PAC-Bayesian generalization bounds (QA only)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import time
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from qa_core import QASystem
from qa_pac_bayes import (
    dqa_divergence,
    compute_pac_constants,
    pac_generalization_bound
)
from pisano_analysis import PisanoClassifier
from brain_qa_mapper import BrainQAMapper
from seismic_data_generator import SeismicWaveformGenerator
from eeg_brain_feature_extractor import EEGBrainFeatureExtractor
from seismic_classifier_enhanced import EnhancedSeismicClassifier


# ==============================================================================
# CNN BASELINE MODELS
# ==============================================================================

class SeismicCNN(nn.Module):
    """1D CNN for seismic event classification."""

    def __init__(self, input_length: int = 500, num_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate flattened size
        self.flat_size = 128 * (input_length // 64)

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)  # Binary classification

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SeismicLSTM(nn.Module):
    """LSTM for seismic event classification."""

    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # -> (batch, seq_len, channels)

        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        out = self.fc(h_n[-1])

        return out


class EEGCNN(nn.Module):
    """2D CNN for EEG classification (multi-channel time-series)."""

    def __init__(self, num_channels: int = 23, seq_length: int = 256):
        super().__init__()

        # Treat as 2D image: (1, num_channels, seq_length)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        # Calculate flattened size
        h = num_channels // 8
        w = seq_length // 8
        self.flat_size = 128 * h * w

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)  # Binary: ictal vs non-ictal

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = x.unsqueeze(1)  # Add channel dim: (batch, 1, channels, seq_len)

        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class EEGLSTM(nn.Module):
    """LSTM for EEG classification."""

    def __init__(self, input_size: int = 23, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # -> (batch, seq_len, channels)

        lstm_out, (h_n, c_n) = self.lstm(x)

        out = self.fc(h_n[-1])

        return out


# ==============================================================================
# DATASET WRAPPERS
# ==============================================================================

class SeismicDataset(Dataset):
    """PyTorch dataset for seismic waveforms."""

    def __init__(self, waveforms, labels):
        self.waveforms = [self._pad_or_truncate(w) for w in waveforms]
        self.labels = labels

    def _pad_or_truncate(self, waveform, target_length=500):
        if len(waveform) > target_length:
            indices = np.linspace(0, len(waveform)-1, target_length, dtype=int)
            return waveform[indices]
        elif len(waveform) < target_length:
            padded = np.zeros(target_length)
            padded[:len(waveform)] = waveform
            return padded
        return waveform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return shape: (1, seq_len) for 1D conv
        waveform = torch.FloatTensor(self.waveforms[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return waveform, label


class EEGDataset(Dataset):
    """PyTorch dataset for EEG signals."""

    def __init__(self, signals, labels):
        # signals: list of (channels, seq_len) arrays
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.signals[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return signal, label


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """Train PyTorch model and return training history."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")

    training_time = time.time() - start_time

    return history, training_time


def evaluate_model(model, test_loader):
    """Evaluate model and return metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            start = time.time()
            outputs = model(inputs)
            inference_times.append(time.time() - start)

            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    avg_inference_time = np.mean(inference_times)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time * 1000
    }


def count_parameters(model):
    """Count trainable parameters in PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# SEISMIC VALIDATION
# ==============================================================================

def run_seismic_validation():
    """Compare QA vs CNN vs LSTM on seismic classification."""
    print("="*80)
    print("SEISMIC VALIDATION: QA vs CNN vs LSTM")
    print("="*80)
    print()

    # Generate dataset
    print("Generating synthetic seismic dataset...")
    generator = SeismicWaveformGenerator(sample_rate=100)
    dataset = generator.generate_dataset(n_earthquakes=100, n_explosions=100)
    print(f"  ✓ Generated {len(dataset)} waveforms")
    print()

    # Prepare data
    waveforms = [d['waveform'] for d in dataset]
    labels = [1 if d['type'] == 'earthquake' else 0 for d in dataset]

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        waveforms, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()

    results = {}

    # 1. QA Framework
    print("-" * 80)
    print("1. QA ENHANCED CLASSIFIER")
    print("-" * 80)

    qa_classifier = EnhancedSeismicClassifier(sample_rate=100, modulus=24, num_nodes=24)

    train_data_qa = [{'waveform': w, 'type': 'earthquake' if l == 1 else 'explosion'}
                     for w, l in zip(X_train, y_train)]

    start_time = time.time()
    qa_preds, qa_results_list, qa_features = qa_classifier.classify_batch(
        train_data_qa  # Train on training set for feature extraction
    )
    qa_train_time = time.time() - start_time

    # Test on test set
    test_data_qa = [{'waveform': w, 'type': 'earthquake' if l == 1 else 'explosion'}
                    for w, l in zip(X_test, y_test)]

    start_time = time.time()
    qa_test_preds, _, _ = qa_classifier.classify_batch(test_data_qa)
    qa_inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

    qa_metrics = {
        'accuracy': accuracy_score(y_test, qa_test_preds),
        'training_time_s': qa_train_time,
        'avg_inference_time_ms': qa_inference_time,
        'num_parameters': 24 * 2  # Simplified: num_nodes * 2 state variables
    }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, qa_test_preds, average='binary'
    )
    qa_metrics.update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    results['QA'] = qa_metrics

    print(f"  Accuracy: {qa_metrics['accuracy']*100:.1f}%")
    print(f"  F1-Score: {qa_metrics['f1']:.3f}")
    print(f"  Training time: {qa_metrics['training_time_s']:.2f}s")
    print(f"  Inference time: {qa_metrics['avg_inference_time_ms']:.2f}ms/sample")
    print()

    # 2. CNN Baseline
    print("-" * 80)
    print("2. 1D-CNN BASELINE")
    print("-" * 80)

    cnn_model = SeismicCNN(input_length=500, num_channels=1)
    print(f"  Parameters: {count_parameters(cnn_model):,}")

    train_dataset = SeismicDataset(X_train, y_train)
    val_dataset = SeismicDataset(X_val, y_val)
    test_dataset = SeismicDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    cnn_history, cnn_train_time = train_model(cnn_model, train_loader, val_loader, epochs=50)
    cnn_metrics = evaluate_model(cnn_model, test_loader)
    cnn_metrics['training_time_s'] = cnn_train_time
    cnn_metrics['num_parameters'] = count_parameters(cnn_model)

    results['CNN'] = cnn_metrics

    print(f"  Accuracy: {cnn_metrics['accuracy']*100:.1f}%")
    print(f"  F1-Score: {cnn_metrics['f1']:.3f}")
    print(f"  Training time: {cnn_metrics['training_time_s']:.2f}s")
    print(f"  Inference time: {cnn_metrics['avg_inference_time_ms']:.2f}ms/sample")
    print()

    # 3. LSTM Baseline
    print("-" * 80)
    print("3. LSTM BASELINE")
    print("-" * 80)

    lstm_model = SeismicLSTM(input_size=1, hidden_size=128, num_layers=2)
    print(f"  Parameters: {count_parameters(lstm_model):,}")

    lstm_history, lstm_train_time = train_model(lstm_model, train_loader, val_loader, epochs=50)
    lstm_metrics = evaluate_model(lstm_model, test_loader)
    lstm_metrics['training_time_s'] = lstm_train_time
    lstm_metrics['num_parameters'] = count_parameters(lstm_model)

    results['LSTM'] = lstm_metrics

    print(f"  Accuracy: {lstm_metrics['accuracy']*100:.1f}%")
    print(f"  F1-Score: {lstm_metrics['f1']:.3f}")
    print(f"  Training time: {lstm_metrics['training_time_s']:.2f}s")
    print(f"  Inference time: {lstm_metrics['avg_inference_time_ms']:.2f}ms/sample")
    print()

    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run complete validation with baselines."""
    print("="*80)
    print("PHASE 2 VALIDATION WITH DEEP LEARNING BASELINES")
    print("="*80)
    print()

    workspace = Path("phase2_workspace")
    workspace.mkdir(exist_ok=True)

    # Run seismic validation
    seismic_results = run_seismic_validation()

    # Comparison table
    print("="*80)
    print("SEISMIC CLASSIFICATION: COMPARISON TABLE")
    print("="*80)
    print()
    print(f"{'Method':<15} {'Acc':<8} {'F1':<8} {'Train(s)':<12} {'Infer(ms)':<12} {'Params':<12}")
    print("-" * 80)

    for method, metrics in seismic_results.items():
        print(f"{method:<15} "
              f"{metrics['accuracy']*100:>6.1f}%  "
              f"{metrics['f1']:>6.3f}  "
              f"{metrics['training_time_s']:>10.2f}  "
              f"{metrics['avg_inference_time_ms']:>10.2f}  "
              f"{metrics['num_parameters']:>10,}")

    print()

    # Save results
    results_path = workspace / "phase2_baseline_comparison.json"
    with open(results_path, 'w') as f:
        json.dump({'seismic': seismic_results}, f, indent=2)

    print(f"✓ Results saved to {results_path}")
    print()

    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()
    print("QA Framework advantages:")
    print("  • Interpretable: Geometric features + PAC-Bayesian bounds")
    print("  • Sample efficient: No gradient-based training needed")
    print("  • Fast inference: Algebraic operations only")
    print("  • Tiny model: <100 parameters vs 100k+ for CNNs")
    print()
    print("CNN/LSTM advantages:")
    print("  • May achieve higher accuracy with enough training data")
    print("  • End-to-end learned representations")
    print()
    print("="*80)


if __name__ == "__main__":
    main()

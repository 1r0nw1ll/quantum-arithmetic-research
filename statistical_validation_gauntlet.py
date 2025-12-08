# statistical_validation_gauntlet.py
#
# This script performs a rigorous, multi-trial validation of the
# "Dynamic Co-Processor" concept on the more challenging CIFAR-10 dataset.
# It corrects the critical initialization flaw from the previous experiment.
#
# Dependencies: numpy, scikit-learn, matplotlib, seaborn, torch, torchvision, scipy

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind

# --- 1. The QA-Harmonic Engine (Co-Processor) ---
from qa_core import QAEngine as QA_Engine

# --- 2. The Neural Network (Adapted for CIFAR-10) ---
class SimpleCNN_CIFAR(nn.Module):
    def __init__(self):
        super(SimpleCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(); self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(); self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(); self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.relu4 = nn.ReLU(); self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x))); x = self.pool2(self.relu2(self.conv2(x))); x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4); x = self.relu4(self.fc1(x)); x = self.fc2(x)
        return x

# --- 3. Data Loading (CIFAR-10) ---
def get_cifar10_loaders(batch_size=128, num_workers: int | None = None):
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    quick = os.environ.get("QUICK")
    if num_workers is None:
        num_workers = 0 if quick else 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("Dataset loaded.")
    return train_loader, test_loader

# --- 4. The Training Loop (Unchanged logic, but now for CIFAR) ---
def train_and_evaluate(model, optimizer, train_loader, test_loader, epochs, is_hybrid=False, qa_engine=None, base_lr=0.01):
    criterion = nn.CrossEntropyLoss(); history = {'loss': [], 'val_accuracy': [], 'lr': []}
    for epoch in range(epochs):
        model.train(); running_loss = 0.0; new_lr = base_lr
        if is_hybrid and qa_engine:
            primary_weights = model.conv1.weight.detach().cpu().numpy().flatten()
            for _ in range(5): qa_engine.step(signal=primary_weights)
            stress = qa_engine.get_geometric_stress()
            stress_factor = 1 / (1 + stress * 10.0); new_lr = base_lr * (stress_factor + 0.5)
            for param_group in optimizer.param_groups: param_group['lr'] = new_lr
        history['lr'].append(new_lr)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data; optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); running_loss += loss.item()
        epoch_loss = running_loss/len(train_loader); history['loss'].append(epoch_loss)
        model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data; outputs = model(images); _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0); correct += (predicted == labels).sum().item()
        val_accuracy = 100*correct/total; history['val_accuracy'].append(val_accuracy)
        print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {new_lr:.6f}")
    return history

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # --- CLI Args ---
    parser = argparse.ArgumentParser(description="CIFAR-10 statistical validation gauntlet")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test (1 trial × 1 epoch)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs per run")
    parser.add_argument("--trials", type=int, default=None, help="Number of independent trials")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker count (default 2; set 0 for restricted env)")
    args = parser.parse_args()

    # --- Hyperparameters ---
    quick = args.quick or (os.environ.get("QUICK") is not None)
    NUM_TRIALS = args.trials if args.trials is not None else (1 if quick else 5)
    EPOCHS = args.epochs if args.epochs is not None else (1 if quick else 15)
    BASE_LR = 0.01

    train_loader, test_loader = get_cifar10_loaders(num_workers=args.workers)
    
    control_final_accuracies = []
    hybrid_final_accuracies = []

    for trial in range(NUM_TRIALS):
        print("\n" + "="*50)
        print(f"STARTING TRIAL {trial + 1}/{NUM_TRIALS}")
        print("="*50)
        
        # Set new random seeds for this trial
        trial_seed = np.random.randint(10000)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        # --- Train Control Model (FRESH INITIALIZATION) ---
        print("\n--- Training Control Model ---")
        control_model = SimpleCNN_CIFAR()
        control_optimizer = optim.SGD(control_model.parameters(), lr=BASE_LR, momentum=0.9)
        control_history = train_and_evaluate(control_model, control_optimizer, train_loader, test_loader, EPOCHS, base_lr=BASE_LR)
        control_final_accuracies.append(control_history['val_accuracy'][-1])

        # --- Train Hybrid Model (FRESH INITIALIZATION) ---
        print("\n--- Training Hybrid Model ---")
        hybrid_model = SimpleCNN_CIFAR()
        # CRITICAL: NO weight loading. Fresh, independent model.
        hybrid_optimizer = optim.SGD(hybrid_model.parameters(), lr=BASE_LR, momentum=0.9)
        qa_coprocessor = QA_Engine()
        hybrid_history = train_and_evaluate(hybrid_model, hybrid_optimizer, train_loader, test_loader, EPOCHS, is_hybrid=True, qa_engine=qa_coprocessor, base_lr=BASE_LR)
        hybrid_final_accuracies.append(hybrid_history['val_accuracy'][-1])

    # --- Final Statistical Analysis and Reporting ---
    print("\n\n" + "="*60)
    print("STATISTICAL VALIDATION GAUNTLET: FINAL REPORT")
    print("="*60)
    
    mean_control_acc = np.mean(control_final_accuracies)
    std_control_acc = np.std(control_final_accuracies)
    mean_hybrid_acc = np.mean(hybrid_final_accuracies)
    std_hybrid_acc = np.std(hybrid_final_accuracies)

    print(f"\nResults across {NUM_TRIALS} independent trials on CIFAR-10:")
    print(f"  - Control Model:  Mean Accuracy = {mean_control_acc:.2f}% (± {std_control_acc:.2f})")
    print(f"  - Hybrid Model:   Mean Accuracy = {mean_hybrid_acc:.2f}% (± {std_hybrid_acc:.2f})")

    # Perform t-test
    t_stat, p_value = ttest_ind(hybrid_final_accuracies, control_final_accuracies)
    
    print("\nIndependent t-test for statistical significance:")
    print(f"  - T-statistic: {t_stat:.4f}")
    print(f"  - P-value:     {p_value:.4f}")

    if p_value < 0.05 and mean_hybrid_acc > mean_control_acc:
        print("\n-> CONCLUSION: VALIDATION SUCCESSFUL")
        print("-> The performance improvement of the Hybrid Model is statistically significant.")
        print("-> The Dynamic Co-Processor concept is robustly validated on a challenging benchmark.")
    else:
        print("\n-> CONCLUSION: VALIDATION FAILED")
        print("-> The performance difference is not statistically significant.")
        print("-> The effect observed on MNIST may have been an artifact or does not generalize.")

    # Visualization of final results
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[control_final_accuracies, hybrid_final_accuracies], palette=["red", "blue"])
    plt.xticks([0, 1], ['Control Model', 'Hybrid Model (QA Co-Processor)'])
    plt.ylabel("Final Validation Accuracy (%)")
    plt.title(f"Performance on CIFAR-10 across {NUM_TRIALS} Trials")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.savefig("statistical_validation_results.png")
    plt.show()

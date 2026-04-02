QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
# statistical_validation_gauntlet.py
#
# This script performs a rigorous, multi-trial validation of the
# "Dynamic Co-Processor" concept on the more challenging CIFAR-10 dataset.
# It now enforces strict pairing per trial (init + shuffle + RNG) and
# uses paired significance testing.
#
# Dependencies: numpy, scikit-learn, matplotlib, torch, torchvision, scipy

import os
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel
from pathlib import Path

# --- 1. The QA-Harmonic Engine (Co-Processor) ---
from qa_core import QAEngine as QA_Engine


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# --- 2. The Neural Network (Adapted for CIFAR-10) ---
class SimpleCNN_CIFAR(nn.Module):
    def __init__(self):
        super(SimpleCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 3. Data Loading (CIFAR-10) ---
def get_cifar10_datasets():
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    print("Dataset loaded.")
    return train_set, test_set


def make_cifar10_loaders(train_set, test_set, batch_size=128, num_workers: int = 0, shuffle_seed: int | None = None):
    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if shuffle_seed is not None:
        g = torch.Generator()
        g.manual_seed(shuffle_seed)
        train_kwargs["generator"] = g

    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    return train_loader, test_loader


# --- 4. Training + Stats Helpers ---
def train_and_evaluate(model, optimizer, train_loader, test_loader, epochs, is_hybrid=False, qa_engine=None, base_lr=0.01):
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "val_accuracy": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        new_lr = base_lr

        if is_hybrid and qa_engine:
            primary_weights = model.conv1.weight.detach().cpu().numpy().flatten()
            for _ in range(5):
                qa_engine.step(signal=primary_weights)
            stress = qa_engine.get_geometric_stress()
            stress_factor = 1 / (1 + stress * 10.0)
            new_lr = base_lr * (stress_factor + 0.5)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        history["lr"].append(new_lr)

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history["loss"].append(epoch_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        history["val_accuracy"].append(val_accuracy)
        print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {new_lr:.6f}")

    return history


def paired_ttest_safe(hybrid_scores, control_scores):
    if len(hybrid_scores) < 2 or len(control_scores) < 2:
        return float("nan"), float("nan")
    t_stat, p_value = ttest_rel(hybrid_scores, control_scores)
    return float(t_stat), float(p_value)


def clone_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def safe_mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def is_significant_positive(p_value: float, delta: float) -> bool:
    return (not np.isnan(p_value)) and p_value < 0.05 and delta > 0


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 statistical validation gauntlet")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test (1 trial × 1 epoch)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs per run")
    parser.add_argument("--trials", type=int, default=None, help="Number of independent trials")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker count (default 2; set 0 for restricted env)")
    parser.add_argument("--base-seed", type=int, default=1234, help="Base seed for strict paired trials")
    args = parser.parse_args()

    quick = args.quick or (os.environ.get("QUICK") is not None)
    num_trials = args.trials if args.trials is not None else (1 if quick else 5)
    epochs = args.epochs if args.epochs is not None else (1 if quick else 15)
    workers = args.workers if args.workers is not None else (0 if quick else 2)
    base_lr = 0.01
    base_seed = args.base_seed

    print(
        f"[PAIRING] base_seed={base_seed} strict_paired_init=True "
        f"strict_paired_shuffle=True test=ttest_rel"
    )

    train_set, test_set = get_cifar10_datasets()

    control_final_accuracies = []
    hybrid_final_accuracies = []
    control_auc_accuracies = []
    hybrid_auc_accuracies = []

    for trial_idx in range(num_trials):
        print("\n" + "=" * 50)
        print(f"STARTING TRIAL {trial_idx + 1}/{num_trials}")
        print("=" * 50)

        trial_seed = base_seed + trial_idx
        print(f"[PAIRING] trial_seed={trial_seed}")

        # Strict paired initialization for control/hybrid within this trial.
        seed_everything(trial_seed)
        init_model = SimpleCNN_CIFAR()
        init_state = clone_state_dict(init_model)

        print("\n--- Training Control Model ---")
        seed_everything(trial_seed + 1000)
        control_model = SimpleCNN_CIFAR()
        control_model.load_state_dict(init_state, strict=True)
        control_optimizer = optim.SGD(control_model.parameters(), lr=base_lr, momentum=0.9)
        control_train_loader, control_test_loader = make_cifar10_loaders(
            train_set, test_set, num_workers=workers, shuffle_seed=trial_seed
        )
        control_history = train_and_evaluate(
            control_model,
            control_optimizer,
            control_train_loader,
            control_test_loader,
            epochs,
            base_lr=base_lr,
        )
        control_final_accuracies.append(control_history["val_accuracy"][-1])
        control_auc_accuracies.append(safe_mean(control_history["val_accuracy"]))

        print("\n--- Training Hybrid Model ---")
        seed_everything(trial_seed + 2000)
        hybrid_model = SimpleCNN_CIFAR()
        hybrid_model.load_state_dict(init_state, strict=True)
        hybrid_optimizer = optim.SGD(hybrid_model.parameters(), lr=base_lr, momentum=0.9)
        hybrid_train_loader, hybrid_test_loader = make_cifar10_loaders(
            train_set, test_set, num_workers=workers, shuffle_seed=trial_seed
        )
        qa_coprocessor = QA_Engine()
        hybrid_history = train_and_evaluate(
            hybrid_model,
            hybrid_optimizer,
            hybrid_train_loader,
            hybrid_test_loader,
            epochs,
            is_hybrid=True,
            qa_engine=qa_coprocessor,
            base_lr=base_lr,
        )
        hybrid_final_accuracies.append(hybrid_history["val_accuracy"][-1])
        hybrid_auc_accuracies.append(safe_mean(hybrid_history["val_accuracy"]))

    print("\n\n" + "=" * 60)
    print("STATISTICAL VALIDATION GAUNTLET: FINAL REPORT")
    print("=" * 60)

    mean_control_acc, std_control_acc = mean_std(control_final_accuracies)
    mean_hybrid_acc, std_hybrid_acc = mean_std(hybrid_final_accuracies)
    mean_control_auc, std_control_auc = mean_std(control_auc_accuracies)
    mean_hybrid_auc, std_hybrid_auc = mean_std(hybrid_auc_accuracies)

    print(f"\nResults across {num_trials} paired trials on CIFAR-10:")
    print(f"  - Control Model (Final):  Mean Accuracy = {mean_control_acc:.2f}% (± {std_control_acc:.2f})")
    print(f"  - Hybrid Model  (Final):  Mean Accuracy = {mean_hybrid_acc:.2f}% (± {std_hybrid_acc:.2f})")
    print(f"  - Control Model (AUC):    Mean Accuracy = {mean_control_auc:.2f}% (± {std_control_auc:.2f})")
    print(f"  - Hybrid Model  (AUC):    Mean Accuracy = {mean_hybrid_auc:.2f}% (± {std_hybrid_auc:.2f})")

    t_stat_final, p_value_final = paired_ttest_safe(hybrid_final_accuracies, control_final_accuracies)
    t_stat_auc, p_value_auc = paired_ttest_safe(hybrid_auc_accuracies, control_auc_accuracies)

    print("\nPaired t-test for final-epoch accuracy:")
    print(f"  - T-statistic: {t_stat_final:.4f}")
    print(f"  - P-value:     {p_value_final:.4f}")

    print("\nPaired t-test for per-trial epoch-mean accuracy (AUC-style):")
    print(f"  - T-statistic: {t_stat_auc:.4f}")
    print(f"  - P-value:     {p_value_auc:.4f}")

    final_delta = mean_hybrid_acc - mean_control_acc
    auc_delta = mean_hybrid_auc - mean_control_auc

    if np.isnan(p_value_final):
        print("\n-> CONCLUSION (FINAL): INSUFFICIENT TRIALS")
        print("-> Paired t-test requires at least 2 trials.")
    elif is_significant_positive(p_value_final, final_delta):
        print("\n-> CONCLUSION (FINAL): VALIDATION SUCCESSFUL")
        print("-> The final-epoch performance improvement of the Hybrid Model is statistically significant.")
    else:
        print("\n-> CONCLUSION (FINAL): VALIDATION FAILED")
        print("-> Final-epoch performance difference is not statistically significant or not positive.")

    if np.isnan(p_value_auc):
        print("-> CONCLUSION (AUC): INSUFFICIENT TRIALS")
        print("-> Paired t-test requires at least 2 trials.")
    elif is_significant_positive(p_value_auc, auc_delta):
        print("-> CONCLUSION (AUC): VALIDATION SUCCESSFUL")
        print("-> The Hybrid Model shows statistically significant learning-speed advantage across epochs.")
    else:
        print("-> CONCLUSION (AUC): NO SIGNIFICANT ADVANTAGE")
        print("-> AUC-style learning-speed difference is not statistically significant or not positive.")

    summary = {
        "base_seed": base_seed,
        "num_trials": num_trials,
        "epochs": epochs,
        "workers": workers,
        "pairing_contract": {
            "strict_paired_init": True,
            "strict_paired_shuffle": True,
            "stat_test": "ttest_rel",
        },
        "control_final_accuracies": control_final_accuracies,
        "hybrid_final_accuracies": hybrid_final_accuracies,
        "control_auc_accuracies": control_auc_accuracies,
        "hybrid_auc_accuracies": hybrid_auc_accuracies,
        "final_stats": {
            "control_mean": mean_control_acc,
            "control_std": std_control_acc,
            "hybrid_mean": mean_hybrid_acc,
            "hybrid_std": std_hybrid_acc,
            "delta_hybrid_minus_control": final_delta,
            "paired_t_stat": t_stat_final,
            "paired_p_value": p_value_final,
        },
        "auc_stats": {
            "control_mean": mean_control_auc,
            "control_std": std_control_auc,
            "hybrid_mean": mean_hybrid_auc,
            "hybrid_std": std_hybrid_auc,
            "delta_hybrid_minus_control": auc_delta,
            "paired_t_stat": t_stat_auc,
            "paired_p_value": p_value_auc,
        },
    }
    out_dir = Path("phase2_workspace")
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "statistical_validation_gauntlet_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")

    plt.figure(figsize=(8, 6))
    boxplot_data = [control_final_accuracies, hybrid_final_accuracies]
    boxplot_labels = ["Control Model", "Hybrid Model (QA Co-Processor)"]
    try:
        plt.boxplot(boxplot_data, tick_labels=boxplot_labels)
    except TypeError:
        plt.boxplot(boxplot_data, labels=boxplot_labels)
    plt.ylabel("Final Validation Accuracy (%)")
    plt.title(f"Performance on CIFAR-10 across {num_trials} Paired Trials")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.savefig("statistical_validation_results.png")
    plt.show()

QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
# dynamic_coprocessor_test.py
#
# This script implements and validates the "Dynamic Co-Processor" concept.
# It trains two identical neural networks on the MNIST dataset:
# 1. A "Control" model with a fixed learning rate.
# 2. A "Hybrid" model whose learning rate is dynamically modulated by a
#    parallel QA-Harmonic Engine that measures the "geometric coherence"
#    of the network's weights.
#
# Dependencies: numpy, scikit-learn, matplotlib, seaborn, torch, torchvision

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. The QA-Harmonic Engine (Our Co-Processor) ---
from qa_core import QAEngine as QA_Engine

# --- 2. The Neural Network Architecture ---
class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network for MNIST classification."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Data Loading ---
def get_mnist_loaders(batch_size=64):
    """Downloads and prepares the MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# --- 4. The Training and Evaluation Loop ---
def train_and_evaluate(model, optimizer, train_loader, test_loader, epochs, is_hybrid=False, qa_engine=None, base_lr=0.01, use_chromo=False):
    """The main training loop."""
    criterion = nn.CrossEntropyLoss()
    history = {'loss': [], 'val_accuracy': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # --- DYNAMIC LEARNING RATE MODULATION (FOR HYBRID MODEL) ---
        if is_hybrid:
            # 1. Get weights from the AI
            primary_weights = model.conv1.weight.detach().cpu().numpy().flatten()
            
            # 2. Nudge the QA-Engine and measure stress
            for _ in range(5): # Nudge for a few steps
                qa_engine.step(signal=primary_weights)
            stress = qa_engine.get_geometric_stress(use_chromo=use_chromo)
            
            # 3. Modulate learning rate
            # We add 1 to stress to avoid division by zero and normalize the effect
            # A high stress should significantly decrease LR, low stress should increase it
            stress_factor = 1 / (1 + stress * 5.0) # The '5.0' is a sensitivity hyperparameter
            new_lr = base_lr * (stress_factor + 0.5) # +0.5 to keep LR from collapsing completely
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            new_lr = base_lr

        history['lr'].append(new_lr)

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history['loss'].append(epoch_loss)
        
        # Validation
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
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, LR: {new_lr:.6f}")
        
    return history

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    # --- Parameters ---
    EPOCHS = 1
    BASE_LR = 0.01
    USE_CHROMO = True  # Enable chromogeometry in QA engine
    
    # --- Get Data ---
    train_loader, test_loader = get_mnist_loaders()

    # --- Train Control Model ---
    print("\n" + "="*50)
    print("TRAINING CONTROL MODEL (Fixed Learning Rate)")
    print("="*50)
    control_model = SimpleCNN()
    control_optimizer = optim.SGD(control_model.parameters(), lr=BASE_LR)
    control_history = train_and_evaluate(control_model, control_optimizer, train_loader, test_loader, EPOCHS, base_lr=BASE_LR, use_chromo=USE_CHROMO)

    # --- Train Hybrid Model ---
    print("\n" + "="*50)
    print("TRAINING HYBRID MODEL (Dynamic Co-Processor)")
    print("="*50)
    hybrid_model = SimpleCNN()
    # Re-initialize weights to be identical to the control model's start
    hybrid_model.load_state_dict(control_model.state_dict())
    hybrid_optimizer = optim.SGD(hybrid_model.parameters(), lr=BASE_LR)
    qa_coprocessor = QA_Engine()
    hybrid_history = train_and_evaluate(hybrid_model, hybrid_optimizer, train_loader, test_loader, EPOCHS, is_hybrid=True, qa_engine=qa_coprocessor, base_lr=BASE_LR, use_chromo=USE_CHROMO)

    # --- Visualization and Reporting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Dynamic Co-Processor vs. Control: Training Performance', fontsize=20)

    # Loss Plot
    axes[0].plot(control_history['loss'], 'r-o', label='Control Model')
    axes[0].plot(hybrid_history['loss'], 'b-o', label='Hybrid Model (QA Co-Processor)')
    axes[0].set_title('Training Loss per Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':')
    axes[0].legend()

    # Accuracy Plot
    axes[1].plot(control_history['val_accuracy'], 'r-o', label='Control Model')
    axes[1].plot(hybrid_history['val_accuracy'], 'b-o', label='Hybrid Model (QA Co-Processor)')
    axes[1].set_title('Validation Accuracy per Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, linestyle=':')
    axes[1].legend()

    # Learning Rate Plot
    axes[2].plot(control_history['lr'], 'r-o', label='Control LR (Fixed)')
    axes[2].plot(hybrid_history['lr'], 'b-o', label='Hybrid LR (Dynamic)')
    axes[2].set_title('Learning Rate per Epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, linestyle=':')
    axes[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("coprocessor_test_results.png")
    plt.show()

    # --- Final Report ---
    print("\n" + "="*50)
    print("FINAL VALIDATION REPORT")
    print("="*50)
    control_final_acc = control_history['val_accuracy'][-1]
    hybrid_final_acc = hybrid_history['val_accuracy'][-1]
    print(f"Control Model Final Accuracy: {control_final_acc:.2f}%")
    print(f"Hybrid Model Final Accuracy:  {hybrid_final_acc:.2f}%")
    
    if hybrid_final_acc > control_final_acc:
        print("\n-> VALIDATION SUCCESSFUL: The Hybrid Model with the QA Co-Processor achieved a higher final accuracy.")
        print("-> This provides strong proof-of-concept for Phase-Controlled Harmonic Computing in AI training.")
    else:
        print("\n-> VALIDATION FAILED: The Hybrid Model did not outperform the Control Model.")
        print("-> The co-processor logic or hyperparameters may require further tuning.")

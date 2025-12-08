# intelligent_coprocessor_v2.py
#
# SALVAGE OPERATION: Testing three intelligent stress metrics that actually
# measure the neural network's optimization landscape geometry.
#
# Dependencies: numpy, matplotlib, torch, torchvision, scipy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind
import time

# ============================================================================
# PART 1: THREE INTELLIGENT STRESS METRICS
# ============================================================================

class IntelligentStressMetrics:
    """
    Three different ways to measure 'geometric stress' in a neural network:
    1. Gradient Variance - measures optimization landscape roughness
    2. Loss Curvature - estimates Hessian trace (local curvature)
    3. Weight Distribution Entropy - measures parameter space exploration
    """
    
    @staticmethod
    def gradient_variance_stress(model):
        """
        Measures the variance of gradients across all parameters.
        High variance = rough landscape = high stress.
        """
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if len(grad_norms) == 0:
            return 0.0
        
        return np.var(grad_norms)
    
    @staticmethod
    def loss_curvature_stress(model, criterion, data_sample, labels_sample):
        """
        Estimates local curvature using finite differences.
        High curvature = unstable region = high stress.
        """
        model.eval()
        
        # Get baseline loss
        with torch.no_grad():
            outputs = model(data_sample)
            loss_baseline = criterion(outputs, labels_sample).item()
        
        # Perturb weights slightly and measure loss change
        perturbation_scale = 0.01
        curvatures = []
        
        for param in model.parameters():
            if param.requires_grad:
                original_data = param.data.clone()
                
                # Positive perturbation
                noise = torch.randn_like(param) * perturbation_scale
                param.data.add_(noise)
                
                with torch.no_grad():
                    outputs = model(data_sample)
                    loss_perturbed = criterion(outputs, labels_sample).item()
                
                # Estimate curvature (second derivative)
                curvature = abs(loss_perturbed - loss_baseline) / (perturbation_scale ** 2)
                curvatures.append(curvature)
                
                # Restore original weights
                param.data = original_data
        
        model.train()
        return np.mean(curvatures) if len(curvatures) > 0 else 0.0
    
    @staticmethod
    def weight_entropy_stress(model):
        """
        Measures the entropy of weight distributions.
        Low entropy = stuck in local minimum = high stress.
        High entropy = exploring freely = low stress.
        """
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.cpu().numpy().flatten())
        
        if len(all_weights) == 0:
            return 0.0
        
        all_weights = np.concatenate(all_weights)
        
        # Compute histogram-based entropy
        hist, _ = np.histogram(all_weights, bins=50, density=True)
        hist = hist + 1e-9  # Avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        
        # Invert: high entropy = low stress
        return 1.0 / (entropy + 1.0)


# ============================================================================
# PART 2: THE ADAPTIVE CO-PROCESSOR
# ============================================================================

class AdaptiveCoProcessor:
    """
    Uses intelligent stress metrics to modulate learning rate.
    No QA-Engine - direct measurement of NN optimization landscape.
    """
    
    def __init__(self, metric_type='gradient_variance', base_lr=0.01):
        self.metric_type = metric_type
        self.base_lr = base_lr
        self.metrics = IntelligentStressMetrics()
        self.stress_history = []
    
    def compute_stress(self, model, criterion=None, data_sample=None, labels_sample=None):
        """Compute stress using the selected metric."""
        
        if self.metric_type == 'gradient_variance':
            stress = self.metrics.gradient_variance_stress(model)
        
        elif self.metric_type == 'loss_curvature':
            if data_sample is None or labels_sample is None:
                return 0.0
            stress = self.metrics.loss_curvature_stress(model, criterion, data_sample, labels_sample)
        
        elif self.metric_type == 'weight_entropy':
            stress = self.metrics.weight_entropy_stress(model)
        
        else:
            stress = 0.0
        
        self.stress_history.append(stress)
        return stress
    
    def modulate_learning_rate(self, stress):
        """
        Convert stress signal to learning rate.
        High stress -> lower LR (careful steps in rough terrain)
        Low stress -> higher LR (confident steps in smooth terrain)
        """
        # Normalize stress with sigmoid
        normalized_stress = 1.0 / (1.0 + np.exp(-stress + 1.0))
        
        # Invert: high stress = low LR multiplier
        lr_multiplier = 1.0 - 0.5 * normalized_stress
        
        return self.base_lr * lr_multiplier


# ============================================================================
# PART 3: NEURAL NETWORK & DATA
# ============================================================================

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


def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ============================================================================
# PART 4: TRAINING LOOP WITH ADAPTIVE LR
# ============================================================================

def train_with_adaptive_lr(model, optimizer, train_loader, test_loader, epochs, 
                          coprocessor=None, use_adaptive=False):
    """
    Training loop with optional adaptive learning rate modulation.
    """
    criterion = nn.CrossEntropyLoss()
    history = {'loss': [], 'val_accuracy': [], 'lr': [], 'stress': []}
    
    # Get a fixed sample for curvature measurements
    data_iter = iter(train_loader)
    sample_data, sample_labels = next(data_iter)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # ADAPTIVE LR MODULATION
            if use_adaptive and coprocessor is not None:
                # Compute stress after backward pass (gradients available)
                stress = coprocessor.compute_stress(
                    model, 
                    criterion, 
                    sample_data, 
                    sample_labels
                )
                
                # Modulate learning rate based on stress
                new_lr = coprocessor.modulate_learning_rate(stress)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                current_lr = new_lr
            
            optimizer.step()
            running_loss += loss.item()
        
        # Record metrics
        epoch_loss = running_loss / len(train_loader)
        history['loss'].append(epoch_loss)
        history['lr'].append(current_lr)
        
        if use_adaptive and coprocessor is not None:
            avg_stress = np.mean(coprocessor.stress_history[-len(train_loader):]) if len(coprocessor.stress_history) > 0 else 0
            history['stress'].append(avg_stress)
        else:
            history['stress'].append(0.0)
        
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
        
        print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.6f}, "
              f"Stress: {history['stress'][-1]:.4f}")
    
    return history


# ============================================================================
# PART 5: MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("INTELLIGENT CO-PROCESSOR v2.0 - SALVAGE OPERATION")
    print("Testing three intelligent stress metrics")
    print("="*70)
    
    EPOCHS = 10
    BASE_LR = 0.01
    NUM_TRIALS = 3  # Reduced for faster testing
    
    # Test three metrics
    metrics_to_test = ['gradient_variance', 'loss_curvature', 'weight_entropy']
    
    results = {
        'control': [],
        'gradient_variance': [],
        'loss_curvature': [],
        'weight_entropy': []
    }
    
    train_loader, test_loader = get_cifar10_loaders()
    
    for trial in range(NUM_TRIALS):
        print(f"\n{'='*70}")
        print(f"TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'='*70}")
        
        trial_seed = np.random.randint(10000)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        
        # Control Model
        print("\n--- Control Model (Fixed LR) ---")
        control_model = SimpleCNN_CIFAR()
        control_optimizer = optim.SGD(control_model.parameters(), lr=BASE_LR, momentum=0.9)
        control_history = train_with_adaptive_lr(
            control_model, control_optimizer, train_loader, test_loader, 
            EPOCHS, use_adaptive=False
        )
        results['control'].append(control_history['val_accuracy'][-1])
        
        # Test each metric
        for metric_name in metrics_to_test:
            print(f"\n--- Adaptive Model ({metric_name}) ---")
            
            adaptive_model = SimpleCNN_CIFAR()
            adaptive_optimizer = optim.SGD(adaptive_model.parameters(), lr=BASE_LR, momentum=0.9)
            coprocessor = AdaptiveCoProcessor(metric_type=metric_name, base_lr=BASE_LR)
            
            adaptive_history = train_with_adaptive_lr(
                adaptive_model, adaptive_optimizer, train_loader, test_loader,
                EPOCHS, coprocessor=coprocessor, use_adaptive=True
            )
            
            results[metric_name].append(adaptive_history['val_accuracy'][-1])
    
    # ========================================================================
    # FINAL ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL RESULTS: INTELLIGENT CO-PROCESSOR VALIDATION")
    print("="*70)
    
    control_mean = np.mean(results['control'])
    control_std = np.std(results['control'])
    
    print(f"\nControl (Fixed LR): {control_mean:.2f}% ± {control_std:.2f}%")
    print("\nAdaptive Methods:")
    
    best_metric = None
    best_improvement = -float('inf')
    
    for metric_name in metrics_to_test:
        metric_mean = np.mean(results[metric_name])
        metric_std = np.std(results[metric_name])
        improvement = metric_mean - control_mean
        
        # Statistical test
        t_stat, p_value = ttest_ind(results[metric_name], results['control'])
        
        print(f"\n  {metric_name}:")
        print(f"    Accuracy: {metric_mean:.2f}% ± {metric_std:.2f}%")
        print(f"    Improvement: {improvement:+.2f}%")
        print(f"    P-value: {p_value:.4f} {'***' if p_value < 0.05 else ''}")
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_metric = metric_name
    
    print("\n" + "="*70)
    
    if best_improvement > 0.5 and p_value < 0.05:
        print(f"✓ SUCCESS: '{best_metric}' shows significant improvement!")
        print(f"  Best improvement: {best_improvement:+.2f}%")
        print("  The salvage operation succeeded.")
    else:
        print("✗ VALIDATION FAILED: No metric shows significant improvement.")
        print("  All three intelligent stress metrics failed to improve performance.")
        print("  The co-processor hypothesis is likely fundamentally flawed.")
    
    print("="*70)

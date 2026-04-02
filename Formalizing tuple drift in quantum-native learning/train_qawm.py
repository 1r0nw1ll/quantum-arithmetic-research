"""
QAWM Training Script
Trains world model on QA transition data with proper evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from qawm import QAWM, QAWMConfig, QAWMLoss
from dataset import QADatasetGenerator, QATransitionDataset, collate_fn, analyze_dataset
from qa_oracle import QAOracle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

# ============================================================================
# Training Loop
# ============================================================================

class QAWMTrainer:
    """Trainer for QAWM"""
    
    def __init__(self, model: QAWM, config: QAWMConfig, device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.loss_fn = QAWMLoss(alpha=1.0, beta=1.0, gamma=2.0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'loss/total': 0.0,
            'loss/legal': 0.0,
            'loss/fail_type': 0.0,
            'loss/return': 0.0
        }
        
        num_batches = 0
        
        for state_features, gen_indices, labels in dataloader:
            # Move to device
            state_features = state_features.to(self.device)
            gen_indices = gen_indices.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass
            outputs = self.model(state_features, gen_indices)
            
            # Compute loss
            loss, metrics = self.loss_fn(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            for key, val in metrics.items():
                epoch_metrics[key] += val
            num_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        all_legal_preds = []
        all_legal_labels = []
        all_return_preds = []
        all_return_labels = []
        
        epoch_metrics = {
            'loss/total': 0.0,
            'loss/legal': 0.0,
            'loss/fail_type': 0.0,
            'loss/return': 0.0
        }
        
        num_batches = 0
        
        for state_features, gen_indices, labels in dataloader:
            # Move to device
            state_features = state_features.to(self.device)
            gen_indices = gen_indices.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass
            outputs = self.model(state_features, gen_indices)
            
            # Compute loss
            loss, metrics = self.loss_fn(outputs, labels)
            
            # Accumulate metrics
            for key, val in metrics.items():
                epoch_metrics[key] += val
            num_batches += 1
            
            # Collect predictions for metrics
            legal_probs = torch.sigmoid(outputs['legal_logits']).squeeze(-1)
            all_legal_preds.extend(legal_probs.cpu().numpy())
            all_legal_labels.extend(labels['legal'].cpu().numpy())
            
            # Return-in-k (only for labeled samples)
            return_mask = labels['return'] != -1
            if return_mask.sum() > 0:
                return_probs = torch.sigmoid(outputs['return_logits']).squeeze(-1)
                all_return_preds.extend(return_probs[return_mask].cpu().numpy())
                all_return_labels.extend(labels['return'][return_mask].cpu().numpy())
        
        # Average loss metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Compute classification metrics
        all_legal_preds = np.array(all_legal_preds)
        all_legal_labels = np.array(all_legal_labels)
        
        legal_pred_binary = (all_legal_preds > 0.5).astype(int)
        legal_acc = accuracy_score(all_legal_labels, legal_pred_binary)
        
        epoch_metrics['acc/legal'] = legal_acc
        
        # Return-in-k AUROC (primary metric)
        if len(all_return_labels) > 10:
            all_return_preds = np.array(all_return_preds)
            all_return_labels = np.array(all_return_labels)
            
            try:
                return_auroc = roc_auc_score(all_return_labels, all_return_preds)
                epoch_metrics['auroc/return'] = return_auroc
            except:
                epoch_metrics['auroc/return'] = 0.5  # Random baseline
        else:
            epoch_metrics['auroc/return'] = 0.5
        
        return epoch_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50):
        """Full training loop"""
        print(f"\nTraining QAWM for {num_epochs} epochs...")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        
        best_val_auroc = 0.0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Store metrics
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train loss: {train_metrics['loss/total']:.4f}")
                print(f"  Val loss: {val_metrics['loss/total']:.4f}")
                print(f"  Val legal acc: {val_metrics['acc/legal']:.3f}")
                print(f"  Val return AUROC: {val_metrics['auroc/return']:.3f}")
            
            # Save best model
            if val_metrics['auroc/return'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc/return']
                torch.save(self.model.state_dict(), '/home/claude/qawm_best.pt')
        
        print(f"\n✅ Training complete")
        print(f"  Best validation AUROC: {best_val_auroc:.3f}")
        
        return self.train_metrics, self.val_metrics

# ============================================================================
# Evaluation Metrics
# ============================================================================

def plot_training_curves(train_metrics: List[Dict], val_metrics: List[Dict]):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(train_metrics) + 1)
    
    # Loss
    axes[0].plot(epochs, [m['loss/total'] for m in train_metrics], label='Train')
    axes[0].plot(epochs, [m['loss/total'] for m in val_metrics], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Legal accuracy
    axes[1].plot(epochs, [m.get('acc/legal', 0) for m in val_metrics])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Legality Prediction Accuracy')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Return-in-k AUROC
    axes[2].plot(epochs, [m.get('auroc/return', 0.5) for m in val_metrics])
    axes[2].axhline(y=0.5, color='r', linestyle='--', label='Random')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUROC')
    axes[2].set_title('Return-in-k AUROC (Primary Metric)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('/home/claude/qawm_training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('/home/claude/qawm_training_curves.png', dpi=300, bbox_inches='tight')
    print("📊 Training curves saved")

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("=" * 70)
    print("QAWM TRAINING - Paper 2 Implementation")
    print("=" * 70)
    
    # Configuration
    oracle = QAOracle(N=30, q_def="none")
    generators = ['sigma', 'mu', 'lambda2', 'nu']
    
    # Generate dataset
    print("\n1. Generating dataset...")
    gen = QADatasetGenerator(oracle, generators)
    dataset = gen.generate_dataset(budget=5000, return_in_k_budget=500)
    
    analyze_dataset(dataset)
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n2. Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("\n3. Initializing QAWM...")
    config = QAWMConfig()
    model = QAWM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Train
    print("\n4. Training...")
    trainer = QAWMTrainer(model, config, device='cpu')
    train_metrics, val_metrics = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=50
    )
    
    # Plot results
    print("\n5. Generating plots...")
    plot_training_curves(train_metrics, val_metrics)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    final_metrics = val_metrics[-1]
    print(f"Legality prediction accuracy: {final_metrics['acc/legal']:.3f}")
    print(f"Return-in-k AUROC: {final_metrics['auroc/return']:.3f}")
    print("\n✅ Paper 2 implementation complete")
    print("   Model saved: qawm_best.pt")
    print("   Curves saved: qawm_training_curves.pdf")

if __name__ == "__main__":
    main()

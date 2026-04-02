"""
QAWM Training Script (Scikit-Learn Implementation)
Paper 2: World Model Learning on QA Manifolds

Trains QAWM to predict:
  1. Legality (legal/illegal boundary learning)
  2. Fail type (5-class structural classification)
  3. Return-in-k (reachability prediction)

NO PYTORCH - uses scikit-learn MLPClassifier
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from qawm import QAWM, QAWMConfig
from dataset import QADatasetGenerator, QATransitionDataset, analyze_dataset
from qa_oracle import QAOracle


# =============================================================================
# Training Loop (Scikit-Learn Style)
# =============================================================================

class QAWMTrainer:
    """Trainer for QAWM using scikit-learn"""

    def __init__(self, model: QAWM, config: QAWMConfig):
        self.model = model
        self.config = config

        self.train_history = {
            'legal_acc': [],
            'fail_acc': [],
            'return_auroc': []
        }

    def prepare_data(self, dataset: QATransitionDataset,
                    train_size: float = 0.8) -> Tuple[Dict, Dict]:
        """
        Prepare training and validation data.

        Returns:
            train_data: dict with X, y for each head
            val_data: dict with X, y for each head
        """
        # Extract all samples
        all_state_features = []
        all_gen_indices = []
        all_labels = {
            'legal': [],
            'fail_type': [],
            'return': [],
            'illegal_mask': []
        }

        for idx in range(len(dataset)):
            state_feat, gen_idx, labels = dataset[idx]
            all_state_features.append(state_feat)
            all_gen_indices.append(gen_idx)

            for key in all_labels:
                all_labels[key].append(labels[key])

        # Stack
        X_state = np.stack(all_state_features)
        X_gen = np.array(all_gen_indices)

        # Combine input (state + generator one-hot)
        X_combined = self.model._prepare_input(X_state, X_gen)

        # Labels
        y_legal = np.array(all_labels['legal'])
        y_fail = np.array(all_labels['fail_type'])
        y_return = np.array(all_labels['return'])
        illegal_mask = np.array(all_labels['illegal_mask'])

        # Train/val split
        indices = np.arange(len(X_combined))
        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_size,
            random_state=self.config.random_state,
            stratify=y_legal  # Stratify by legality
        )

        train_data = {
            'X': X_combined[train_idx],
            'y_legal': y_legal[train_idx],
            'y_fail': y_fail[train_idx],
            'y_return': y_return[train_idx],
            'illegal_mask': illegal_mask[train_idx]
        }

        val_data = {
            'X': X_combined[val_idx],
            'y_legal': y_legal[val_idx],
            'y_fail': y_fail[val_idx],
            'y_return': y_return[val_idx],
            'illegal_mask': illegal_mask[val_idx]
        }

        return train_data, val_data

    def train(self, train_data: Dict, val_data: Dict,
              num_epochs: int = 50):
        """
        Train all three heads.

        Note: Scikit-learn MLPClassifier doesn't support true multi-epoch training,
        but we can use warm_start to continue training.
        """
        print(f"\nTraining QAWM for {num_epochs} epochs...")
        print(f"  Training samples: {len(train_data['X'])}")
        print(f"  Validation samples: {len(val_data['X'])}")

        # =====================================================================
        # Head 1: Legality Prediction (All Samples)
        # =====================================================================
        print("\n[1/3] Training legality head...")
        X_train = train_data['X']
        y_train_legal = train_data['y_legal']

        self.model.legal_head.fit(X_train, y_train_legal)
        self.model.legal_fitted = True

        # Evaluate
        y_pred_legal = self.model.legal_head.predict(val_data['X'])
        legal_acc = accuracy_score(val_data['y_legal'], y_pred_legal)
        print(f"  ✓ Legality accuracy: {legal_acc:.3f}")

        # =====================================================================
        # Head 2: Fail Type Prediction (Illegal Samples Only)
        # =====================================================================
        print("\n[2/3] Training fail type head...")

        # Filter to illegal moves only
        train_illegal_mask = train_data['illegal_mask']
        val_illegal_mask = val_data['illegal_mask']

        if np.sum(train_illegal_mask) > 10:
            X_train_illegal = X_train[train_illegal_mask]
            y_train_fail = train_data['y_fail'][train_illegal_mask]

            self.model.fail_type_head.fit(X_train_illegal, y_train_fail)
            self.model.fail_type_fitted = True

            # Evaluate
            if np.sum(val_illegal_mask) > 0:
                X_val_illegal = val_data['X'][val_illegal_mask]
                y_val_fail = val_data['y_fail'][val_illegal_mask]

                y_pred_fail = self.model.fail_type_head.predict(X_val_illegal)
                fail_acc = accuracy_score(y_val_fail, y_pred_fail)
                print(f"  ✓ Fail type accuracy: {fail_acc:.3f}")
            else:
                fail_acc = 0.0
                print(f"  ! No illegal samples in validation set")
        else:
            fail_acc = 0.0
            print(f"  ! Insufficient illegal samples for training")

        # =====================================================================
        # Head 3: Return-in-k Prediction (Labeled Subset Only)
        # =====================================================================
        print("\n[3/3] Training return-in-k head...")

        # Filter to labeled samples (y_return != -1)
        train_return_mask = train_data['y_return'] != -1
        val_return_mask = val_data['y_return'] != -1

        if np.sum(train_return_mask) > 10:
            X_train_return = X_train[train_return_mask]
            y_train_return = train_data['y_return'][train_return_mask]

            self.model.return_head.fit(X_train_return, y_train_return)
            self.model.return_fitted = True

            # Evaluate with AUROC (primary metric for Paper 2)
            if np.sum(val_return_mask) > 10:
                X_val_return = val_data['X'][val_return_mask]
                y_val_return = val_data['y_return'][val_return_mask]

                # Get probabilities for AUROC
                y_pred_return_proba = self.model.return_head.predict_proba(X_val_return)[:, 1]

                try:
                    return_auroc = roc_auc_score(y_val_return, y_pred_return_proba)
                    print(f"  ✓ Return-in-k AUROC: {return_auroc:.3f} (PRIMARY METRIC)")
                except:
                    return_auroc = 0.5
                    print(f"  ! Could not compute AUROC (single class?)")
            else:
                return_auroc = 0.5
                print(f"  ! No labeled return-in-k samples in validation set")
        else:
            return_auroc = 0.5
            print(f"  ! Insufficient labeled samples for return-in-k training")

        # Store metrics
        self.train_history['legal_acc'].append(legal_acc)
        self.train_history['fail_acc'].append(fail_acc)
        self.train_history['return_auroc'].append(return_auroc)

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Final Results:")
        print(f"  Legality accuracy:     {legal_acc:.3f}")
        print(f"  Fail type accuracy:    {fail_acc:.3f}")
        print(f"  Return-in-k AUROC:     {return_auroc:.3f} ← PRIMARY")

        return self.train_history


# =============================================================================
# Visualization
# =============================================================================

def plot_results(trainer: QAWMTrainer, save_path: str = None):
    """Plot training results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Legality accuracy
    axes[0].bar(['Legality'], trainer.train_history['legal_acc'], color='green', alpha=0.7)
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Legality Prediction')
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Fail type accuracy
    axes[1].bar(['Fail Type'], trainer.train_history['fail_acc'], color='orange', alpha=0.7)
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Fail Type Classification')
    axes[1].axhline(y=0.2, color='r', linestyle='--', label='Random (5 classes)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Return-in-k AUROC (PRIMARY METRIC)
    axes[2].bar(['Return-in-k'], trainer.train_history['return_auroc'],
                color='blue', alpha=0.7, label='QAWM')
    axes[2].set_ylim([0, 1])
    axes[2].set_ylabel('AUROC')
    axes[2].set_title('Return-in-k Prediction (PRIMARY)')
    axes[2].axhline(y=0.5, color='r', linestyle='--', label='Random')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Results saved to: {save_path}")
    else:
        plt.savefig('qawm_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Results saved to: qawm_results.png")


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    print("=" * 70)
    print("QAWM TRAINING - Paper 2 Implementation (Scikit-Learn)")
    print("=" * 70)

    # Configuration
    oracle = QAOracle(N=30, q_def="none")
    generators = ['sigma', 'mu', 'lambda2', 'nu']

    # Generate dataset
    print("\n[STEP 1] Generating dataset...")
    gen = QADatasetGenerator(oracle, generators)
    dataset = gen.generate_dataset(budget=5000, return_in_k_budget=500)

    analyze_dataset(dataset)

    # Initialize model
    print("\n[STEP 2] Initializing QAWM...")
    config = QAWMConfig()
    model = QAWM(config)

    print(f"  Model config:")
    print(f"    State dimension: {config.state_dim}")
    print(f"    Hidden dimension: {config.hidden_dim}")
    print(f"    Generator dimension: {config.gen_dim}")

    # Train
    print("\n[STEP 3] Training...")
    trainer = QAWMTrainer(model, config)
    train_data, val_data = trainer.prepare_data(dataset, train_size=0.8)
    history = trainer.train(train_data, val_data, num_epochs=50)

    # Plot results
    print("\n[STEP 4] Generating visualizations...")
    plot_results(trainer, save_path='qawm_results.png')

    # Save model (using scikit-learn's joblib)
    try:
        import joblib
        joblib.dump(model, 'qawm_model.pkl')
        print(f"\n💾 Model saved to: qawm_model.pkl")
    except:
        print(f"\n⚠ Could not save model (joblib not available)")

    # Final summary
    print("\n" + "=" * 70)
    print("PAPER 2 RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: {len(dataset)} samples (80/20 train/val split)")
    print(f"\nTopology Learning (from sparse interaction):")
    print(f"  → Legality prediction:  {history['legal_acc'][0]:.3f} accuracy")
    print(f"  → Fail type learning:   {history['fail_acc'][0]:.3f} accuracy")
    print(f"  → Return-in-k AUROC:    {history['return_auroc'][0]:.3f} ← PRIMARY")
    print(f"\nInterpretation:")

    auroc = history['return_auroc'][0]
    if auroc > 0.7:
        print(f"  ✅ STRONG: QAWM successfully learns reachability structure")
    elif auroc > 0.6:
        print(f"  ✓ MODERATE: QAWM shows learning but could improve")
    else:
        print(f"  ⚠ WEAK: May need more data or feature engineering")

    print(f"\nThis demonstrates: Unknown (topology) → Known (from sparse probes)")
    print(f"Generalization: QAWM predicts on unseen state-generator pairs")
    print("\n✅ Paper 2 training complete!")


if __name__ == "__main__":
    main()

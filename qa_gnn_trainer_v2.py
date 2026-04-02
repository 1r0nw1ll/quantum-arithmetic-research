#!/usr/bin/env python3
"""
QA GNN Trainer v2 - Instrumented Version
Fixed version with checkpointing, early stopping, and real-time monitoring
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import logging
import time
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GNNTrainer")


class GNNGenerator(nn.Module):
    """
    Graph Neural Network for QA tuple classification
    """

    def __init__(self, node_feature_dim=8, gcn_hidden_dim=32, final_hidden_dim=64, output_dim=4):
        super(GNNGenerator, self).__init__()

        # GIN layers
        self.gin1 = GINConv(
            nn.Sequential(
                nn.Linear(node_feature_dim, gcn_hidden_dim),
                nn.ReLU(),
                nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
            )
        )

        self.gin2 = GINConv(
            nn.Sequential(
                nn.Linear(gcn_hidden_dim, gcn_hidden_dim),
                nn.ReLU(),
                nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
            )
        )

        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(gcn_hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(final_hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):
        """
        Forward pass - node classification (NOT graph classification)
        """
        # GIN layer 1
        x = self.gin1(x, edge_index)
        x = F.relu(x)

        # GIN layer 2
        x = self.gin2(x, edge_index)
        x = F.relu(x)

        # MLP classification (per node)
        x = self.mlp(x)

        return x

    def get_embeddings(self, x, edge_index):
        """
        Extract embeddings (before final classification layer)
        """
        x = self.gin1(x, edge_index)
        x = F.relu(x)
        x = self.gin2(x, edge_index)
        x = F.relu(x)
        return x


class QAGNNTrainer:
    """
    Trainer with checkpointing, early stopping, and monitoring
    """

    def __init__(self, model, graph, config):
        self.model = model
        self.graph = graph
        self.config = config

        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=config.get('learning_rate', 0.005))
        self.criterion = nn.CrossEntropyLoss()

        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []

    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()

        # Forward pass
        out = self.model(self.graph.x, self.graph.edge_index)

        # Calculate loss
        loss = self.criterion(out, self.graph.y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct = (pred == self.graph.y).sum().item()
        accuracy = correct / self.graph.y.size(0)

        return loss.item(), accuracy

    def save_checkpoint(self, epoch, loss, accuracy):
        """
        Save model checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✓ Checkpoint saved: {checkpoint_path.name}")

    def early_stopping_check(self, loss):
        """
        Check if training should stop early
        """
        patience = self.config.get('patience', 20)

        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered (patience={patience})")
                return True

        return False

    def train(self):
        """
        Main training loop with full instrumentation
        """
        epochs = self.config.get('epochs', 300)
        checkpoint_interval = self.config.get('checkpoint_interval', 50)
        log_interval = self.config.get('log_interval', 10)

        logger.info("="*60)
        logger.info("QA GNN TRAINING - Starting")
        logger.info("="*60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning rate: {self.config.get('learning_rate', 0.005)}")
        logger.info(f"Checkpoint interval: {checkpoint_interval}")
        logger.info(f"Graph: {self.graph.num_nodes} nodes, {self.graph.num_edges} edges")
        logger.info("="*60)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            loss, accuracy = self.train_epoch()

            # Record history
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'accuracy': accuracy
            })

            # Log progress
            if epoch % log_interval == 0 or epoch == 1:
                epoch_time = time.time() - epoch_start
                logger.info(f"Epoch {epoch:3d}/{epochs} | "
                          f"Loss: {loss:.4f} | "
                          f"Acc: {accuracy:.2%} | "
                          f"Time: {epoch_time:.2f}s")

            # Save checkpoint
            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch, loss, accuracy)

            # Early stopping check
            if self.config.get('early_stopping', False):
                if self.early_stopping_check(loss):
                    logger.info(f"Training stopped early at epoch {epoch}")
                    break

        # Final statistics
        total_time = time.time() - start_time
        final_loss = self.training_history[-1]['loss']
        final_acc = self.training_history[-1]['accuracy']

        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total epochs: {len(self.training_history)}")
        logger.info(f"Final loss: {final_loss:.4f}")
        logger.info(f"Final accuracy: {final_acc:.2%}")
        logger.info(f"Best loss: {self.best_loss:.4f}")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        logger.info("="*60)

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, final_path)
        logger.info(f"✓ Final model saved: {final_path}")

        # Save training history as JSON
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"✓ Training history saved: {history_path}")

        return self.model

    def extract_embeddings(self, save_path="qa_embeddings.pt"):
        """
        Extract and save node embeddings
        """
        logger.info("Extracting node embeddings...")

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(self.graph.x, self.graph.edge_index)

        logger.info(f"✓ Embeddings shape: {embeddings.shape}")

        if save_path:
            np.save(save_path.replace(".pt", ".npy"), embeddings.cpu().numpy())
            logger.info(f"✓ Embeddings saved to {save_path.replace(".pt", ".npy")}")

        return embeddings


def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train QA GNN with monitoring')
    parser.add_argument('--graph', default='qa_graph.pt',
                       help='Input graph file')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Checkpoint save interval')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log interval')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--output-dir', default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--embeddings-output', default='qa_embeddings.pt',
                       help='Output path for embeddings')

    args = parser.parse_args()

    try:
        # Load graph
        logger.info(f"Loading graph from {args.graph}")
        graph = torch.load(args.graph, weights_only=False)
        logger.info(f"✓ Graph loaded: {graph.num_nodes} nodes, {graph.num_edges} edges")

        # Create model
        num_classes = len(torch.unique(graph.y))
        logger.info(f"Number of classes: {num_classes}")

        model = GNNGenerator(
            node_feature_dim=graph.x.size(1),
            gcn_hidden_dim=32,
            final_hidden_dim=64,
            output_dim=num_classes
        )

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training config
        config = {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'checkpoint_interval': args.checkpoint_interval,
            'log_interval': args.log_interval,
            'early_stopping': args.early_stopping,
            'patience': args.patience,
            'checkpoint_dir': args.output_dir
        }

        # Train
        trainer = QAGNNTrainer(model, graph, config)
        trained_model = trainer.train()

        # Extract embeddings
        trainer.extract_embeddings(args.embeddings_output)

        logger.info("✓ SUCCESS: Training completed")
        return 0

    except Exception as e:
        logger.error(f"✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

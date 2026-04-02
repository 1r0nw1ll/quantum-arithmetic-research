#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
QALM Production Training Script
Trains QA Language Model v1.0 on full dataset with checkpointing and monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/player2/signal_experiments/qa_lab')

from qa_dataloader import QAJSONLDataset, create_dataloaders
from qa_model_architecture import QALanguageModel, QAConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qalm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QALMTrainer:
    """Production trainer for QALM"""

    def __init__(
        self,
        config: QAConfig,
        dataset_path: str,
        checkpoint_dir: str = 'checkpoints',
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        max_length: int = 512,
        vocab_size: int = 10000
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create dataloaders
        logger.info("Loading dataset...")
        self.train_loader, self.val_loader, self.vocab = create_dataloaders(
            data_path=dataset_path,
            batch_size=batch_size,
            train_split=0.9,
            max_length=max_length,
            vocab_size=vocab_size
        )

        # Update config with actual vocab size
        self.config.vocab_size = len(self.vocab)

        # Initialize model
        logger.info("Initializing model...")
        self.model = QALanguageModel(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model initialized with {param_count:,} parameters")
        logger.info(f"Using device: {self.device}")

        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])

        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'vocab': self.vocab,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'qalm_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'qalm_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))

        logger.info(f"Resumed from epoch {self.start_epoch}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            qa_tuples = batch['qa_tuples'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                qa_tuples=qa_tuples,
                attention_mask=attention_mask
            )
            logits = outputs[0]

            # Compute loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        return epoch_loss / num_batches

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                qa_tuples = batch['qa_tuples'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    qa_tuples=qa_tuples,
                    attention_mask=attention_mask
                )
                logits = outputs[0]

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                val_loss += loss.item()
                num_batches += 1

        return val_loss / num_batches

    def train(self, num_epochs: int, save_every: int = 10):
        """Full training loop"""
        logger.info("=" * 70)
        logger.info("Starting QALM Production Training")
        logger.info("=" * 70)
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Starting from epoch: {self.start_epoch}")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        logger.info("=" * 70)

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = datetime.now()

            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            epoch_time = (datetime.now() - epoch_start).total_seconds()

            # Log epoch results
            logger.info("=" * 70)
            logger.info(
                f"Epoch {epoch}/{num_epochs-1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Best Val: {self.best_val_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            if is_best:
                logger.info("🌟 New best model!")
            logger.info("=" * 70)

            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # Plot progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_progress()

        logger.info("=" * 70)
        logger.info("✅ Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Final training loss: {self.train_losses[-1]:.4f}")
        logger.info("=" * 70)

        # Save final checkpoint
        self.save_checkpoint(num_epochs - 1, self.val_losses[-1])
        self.plot_progress()

        return self.train_losses, self.val_losses

    def plot_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(12, 5))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('QALM Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Log scale
        plt.subplot(1, 2, 2)
        plt.semilogy(self.train_losses, label='Train Loss', linewidth=2)
        plt.semilogy(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('QALM Training Progress (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_progress.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train QALM v1.0')
    parser.add_argument('--dataset', type=str,
                       default='qa_training_dataset.jsonl',
                       help='Path to JSONL dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=768,
                       help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size')

    args = parser.parse_args()

    # Create config
    config = QAConfig(
        vocab_size=args.vocab_size,  # Will be updated from dataset
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        qa_tuple_dim=4,
        invariant_heads=args.num_heads // 2,
        modular_bases=[24, 72, 288],
    )

    # Create trainer
    trainer = QALMTrainer(
        config=config,
        dataset_path=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        vocab_size=args.vocab_size
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    train_losses, val_losses = trainer.train(
        num_epochs=args.epochs,
        save_every=10
    )

    # Save training summary
    summary = {
        'config': config.__dict__,
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'best_val_loss': float(trainer.best_val_loss),
        'total_epochs': args.epochs,
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses]
    }

    summary_path = Path(args.checkpoint_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training summary saved to {summary_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
Test QALM Training
Quick test to verify training pipeline with small batch
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/player2/signal_experiments/qa_lab')

from qa_dataloader import QAJSONLDataset, create_dataloaders
from qa_model_architecture import QALanguageModel, QAConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_training_step():
    """Test a single training step"""
    logger.info("=" * 70)
    logger.info("QALM Training Test")
    logger.info("=" * 70)

    # 1. Create small config for testing
    config = QAConfig(
        vocab_size=5000,
        hidden_size=128,  # Small for testing
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        qa_tuple_dim=4,
        invariant_heads=2,
        modular_bases=[24, 72],
    )

    logger.info("\n[1/5] Loading dataset...")
    dataset = QAJSONLDataset(
        data_path='/home/player2/signal_experiments/qa_training_dataset.jsonl',
        max_length=128,
        vocab_size=5000
    )
    logger.info(f"  ✓ Loaded {len(dataset)} examples")
    logger.info(f"  ✓ Vocabulary size: {len(dataset.vocab)}")

    # 2. Create model
    logger.info("\n[2/5] Initializing model...")
    # Update config with actual vocab size
    config.vocab_size = len(dataset.vocab)
    model = QALanguageModel(config)
    logger.info(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Test forward pass
    logger.info("\n[3/5] Testing forward pass...")
    batch = dataset[0]
    input_ids = batch['input_ids'].unsqueeze(0)  # Add batch dim
    attention_mask = batch['attention_mask'].unsqueeze(0)
    qa_tuples = batch['qa_tuples'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            qa_tuples=qa_tuples,
            attention_mask=attention_mask
        )
        logits = outputs[0]

    logger.info(f"  ✓ Input shape: {input_ids.shape}")
    logger.info(f"  ✓ Output logits shape: {logits.shape}")
    logger.info(f"  ✓ Expected: (batch=1, seq_len=128, vocab={config.vocab_size})")

    # 4. Test training step
    logger.info("\n[4/5] Testing training step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.token_to_id['<pad>'])

    model.train()
    optimizer.zero_grad()

    # Forward
    outputs = model(
        input_ids=input_ids,
        qa_tuples=qa_tuples,
        attention_mask=attention_mask
    )
    logits = outputs[0]

    # Loss
    loss = criterion(
        logits.view(-1, logits.size(-1)),
        batch['labels'].unsqueeze(0).view(-1)
    )

    # Backward
    loss.backward()
    optimizer.step()

    logger.info(f"  ✓ Loss: {loss.item():.4f}")
    logger.info(f"  ✓ Gradients computed successfully")

    # 5. Test dataloader with batches
    logger.info("\n[5/5] Testing batch dataloader...")
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_iter = iter(dataloader)
    batch = next(batch_iter)

    logger.info(f"  ✓ Batch input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  ✓ Batch qa_tuples shape: {batch['qa_tuples'].shape}")

    # Forward with batch
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            qa_tuples=batch['qa_tuples'],
            attention_mask=batch['attention_mask']
        )
        logits = outputs[0]

    logger.info(f"  ✓ Batch output shape: {logits.shape}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("=" * 70)

    return True


def run_mini_training(num_epochs=3, batch_size=4):
    """Run a mini training session"""
    logger.info("\n" + "=" * 70)
    logger.info("Mini Training Session (3 epochs, batch_size=4)")
    logger.info("=" * 70)

    # Config
    config = QAConfig(
        vocab_size=5000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
    )

    # Dataset
    dataset = QAJSONLDataset(
        data_path='/home/player2/signal_experiments/qa_training_dataset.jsonl',
        max_length=128,
        vocab_size=5000
    )
    config.vocab_size = len(dataset.vocab)

    # Model
    model = QALanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.token_to_id['<pad>'])

    # Dataloader (small subset for speed)
    from torch.utils.data import Subset, DataLoader
    small_dataset = Subset(dataset, range(100))  # Just 100 examples
    dataloader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Training on {len(small_dataset)} examples...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                qa_tuples=batch['qa_tuples'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs[0]

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    logger.info("\n✅ Mini training completed successfully!")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test QALM Training')
    parser.add_argument('--mini', action='store_true',
                       help='Run mini training session (3 epochs)')
    args = parser.parse_args()

    # Run forward/backward test
    test_training_step()

    # Optionally run mini training
    if args.mini:
        run_mini_training()

    print("\n🎉 QALM training test successful! Ready for full training.")

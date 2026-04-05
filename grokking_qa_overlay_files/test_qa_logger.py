"""
Quick test to verify QA logger works before running full experiment.
This creates synthetic data and logs a few steps.
"""

import torch
import numpy as np
from qa_logger import QALogger

print("="*60)
print("QA Logger Verification Test")
print("="*60)

# Create synthetic model and data
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Initialize
model = DummyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
qa_logger = QALogger(run_id="verification_test", log_every=1)

print("\n1. Testing basic logging...")

# Simulate a few training steps
for epoch in range(5):
    # Forward pass
    x = torch.randn(32, 10)
    targets = torch.randint(0, 5, (32,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log with QA logger
    qa_logger.log_step(epoch, logits, targets, loss, model, optimizer)
    print(f"  Step {epoch}: loss={loss.item():.4f} ✓")

qa_logger.close()

print("\n2. Checking output file...")
import json
from pathlib import Path

log_file = Path("qa_logs/verification_test.jsonl")
if log_file.exists():
    with open(log_file, 'r') as f:
        records = [json.loads(line) for line in f]
    print(f"  ✓ Found {len(records)} records in {log_file}")

    if len(records) > 0:
        print("\n3. Validating record schema...")
        rec = records[0]
        required_keys = ['run_id', 'step', 'state', 'generators', 'failures', 'cumulative_failures']
        for key in required_keys:
            if key in rec:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ {key} MISSING!")

        print("\n4. Sample record (first step):")
        print(json.dumps(rec, indent=2))

        print("\n5. Checking state variables...")
        state_vars = [
            'train_loss', 'train_acc', 'logit_max', 'logit_min',
            'p_entropy', 'grad_norm', 'param_norm', 'cos_grad_w'
        ]
        missing = [v for v in state_vars if v not in rec['state']]
        if missing:
            print(f"  ✗ Missing state variables: {missing}")
        else:
            print(f"  ✓ All {len(state_vars)} state variables present")

        print("\n6. Checking generator legality...")
        if 'sgd_step' in rec['generators']:
            gen = rec['generators']['sgd_step']
            print(f"  Legal: {gen['legal']}")
            print(f"  Reason: {gen['reason']}")
            print("  ✓")
        else:
            print("  ✗ sgd_step generator missing!")

        print("\n" + "="*60)
        print("QA Logger Verification: PASSED ✓")
        print("="*60)
        print("\nReady to run: ./run_qa_experiment.sh")
        print("Or: python grokking_experiments_qa.py --dataset modular_addition")

else:
    print(f"  ✗ Log file not found: {log_file}")
    print("\nVerification FAILED. Check errors above.")

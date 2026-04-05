# QA Logger Patch Instructions

## Changes to `grokking_experiments.py`

### 1. Add import (after line 7)

```python
from logger import MetricsLogger
from qa_logger import QALogger  # ADD THIS LINE
from torch.utils.data import DataLoader
```

### 2. Initialize QA logger (after line 44)

```python
model = get_model(args)
logger = MetricsLogger(args.num_epochs, args.log_frequency)
# ADD THESE LINES:
qa_log_id = f"{args.dataset}_{args.loss_function}_seed{args.seed}"
qa_logger = QALogger(run_id=qa_log_id, log_every=args.log_frequency)
optimizer = get_optimizer(model, args)
```

### 3. Log each training step (after line 90, before the logging block)

```python
    loss.backward()
    optimizer.step()

    # ADD THIS LINE (log BEFORE the if block so we capture the step):
    qa_logger.log_step(epoch, output, shuffled_targets, loss, model, optimizer)

    if epoch % logger.log_frequency == 0:
```

### 4. Close logger at end (after line 128)

```python
logger.metrics_df.to_csv(f"loggs/{experiment_key}/metrics.csv", index=False)

# ADD THIS LINE:
qa_logger.close()

with open(f"loggs/{experiment_key}/args.json", 'w') as f:
```

---

## Complete Modified Training Loop (lines 76-110)

Here's what the modified section looks like:

```python
loss = torch.inf
start_time = time.time()
model.to(device).to(train_dtype)
for epoch in range(args.num_epochs):
    #Shuffling the data should not matter for full batch GD,
    #but it sometimes does matter because of floating point errors
    permutation = torch.randperm(all_data.size(0))
    shuffled_data = all_data[permutation]
    shuffled_targets = all_targets[permutation]
    model.train()
    optimizer.zero_grad()
    output = model(shuffled_data)
    if args.use_transformer:
        output = output[:, -1]
    output = output*args.alpha
    loss = loss_function(output, shuffled_targets, dtype=ce_dtype)
    loss.backward()
    optimizer.step()

    # QA LOGGER HOOK
    qa_logger.log_step(epoch, output, shuffled_targets, loss, model, optimizer)

    if epoch % logger.log_frequency == 0:
        logger.log_metrics(
            model=model,
            epoch=epoch,
            save_model_checkpoints=save_model_checkpoints,
            saved_models=saved_models,
            all_data=shuffled_data,
            all_targets=shuffled_targets,
            all_test_data=all_test_data,
            all_test_targets=all_test_targets,
            args=args,
            loss_function=loss_function,
        )

        print(f'Epoch {epoch}: Training loss: {loss.item():.4f}')
        if epoch > 0:
            print(f"Time taken for the last {args.log_frequency} epochs: {(time.time() - start_time)/60:.2f} min")
        start_time = time.time()
```

---

## Output

After running with this patch, you'll find:
- `qa_logs/<run_id>.jsonl` - JSONL file with QA state/legality/failure logs
- Each line is a JSON record with the schema described in the plan

## Example JSONL Record

```json
{
  "run_id": "modular_addition_cross_entropy_seed0",
  "step": 1000,
  "state": {
    "train_loss": 0.523,
    "train_acc": 0.88,
    "logit_max": 12.4,
    "logit_min": -8.2,
    "logit_std": 4.1,
    "logit_norm": 103.2,
    "p_max": 0.95,
    "p_entropy": 0.42,
    "num_exact_ones": 0,
    "num_exact_zeros": 12,
    "grad_norm": 2.3,
    "grad_nan_count": 0,
    "grad_inf_count": 0,
    "param_norm": 45.2,
    "param_nan_count": 0,
    "param_inf_count": 0,
    "cos_grad_w": 0.23
  },
  "generators": {
    "sgd_step": {"legal": true, "reason": null}
  },
  "failures": [],
  "cumulative_failures": {
    "SOFTMAX_COLLAPSE": 0,
    "NAN_LOSS": 0,
    "INF_GRAD": 0,
    "GRAD_EXPLOSION": 0,
    "PARAM_EXPLOSION": 0,
    "LOGIT_EXPLOSION": 0
  }
}
```

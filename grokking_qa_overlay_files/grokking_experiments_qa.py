import random
import time
import torch
import torch.nn as nn
import json
import os
from logger import MetricsLogger
from qa_logger import QALogger  # QA INSTRUMENTATION
from torch.utils.data import DataLoader
from utils import (evaluate,
                   softmax_cross_entropy,
                   get_specified_args,
                   get_dataset,
                   get_model,
                   parse_args,
                   get_optimizer,
                   stablemax_cross_entropy)


torch.set_num_threads(5)

parser, args = parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

train_dtype = getattr(torch, args.train_dtype)

device = args.device
print("Using device:", device)

train_dataset, test_dataset = get_dataset(args)
if args.full_batch:
    args.batch_size = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

torch.save(train_dataset, "last_train_loader.pt")
torch.save(test_dataset, "last_test_loader.pt")

args.lr = args.lr/(args.alpha**2)

model = get_model(args)
logger = MetricsLogger(args.num_epochs, args.log_frequency)

# QA INSTRUMENTATION: Initialize QA logger
qa_log_id = f"{args.dataset}_{args.loss_function}_seed{args.seed}"
qa_logger = QALogger(run_id=qa_log_id, log_every=args.log_frequency)

optimizer = get_optimizer(model, args)

print(args.loss_function)

loss_functions = {
    "cross_entropy": softmax_cross_entropy,
    "stablemax": stablemax_cross_entropy
}
loss_function = loss_functions[args.loss_function]
ce_dtype = getattr(torch, args.cross_entropy_dtype)
save_model_checkpoints = range(0, args.num_epochs, args.log_frequency)
saved_models = {epoch: None for epoch in save_model_checkpoints}

softmax_temperature = 1

if args.full_batch:
    all_data = train_dataset.dataset.data[train_dataset.indices].to(device)
    all_targets = train_dataset.dataset.targets[train_dataset.indices].to(device).long()

    all_test_data = test_dataset.dataset.data[test_dataset.indices].to(device)
    all_test_targets = test_dataset.dataset.targets[test_dataset.indices].to(device).long()

    if not (args.use_transformer or args.use_embedding):
        all_data = all_data.to(train_dtype)
        all_test_data = all_test_data.to(train_dtype)
else:
    raise ValueError("Current implementation only supports full batch training.")

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

    # QA INSTRUMENTATION: Log each training step
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

model.eval().to('cpu')
test_loss, test_accuracy = evaluate(model, test_loader, use_embedding=args.use_transformer or args.use_embedding)
print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}')
args.lr = args.lr

specified_args = get_specified_args(parser, args)
if len(specified_args.keys()) == 0:
    experiment_key = f'{args.dataset}_default'
else:
    experiment_key = f'{args.dataset}|' + '|'.join([f'{key}-{str(specified_args[key])}' for key in specified_args.keys()])

experiment_key = experiment_key[:255]  # Unfortunately most Linux systems do not allow longer directory / filenames

torch.save(saved_models, 'last_run_saved_model_checkpoints.pt')
torch.save(optimizer, 'last_optimizer.pt')

os.makedirs(f"loggs/{experiment_key}", exist_ok=True)
logger.metrics_df.to_csv(f"loggs/{experiment_key}/metrics.csv", index=False)

# QA INSTRUMENTATION: Close QA logger
qa_logger.close()

with open(f"loggs/{experiment_key}/args.json", 'w') as f:
    json.dump(vars(args), f, indent=4)

print(f"Saving run: {experiment_key}")

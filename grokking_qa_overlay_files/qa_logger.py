"""
QA (Quantum Arithmetic) Logger for Grokking Experiments
Instruments training with discrete state logging, generator legality, and failure counters.
"""
import torch
import json
import numpy as np
from pathlib import Path


class QALogger:
    """
    Logs training as a QA reachability process:
    - State: numerical quantities at each step
    - Generators: SGD step legality
    - Failures: boundary violations
    """

    def __init__(self, run_id, output_dir="qa_logs", log_every=1):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_every = log_every

        # JSONL output file
        self.log_file = self.output_dir / f"{run_id}.jsonl"
        self.log_handle = open(self.log_file, 'w')

        # Failure counters (persistent across training)
        self.failure_counts = {
            "SOFTMAX_COLLAPSE": 0,
            "NAN_LOSS": 0,
            "INF_GRAD": 0,
            "GRAD_EXPLOSION": 0,
            "PARAM_EXPLOSION": 0,
            "LOGIT_EXPLOSION": 0,
        }

        # Legality thresholds (tune after first run if needed)
        self.LMAX = 85.0  # fp32 logit max threshold
        self.HMIN = 0.01  # entropy floor for softmax collapse
        self.GN_MAX = 1e6  # gradient norm explosion threshold
        self.PN_MAX = 1e6  # parameter norm explosion threshold

        print(f"[QA Logger] Initialized: {self.log_file}")
        print(f"[QA Logger] Thresholds: LMAX={self.LMAX}, HMIN={self.HMIN}, GN_MAX={self.GN_MAX}")

    def log_step(self, epoch, logits, targets, loss, model, optimizer):
        """
        Log one training step.

        Args:
            epoch: training epoch number
            logits: model output (before softmax), shape [batch, num_classes]
            targets: ground truth labels
            loss: scalar loss value
            model: the neural network model
            optimizer: the optimizer (for gradient access)
        """
        if epoch % self.log_every != 0:
            return

        # === 1. Compute State Variables ===
        state = self._compute_state(epoch, logits, targets, loss, model, optimizer)

        # === 2. Check Generator Legality ===
        generators, current_failures = self._check_legality(state)

        # === 3. Update Failure Counters ===
        for fail_type in current_failures:
            self.failure_counts[fail_type] += 1

        # === 4. Write JSONL Record ===
        record = {
            "run_id": self.run_id,
            "step": epoch,
            "state": state,
            "generators": generators,
            "failures": current_failures,
            "cumulative_failures": self.failure_counts.copy()
        }
        self.log_handle.write(json.dumps(record) + "\n")
        self.log_handle.flush()

    def _compute_state(self, epoch, logits, targets, loss, model, optimizer):
        """Compute all state variables for this step."""

        # Basic training metrics
        with torch.no_grad():
            train_loss = loss.item() if torch.isfinite(loss) else float('inf')
            preds = logits.argmax(dim=1)
            train_acc = (preds == targets).float().mean().item()

        # Logit statistics
        logit_max = logits.max().item()
        logit_min = logits.min().item()
        logit_std = logits.std().item()
        logit_norm = logits.norm().item()

        # Softmax stability (replicate their softmax_collapse logic)
        with torch.no_grad():
            output_off = logits - logits.amax(dim=1, keepdim=True)
            exp_output = torch.exp(output_off)
            probs = exp_output / exp_output.sum(dim=-1, keepdim=True)

            p_max = probs.max(dim=1).values.mean().item()

            # Entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-12)  # numerical stability
            entropy = -(probs * log_probs).sum(dim=-1)
            p_entropy = entropy.mean().item()

            # Count exact 0s and 1s (softmax collapse indicators)
            num_exact_ones = (probs == 1.0).sum().item()
            num_exact_zeros = (probs == 0.0).sum().item()

        # Gradient statistics
        grad_norm = 0.0
        grad_nan_count = 0
        grad_inf_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                grad_nan_count += torch.isnan(param.grad).sum().item()
                grad_inf_count += torch.isinf(param.grad).sum().item()
        grad_norm = np.sqrt(grad_norm)

        # Parameter statistics
        param_norm = 0.0
        param_nan_count = 0
        param_inf_count = 0
        for param in model.parameters():
            param_norm += param.norm().item() ** 2
            param_nan_count += torch.isnan(param).sum().item()
            param_inf_count += torch.isinf(param).sum().item()
        param_norm = np.sqrt(param_norm)

        # Gradient-weight alignment (NLM proxy)
        # cos(θ) between gradient direction and weight direction
        cos_grad_w = self._compute_gradient_weight_alignment(model)

        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "logit_max": logit_max,
            "logit_min": logit_min,
            "logit_std": logit_std,
            "logit_norm": logit_norm,
            "p_max": p_max,
            "p_entropy": p_entropy,
            "num_exact_ones": num_exact_ones,
            "num_exact_zeros": num_exact_zeros,
            "grad_norm": grad_norm,
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "param_norm": param_norm,
            "param_nan_count": param_nan_count,
            "param_inf_count": param_inf_count,
            "cos_grad_w": cos_grad_w,
        }

    def _compute_gradient_weight_alignment(self, model):
        """
        Compute cosine similarity between gradient and weight vectors.
        High alignment suggests gradient is scaling weights (NLM direction).
        """
        grad_vec = []
        weight_vec = []

        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.flatten())
                weight_vec.append(param.data.flatten())

        if len(grad_vec) == 0:
            return 0.0

        grad_vec = torch.cat(grad_vec)
        weight_vec = torch.cat(weight_vec)

        # Cosine similarity
        cos_sim = (grad_vec * weight_vec).sum() / (grad_vec.norm() * weight_vec.norm() + 1e-12)
        return cos_sim.item()

    def _check_legality(self, state):
        """
        Check if the SGD step is 'legal' (not near numerical boundaries).
        Returns (generators_dict, failure_list).
        """
        failures = []

        # Check softmax collapse
        if state["p_entropy"] < self.HMIN or state["num_exact_ones"] > 0:
            failures.append("SOFTMAX_COLLAPSE")

        # Check logit explosion
        if state["logit_max"] > self.LMAX or state["logit_max"] < -self.LMAX:
            failures.append("LOGIT_EXPLOSION")

        # Check NaN loss
        if not np.isfinite(state["train_loss"]):
            failures.append("NAN_LOSS")

        # Check gradient health
        if state["grad_nan_count"] > 0 or state["grad_inf_count"] > 0:
            failures.append("INF_GRAD")

        if state["grad_norm"] > self.GN_MAX:
            failures.append("GRAD_EXPLOSION")

        # Check parameter health
        if state["param_norm"] > self.PN_MAX:
            failures.append("PARAM_EXPLOSION")

        # Overall legality
        legal = len(failures) == 0
        reason = None if legal else failures[0]  # report first failure

        generators = {
            "sgd_step": {
                "legal": legal,
                "reason": reason
            }
        }

        return generators, failures

    def close(self):
        """Close the log file."""
        self.log_handle.close()
        print(f"[QA Logger] Closed: {self.log_file}")
        print(f"[QA Logger] Total failures: {self.failure_counts}")

    def __del__(self):
        """Ensure file is closed on deletion."""
        if hasattr(self, 'log_handle') and not self.log_handle.closed:
            self.close()

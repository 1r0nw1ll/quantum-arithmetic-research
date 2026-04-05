"""
QA Logger - FIXED VERSION
Addresses: logit magnitude, test accuracy tracking, adaptive logging density
"""
import torch
import json
import numpy as np
from pathlib import Path


class QALogger:
    """Logs training as a QA reachability process"""

    def __init__(self, run_id, output_dir="qa_logs", log_every=1, log_dense_until=1000):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_every = log_every
        self.log_dense_until = log_dense_until  # Log every step until this epoch

        self.log_file = self.output_dir / f"{run_id}.jsonl"
        self.log_handle = open(self.log_file, 'w')

        self.failure_counts = {
            "SOFTMAX_COLLAPSE": 0,
            "NAN_LOSS": 0,
            "INF_GRAD": 0,
            "GRAD_EXPLOSION": 0,
            "PARAM_EXPLOSION": 0,
            "LOGIT_EXPLOSION": 0,
        }

        # Thresholds
        self.LMAX = 85.0           # Max absolute logit
        self.HMIN = 0.01           # Entropy floor
        self.GN_MAX = 1e6          # Gradient norm explosion
        self.PN_MAX = 1e6          # Parameter norm explosion
        self.GRAD_DEAD = 1e-6      # Gradient effectively zero

    def log_step(self, epoch, logits, targets, loss, model, optimizer,
                 test_logits=None, test_targets=None):
        """
        Log one training step with optional test metrics.

        Args:
            epoch: training epoch
            logits: train logits
            targets: train targets
            loss: train loss
            model: the model
            optimizer: optimizer
            test_logits: (optional) test set logits
            test_targets: (optional) test set targets
        """
        # Adaptive logging: dense early, sparse later
        if epoch < self.log_dense_until:
            should_log = True
        else:
            should_log = (epoch % self.log_every == 0)

        if not should_log:
            return

        state = self._compute_state(epoch, logits, targets, loss, model, optimizer,
                                    test_logits, test_targets)
        generators, failures = self._check_legality(state)

        for fail_type in failures:
            self.failure_counts[fail_type] += 1

        record = {
            "run_id": self.run_id,
            "step": epoch,
            "state": state,
            "generators": generators,
            "failures": failures,
            "cumulative_failures": self.failure_counts.copy()
        }
        self.log_handle.write(json.dumps(record) + "\n")
        self.log_handle.flush()

    def _compute_state(self, epoch, logits, targets, loss, model, optimizer,
                       test_logits, test_targets):
        # Train metrics
        with torch.no_grad():
            train_loss = loss.item() if torch.isfinite(loss) else float('inf')
            preds = logits.argmax(dim=1)
            train_acc = (preds == targets).float().mean().item()

            # FIXED: Logit statistics (use absolute values and norms)
            logit_max_abs = logits.abs().max().item()  # Max magnitude
            logit_range = (logits.max() - logits.min()).item()  # Range
            logit_std = logits.std().item()
            logit_norm = logits.norm().item()  # Frobenius norm

            # Softmax stability
            output_off = logits - logits.amax(dim=1, keepdim=True)
            exp_output = torch.exp(output_off.clamp(min=-88))  # Prevent underflow
            probs = exp_output / exp_output.sum(dim=-1, keepdim=True)

            p_max = probs.max(dim=1).values.mean().item()
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum(dim=-1)
            p_entropy = entropy.mean().item()

            num_exact_ones = (probs == 1.0).sum().item()
            num_exact_zeros = (probs == 0.0).sum().item()

        # Test metrics (if provided)
        test_loss = None
        test_acc = None
        if test_logits is not None and test_targets is not None:
            with torch.no_grad():
                test_preds = test_logits.argmax(dim=1)
                test_acc = (test_preds == test_targets).float().mean().item()
                # Could compute test loss here if needed

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
        for param in model.parameters():
            param_norm += param.norm().item() ** 2
        param_norm = np.sqrt(param_norm)

        # Gradient-weight alignment
        cos_grad_w = self._compute_gradient_weight_alignment(model)

        state = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "logit_max_abs": logit_max_abs,  # FIXED: absolute value
            "logit_range": logit_range,       # FIXED: range
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
            "cos_grad_w": cos_grad_w,
        }

        # Add test metrics if available
        if test_acc is not None:
            state["test_acc"] = test_acc
        if test_loss is not None:
            state["test_loss"] = test_loss

        return state

    def _compute_gradient_weight_alignment(self, model):
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
        cos_sim = (grad_vec * weight_vec).sum() / (grad_vec.norm() * weight_vec.norm() + 1e-12)
        return cos_sim.item()

    def _check_legality(self, state):
        """Check if SGD step is legal (not near numerical boundaries)"""
        failures = []

        # Softmax collapse
        if state["p_entropy"] < self.HMIN or state["num_exact_ones"] > 0:
            failures.append("SOFTMAX_COLLAPSE")

        # Logit explosion (use absolute value)
        if state["logit_max_abs"] > self.LMAX:
            failures.append("LOGIT_EXPLOSION")

        # NaN loss
        if not np.isfinite(state["train_loss"]):
            failures.append("NAN_LOSS")

        # Gradient health
        if state["grad_nan_count"] > 0 or state["grad_inf_count"] > 0:
            failures.append("INF_GRAD")

        # Gradient explosion
        if state["grad_norm"] > self.GN_MAX:
            failures.append("GRAD_EXPLOSION")

        # Parameter explosion
        if state["param_norm"] > self.PN_MAX:
            failures.append("PARAM_EXPLOSION")

        legal = len(failures) == 0
        reason = None if legal else failures[0]

        return {"sgd_step": {"legal": legal, "reason": reason}}, failures

    def close(self):
        self.log_handle.close()
        print(f"[QA Logger] Closed: {self.log_file}")
        print(f"[QA Logger] Total failures: {self.failure_counts}")

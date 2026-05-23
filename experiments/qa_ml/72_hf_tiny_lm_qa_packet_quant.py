"""HF tiny causal-LM QA packet quantization rung.

This is the next rung after the local TinyDigitTransformer probe: use an actual
Hugging Face causal language model and compare QA packet-family quantizers
against GPTQ/AWQ-shaped activation-aware post-training baselines.

Claim boundary: this is a real HF model, but the GPTQ/AWQ comparisons here are
local proxy implementations, not the auto-gptq/awq libraries.

QA_COMPLIANCE = "hf_tiny_lm_qa_packet_quant - real HF causal LM; activation-aware proxy baselines; no LLM-scale claim"
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn


REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


SEED = 72072
MODEL_ID = "sshleifer/tiny-gpt2"
BITS = [2, 3, 4]
LLOYD_STEPS = 8
CALIBRATION_TEXTS = [
    "Quantum arithmetic keeps exact packets while observers project decimals.",
    "Robust geometry needs predicates that preserve signs near degeneracy.",
    "Low bit transformer quantization should preserve logits on calibration text.",
    "A tiny language model is not a large language model, but it is a real transformer.",
]
EVAL_TEXTS = [
    "Exact invariant packets avoid drift in long horizon simulations.",
    "Activation aware quantization uses calibration statistics to protect important channels.",
    "The next token distribution changes when low precision destroys transformer weights.",
    "A fair experiment reports losses, logits, and reconstruction errors with caveats.",
    "Packet codebooks can preserve local structure better than uniform scalar bins.",
    "This benchmark is a small causal language model loaded from Hugging Face.",
]
OUT_PATH = Path(__file__).with_name("results_hf_tiny_lm_qa_packet_quant.json")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)


def tokenize_texts(tokenizer, texts: list[str]) -> list[dict[str, torch.Tensor]]:
    batches = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt")
        batches.append({k: v for k, v in encoded.items()})
    return batches


def causal_loss_and_logits(model: nn.Module, batches: list[dict[str, torch.Tensor]]) -> tuple[float, list[torch.Tensor]]:
    model.eval()
    losses = []
    logits_list = []
    with torch.no_grad():
        for batch in batches:
            out = model(**batch, labels=batch["input_ids"])
            losses.append(float(out.loss))
            logits_list.append(out.logits.detach().cpu())
    return float(np.mean(losses)), logits_list


def logit_mse(fp_logits: list[torch.Tensor], q_logits: list[torch.Tensor]) -> float:
    vals = []
    for a, b in zip(fp_logits, q_logits):
        vals.append(float(torch.mean((a - b) * (a - b))))
    return float(np.mean(vals))


def iter_quant_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    tensors = []
    for name, param in model.named_parameters():
        if ".wte." in name or ".wpe." in name or name.endswith("wte.weight") or name.endswith("wpe.weight"):
            continue
        if param.dtype.is_floating_point and param.ndim >= 2:
            tensors.append((name, param.detach()))
    return tensors


def _quantize_with_input_axis(
    t: torch.Tensor,
    bits: int,
    importance: torch.Tensor | None,
    quantizer: Callable[[torch.Tensor, int, torch.Tensor | None], torch.Tensor],
) -> torch.Tensor:
    if importance is None or t.ndim < 2:
        return quantizer(t, bits, importance)
    flat = t.reshape(t.shape[0], -1)
    if importance.numel() == flat.shape[1]:
        return quantizer(t, bits, importance)
    if importance.numel() == t.shape[0]:
        qt = quantizer(t.transpose(0, 1).contiguous(), bits, importance)
        return qt.transpose(0, 1).contiguous().reshape_as(t)
    return quantizer(t, bits, None)


def _levels(bits: int) -> int:
    return 1 << bits


def minmax_quantize(t: torch.Tensor, bits: int) -> torch.Tensor:
    levels = _levels(bits) - 1
    mn = t.min()
    mx = t.max()
    if torch.isclose(mx, mn):
        return t.clone()
    q = torch.round((t - mn) / (mx - mn) * levels).clamp(0, levels)
    return q / levels * (mx - mn) + mn


def gptq_proxy_weighted_clip(t: torch.Tensor, bits: int, importance: torch.Tensor | None) -> torch.Tensor:
    if t.ndim < 2 or importance is None:
        return minmax_quantize(t, bits)
    rows = t.reshape(t.shape[0], -1)
    if importance.numel() != rows.shape[1]:
        return minmax_quantize(t, bits)
    out = torch.empty_like(rows)
    positive = (1 << (bits - 1)) - 1
    clip_grid = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1]
    w = torch.clamp(importance.to(dtype=t.dtype), min=1e-12)
    for i in range(rows.shape[0]):
        row = rows[i]
        max_abs = row.abs().max()
        if torch.isclose(max_abs, torch.zeros_like(max_abs)):
            out[i] = row
            continue
        best_err = None
        best = None
        for ratio in clip_grid:
            scale = max_abs * ratio / positive
            q = torch.round(row / scale).clamp(-positive, positive) * scale
            err = torch.sum(w * (row - q) * (row - q))
            if best_err is None or float(err) < best_err:
                best_err = float(err)
                best = q
        assert best is not None
        out[i] = best
    return out.reshape_as(t)


def awq_proxy_scaled_minmax(t: torch.Tensor, bits: int, importance: torch.Tensor | None) -> torch.Tensor:
    if t.ndim < 2 or importance is None:
        return minmax_quantize(t, bits)
    rows = t.reshape(t.shape[0], -1)
    if importance.numel() != rows.shape[1]:
        return minmax_quantize(t, bits)
    imp = torch.sqrt(torch.clamp(importance.to(dtype=t.dtype), min=1e-12))
    best_err = None
    best = None
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        scale = imp.pow(alpha)
        scale = scale / torch.mean(scale)
        scaled = rows * scale.reshape(1, -1)
        q_scaled = torch.empty_like(scaled)
        for i in range(scaled.shape[0]):
            q_scaled[i] = minmax_quantize(scaled[i], bits)
        recon = q_scaled / scale.reshape(1, -1)
        err = torch.mean(torch.sum(importance.reshape(1, -1) * (rows - recon) * (rows - recon), dim=1))
        if best_err is None or float(err) < best_err:
            best_err = float(err)
            best = recon
    assert best is not None
    return best.reshape_as(t)


def _lloyd(block: torch.Tensor, levels: int, weights: torch.Tensor | None = None) -> torch.Tensor:
    if block.numel() <= 1:
        return block.clone()
    if weights is None:
        weights = torch.ones_like(block)
    weights = torch.clamp(weights.to(dtype=block.dtype), min=1e-12)
    unique = torch.unique(block)
    if unique.numel() <= levels:
        distances = weights[:, None] * (block[:, None] - unique[None, :]).abs()
        return unique[torch.argmin(distances, dim=1)]
    order = torch.argsort(block)
    sorted_block = block[order]
    centers = []
    for idx in range(levels):
        lo = int(idx * block.numel() / levels)
        hi = int((idx + 1) * block.numel() / levels)
        if hi <= lo:
            hi = min(lo + 1, block.numel())
        centers.append(sorted_block[lo:hi].mean())
    centroids = torch.stack(centers)
    assignments = torch.zeros(block.numel(), dtype=torch.long)
    for _ in range(LLOYD_STEPS):
        distances = weights[:, None] * (block[:, None] - centroids[None, :]).abs()
        assignments = torch.argmin(distances, dim=1)
        new_centroids = centroids.clone()
        for level in range(levels):
            mask = assignments == level
            if torch.any(mask):
                w = weights[mask]
                new_centroids[level] = torch.sum(w * block[mask]) / torch.sum(w)
        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids[assignments]


def qa_lloyd_packet(t: torch.Tensor, bits: int, importance: torch.Tensor | None) -> torch.Tensor:
    if t.ndim < 2:
        return _lloyd(t.flatten(), _levels(bits)).reshape_as(t)
    rows = t.reshape(t.shape[0], -1)
    out = torch.empty_like(rows)
    use_importance = importance is not None and importance.numel() == rows.shape[1]
    weights = importance.to(dtype=t.dtype) if use_importance else None
    for i in range(rows.shape[0]):
        out[i] = _lloyd(rows[i], _levels(bits), weights)
    return out.reshape_as(t)


def qa_unweighted_lloyd_packet(t: torch.Tensor, bits: int, _importance: torch.Tensor | None) -> torch.Tensor:
    return qa_lloyd_packet(t, bits, None)


def qa_affine_residual_packet(t: torch.Tensor, bits: int, importance: torch.Tensor | None) -> torch.Tensor:
    if t.ndim < 2:
        return qa_lloyd_packet(t, bits, importance)
    rows = t.reshape(t.shape[0], -1)
    out = torch.empty_like(rows)
    positive = (1 << (bits - 1)) - 1
    if positive <= 0:
        raise ValueError("bits must be >= 2")
    for idx in range(rows.shape[0]):
        row = rows[idx]
        x = torch.linspace(-1.0, 1.0, row.numel(), dtype=row.dtype, device=row.device)
        xc = x - x.mean()
        yc = row - row.mean()
        denom = torch.sum(xc * xc)
        slope = torch.sum(xc * yc) / denom if not torch.isclose(denom, torch.zeros_like(denom)) else torch.zeros_like(row.mean())
        base = row.mean() + slope * xc
        residual = row - base
        radius = residual.abs().max()
        if torch.isclose(radius, torch.zeros_like(radius)):
            out[idx] = row
        else:
            q = torch.round(residual / radius * positive).clamp(-positive, positive)
            out[idx] = base + q / positive * radius
    return out.reshape_as(t)


def qa_symmetric_packet(t: torch.Tensor, bits: int, importance: torch.Tensor | None) -> torch.Tensor:
    if t.ndim < 2:
        return gptq_proxy_weighted_clip(t, bits, importance)
    rows = t.reshape(t.shape[0], -1)
    out = torch.empty_like(rows)
    positive = (1 << (bits - 1)) - 1
    for idx in range(rows.shape[0]):
        row = rows[idx]
        radius = row.abs().max()
        if torch.isclose(radius, torch.zeros_like(radius)):
            out[idx] = row
        else:
            scale = radius / positive
            out[idx] = torch.round(row / scale).clamp(-positive, positive) * scale
    return out.reshape_as(t)


def transformer_weight_modules() -> tuple[type[nn.Module], ...]:
    from transformers.pytorch_utils import Conv1D

    return (nn.Linear, Conv1D)


def collect_param_importance(model: nn.Module, calibration_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    module_to_param = {}
    weight_modules = transformer_weight_modules()
    for module_name, module in model.named_modules():
        if isinstance(module, weight_modules):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name == "weight":
                    module_to_param[module] = f"{module_name}.weight"

    hooks = []
    for module, param_name in module_to_param.items():
        def make_hook(name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                x = inputs[0].detach()
                x = x.reshape(-1, x.shape[-1])
                sq = torch.sum(x * x, dim=0)
                if name not in sums:
                    sums[name] = sq.clone()
                    counts[name] = int(x.shape[0])
                else:
                    sums[name] += sq
                    counts[name] += int(x.shape[0])

            return hook

        hooks.append(module.register_forward_pre_hook(make_hook(param_name)))

    model.eval()
    with torch.no_grad():
        for batch in calibration_batches:
            model(**batch)
    for hook in hooks:
        hook.remove()

    return {name: sums[name] / max(1, counts[name]) for name in sums}


def apply_quantization(
    model: nn.Module,
    bits: int,
    method: str,
    importance: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict]:
    q_model = copy.deepcopy(model)
    q_state = q_model.state_dict()
    mse = {}
    weighted_mse = {}
    values = 0
    tensors = 0
    quantizers = {
        "gptq_proxy_weighted_clip": gptq_proxy_weighted_clip,
        "awq_proxy_scaled_minmax": awq_proxy_scaled_minmax,
        "qa_lloyd_packet": qa_lloyd_packet,
        "qa_unweighted_lloyd_packet": qa_unweighted_lloyd_packet,
        "qa_affine_residual_packet": qa_affine_residual_packet,
        "qa_symmetric_packet": qa_symmetric_packet,
    }
    quantizer = quantizers[method]
    for name, original in iter_quant_tensors(model):
        imp = importance.get(name)
        q = _quantize_with_input_axis(original, bits, imp, quantizer)
        q_state[name].copy_(q)
        diff = original - q
        mse[name] = float(torch.mean(diff * diff))
        if imp is not None and original.reshape(original.shape[0], -1).shape[1] == imp.numel():
            rows = diff.reshape(diff.shape[0], -1)
            weighted_mse[name] = float(torch.mean(torch.sum(rows * rows * imp.reshape(1, -1), dim=1)))
        else:
            weighted_mse[name] = mse[name]
        values += int(original.numel())
        tensors += 1
    q_model.load_state_dict(q_state)
    return q_model, {
        "method": method,
        "bits": bits,
        "quantized_param_tensors": tensors,
        "quantized_values": values,
        "weight_mse_mean": float(np.mean(list(mse.values()))),
        "weighted_mse_mean": float(np.mean(list(weighted_mse.values()))),
        "mse_by_tensor": mse,
        "weighted_mse_by_tensor": weighted_mse,
    }


def apply_qa_calibrated_selector(
    model: nn.Module,
    bits: int,
    importance: dict[str, torch.Tensor],
    calibration_batches: list[dict[str, torch.Tensor]],
) -> tuple[nn.Module, dict]:
    candidates: dict[str, Callable[[torch.Tensor, int, torch.Tensor | None], torch.Tensor]] = {
        "qa_lloyd_packet": qa_lloyd_packet,
        "qa_unweighted_lloyd_packet": qa_unweighted_lloyd_packet,
        "qa_affine_residual_packet": qa_affine_residual_packet,
        "qa_symmetric_packet": qa_symmetric_packet,
    }
    current = copy.deepcopy(model)
    q_state = current.state_dict()
    selected = {}
    mse = {}
    weighted_mse = {}
    tensors = 0
    values = 0

    for name, original in iter_quant_tensors(model):
        imp = importance.get(name)
        best_loss = None
        best_method = None
        best_tensor = None
        for method, quantizer in candidates.items():
            trial = copy.deepcopy(current)
            trial_state = trial.state_dict()
            q = _quantize_with_input_axis(original, bits, imp, quantizer)
            trial_state[name].copy_(q)
            trial.load_state_dict(trial_state)
            loss, _logits = causal_loss_and_logits(trial, calibration_batches)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_method = method
                best_tensor = q
        assert best_tensor is not None and best_method is not None
        q_state[name].copy_(best_tensor)
        current.load_state_dict(q_state)
        selected[name] = best_method
        diff = original - best_tensor
        mse[name] = float(torch.mean(diff * diff))
        if imp is not None:
            flat = diff.reshape(diff.shape[0], -1)
            if flat.shape[1] == imp.numel():
                weighted_mse[name] = float(torch.mean(torch.sum(flat * flat * imp.reshape(1, -1), dim=1)))
            elif diff.shape[0] == imp.numel():
                flat_t = diff.transpose(0, 1).contiguous().reshape(diff.shape[1], -1)
                weighted_mse[name] = float(torch.mean(torch.sum(flat_t * flat_t * imp.reshape(1, -1), dim=1)))
            else:
                weighted_mse[name] = mse[name]
        else:
            weighted_mse[name] = mse[name]
        tensors += 1
        values += int(original.numel())

    return current, {
        "method": "qa_calibrated_packet_selector",
        "bits": bits,
        "selected_methods": selected,
        "quantized_param_tensors": tensors,
        "quantized_values": values,
        "weight_mse_mean": float(np.mean(list(mse.values()))),
        "weighted_mse_mean": float(np.mean(list(weighted_mse.values()))),
        "mse_by_tensor": mse,
        "weighted_mse_by_tensor": weighted_mse,
    }


def run(model_id: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seeds()
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    calibration = tokenize_texts(tokenizer, CALIBRATION_TEXTS)
    evaluation = tokenize_texts(tokenizer, EVAL_TEXTS)
    load_s = time.time() - t0

    fp_loss, fp_logits = causal_loss_and_logits(model, evaluation)
    importance = collect_param_importance(model, calibration)

    rows = []
    for bits in BITS:
        for method in (
            "gptq_proxy_weighted_clip",
            "awq_proxy_scaled_minmax",
            "qa_lloyd_packet",
            "qa_unweighted_lloyd_packet",
            "qa_affine_residual_packet",
            "qa_symmetric_packet",
        ):
            q_model, meta = apply_quantization(model, bits, method, importance)
            q_loss, q_logits = causal_loss_and_logits(q_model, evaluation)
            meta["eval_loss"] = q_loss
            meta["eval_ppl"] = float(math.exp(q_loss)) if q_loss < 50 else float("inf")
            meta["loss_delta_vs_fp32"] = q_loss - fp_loss
            meta["logit_mse_vs_fp32"] = logit_mse(fp_logits, q_logits)
            rows.append(meta)
        q_model, meta = apply_qa_calibrated_selector(model, bits, importance, calibration)
        q_loss, q_logits = causal_loss_and_logits(q_model, evaluation)
        meta["eval_loss"] = q_loss
        meta["eval_ppl"] = float(math.exp(q_loss)) if q_loss < 50 else float("inf")
        meta["loss_delta_vs_fp32"] = q_loss - fp_loss
        meta["logit_mse_vs_fp32"] = logit_mse(fp_logits, q_logits)
        rows.append(meta)

    summary = {}
    proxy_methods = {"gptq_proxy_weighted_clip", "awq_proxy_scaled_minmax"}
    for bits in BITS:
        subset = [r for r in rows if r["bits"] == bits]
        qa_family = [r for r in subset if r["method"].startswith("qa_")]
        best_qa_loss = min(qa_family, key=lambda r: (r["eval_loss"], r["logit_mse_vs_fp32"]))
        best_qa_logit = min(qa_family, key=lambda r: (r["logit_mse_vs_fp32"], r["eval_loss"]))
        best_qa_wmse = min(qa_family, key=lambda r: (r["weighted_mse_mean"], r["eval_loss"]))
        proxies = [r for r in subset if r["method"] in proxy_methods]
        best_proxy_loss = min(proxies, key=lambda r: (r["eval_loss"], r["logit_mse_vs_fp32"]))
        best_proxy_logit = min(proxies, key=lambda r: (r["logit_mse_vs_fp32"], r["eval_loss"]))
        best_proxy_wmse = min(proxies, key=lambda r: (r["weighted_mse_mean"], r["eval_loss"]))
        summary[str(bits)] = {
            "best_qa_loss_method": best_qa_loss["method"],
            "best_qa_logit_method": best_qa_logit["method"],
            "best_qa_weighted_mse_method": best_qa_wmse["method"],
            "best_proxy_loss_method": best_proxy_loss["method"],
            "best_proxy_logit_method": best_proxy_logit["method"],
            "best_proxy_weighted_mse_method": best_proxy_wmse["method"],
            "qa_loss_minus_best_proxy": best_qa_loss["eval_loss"] - best_proxy_loss["eval_loss"],
            "qa_logit_mse_minus_best_proxy": best_qa_logit["logit_mse_vs_fp32"] - best_proxy_logit["logit_mse_vs_fp32"],
            "qa_weighted_mse_minus_best_proxy": best_qa_wmse["weighted_mse_mean"] - best_proxy_wmse["weighted_mse_mean"],
        }

    qa_loss_bits = [bit for bit, row in summary.items() if row["qa_loss_minus_best_proxy"] < 0.0]
    qa_logit_bits = [bit for bit, row in summary.items() if row["qa_logit_mse_minus_best_proxy"] < 0.0]
    qa_wmse_bits = [bit for bit, row in summary.items() if row["qa_weighted_mse_minus_best_proxy"] < 0.0]
    if qa_loss_bits and qa_logit_bits and qa_wmse_bits:
        status = "PASS_REAL_HF_LOSS_LOGIT_AND_WMSE"
    elif qa_loss_bits or qa_logit_bits or qa_wmse_bits:
        status = "PARTIAL_REAL_HF_PROXY"
    else:
        status = "NULL_REAL_HF_PROXY"

    return {
        "experiment": "hf_tiny_lm_qa_packet_quant",
        "status": status,
        "seed": SEED,
        "model_id": model_id,
        "bits": BITS,
        "load_time_s": load_s,
        "fp32_eval_loss": fp_loss,
        "fp32_eval_ppl": float(math.exp(fp_loss)) if fp_loss < 50 else float("inf"),
        "claim_boundary": (
            "Real Hugging Face causal LM loaded through transformers; GPTQ/AWQ entries "
            "are local activation-aware proxy algorithms, not auto-gptq/awq library runs."
        ),
        "success_criteria": (
            "PASS_REAL_HF_LOSS_LOGIT_AND_WMSE if QA packet quantization beats the best "
            "GPTQ/AWQ-style proxy on eval loss, logit MSE, and weighted MSE for at least one bit width each."
        ),
        "quantization_results": rows,
        "summary_by_bits": summary,
        "qa_loss_bits": qa_loss_bits,
        "qa_logit_bits": qa_logit_bits,
        "qa_weighted_mse_bits": qa_wmse_bits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        t = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]])
        imp = torch.tensor([1.0, 2.0, 3.0])
        for bits in BITS:
            for fn in (
                gptq_proxy_weighted_clip,
                awq_proxy_scaled_minmax,
                qa_lloyd_packet,
                qa_unweighted_lloyd_packet,
                qa_affine_residual_packet,
                qa_symmetric_packet,
            ):
                q = fn(t, bits, imp)
                assert q.shape == t.shape
                assert torch.isfinite(q).all()
        print(canonical_json({"ok": True}))
        return
    result = run(args.model_id)
    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "model_id": result["model_id"],
        "fp32_eval_loss": result["fp32_eval_loss"],
        "summary_by_bits": result["summary_by_bits"],
        "out_path": str(OUT_PATH),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

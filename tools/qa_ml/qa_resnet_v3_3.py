"""QA-ML v3.3 E4 generator-residual feature stack.

This is the fixed-observer analog of Pepe's STAResNet lesson: residual
blocks should operate in the algebra of the task. Each block applies one
QA generator to the current integer state, computes the strict v3 packet
delta, and appends that delta as a residual channel.

QA_COMPLIANCE = "qa_ml_resnet_v3_3 — integer generator residual observer features"
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .qa_features_v3 import FEATURE_NAMES_V3, qa_packet_v3
from .qa_generators import GENERATORS


GENERATOR_CYCLE: tuple[str, ...] = ("sigma", "mu", "lambda_2", "nu")


def residual_feature_names(
    active_names: Iterable[str],
    depth: int,
    generator_cycle: tuple[str, ...] = GENERATOR_CYCLE,
) -> tuple[str, ...]:
    """Names for base active features plus one delta packet per block."""
    active = tuple(active_names)
    names = list(active)
    for block_idx in range(depth):
        gen_name = generator_cycle[block_idx % len(generator_cycle)]
        names.extend(f"res{block_idx + 1}_{gen_name}_delta_{name}" for name in active)
    return tuple(names)


def qa_residual_stack(
    triples: Iterable[tuple[int, int, int]],
    active_names: tuple[str, ...],
    depth: int,
    generator_cycle: tuple[str, ...] = GENERATOR_CYCLE,
) -> np.ndarray:
    """Build a strict packet plus QA generator residual deltas.

    Partial generators that are undefined at the current state append a zero
    residual and leave the current state unchanged. Zero is reserved outside
    the A1 state domain, so the residual channel keeps undefined blocks
    distinguishable without inventing a wrapped state.
    """
    full_idx = {name: i for i, name in enumerate(FEATURE_NAMES_V3)}
    active_idx = [full_idx[name] for name in active_names]
    rows: list[list[int]] = []

    for b, e, m in triples:
        cur_b, cur_e = b, e
        cur_packet = qa_packet_v3(cur_b, cur_e, m)
        row = [cur_packet[i] for i in active_idx]

        for block_idx in range(depth):
            gen_name = generator_cycle[block_idx % len(generator_cycle)]
            next_state = GENERATORS[gen_name](cur_b, cur_e, m)
            if next_state is None:
                row.extend(0 for _ in active_idx)
                continue

            next_b, next_e = next_state
            next_packet = qa_packet_v3(next_b, next_e, m)
            row.extend(next_packet[i] - cur_packet[i] for i in active_idx)
            cur_b, cur_e = next_b, next_e
            cur_packet = next_packet

        rows.append(row)

    return np.asarray(rows, dtype=np.int64)

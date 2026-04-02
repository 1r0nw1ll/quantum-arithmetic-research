#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_compression_benchmark.py
===========================

Benchmark a QA-specific reversible trajectory codec against strong general-purpose
lossless compressors on QA-native state sequences.

Scope:
- Domain-specific data only: sequences of QA states (b, e) on Caps(N, N)
- Baselines: gzip, bzip2, xz/lzma, brotli, zstd
- QA codec: initial state + packed generator tokens + sparse escape states
- Fairness: all codecs see the same raw binary state serialization

Pre-registered benchmark intent:
- Hypothesis: QA compression wins only when trajectories contain sufficiently long
  lawful spans under the declared generator set; strong generic codecs dominate
  short or off-manifold traces.
- Success criteria: on real repo-native traces, at least one ordered trace slice
  should show a QA advantage if lawful-span structure is strong enough; otherwise
  the run records a FAIL boundary result rather than forcing a theorem.

The QA codec is intentionally narrow. If it wins on QA-native traces and loses on
non-QA controls, that is useful evidence rather than a defect.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import brotli
import zstandard as zstd


MAGIC = b"QAC1"
SEG_MAGIC = b"QAS1"
QSTEP_MAGIC = b"QAT1"
QSTEP_SEG_MAGIC = b"QAU1"
MICROTRACE_VERSION = 1
TOKEN_BITS = 3
MIN_QA_SEGMENT_STATES = 128
MIN_QSTEP_SEGMENT_STATES = 8
MOVE_TO_CODE = {
    "sigma": 0,
    "mu": 1,
    "lambda2": 2,
    "nu": 3,
    "escape": 4,
}
CODE_TO_MOVE = {value: key for key, value in MOVE_TO_CODE.items()}
MOVE_ORDER = ("sigma", "mu", "lambda2", "nu")


State = tuple[int, int]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    length: int
    description: str
    generator: Callable[[int, random.Random], list[State]]


@dataclass(frozen=True)
class DatasetInstance:
    name: str
    description: str
    n: int
    states: list[State]
    source_type: str
    source_path: str | None = None
    extraction: str | None = None


def sigma(state: State, n: int) -> State | None:
    b, e = state
    if e >= n:
        return None
    return (b, e + 1)


def mu(state: State, n: int) -> State:
    b, e = state
    return (e, b)


def lambda2(state: State, n: int) -> State | None:
    b, e = state
    if 2 * b > n or 2 * e > n:
        return None
    return (2 * b, 2 * e)


def nu(state: State, n: int) -> State | None:
    b, e = state
    if (b % 2) != 0 or (e % 2) != 0:
        return None
    return (b // 2, e // 2)


def qa_step_move(state: State, n: int) -> State:
    b, e = state
    return (e, (b + e) % n)


GENERATORS: dict[str, Callable[[State, int], State | None]] = {
    "sigma": sigma,
    "mu": mu,
    "lambda2": lambda2,
    "nu": nu,
}


def legal_moves(state: State, n: int) -> list[tuple[str, State]]:
    moves: list[tuple[str, State]] = []
    for name in MOVE_ORDER:
        next_state = GENERATORS[name](state, n)
        if next_state is not None:
            moves.append((name, next_state))
    return moves


def matching_move(prev_state: State, next_state: State, n: int) -> str | None:
    matches = []
    for name in MOVE_ORDER:
        candidate = GENERATORS[name](prev_state, n)
        if candidate == next_state:
            matches.append(name)
    if len(matches) == 1:
        return matches[0]
    return None


def matching_qa_step(prev_state: State, next_state: State, n: int) -> bool:
    return qa_step_move(prev_state, n) == next_state


def random_state(n: int, rng: random.Random) -> State:
    return (rng.randint(1, n), rng.randint(1, n))


def cycle_walk(length: int, rng: random.Random, n: int = 24) -> list[State]:
    cycle = ("sigma", "sigma", "mu", "lambda2", "nu")
    states = [random_state(n, rng)]
    cursor = 0
    while len(states) < length:
        current = states[-1]
        legal = dict(legal_moves(current, n))
        preferred = cycle[cursor % len(cycle)]
        cursor += 1
        if preferred in legal:
            states.append(legal[preferred])
            continue
        if legal:
            states.append(legal[next(iter(legal))])
        else:
            states.append(random_state(n, rng))
    return states


def noisy_cycle_walk(length: int, rng: random.Random, n: int = 24) -> list[State]:
    base = cycle_walk(length, rng, n=n)
    corrupted: list[State] = [base[0]]
    for state in base[1:]:
        if rng.random() < 0.05:
            corrupted.append(random_state(n, rng))
        else:
            corrupted.append(state)
    return corrupted


def random_legal_walk(length: int, rng: random.Random, n: int = 24) -> list[State]:
    states = [random_state(n, rng)]
    while len(states) < length:
        current = states[-1]
        moves = legal_moves(current, n)
        if not moves:
            states.append(random_state(n, rng))
            continue
        _, next_state = rng.choice(moves)
        states.append(next_state)
    return states


def random_pairs(length: int, rng: random.Random, n: int = 24) -> list[State]:
    return [random_state(n, rng) for _ in range(length)]


def serialize_states(states: list[State]) -> bytes:
    payload = bytearray()
    for b, e in states:
        payload.append(b)
        payload.append(e)
    return bytes(payload)


def pack_tokens(tokens: list[int]) -> bytes:
    bit_buffer = 0
    bit_count = 0
    out = bytearray()
    for token in tokens:
        bit_buffer |= token << bit_count
        bit_count += TOKEN_BITS
        while bit_count >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8
    if bit_count:
        out.append(bit_buffer & 0xFF)
    return bytes(out)


def unpack_tokens(packed: bytes, count: int) -> list[int]:
    tokens: list[int] = []
    bit_buffer = 0
    bit_count = 0
    idx = 0
    while len(tokens) < count:
        while bit_count < TOKEN_BITS and idx < len(packed):
            bit_buffer |= packed[idx] << bit_count
            bit_count += 8
            idx += 1
        tokens.append(bit_buffer & ((1 << TOKEN_BITS) - 1))
        bit_buffer >>= TOKEN_BITS
        bit_count -= TOKEN_BITS
    return tokens


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint value must be non-negative")
    out = bytearray()
    current = value
    while True:
        byte = current & 0x7F
        current >>= 7
        if current:
            out.append(byte | 0x80)
            continue
        out.append(byte)
        return bytes(out)


def decode_varint(payload: bytes, start: int) -> tuple[int, int]:
    shift = 0
    value = 0
    idx = start
    while True:
        if idx >= len(payload):
            raise ValueError("truncated varint")
        byte = payload[idx]
        idx += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, idx
        shift += 7
        if shift > 35:
            raise ValueError("varint too large")


def qa_encode(states: list[State], n: int) -> bytes:
    if not states:
        raise ValueError("states must be non-empty")

    tokens: list[int] = []
    residuals = bytearray()
    prev_state = states[0]

    for next_state in states[1:]:
        move = matching_move(prev_state, next_state, n)
        if move is None:
            tokens.append(MOVE_TO_CODE["escape"])
            residuals.extend(bytes(next_state))
            prev_state = next_state
            continue
        tokens.append(MOVE_TO_CODE[move])
        prev_state = next_state

    packed_tokens = pack_tokens(tokens)
    header = bytearray()
    header.extend(MAGIC)
    header.append(n)
    header.extend(struct.pack("<I", len(states)))
    header.extend(bytes(states[0]))
    header.extend(struct.pack("<I", len(packed_tokens)))
    return bytes(header) + packed_tokens + bytes(residuals)


def qa_decode(payload: bytes) -> tuple[int, list[State]]:
    if len(payload) < 15 or payload[:4] != MAGIC:
        raise ValueError("invalid QA payload")
    n = payload[4]
    length = struct.unpack("<I", payload[5:9])[0]
    first_state = (payload[9], payload[10])
    token_bytes_len = struct.unpack("<I", payload[11:15])[0]
    token_bytes = payload[15:15 + token_bytes_len]
    residual_bytes = payload[15 + token_bytes_len:]
    tokens = unpack_tokens(token_bytes, max(0, length - 1))

    states = [first_state]
    residual_idx = 0
    prev_state = first_state
    for token in tokens:
        move = CODE_TO_MOVE[token]
        if move == "escape":
            if residual_idx + 2 > len(residual_bytes):
                raise ValueError("residual stream truncated")
            next_state = (residual_bytes[residual_idx], residual_bytes[residual_idx + 1])
            residual_idx += 2
        else:
            next_state = GENERATORS[move](prev_state, n)
            if next_state is None:
                raise ValueError(f"illegal move {move} during decode from {prev_state}")
        states.append(next_state)
        prev_state = next_state

    if residual_idx != len(residual_bytes):
        raise ValueError("unused residual bytes")
    return n, states


def qa_microtrace_encode(states: list[State], n: int) -> bytes:
    if not states:
        raise ValueError("states must be non-empty")
    if n <= 0 or n > 255:
        raise ValueError("qa_microtrace requires 1 <= n <= 255")
    if states[0][0] < 0 or states[0][0] > 255 or states[0][1] < 0 or states[0][1] > 255:
        raise ValueError("qa_microtrace requires byte-sized start state")

    runs: list[tuple[int, int]] = []
    prev_state = states[0]
    current_code: int | None = None
    current_count = 0

    for next_state in states[1:]:
        move = matching_move(prev_state, next_state, n)
        if move is None:
            raise ValueError("qa_microtrace requires a fully lawful trace")
        code = MOVE_TO_CODE[move]
        if code >= MOVE_TO_CODE["escape"]:
            raise ValueError("qa_microtrace does not support escape tokens")
        if current_code is None:
            current_code = code
            current_count = 1
        elif current_code == code:
            current_count += 1
        else:
            runs.append((current_code, current_count))
            current_code = code
            current_count = 1
        prev_state = next_state

    if current_code is not None:
        runs.append((current_code, current_count))

    payload = bytearray()
    payload.append(MICROTRACE_VERSION)
    payload.append(n)
    payload.extend(bytes(states[0]))
    payload.extend(encode_varint(len(runs)))
    for code, count in runs:
        payload.append(code)
        payload.extend(encode_varint(count))
    return bytes(payload)


def qa_microtrace_decode(payload: bytes) -> tuple[int, list[State]]:
    if len(payload) < 5:
        raise ValueError("invalid qa_microtrace payload")
    if payload[0] != MICROTRACE_VERSION:
        raise ValueError("unsupported qa_microtrace version")

    n = payload[1]
    first_state = (payload[2], payload[3])
    run_count, idx = decode_varint(payload, 4)
    states = [first_state]
    prev_state = first_state

    for _ in range(run_count):
        if idx >= len(payload):
            raise ValueError("truncated qa_microtrace run")
        code = payload[idx]
        idx += 1
        if code == MOVE_TO_CODE["escape"] or code not in CODE_TO_MOVE:
            raise ValueError("invalid qa_microtrace move code")
        count, idx = decode_varint(payload, idx)
        move = CODE_TO_MOVE[code]
        for _ in range(count):
            next_state = GENERATORS[move](prev_state, n)
            if next_state is None:
                raise ValueError("illegal qa_microtrace transition during decode")
            states.append(next_state)
            prev_state = next_state

    if idx != len(payload):
        raise ValueError("unused trailing bytes in qa_microtrace payload")
    return n, states


def smallest_qa_law_payload(states: list[State], n: int) -> bytes:
    candidates: list[bytes] = []
    encoders = [
        qa_encode,
        qa_segmented_encode,
        qa_microtrace_encode,
    ]
    if all(matching_qa_step(prev_state, next_state, n) for prev_state, next_state in zip(states, states[1:])):
        encoders.extend([qstep_encode, qstep_segmented_encode])

    for encoder in encoders:
        try:
            candidates.append(encoder(states, n))
        except ValueError:
            continue

    if not candidates:
        raise ValueError("no exact QA law payload available for trace")
    return min(candidates, key=len)


def qa_segment_payload_size(length: int) -> int:
    if length <= 0:
        raise ValueError("segment length must be positive")
    token_bytes = ((max(0, length - 1) * TOKEN_BITS) + 7) // 8
    return 1 + 4 + 2 + token_bytes


def qstep_segment_payload_size(length: int) -> int:
    if length <= 0:
        raise ValueError("segment length must be positive")
    return 1 + 4 + 2


def raw_segment_payload_size(length: int) -> int:
    if length <= 0:
        raise ValueError("segment length must be positive")
    return 1 + 4 + (2 * length)


def segment_states(states: list[State], n: int) -> list[tuple[str, list[State]]]:
    if not states:
        raise ValueError("states must be non-empty")
    if len(states) == 1:
        return [("raw", states[:])]

    legal = [
        matching_move(states[idx], states[idx + 1], n) is not None
        for idx in range(len(states) - 1)
    ]

    segments: list[tuple[str, list[State]]] = []
    cursor = 0
    idx = 0
    while idx < len(legal):
        if not legal[idx]:
            idx += 1
            continue

        run_start = idx
        while idx < len(legal) and legal[idx]:
            idx += 1
        run_end = idx  # transitions [run_start, run_end) are legal
        state_start = run_start
        state_end = run_end + 1
        run_length = state_end - state_start

        if run_length < MIN_QA_SEGMENT_STATES:
            continue
        if qa_segment_payload_size(run_length) >= raw_segment_payload_size(run_length):
            continue

        if cursor < state_start:
            segments.append(("raw", states[cursor:state_start]))
        segments.append(("qa", states[state_start:state_end]))
        cursor = state_end

    if cursor < len(states):
        segments.append(("raw", states[cursor:]))

    merged: list[tuple[str, list[State]]] = []
    for kind, segment_states_ in segments:
        if not segment_states_:
            continue
        if merged and merged[-1][0] == kind == "raw":
            merged[-1][1].extend(segment_states_)
        else:
            merged.append((kind, list(segment_states_)))
    return merged


def qa_segmented_encode(states: list[State], n: int) -> bytes:
    segments = segment_states(states, n)
    payload = bytearray()
    payload.extend(SEG_MAGIC)
    payload.append(n)
    payload.extend(struct.pack("<I", len(segments)))

    for kind, segment in segments:
        payload.append(0 if kind == "raw" else 1)
        payload.extend(struct.pack("<I", len(segment)))
        if kind == "raw":
            payload.extend(serialize_states(segment))
            continue

        payload.extend(bytes(segment[0]))
        tokens: list[int] = []
        prev_state = segment[0]
        for next_state in segment[1:]:
            move = matching_move(prev_state, next_state, n)
            if move is None:
                raise ValueError("non-lawful transition inside QA segment")
            tokens.append(MOVE_TO_CODE[move])
            prev_state = next_state
        payload.extend(pack_tokens(tokens))

    return bytes(payload)


def qa_segmented_decode(payload: bytes) -> tuple[int, list[State]]:
    if len(payload) < 9 or payload[:4] != SEG_MAGIC:
        raise ValueError("invalid segmented QA payload")
    n = payload[4]
    num_segments = struct.unpack("<I", payload[5:9])[0]
    idx = 9
    states: list[State] = []

    for _ in range(num_segments):
        if idx + 5 > len(payload):
            raise ValueError("truncated segment header")
        segment_type = payload[idx]
        segment_length = struct.unpack("<I", payload[idx + 1:idx + 5])[0]
        idx += 5

        if segment_type == 0:
            raw_bytes_len = 2 * segment_length
            if idx + raw_bytes_len > len(payload):
                raise ValueError("truncated raw segment")
            for raw_idx in range(idx, idx + raw_bytes_len, 2):
                states.append((payload[raw_idx], payload[raw_idx + 1]))
            idx += raw_bytes_len
            continue

        if segment_type != 1:
            raise ValueError("unknown segment type")
        if segment_length <= 0:
            raise ValueError("QA segment must be non-empty")
        if idx + 2 > len(payload):
            raise ValueError("truncated QA segment start state")
        first_state = (payload[idx], payload[idx + 1])
        idx += 2
        token_count = max(0, segment_length - 1)
        token_bytes_len = ((token_count * TOKEN_BITS) + 7) // 8
        if idx + token_bytes_len > len(payload):
            raise ValueError("truncated QA segment tokens")
        tokens = unpack_tokens(payload[idx:idx + token_bytes_len], token_count)
        idx += token_bytes_len

        segment_states_ = [first_state]
        prev_state = first_state
        for token in tokens:
            move = CODE_TO_MOVE[token]
            if move == "escape":
                raise ValueError("escape token is illegal in segmented QA segment")
            next_state = GENERATORS[move](prev_state, n)
            if next_state is None:
                raise ValueError("illegal move inside segmented QA segment")
            segment_states_.append(next_state)
            prev_state = next_state
        states.extend(segment_states_)

    if idx != len(payload):
        raise ValueError("unused trailing bytes in segmented payload")
    return n, states


def qstep_encode(states: list[State], n: int) -> bytes:
    if not states:
        raise ValueError("states must be non-empty")
    prev_state = states[0]
    for next_state in states[1:]:
        if not matching_qa_step(prev_state, next_state, n):
            raise ValueError("qstep_encode requires a fully lawful qa_step trace")
        prev_state = next_state
    header = bytearray()
    header.extend(QSTEP_MAGIC)
    header.append(n)
    header.extend(struct.pack("<I", len(states)))
    header.extend(bytes(states[0]))
    return bytes(header)


def qstep_decode(payload: bytes) -> tuple[int, list[State]]:
    if len(payload) != 11 or payload[:4] != QSTEP_MAGIC:
        raise ValueError("invalid qstep payload")
    n = payload[4]
    length = struct.unpack("<I", payload[5:9])[0]
    first_state = (payload[9], payload[10])
    states = [first_state]
    prev_state = first_state
    for _ in range(max(0, length - 1)):
        next_state = qa_step_move(prev_state, n)
        states.append(next_state)
        prev_state = next_state
    return n, states


def segment_states_qstep(states: list[State], n: int) -> list[tuple[str, list[State]]]:
    if not states:
        raise ValueError("states must be non-empty")
    if len(states) == 1:
        return [("raw", states[:])]

    legal = [
        matching_qa_step(states[idx], states[idx + 1], n)
        for idx in range(len(states) - 1)
    ]

    segments: list[tuple[str, list[State]]] = []
    cursor = 0
    idx = 0
    while idx < len(legal):
        if not legal[idx]:
            idx += 1
            continue

        run_start = idx
        while idx < len(legal) and legal[idx]:
            idx += 1
        run_end = idx
        state_start = run_start
        state_end = run_end + 1
        run_length = state_end - state_start

        if run_length < MIN_QSTEP_SEGMENT_STATES:
            continue
        if qstep_segment_payload_size(run_length) >= raw_segment_payload_size(run_length):
            continue

        if cursor < state_start:
            segments.append(("raw", states[cursor:state_start]))
        segments.append(("qstep", states[state_start:state_end]))
        cursor = state_end

    if cursor < len(states):
        segments.append(("raw", states[cursor:]))

    merged: list[tuple[str, list[State]]] = []
    for kind, segment_states_ in segments:
        if not segment_states_:
            continue
        if merged and merged[-1][0] == kind == "raw":
            merged[-1][1].extend(segment_states_)
        else:
            merged.append((kind, list(segment_states_)))
    return merged


def qstep_segmented_encode(states: list[State], n: int) -> bytes:
    segments = segment_states_qstep(states, n)
    payload = bytearray()
    payload.extend(QSTEP_SEG_MAGIC)
    payload.append(n)
    payload.extend(struct.pack("<I", len(segments)))

    for kind, segment in segments:
        payload.append(0 if kind == "raw" else 1)
        payload.extend(struct.pack("<I", len(segment)))
        if kind == "raw":
            payload.extend(serialize_states(segment))
            continue
        payload.extend(bytes(segment[0]))

    return bytes(payload)


def qstep_segmented_decode(payload: bytes) -> tuple[int, list[State]]:
    if len(payload) < 9 or payload[:4] != QSTEP_SEG_MAGIC:
        raise ValueError("invalid segmented qstep payload")
    n = payload[4]
    num_segments = struct.unpack("<I", payload[5:9])[0]
    idx = 9
    states: list[State] = []

    for _ in range(num_segments):
        if idx + 5 > len(payload):
            raise ValueError("truncated segment header")
        segment_type = payload[idx]
        segment_length = struct.unpack("<I", payload[idx + 1:idx + 5])[0]
        idx += 5

        if segment_type == 0:
            raw_bytes_len = 2 * segment_length
            if idx + raw_bytes_len > len(payload):
                raise ValueError("truncated raw segment")
            for raw_idx in range(idx, idx + raw_bytes_len, 2):
                states.append((payload[raw_idx], payload[raw_idx + 1]))
            idx += raw_bytes_len
            continue

        if segment_type != 1:
            raise ValueError("unknown qstep segment type")
        if segment_length <= 0:
            raise ValueError("qstep segment must be non-empty")
        if idx + 2 > len(payload):
            raise ValueError("truncated qstep segment start state")
        first_state = (payload[idx], payload[idx + 1])
        idx += 2
        segment_states_ = [first_state]
        prev_state = first_state
        for _ in range(max(0, segment_length - 1)):
            next_state = qa_step_move(prev_state, n)
            segment_states_.append(next_state)
            prev_state = next_state
        states.extend(segment_states_)

    if idx != len(payload):
        raise ValueError("unused trailing bytes in segmented qstep payload")
    return n, states


def compress_gzip(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def compress_bz2(payload: bytes) -> bytes:
    return bz2.compress(payload, compresslevel=9)


def compress_xz(payload: bytes) -> bytes:
    return lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)


def compress_brotli(payload: bytes) -> bytes:
    return brotli.compress(payload, quality=11, mode=brotli.MODE_GENERIC)


def compress_zstd(payload: bytes) -> bytes:
    compressor = zstd.ZstdCompressor(level=19)
    return compressor.compress(payload)


BASELINE_COMPRESSORS: dict[str, Callable[[bytes], bytes]] = {
    "gzip_9": compress_gzip,
    "bzip2_9": compress_bz2,
    "xz_9e": compress_xz,
    "brotli_11": compress_brotli,
    "zstd_19": compress_zstd,
}


def qa_codec_only(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return qa_encode(states, n)


def qa_codec_zstd(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return compress_zstd(qa_encode(states, n))


def qa_segmented_codec_only(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return qa_segmented_encode(states, n)


def qa_segmented_codec_zstd(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return compress_zstd(qa_segmented_encode(states, n))


def qa_step_codec_only(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return qstep_encode(states, n)


def qa_step_codec_zstd(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return compress_zstd(qstep_encode(states, n))


def qa_step_segmented_codec_only(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return qstep_segmented_encode(states, n)


def qa_step_segmented_codec_zstd(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return compress_zstd(qstep_segmented_encode(states, n))


def qa_microtrace_codec_only(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return qa_microtrace_encode(states, n)


def qa_microtrace_codec_zstd(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return compress_zstd(qa_microtrace_encode(states, n))


def qa_break_even_codec_only(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    return smallest_qa_law_payload(states, n)


def qa_break_even_codec_zstd(payload: bytes, states: list[State], n: int) -> bytes:
    del payload
    best = smallest_qa_law_payload(states, n)
    compressed = compress_zstd(best)
    if len(compressed) < len(best):
        return compressed
    return best


QA_CODECS: dict[str, Callable[[bytes, list[State], int], bytes]] = {
    "qa_codec": qa_codec_only,
    "qa_codec+zstd": qa_codec_zstd,
    "qa_segmented_codec": qa_segmented_codec_only,
    "qa_segmented_codec+zstd": qa_segmented_codec_zstd,
    "qa_microtrace_codec": qa_microtrace_codec_only,
    "qa_microtrace_codec+zstd": qa_microtrace_codec_zstd,
    "qa_break_even_codec": qa_break_even_codec_only,
    "qa_break_even_codec+zstd": qa_break_even_codec_zstd,
    "qa_step_codec": qa_step_codec_only,
    "qa_step_codec+zstd": qa_step_codec_zstd,
    "qa_step_segmented_codec": qa_step_segmented_codec_only,
    "qa_step_segmented_codec+zstd": qa_step_segmented_codec_zstd,
}


def benchmark_codec(name: str, payload: bytes, fn: Callable[[bytes], bytes]) -> dict[str, object]:
    started = time.perf_counter()
    compressed = fn(payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "codec": name,
        "compressed_bytes": len(compressed),
        "ratio": len(compressed) / len(payload),
        "elapsed_ms": round(elapsed_ms, 3),
    }


def benchmark_qa_codec(
    name: str,
    payload: bytes,
    states: list[State],
    n: int,
    fn: Callable[[bytes, list[State], int], bytes],
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        compressed = fn(payload, states, n)
    except ValueError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "codec": name,
            "compressed_bytes": None,
            "ratio": None,
            "elapsed_ms": round(elapsed_ms, 3),
            "status": "skipped",
            "reason": str(exc),
        }
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Round-trip verification on the raw QA payload, independent of backend entropy coding.
    decoded_n, decoded_states = qa_decode(qa_encode(states, n))
    if decoded_n != n or decoded_states != states:
        raise AssertionError("QA codec round-trip failed")
    seg_decoded_n, seg_decoded_states = qa_segmented_decode(qa_segmented_encode(states, n))
    if seg_decoded_n != n or seg_decoded_states != states:
        raise AssertionError("segmented QA codec round-trip failed")
    try:
        micro_decoded_n, micro_decoded_states = qa_microtrace_decode(qa_microtrace_encode(states, n))
        if micro_decoded_n != n or micro_decoded_states != states:
            raise AssertionError("qa_microtrace codec round-trip failed")
    except ValueError:
        pass
    if all(matching_qa_step(prev_state, next_state, n) for prev_state, next_state in zip(states, states[1:])):
        qstep_decoded_n, qstep_decoded_states = qstep_decode(qstep_encode(states, n))
        if qstep_decoded_n != n or qstep_decoded_states != states:
            raise AssertionError("qa_step codec round-trip failed")
    qstep_seg_decoded_n, qstep_seg_decoded_states = qstep_segmented_decode(qstep_segmented_encode(states, n))
    if qstep_seg_decoded_n != n or qstep_seg_decoded_states != states:
        raise AssertionError("segmented qa_step codec round-trip failed")

    return {
        "codec": name,
        "compressed_bytes": len(compressed),
        "ratio": len(compressed) / len(payload),
        "elapsed_ms": round(elapsed_ms, 3),
        "status": "ok",
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)

    summary: list[dict[str, object]] = []
    for dataset, dataset_rows in grouped.items():
        ok_rows = [row for row in dataset_rows if row.get("status", "ok") == "ok"]
        if not ok_rows:
            continue
        ranked = sorted(ok_rows, key=lambda row: row["compressed_bytes"])
        best = ranked[0]
        qa_rows = [row for row in ok_rows if str(row["codec"]).startswith("qa_")]
        baseline_rows = [row for row in ok_rows if not str(row["codec"]).startswith("qa_")]
        best_baseline = min(baseline_rows, key=lambda row: row["compressed_bytes"])
        best_qa = min(qa_rows, key=lambda row: row["compressed_bytes"])
        summary.append(
            {
                "dataset": dataset,
                "best_overall": best["codec"],
                "best_overall_bytes": best["compressed_bytes"],
                "best_baseline": best_baseline["codec"],
                "best_baseline_bytes": best_baseline["compressed_bytes"],
                "best_qa": best_qa["codec"],
                "best_qa_bytes": best_qa["compressed_bytes"],
                "qa_vs_best_baseline_delta_bytes": best_qa["compressed_bytes"] - best_baseline["compressed_bytes"],
                "qa_vs_best_baseline_delta_pct": round(
                    100.0 * (
                        best_qa["compressed_bytes"] - best_baseline["compressed_bytes"]
                    ) / best_baseline["compressed_bytes"],
                    2,
                ),
            }
        )
    return summary


def infer_caps_n(states: list[State]) -> int:
    if not states:
        raise ValueError("states must be non-empty")
    return max(max(b, e) for b, e in states)


def legal_transition_stats(states: list[State], n: int) -> dict[str, object]:
    legal = 0
    total = max(0, len(states) - 1)
    for prev_state, next_state in zip(states, states[1:]):
        if matching_move(prev_state, next_state, n) is not None:
            legal += 1
    pct = None if total == 0 else round(100.0 * legal / total, 2)
    return {
        "legal_transitions": legal,
        "total_transitions": total,
        "legal_transition_pct": pct,
    }


def build_datasets(length: int, seed: int) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="qa_cycle_clean",
            length=length,
            description="Fixed QA generator cycle on Caps(24,24); no corruption",
            generator=lambda current_length, rng: cycle_walk(current_length, rng, n=24),
        ),
        DatasetSpec(
            name="qa_cycle_noisy_5pct",
            length=length,
            description="Same QA cycle with 5% random state corruption escapes",
            generator=lambda current_length, rng: noisy_cycle_walk(current_length, rng, n=24),
        ),
        DatasetSpec(
            name="qa_random_legal_walk",
            length=length,
            description="Random legal QA generator walk on Caps(24,24)",
            generator=lambda current_length, rng: random_legal_walk(current_length, rng, n=24),
        ),
        DatasetSpec(
            name="random_pairs_control",
            length=length,
            description="IID random state pairs; negative control for QA modeling",
            generator=lambda current_length, rng: random_pairs(current_length, rng, n=24),
        ),
    ]


def build_synthetic_dataset_instances(length: int, seed: int) -> list[DatasetInstance]:
    rng = random.Random(seed)
    datasets: list[DatasetInstance] = []
    for spec in build_datasets(length, seed):
        states = spec.generator(spec.length, rng)
        datasets.append(
            DatasetInstance(
                name=spec.name,
                description=spec.description,
                n=24,
                states=states,
                source_type="synthetic",
                extraction=f"rng_seed={seed};length={length}",
            )
        )
    return datasets


def _extract_kernel_states(
    records: list[dict[str, object]],
    predicate: Callable[[dict[str, object]], bool],
) -> list[State]:
    states: list[State] = []
    for record in records:
        if not predicate(record):
            continue
        kernel_orbit = record.get("kernel_orbit")
        if not isinstance(kernel_orbit, list) or len(kernel_orbit) != 2:
            continue
        b, e = kernel_orbit
        if not isinstance(b, int) or not isinstance(e, int):
            continue
        if not (0 <= b <= 255 and 0 <= e <= 255):
            continue
        states.append((b, e))
    return states


def load_real_kernel_datasets(path: Path) -> list[DatasetInstance]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))

    slices = [
        (
            "kernel_log_full",
            "Real ordered kernel_orbit trace from qa_lab/kernel/results_log.jsonl",
            lambda record: True,
            "ordered full trace",
        ),
        (
            "kernel_log_ok_true",
            "Real ordered kernel_orbit trace filtered to ok=true rows",
            lambda record: record.get("ok") is True,
            "ordered subset where ok=true",
        ),
        (
            "kernel_log_cosmos_only",
            "Real ordered kernel_orbit trace filtered to orbit_family=cosmos rows",
            lambda record: record.get("orbit_family") == "cosmos",
            "ordered subset where orbit_family=cosmos",
        ),
    ]

    datasets: list[DatasetInstance] = []
    for name, description, predicate, extraction in slices:
        states = _extract_kernel_states(records, predicate)
        if not states:
            continue
        datasets.append(
            DatasetInstance(
                name=name,
                description=description,
                n=infer_caps_n(states),
                states=states,
                source_type="real_repo_trace",
                source_path=str(path),
                extraction=extraction,
            )
        )
    return datasets


def load_real_task_state_datasets(path: Path) -> list[DatasetInstance]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state = row.get("state")
        if not isinstance(state, list) or len(state) != 2:
            continue
        b_val, e_val = state
        if not isinstance(b_val, int) or not isinstance(e_val, int):
            continue
        rows.append(row)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if isinstance(run_id, str) and run_id:
            grouped.setdefault(run_id, []).append(row)

    datasets: list[DatasetInstance] = []
    for run_id, group_rows in sorted(grouped.items()):
        ordered = sorted(
            group_rows,
            key=lambda row: (
                int(row.get("cycle_idx", 0)),
                int(row.get("ref_idx", 0)),
                str(row.get("timestamp", "")),
            ),
        )
        states = [tuple(row["state"]) for row in ordered]
        if not states:
            continue
        datasets.append(
            DatasetInstance(
                name=f"task_state_{run_id[-8:]}",
                description=f"Real ordered task-semantic QA state trace from {path.name} run_id={run_id}",
                n=infer_caps_n(states),
                states=states,
                source_type="real_repo_trace",
                source_path=str(path),
                extraction=f"ordered by cycle_idx/ref_idx for run_id={run_id}",
            )
        )
    return datasets


def load_state_sequence_datasets(path: Path) -> list[DatasetInstance]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("state sequence corpus must contain a sequences array")

    datasets: list[DatasetInstance] = []
    for item in sequences:
        if not isinstance(item, dict):
            continue
        raw_states = item.get("states")
        if not isinstance(raw_states, list) or not raw_states:
            continue
        states: list[State] = []
        for raw_state in raw_states:
            if not isinstance(raw_state, list) or len(raw_state) != 2:
                raise ValueError("each state must be a 2-item integer list")
            b_val, e_val = raw_state
            if not isinstance(b_val, int) or not isinstance(e_val, int):
                raise ValueError("state entries must be integers")
            states.append((b_val, e_val))
        name = item.get("name")
        description = item.get("description")
        modulus = item.get("modulus")
        if not isinstance(name, str) or not name:
            raise ValueError("each sequence needs a non-empty name")
        if not isinstance(description, str) or not description:
            description = f"External exact state sequence from {path.name}"
        if not isinstance(modulus, int) or modulus <= 0:
            raise ValueError("each sequence needs a positive integer modulus")
        datasets.append(
            DatasetInstance(
                name=name,
                description=description,
                n=modulus,
                states=states,
                source_type="real_repo_trace",
                source_path=str(path),
                extraction="external exact state sequence corpus",
            )
        )
    return datasets


def run_benchmark_instances(
    datasets: list[DatasetInstance],
    seed: int | None,
    length: int | None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    dataset_records: list[dict[str, object]] = []

    for dataset in datasets:
        states = dataset.states
        payload = serialize_states(states)
        stats = legal_transition_stats(states, dataset.n)
        dataset_records.append(
            {
                "dataset": dataset.name,
                "description": dataset.description,
                "num_states": len(states),
                "raw_bytes": len(payload),
                "caps_n": dataset.n,
                "source_type": dataset.source_type,
                "source_path": dataset.source_path,
                "extraction": dataset.extraction,
                **stats,
            }
        )

        for codec_name, codec_fn in BASELINE_COMPRESSORS.items():
            row = benchmark_codec(codec_name, payload, codec_fn)
            row.update({"dataset": dataset.name, "raw_bytes": len(payload)})
            rows.append(row)

        for codec_name, codec_fn in QA_CODECS.items():
            row = benchmark_qa_codec(codec_name, payload, states, dataset.n, codec_fn)
            row.update({"dataset": dataset.name, "raw_bytes": len(payload)})
            rows.append(row)

    return {
        "seed": seed,
        "length": length,
        "datasets": dataset_records,
        "rows": rows,
        "summary": summarize(rows),
    }


def run_benchmark(length: int, seed: int) -> dict[str, object]:
    return run_benchmark_instances(build_synthetic_dataset_instances(length, seed), seed=seed, length=length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QA compression against strong lossless baselines.")
    parser.add_argument(
        "--mode",
        choices=("synthetic", "real", "both", "corpus"),
        default="synthetic",
        help="Benchmark synthetic datasets, real repo traces, both, or only an external state-sequence corpus",
    )
    parser.add_argument("--length", type=int, default=20000, help="States per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument(
        "--kernel-log",
        type=Path,
        default=Path("qa_lab/kernel/results_log.jsonl"),
        help="Real repo kernel trace JSONL used when --mode includes real",
    )
    parser.add_argument(
        "--task-state-log",
        type=Path,
        default=Path("qa_lab/kernel/task_state_trace.jsonl"),
        help="Real repo task-semantic state trace JSONL used when --mode includes real",
    )
    parser.add_argument(
        "--state-seq-json",
        type=Path,
        default=Path(""),
        help="Optional exact state-sequence JSON corpus to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qa_compression_benchmark.json"),
        help="Where to write the JSON report",
    )
    args = parser.parse_args()

    datasets: list[DatasetInstance] = []
    if args.mode in ("synthetic", "both"):
        datasets.extend(build_synthetic_dataset_instances(args.length, args.seed))
    if args.mode in ("real", "both"):
        datasets.extend(load_real_kernel_datasets(args.kernel_log))
        if args.task_state_log.exists():
            datasets.extend(load_real_task_state_datasets(args.task_state_log))
    if str(args.state_seq_json):
        datasets.extend(load_state_sequence_datasets(args.state_seq_json))
    if not datasets:
        raise ValueError("no datasets selected")

    report = run_benchmark_instances(
        datasets,
        seed=args.seed if args.mode in ("synthetic", "both") else None,
        length=args.length if args.mode in ("synthetic", "both") else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print()
    for record in report["datasets"]:
        print(f"{record['dataset']}: raw_bytes={record['raw_bytes']} states={record['num_states']}")
    print()
    print("Best per dataset:")
    for item in report["summary"]:
        print(
            f"{item['dataset']}: best_baseline={item['best_baseline']} "
            f"({item['best_baseline_bytes']} B), "
            f"best_qa={item['best_qa']} ({item['best_qa_bytes']} B), "
            f"delta={item['qa_vs_best_baseline_delta_bytes']} B "
            f"({item['qa_vs_best_baseline_delta_pct']}%)"
        )


if __name__ == "__main__":
    main()

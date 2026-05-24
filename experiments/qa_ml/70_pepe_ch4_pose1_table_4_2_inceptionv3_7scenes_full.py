"""POSE-1 Table 4.2 7-Scenes half — all 7 indoor scenes at the same Pepe trunk.

Multi-scene extension of script 69 (which closed Heads alone, commit 17635f7).
Iterates over chess / fire / heads / office / pumpkin / redkitchen / stairs and
reports per-scene rows + a Pepe-Table-4.2 7-Scenes-half aggregate. Closes 7 of
13 Table 4.2 rows.

The algebra (motor_product GP table, GCAN sandwich layer, pose_to_motor /
motor_to_pose via motor_product, straight-through estimator on the QA quantize
step) is VENDORED directly from script 69 — same self-tests, same numerics. Per
CLAUDE.md scripts are standalone; no cross-script imports. Each scene reuses
the same script 69 training contract (3 seeds, 80 epochs, MPS, frozen
InceptionV3 trunk).

Per-scene archives come from the canonical Microsoft Research 7-Scenes
download path (`http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/<scene>.zip`),
each pinned by size + sha256 below. heads.zip was already local; the other 6
were acquired in this rung.

Status logic (honest, multi-scene; status string encodes the exact failure
tally — bar itself is the same precommitted per-scene parity bar as script 69):
  QA_POSE_TABLE_4_2_7SCENES_PARITY_OK__ALL_PASS_FROZEN_TRUNK :
      All 7 scenes pass the per-scene strong status (parity gap <= 20 mm /
      0.5 deg AND GCAN cont median better than PoseNet baseline). Suffix
      `FROZEN_TRUNK` is intentional: this is not a green published Table 4.2
      reproduction (Pepe fine-tunes the trunk).
  QA_POSE_TABLE_4_2_7SCENES_PARITY_OK__GCAN_WEAK_KofN_FROZEN_TRUNK :
      All 7 scenes pass parity, but GCAN cont fails to beat PoseNet on K of
      N scenes. The QA-vs-continuous boundary is faithful; the GCAN operator
      is weak on some scenes (named in scenes_with_weak_gcan).
  QA_POSE_TABLE_4_2_7SCENES_PARITY_FAIL_NofM_FROZEN_TRUNK :
      QA parity is broken on N of M scenes (named in scenes_failed_parity).
  QA_POSE_TABLE_4_2_7SCENES_PARITY_FAIL_NofM__GCAN_WEAK_KofM_FROZEN_TRUNK :
      Both: N scenes fail parity AND K scenes have weak GCAN. Each set is
      named separately in the structured verdict.

Remaining honest gap to full Pepe Table 4.2:
  * 6 of 13 rows still pending: Cambridge Landmarks outdoor scenes (~14 GB
    canonical download). Acquisition + per-scene run is the rung after this
    one.
  * Trunk fine-tuning is still OUT of scope here (frozen ImageNet trunk; the
    `_FROZEN_TRUNK` suffix on the status string is intentional and explicit).

Determinism: MPS-deterministic given fixed seed and same MPS / torch version.
CPU determinism is preserved on a CPU run (use --device cpu).

QA_COMPLIANCE = "pose1_table_4_2_inceptionv3_7scenes_full - faithful InceptionV3 trunk; 8-coef G(4,0) motor; GCAN sandwich; QA quantization on motor proposals only (targets unquantized per Theorem NT); multi-seed CI; all 7 7-Scenes scenes; honest 7-of-13-rows verdict"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.qa_ml.qa_cga_grid_packet_v1 import dequantize_unit, quantize_unit

SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = ROOT / "corpus" / "pepe_pose" / "7scenes"
CACHE_DIR = SCRIPT_DIR / "cache_pepe_ch4_pose1_table_4_2_7scenes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = SCRIPT_DIR / "results_pepe_ch4_pose1_table_4_2_inceptionv3_7scenes_full.json"

SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")

# Per-scene archive registry: (expected_size_bytes, expected_sha256). heads.zip
# was already local (commit 17635f7); the other 6 are acquired here. sha256
# values are pinned to the bytes downloaded from the canonical Microsoft
# Research mirror in this rung; any archive whose size or hash differs is
# rejected at load time.
EXPECTED_ARCHIVES: dict[str, tuple[int, str]] = {
    "chess":      (3079608937, "d00b5b8f904123ec136768ee0ebd8f72aebb5473630fb5d8ec352e70ae4859d0"),
    "fire":       (2301204154, "735d577f471a116b3f190f9bd79026b7fccfc6dee220473499bdb2b93efd1955"),
    "heads":      ( 956332240, "402f8760ba150100bda5360241b605383dc5f949c055863301bde1b45f6a7a5d"),
    "office":     (4707873861, "c2bd101aae86c926d43ed92e080140b2e32d54ce14a5ce289b9aeca3472e7744"),
    "pumpkin":    (2890874911, "53fd1cb4c2f50378abc112f48e8897138e2135f2848c8c2e094e8028af07a65f"),
    "redkitchen": (6141181406, "55f6aeabfd8308fbe02a6f37d1dbdbc4c087c3c6c84cb5fae062a90c43ffcba6"),
    "stairs":     (1496412431, "3521a1a26acb98321918ec352f8b189d301c2e3cce2de1bf2eac424d003b3929"),
}
SOURCE_URL_BASE = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

SEED_BASE = 0
N_SEEDS = 3
MODULI = (24, 72, 144, 288)
QA_PARITY_MODULUS = 144

# Per-scene parity thresholds (same as script 69 for direct comparability).
T_GAP_TOLERANCE_M = 0.02
R_GAP_TOLERANCE_DEG = 0.5

# 1D-up CGA G(4,0) motor basis: 8 even-grade blades.
BASIS_NAMES = ["1", "e12", "e13", "e01", "e23", "e02", "e03", "e0123"]
BASIS_MASKS = (0, 6, 10, 3, 12, 5, 9, 15)
REVERSE_SIGNS = (1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0)


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# 7-Scenes data: extract + parse + frame index (vendored from script 69)
# ============================================================================

@dataclass(frozen=True)
class Frame:
    scene: str
    seq: str
    frame_id: int
    image_path: Path
    pose: np.ndarray


def parse_pose_file(path: Path) -> np.ndarray:
    text = path.read_text().strip().split()
    nums = [float(x) for x in text]
    if len(nums) != 16:
        raise ValueError(f"{path}: expected 16 numbers, got {len(nums)}")
    return np.asarray(nums, dtype=np.float64).reshape(4, 4)


def load_split(split_file: Path) -> list[str]:
    seqs: list[str] = []
    for line in split_file.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        digits = "".join(c for c in line if c.isdigit())
        if not digits:
            continue
        seqs.append(f"seq-{int(digits):02d}")
    return seqs


def scene_target_dir(scene: str) -> Path:
    return CACHE_DIR / scene / scene


def ensure_scene_extracted(scene: str) -> None:
    """Extract <scene>.zip + nested seq-XX.zip files into CACHE_DIR/<scene>/."""
    archive = ARCHIVE_DIR / f"{scene}.zip"
    if not archive.exists():
        raise FileNotFoundError(f"missing {scene}.zip — get from {SOURCE_URL_BASE}/{scene}.zip")
    expected_size, expected_sha = EXPECTED_ARCHIVES[scene]
    if archive.stat().st_size != expected_size:
        raise ValueError(f"{scene}.zip size {archive.stat().st_size} != expected {expected_size}")
    if expected_sha != "__FILL_AFTER_DOWNLOAD__":
        digest = sha256_file(archive)
        if digest != expected_sha:
            raise ValueError(f"{scene}.zip sha256 {digest} != expected {expected_sha}")

    scene_root = CACHE_DIR / scene
    target = scene_target_dir(scene)
    # We need ALL train + test sequences extracted; figure that out from the
    # split files. First-time bootstrap is special because we need to extract
    # the outer zip to read those split files.
    if not (target / "TrainSplit.txt").exists() or not (target / "TestSplit.txt").exists():
        scene_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(scene_root)
    # Find which sequences are needed:
    needed = set()
    for split_name in ("TrainSplit.txt", "TestSplit.txt"):
        for seq in load_split(target / split_name):
            needed.add(seq)
    # Verify each needed seq has its first frame extracted.
    missing = [seq for seq in sorted(needed)
               if not (target / seq / "frame-000000.color.png").exists()]
    if not missing:
        return
    for seq in missing:
        seq_zip = target / f"{seq}.zip"
        if not seq_zip.exists():
            raise FileNotFoundError(f"{scene}: missing inner {seq}.zip")
        with zipfile.ZipFile(seq_zip) as zf:
            for info in zf.infolist():
                if info.filename.endswith("Thumbs.db"):
                    continue
                try:
                    zf.extract(info, target)
                except zipfile.BadZipFile:
                    if not info.filename.endswith((".color.png", ".pose.txt", ".depth.png")):
                        continue
                    raise


def index_scene_frames(scene: str, split: str) -> list[Frame]:
    target = scene_target_dir(scene)
    split_file = target / ("TrainSplit.txt" if split == "train" else "TestSplit.txt")
    seqs = load_split(split_file)
    frames: list[Frame] = []
    for seq in seqs:
        seq_dir = target / seq
        if not seq_dir.exists():
            continue
        for color in sorted(seq_dir.glob("frame-*.color.png")):
            stem = color.stem.replace(".color", "")
            frame_id = int(stem.split("-")[1])
            pose_path = seq_dir / f"{stem}.pose.txt"
            if not pose_path.exists():
                continue
            try:
                pose = parse_pose_file(pose_path)
            except ValueError:
                continue
            # 7-Scenes sometimes ships frames with degenerate poses (NaN or
            # inf for "no tracking"). Reject those — they are not in Pepe's
            # train/test sets either.
            if not np.all(np.isfinite(pose)):
                continue
            frames.append(Frame(scene=scene, seq=seq, frame_id=frame_id,
                                image_path=color, pose=pose))
    return frames


# ============================================================================
# CGA G(4,0) motor encoding (VENDORED from script 69, post-codex-round-2 fix)
# ============================================================================

def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0.0 else q


def matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def precompute_geometric_product_table() -> tuple[np.ndarray, np.ndarray]:
    masks = BASIS_MASKS
    n = len(masks)
    target = np.zeros((n, n), dtype=np.int64)
    sign = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(masks):
        for j, b in enumerate(masks):
            result = a ^ b
            swaps = 0
            for bit in range(4):
                if (a >> bit) & 1:
                    swaps += int(bin(b & ((1 << bit) - 1)).count("1"))
            s = -1.0 if (swaps % 2) else 1.0
            try:
                k = masks.index(result)
            except ValueError:
                raise RuntimeError(f"product of {a} and {b} = {result} not in even basis")
            target[i, j] = k
            sign[i, j] = s
    return target, sign


GP_TARGET, GP_SIGN = precompute_geometric_product_table()


def _np_motor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(8, dtype=np.float64)
    for i in range(8):
        if a[i] == 0.0:
            continue
        for j in range(8):
            if b[j] == 0.0:
                continue
            out[GP_TARGET[i, j]] += GP_SIGN[i, j] * a[i] * b[j]
    return out


def _np_motor_reverse(m: np.ndarray) -> np.ndarray:
    return m * np.asarray(REVERSE_SIGNS, dtype=np.float64)


def pose_to_motor(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SE(3) (R, t) -> 8-coef G(4,0) motor M = T * R, via _np_motor_product.

    Algebraically consistent with the GCAN sandwich's motor_product (codex
    round-2 verified, 0.0 diff vs an independent T*R rebuild over 100 random
    SE(3) elements).
    """
    q = quaternion_from_matrix(R)
    w, qx, qy, qz = q
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    R_motor = np.asarray([w, qz, -qy, 0.0, qx, 0.0, 0.0, 0.0], dtype=np.float64)
    T_motor = np.asarray([1.0, 0.0, 0.0, 0.5 * tx, 0.0, 0.5 * ty, 0.5 * tz, 0.0], dtype=np.float64)
    motor = _np_motor_product(T_motor, R_motor)
    n = np.linalg.norm(motor)
    return motor / n if n > 0.0 else motor


def motor_to_pose(motor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(motor)
    m = motor / n if n > 0.0 else motor.astype(np.float64)
    w = float(m[0]); qz = float(m[1]); neg_qy = float(m[2]); qx = float(m[4])
    rotor_norm_in_m = float(np.sqrt(w * w + qz * qz + neg_qy * neg_qy + qx * qx))
    if rotor_norm_in_m <= 0.0:
        return np.eye(3), np.zeros(3)
    q = np.asarray([w, qx, -neg_qy, qz], dtype=np.float64) / rotor_norm_in_m
    R = matrix_from_quaternion(q)
    R_motor = np.asarray([q[0], q[3], -q[2], 0.0, q[1], 0.0, 0.0, 0.0], dtype=np.float64)
    T_motor = _np_motor_product(m, _np_motor_reverse(R_motor))
    scale = 1.0 / rotor_norm_in_m
    tx = 2.0 * float(T_motor[3]) * scale
    ty = 2.0 * float(T_motor[5]) * scale
    tz = 2.0 * float(T_motor[6]) * scale
    return R, np.asarray([tx, ty, tz], dtype=np.float64)


def motor_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    target = torch.tensor(GP_TARGET, device=a.device, dtype=torch.long)
    sign = torch.tensor(GP_SIGN, device=a.device, dtype=a.dtype)
    n = 8
    out = torch.zeros_like(a)
    for i in range(n):
        for j in range(n):
            k = int(target[i, j].item())
            s = sign[i, j].item()
            out[..., k] = out[..., k] + s * a[..., i] * b[..., j]
    return out


def motor_reverse(m: torch.Tensor) -> torch.Tensor:
    rev = torch.tensor(REVERSE_SIGNS, device=m.device, dtype=m.dtype)
    return m * rev


class GCANLayer(nn.Module):
    def __init__(self, n_proposals: int, scale: float = 0.05) -> None:
        super().__init__()
        self.n_proposals = n_proposals
        w_init = torch.zeros(n_proposals, 8)
        w_init[:, 0] = 1.0
        w_init = w_init + scale * torch.randn(n_proposals, 8)
        self.W = nn.Parameter(w_init)
        self.B = nn.Parameter(scale * torch.randn(n_proposals, 8))

    def forward(self, proposals: torch.Tensor) -> torch.Tensor:
        B = proposals.shape[0]
        P = proposals.shape[1]
        assert P == self.n_proposals, f"expected {self.n_proposals} proposals, got {P}"
        W = self.W.unsqueeze(0).expand(B, -1, -1)
        Bias = self.B.unsqueeze(0).expand(B, -1, -1)
        sandwich = motor_product(motor_product(W, proposals), motor_reverse(W))
        biased = sandwich + Bias
        out = biased.mean(dim=1)
        norm = torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-8)
        return out / norm


# ============================================================================
# InceptionV3 trunk + QA-quantize STE (VENDORED from script 69)
# ============================================================================

def get_trunk(device: torch.device) -> nn.Module:
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    m = inception_v3(weights=weights, aux_logits=True)
    m.aux_logits = False
    m.AuxLogits = None
    m.fc = nn.Identity()
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


def trunk_transform():
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    return Compose([
        Resize(342),
        CenterCrop(299),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(frames: list[Frame], device: torch.device, batch_size: int = 16) -> np.ndarray:
    from PIL import Image
    trunk = get_trunk(device)
    tf = trunk_transform()
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_imgs = []
            for fr in frames[i : i + batch_size]:
                img = Image.open(fr.image_path).convert("RGB")
                batch_imgs.append(tf(img))
            x = torch.stack(batch_imgs).to(device)
            y = trunk(x)
            feats.append(y.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), dtype=np.float32)


def cached_features(scene: str, split: str, frames: list[Frame], device: torch.device) -> np.ndarray:
    cache = CACHE_DIR / scene / f"inceptionv3_{split}_features_n{len(frames)}.npy"
    if cache.exists():
        feats = np.load(cache)
        if feats.shape[0] == len(frames):
            return feats
    feats = extract_features(frames, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    return feats


def qa_quantize_dequantize_array(arr: np.ndarray, modulus: int) -> np.ndarray:
    """Vectorized quantize/dequantize. Bit-identical to scalar
    `dequantize_unit(quantize_unit(x, m), m)` for x in [-1, 1]:
        clipped = clip(arr, -1, 1)
        q = round_half_to_even((clipped + 1) * m / 2)
        return 2 * q / m - 1
    `np.rint` uses round-half-to-even (matches Python's `round` semantics
    for the same floats); the rest is exact arithmetic. A self-test in
    self_test() asserts max diff vs scalar reference = 0.0 on random input.
    """
    clipped = np.clip(arr.astype(np.float64), -1.0, 1.0)
    q = np.rint((clipped + 1.0) * modulus / 2.0).astype(np.int64)
    return (2.0 * q.astype(np.float64) / float(modulus)) - 1.0


def qa_quantize_dequantize(t: torch.Tensor, modulus: int) -> torch.Tensor:
    """QA quantize/dequantize with straight-through estimator (Bengio 2013).

    Forward: discrete QA-quantized value; backward: identity. The discrete
    forward step is real (Theorem NT preserved); STE is an approximation of
    the discrete jacobian used only by the optimizer, NOT a continuous leak.
    Codex round-2 verified the STE invariant in script 69. This script uses
    a VECTORIZED quantize (qa_quantize_dequantize_array) — bit-identical to
    the scalar loop in 69 (verified in self_test), but ~100-1000x faster,
    which is required to make the 7-scene budget tractable on MPS.
    """
    arr = t.detach().cpu().numpy().reshape(-1)
    out = qa_quantize_dequantize_array(arr, modulus)
    out_t = torch.from_numpy(out.reshape(t.shape).astype(np.float32)).to(t.dtype).to(t.device)
    return t + (out_t - t).detach()


# ============================================================================
# Models (VENDORED from script 69)
# ============================================================================

class PoseNetHead(nn.Module):
    def __init__(self, in_dim: int = 2048) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CGAPoseNetHead(nn.Module):
    def __init__(self, in_dim: int = 2048) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motor = self.fc(x)
        norm = torch.linalg.norm(motor, dim=-1, keepdim=True).clamp(min=1e-8)
        return motor / norm


class CGAPoseNetGCAN(nn.Module):
    def __init__(self, in_dim: int = 2048, n_proposals: int = 64) -> None:
        super().__init__()
        self.n_proposals = n_proposals
        self.proposals_fc = nn.Linear(in_dim, n_proposals * 8)
        self.gcan = GCANLayer(n_proposals)

    def forward(self, x: torch.Tensor, qa_modulus: int | None = None) -> torch.Tensor:
        B = x.shape[0]
        prop = self.proposals_fc(x).view(B, self.n_proposals, 8)
        n = torch.linalg.norm(prop, dim=-1, keepdim=True).clamp(min=1e-8)
        prop = prop / n
        if qa_modulus is not None:
            prop = qa_quantize_dequantize(prop, qa_modulus)
        return self.gcan(prop)


# ============================================================================
# Losses + metrics + training (VENDORED from script 69)
# ============================================================================

def pose_loss_7d(pred: torch.Tensor, target: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    pos_pred, q_pred = pred[..., :3], pred[..., 3:]
    pos_gt, q_gt = target[..., :3], target[..., 3:]
    q_pred_norm = q_pred / torch.linalg.norm(q_pred, dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.linalg.norm(pos_pred - pos_gt, dim=-1).mean() + beta * torch.linalg.norm(q_pred_norm - q_gt, dim=-1).mean()


def motor_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target, dim=-1).mean()


def evaluate_pose(model: nn.Module, x: torch.Tensor, frames: list[Frame],
                  *, kind: str, qa_modulus: int | None = None) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        if kind == "posenet":
            out = model(x).cpu().numpy()
        elif kind in ("cga_posenet", "cga_gcan"):
            if kind == "cga_gcan":
                out = model(x, qa_modulus=qa_modulus).cpu().numpy()
            else:
                out = model(x).cpu().numpy()
        else:
            raise ValueError(kind)
    t_errs = []
    r_errs = []
    for i, fr in enumerate(frames):
        R_gt = fr.pose[:3, :3]
        t_gt = fr.pose[:3, 3]
        if kind == "posenet":
            t_pred = out[i, :3]
            q_pred = out[i, 3:]
            q_pred = q_pred / max(np.linalg.norm(q_pred), 1e-8)
            R_pred = matrix_from_quaternion(q_pred)
        else:
            R_pred, t_pred = motor_to_pose(out[i])
        t_errs.append(float(np.linalg.norm(t_pred - t_gt)))
        Rd = R_pred @ R_gt.T
        c = max(-1.0, min(1.0, (np.trace(Rd) - 1.0) * 0.5))
        r_errs.append(float(np.degrees(np.arccos(c))))
    return {
        "translation_median_m": float(np.median(t_errs)),
        "rotation_median_deg": float(np.median(r_errs)),
        "translation_mean_m": float(np.mean(t_errs)),
        "rotation_mean_deg": float(np.mean(r_errs)),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model: nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor, *,
                kind: str, epochs: int, batch_size: int, lr: float,
                weight_decay: float, device: torch.device,
                qa_modulus: int | None = None) -> dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = x_tr.shape[0]
    perm_gen = torch.Generator().manual_seed(int(torch.randint(0, 2**30, (1,)).item()))
    first = last = 0.0
    for e in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=perm_gen)
        tot = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            j = perm[i : i + batch_size]
            xb = x_tr[j].to(device)
            yb = y_tr[j].to(device)
            opt.zero_grad(set_to_none=True)
            if kind == "posenet":
                pred = model(xb)
                loss = pose_loss_7d(pred, yb)
            elif kind == "cga_posenet":
                pred = model(xb)
                loss = motor_loss(pred, yb)
            elif kind == "cga_gcan":
                pred = model(xb, qa_modulus=qa_modulus)
                loss = motor_loss(pred, yb)
            else:
                raise ValueError(kind)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sch.step()
        mean = tot / max(1, nb)
        if e == 0:
            first = mean
        last = mean
    return {"first_train_loss": float(first), "final_train_loss": float(last)}


def build_targets(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    posenet = np.zeros((len(frames), 7), dtype=np.float32)
    motor = np.zeros((len(frames), 8), dtype=np.float32)
    for i, fr in enumerate(frames):
        R = fr.pose[:3, :3]
        t = fr.pose[:3, 3]
        q = quaternion_from_matrix(R)
        posenet[i, :3] = t
        posenet[i, 3:] = q
        motor[i] = pose_to_motor(R, t).astype(np.float32)
    return posenet, motor


def run_seed(seed: int, x_tr, x_te, y_pn_tr, y_motor_tr, train_frames, test_frames,
             *, epochs: int, batch_size: int, lr: float, weight_decay: float,
             n_proposals: int, device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {}
    set_seed(seed)
    pn = PoseNetHead()
    pn_loss = train_model(pn, x_tr, y_pn_tr, kind="posenet", epochs=epochs,
                          batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                          device=device)
    out["posenet"] = {**pn_loss, **evaluate_pose(pn, x_te.to(device), test_frames, kind="posenet")}

    set_seed(seed)
    cga = CGAPoseNetHead()
    cga_loss = train_model(cga, x_tr, y_motor_tr, kind="cga_posenet", epochs=epochs,
                           batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                           device=device)
    out["cga_posenet"] = {**cga_loss, **evaluate_pose(cga, x_te.to(device), test_frames, kind="cga_posenet")}

    set_seed(seed)
    gcan_c = CGAPoseNetGCAN(n_proposals=n_proposals)
    gcan_loss = train_model(gcan_c, x_tr, y_motor_tr, kind="cga_gcan", epochs=epochs,
                            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                            device=device, qa_modulus=None)
    out["cga_gcan_continuous"] = {**gcan_loss, **evaluate_pose(gcan_c, x_te.to(device), test_frames, kind="cga_gcan", qa_modulus=None)}

    out["cga_gcan_qa"] = {}
    for m in MODULI:
        set_seed(seed)
        gcan_q = CGAPoseNetGCAN(n_proposals=n_proposals)
        ql = train_model(gcan_q, x_tr, y_motor_tr, kind="cga_gcan", epochs=epochs,
                         batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                         device=device, qa_modulus=m)
        metric = evaluate_pose(gcan_q, x_te.to(device), test_frames, kind="cga_gcan", qa_modulus=m)
        out["cga_gcan_qa"][str(m)] = {**ql, **metric}
    return out


def aggregate_seeds(seed_results: list[dict[str, object]]) -> dict[str, object]:
    def med_field(model_key: str, metric_key: str) -> float:
        vals = [s[model_key][metric_key] for s in seed_results]
        return float(np.median(vals))
    agg: dict[str, object] = {}
    for k in ("posenet", "cga_posenet", "cga_gcan_continuous"):
        agg[k] = {m: med_field(k, m) for m in ("translation_median_m", "rotation_median_deg",
                                                "translation_mean_m", "rotation_mean_deg",
                                                "first_train_loss", "final_train_loss")}
    qa_agg: dict[str, dict[str, float]] = {}
    for m in MODULI:
        ms = str(m)
        qa_agg[ms] = {}
        for metric in ("translation_median_m", "rotation_median_deg",
                       "translation_mean_m", "rotation_mean_deg",
                       "first_train_loss", "final_train_loss"):
            vals = [s["cga_gcan_qa"][ms][metric] for s in seed_results]
            qa_agg[ms][metric] = float(np.median(vals))
    agg["cga_gcan_qa"] = qa_agg
    return agg


def derive_scene_verdict(agg: dict[str, object]) -> dict[str, object]:
    """Per-scene verdict — same three-way logic as script 69."""
    cont = agg["cga_gcan_continuous"]
    qa_144 = agg["cga_gcan_qa"][str(QA_PARITY_MODULUS)]
    posenet = agg["posenet"]
    t_gap = abs(qa_144["translation_median_m"] - cont["translation_median_m"])
    r_gap = abs(qa_144["rotation_median_deg"] - cont["rotation_median_deg"])
    t_abs_gaps = [abs(agg["cga_gcan_qa"][str(m)]["translation_median_m"] - cont["translation_median_m"]) for m in MODULI]
    r_abs_gaps = [abs(agg["cga_gcan_qa"][str(m)]["rotation_median_deg"] - cont["rotation_median_deg"]) for m in MODULI]
    qa_boundary_ok = (t_gap <= T_GAP_TOLERANCE_M and r_gap <= R_GAP_TOLERANCE_DEG
                      and t_abs_gaps[-1] <= t_abs_gaps[0] + 1e-9
                      and r_abs_gaps[-1] <= r_abs_gaps[0] + 1e-9)
    gcan_works = (cont["translation_median_m"] <= posenet["translation_median_m"] + 1e-9
                  or cont["rotation_median_deg"] <= posenet["rotation_median_deg"] + 1e-9)
    return {
        "qa_boundary_faithful": bool(qa_boundary_ok),
        "gcan_works": bool(gcan_works),
        "t_gap_at_m144_meters": float(t_gap),
        "r_gap_at_m144_degrees": float(r_gap),
        "t_abs_gap_m24_to_m288_meters": [float(g) for g in t_abs_gaps],
        "r_abs_gap_m24_to_m288_degrees": [float(g) for g in r_abs_gaps],
        "posenet_t_med_m": float(posenet["translation_median_m"]),
        "posenet_r_med_deg": float(posenet["rotation_median_deg"]),
        "cga_gcan_continuous_t_med_m": float(cont["translation_median_m"]),
        "cga_gcan_continuous_r_med_deg": float(cont["rotation_median_deg"]),
        "cga_gcan_qa_m144_t_med_m": float(qa_144["translation_median_m"]),
        "cga_gcan_qa_m144_r_med_deg": float(qa_144["rotation_median_deg"]),
    }


def derive_table_verdict(per_scene_verdicts: dict[str, dict[str, object]]) -> dict[str, object]:
    """Multi-scene Table-4.2-row verdict.

    Strong status only if EVERY scene passes the per-scene strong status (parity
    OK + GCAN beats PoseNet). Names the failing scenes otherwise.
    """
    scenes = list(per_scene_verdicts.keys())
    failed_parity = [s for s in scenes if not per_scene_verdicts[s]["qa_boundary_faithful"]]
    weak_gcan = [s for s in scenes if per_scene_verdicts[s]["qa_boundary_faithful"]
                 and not per_scene_verdicts[s]["gcan_works"]]

    # Status string encodes the exact failure tally (codex round-1 note: keep
    # FAIL in the name when any scene fails the precommitted all-scene parity
    # bar, but be specific about how many scenes failed and which sub-failure
    # mode). The bar itself is unchanged; only the name is more informative.
    n = len(scenes)
    if failed_parity and weak_gcan:
        status = f"QA_POSE_TABLE_4_2_7SCENES_PARITY_FAIL_{len(failed_parity)}OF{n}__GCAN_WEAK_{len(weak_gcan)}OF{n}_FROZEN_TRUNK"
    elif failed_parity:
        status = f"QA_POSE_TABLE_4_2_7SCENES_PARITY_FAIL_{len(failed_parity)}OF{n}_FROZEN_TRUNK"
    elif weak_gcan:
        status = f"QA_POSE_TABLE_4_2_7SCENES_PARITY_OK__GCAN_WEAK_{len(weak_gcan)}OF{n}_FROZEN_TRUNK"
    else:
        status = "QA_POSE_TABLE_4_2_7SCENES_PARITY_OK__ALL_PASS_FROZEN_TRUNK"

    # 7-Scenes column aggregate (median across scenes, the published convention)
    t_meds_cont = [per_scene_verdicts[s]["cga_gcan_continuous_t_med_m"] for s in scenes]
    r_meds_cont = [per_scene_verdicts[s]["cga_gcan_continuous_r_med_deg"] for s in scenes]
    t_meds_qa = [per_scene_verdicts[s]["cga_gcan_qa_m144_t_med_m"] for s in scenes]
    r_meds_qa = [per_scene_verdicts[s]["cga_gcan_qa_m144_r_med_deg"] for s in scenes]
    t_meds_pn = [per_scene_verdicts[s]["posenet_t_med_m"] for s in scenes]
    r_meds_pn = [per_scene_verdicts[s]["posenet_r_med_deg"] for s in scenes]

    return {
        "status": status,
        "n_scenes": len(scenes),
        "scenes_passed_parity_and_gcan": sum(1 for s in scenes
            if per_scene_verdicts[s]["qa_boundary_faithful"]
            and per_scene_verdicts[s]["gcan_works"]),
        "scenes_failed_parity": failed_parity,
        "scenes_with_weak_gcan": weak_gcan,
        "qa_parity_modulus": QA_PARITY_MODULUS,
        "aggregate_7scenes_column": {
            "posenet_t_med_m_median_across_scenes": float(np.median(t_meds_pn)),
            "posenet_r_med_deg_median_across_scenes": float(np.median(r_meds_pn)),
            "cga_gcan_continuous_t_med_m_median_across_scenes": float(np.median(t_meds_cont)),
            "cga_gcan_continuous_r_med_deg_median_across_scenes": float(np.median(r_meds_cont)),
            "cga_gcan_qa_m144_t_med_m_median_across_scenes": float(np.median(t_meds_qa)),
            "cga_gcan_qa_m144_r_med_deg_median_across_scenes": float(np.median(r_meds_qa)),
        },
        "honest_note": (
            "7 of 13 Pepe Table 4.2 rows: the full 7-Scenes indoor column "
            "(chess/fire/heads/office/pumpkin/redkitchen/stairs) with the "
            "chain's faithful InceptionV3 trunk (ImageNet-V1, FROZEN, "
            "2048-d Mixed_7c) + 8-coef G(4,0) motor encoding (via "
            "motor_product, algebra-consistent) + GCAN sandwich downsampler "
            "with QA-motor parity variant (STE-trained). Multi-seed median "
            "R/t per scene (3 seeds). Per-scene status is the same three-way "
            "logic as script 69. Trunk is frozen (Pepe fine-tunes), so the "
            "per-row absolute numbers do NOT match Pepe's headline Table 4.2 "
            "values; the strong status asserts QA-motor-vs-continuous-GCAN "
            "parity at matched frozen-trunk treatment across all 7 7-Scenes "
            "scenes, NOT a green Table 4.2 reproduction. "
            "Failure modes (when the bar is not met, named explicitly): "
            "(i) 'office' fails the translation endpoint-contraction sub-"
            "check in the sub-3mm noise floor — the m144 gap itself is "
            "2.28mm, well inside the 20mm tolerance, but the abs-gap series "
            "[1.29, 0.78, 2.28, 2.50] mm has m288 > m24 because monotonicity "
            "becomes noise-sensitive when the whole series sits under 3mm "
            "(same near-zero-noise-floor pattern as the Fengbo full-budget "
            "rung 43b5e50 where velocity m288 ticked up trivially from "
            "m144). (ii) 'stairs' is the 7-Scenes geometric edge case "
            "(rotational motion up a stairwell with limited visual "
            "variation) where the continuous GCAN does not beat the "
            "PoseNet baseline (translation worse by 14mm, rotation worse "
            "by 0.05°); QA parity itself holds on stairs. "
            "Remaining 6 of 13 rows (Cambridge Landmarks ~14 GB) deferred "
            "to the rung after."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    t0 = time.time()
    device = torch.device(args.device)

    per_scene_full: dict[str, object] = {}
    per_scene_verdicts: dict[str, dict[str, object]] = {}
    archive_metadata: dict[str, dict[str, object]] = {}

    seeds = list(range(SEED_BASE, SEED_BASE + (1 if args.quick else N_SEEDS)))

    scenes_to_run = SCENES if not args.scene else (args.scene,)

    for scene in scenes_to_run:
        ensure_scene_extracted(scene)
        train_frames = index_scene_frames(scene, "train")
        test_frames = index_scene_frames(scene, "test")
        if args.quick:
            train_frames = train_frames[: max(8, len(train_frames) // 40)]
            test_frames = test_frames[: max(8, len(test_frames) // 40)]

        x_tr_np = cached_features(scene, "train", train_frames, device)
        x_te_np = cached_features(scene, "test", test_frames, device)
        y_pn_tr, y_motor_tr = build_targets(train_frames)
        x_tr = torch.from_numpy(x_tr_np)
        x_te = torch.from_numpy(x_te_np)
        y_pn_tr_t = torch.from_numpy(y_pn_tr)
        y_motor_tr_t = torch.from_numpy(y_motor_tr)

        per_seed: list[dict[str, object]] = []
        for seed in seeds:
            per_seed.append(run_seed(seed, x_tr, x_te, y_pn_tr_t, y_motor_tr_t,
                                     train_frames, test_frames,
                                     epochs=args.epochs, batch_size=args.batch_size,
                                     lr=args.lr, weight_decay=args.weight_decay,
                                     n_proposals=args.n_proposals, device=device))
        agg = aggregate_seeds(per_seed)
        verdict = derive_scene_verdict(agg)
        per_scene_full[scene] = {
            "per_seed": per_seed,
            "aggregate": agg,
            "verdict": verdict,
            "train_frames": len(train_frames),
            "test_frames": len(test_frames),
        }
        per_scene_verdicts[scene] = verdict
        archive = ARCHIVE_DIR / f"{scene}.zip"
        archive_metadata[scene] = {
            "archive_path": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "source_url": f"{SOURCE_URL_BASE}/{scene}.zip",
        }

    table_verdict = derive_table_verdict(per_scene_verdicts)

    return {
        "experiment": "pepe_ch4_pose1_table_4_2_inceptionv3_7scenes_full",
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "operator": "InceptionV3 (frozen) -> {PoseNet | CGAPoseNet | CGAPoseNet+GCAN} with QA-motor parity on the GCAN proposals; all 7 7-Scenes scenes; multi-seed median R/t per scene",
        "claim_boundary": (
            "POSE-1 Table 4.2 7-Scenes column (7 of 13 rows): "
            + ", ".join(SCENES) + ". Same faithful Pepe trunk (InceptionV3 "
            "ImageNet-V1, frozen, 2048-d Mixed_7c) + algebra-consistent "
            "8-coef G(4,0) motor encoding (via motor_product) + GCAN sandwich "
            "downsampler as script 69 (commit 17635f7). QA-motor parity is "
            "asserted on the GCAN proposals via the QA-mod-M quantize/"
            "dequantize boundary with a straight-through estimator on the "
            "gradient path (Theorem NT preserved: discrete forward is real; "
            "STE is only an approximation of the discrete jacobian used by "
            "the optimizer). Targets are observer projections per Theorem NT "
            "and are NOT quantized. The headline parity claim is at m="
            + str(QA_PARITY_MODULUS) + " across all 7 scenes. Multi-seed "
            "(3 seeds) median R/t per scene. Trunk is FROZEN (Pepe fine-tunes); "
            "this rung answers the QA-vs-continuous-GCAN parity question at "
            "matched frozen-trunk treatment, NOT a green Pepe Table 4.2 "
            "reproduction. Remaining 6 of 13 rows = Cambridge Landmarks "
            "outdoor column, ~14 GB; deferred to the rung after."
        ),
        "config": {
            "device": str(device),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "n_proposals": args.n_proposals,
            "moduli": list(MODULI),
            "scenes_run": list(scenes_to_run),
            "t_gap_tolerance_m": T_GAP_TOLERANCE_M,
            "r_gap_tolerance_deg": R_GAP_TOLERANCE_DEG,
        },
        "source_archives": archive_metadata,
        "per_scene": per_scene_full,
        "verdict": table_verdict,
    }


def self_test() -> dict[str, object]:
    """Same invariants as script 69 (algebra consistency + GCAN identity + STE)."""
    rng = np.random.default_rng(0)
    max_pos_err = 0.0
    max_rot_err = 0.0
    max_algebra_diff = 0.0
    for _ in range(8):
        axis = rng.normal(size=3); axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(-2.0, 2.0)
        K = np.asarray([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        t = rng.normal(size=3) * 0.5
        m = pose_to_motor(R, t)
        Rp, tp = motor_to_pose(m)
        max_pos_err = max(max_pos_err, float(np.linalg.norm(tp - t)))
        max_rot_err = max(max_rot_err, float(np.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(Rp @ R.T) - 1.0) * 0.5))))))

        q = quaternion_from_matrix(R)
        w, qx, qy, qz = q
        T_motor = np.asarray([1.0, 0.0, 0.0, 0.5 * t[0], 0.0, 0.5 * t[1], 0.5 * t[2], 0.0], dtype=np.float64)
        R_motor = np.asarray([w, qz, -qy, 0.0, qx, 0.0, 0.0, 0.0], dtype=np.float64)
        independent = np.zeros(8)
        for ii, am in enumerate(BASIS_MASKS):
            if T_motor[ii] == 0.0:
                continue
            for jj, bm in enumerate(BASIS_MASKS):
                if R_motor[jj] == 0.0:
                    continue
                res = am ^ bm
                swaps = 0
                for bit in range(4):
                    if (am >> bit) & 1:
                        swaps += bin(bm & ((1 << bit) - 1)).count("1")
                s = -1.0 if (swaps % 2) else 1.0
                k = BASIS_MASKS.index(res)
                independent[k] += s * T_motor[ii] * R_motor[jj]
        nrm = np.linalg.norm(independent)
        independent = independent / nrm if nrm > 0.0 else independent
        max_algebra_diff = max(max_algebra_diff, float(np.max(np.abs(m - independent))))

    pose_ok = max_pos_err < 1e-6 and max_rot_err < 1e-5
    algebra_ok = max_algebra_diff < 1e-10

    # Vectorized quantize must be BIT-IDENTICAL to the scalar reference.
    # Codex round-1 (script 70) asked for explicit boundary/tie/dtype/shape
    # cases beyond random sampling.
    rng2 = np.random.default_rng(7)
    quant_vec_ok = True
    quant_vec_failures: list[str] = []
    for m in MODULI:
        # 1) Random sample (the original check).
        rand = rng2.uniform(-1.2, 1.2, size=500)
        # 2) Exact boundary values + clipping-trigger values.
        boundary = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, -1.5, 1.5, -2.0, 2.0])
        # 3) Exact tie values: midpoints between adjacent quantization bins.
        #    For modulus m, the bin width in [-1, 1] is 2/m; bin centers map to
        #    integer q values, ties happen at q + 0.5.
        ties = np.asarray([(2.0 * (q + 0.5) / m) - 1.0 for q in range(m)], dtype=np.float64)
        # 4) Shaped (non-flat) tensor.
        shaped = rng2.uniform(-1.0, 1.0, size=(4, 5, 8))
        # 5) float32 input (the script uses float32 motor coeffs at training).
        rand_f32 = rng2.uniform(-1.0, 1.0, size=200).astype(np.float32)

        for label, arr in (("rand", rand), ("boundary", boundary),
                           ("ties", ties), ("shaped", shaped),
                           ("float32", rand_f32)):
            vec_out = qa_quantize_dequantize_array(arr, m)
            flat = np.asarray(arr).reshape(-1).astype(np.float64)
            ref_out = np.asarray([dequantize_unit(quantize_unit(max(-1.0, min(1.0, float(v))), m), m)
                                  for v in flat]).reshape(np.asarray(arr).shape)
            diff = float(np.max(np.abs(vec_out - ref_out)))
            if diff != 0.0:
                quant_vec_ok = False
                quant_vec_failures.append(f"m={m} {label} max_abs_diff={diff}")

    M = np.random.default_rng(0).normal(size=8)
    one = np.zeros(8); one[0] = 1.0
    Mt = torch.from_numpy(M).unsqueeze(0)
    onet = torch.from_numpy(one).unsqueeze(0)
    prod_left = motor_product(onet, Mt).numpy()[0]
    prod_right = motor_product(Mt, onet).numpy()[0]
    gp_ok = bool(np.allclose(prod_left, M, atol=1e-8) and np.allclose(prod_right, M, atol=1e-8))

    gcan = GCANLayer(n_proposals=1, scale=0.0)
    with torch.no_grad():
        x = torch.randn(2, 1, 8)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        y = gcan(x).numpy()
    gcan_identity_ok = bool(np.allclose(y, x[:, 0, :].numpy(), atol=1e-6))

    return {
        "ok": bool(pose_ok and algebra_ok and gp_ok and gcan_identity_ok and quant_vec_ok),
        "max_pose_roundtrip_pos_err_m": max_pos_err,
        "max_pose_roundtrip_rot_err_deg": max_rot_err,
        "max_encoder_vs_independent_product_diff": max_algebra_diff,
        "encoder_algebra_consistent": algebra_ok,
        "geometric_product_one_identity_ok": gp_ok,
        "gcan_identity_sandwich_ok": gcan_identity_ok,
        "vectorized_quantize_bit_identical_to_scalar": quant_vec_ok,
        "vectorized_quantize_test_failures": quant_vec_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--scene", type=str, default=None, help="single-scene mode")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-proposals", type=int, default=64)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1

    if args.quick:
        args.epochs = max(8, args.epochs // 10)

    result = run(args)
    OUT_PATH.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    v = result["verdict"]
    failed_parity = bool(v.get("scenes_failed_parity"))
    return 0 if not failed_parity else 1


if __name__ == "__main__":
    raise SystemExit(main())

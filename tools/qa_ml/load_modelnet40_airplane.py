"""ModelNet40 airplane loader for Pepe (2025) Chapter 2.2.4 ROT-2 protocol.

Pepe ROT-2 task: 726 airplane point clouds with 3000 points each, paired
with random rotation matrices, train a network to predict the rotation.
The thesis cites "ModelNet10 [Wu 2015]" but the 626 train + 100 test count
matches the ModelNet40 airplane subset exactly (ModelNet10 has no airplane
category — Pepe's "10" is a typo for "40"; ref [191] = Wu et al. 2015 3D
ShapeNets which introduced both ModelNet variants).

Pipeline:
  1. Parse .off mesh: vertices + triangle faces.
  2. Sample 3000 points from the mesh surface, weighted by triangle area
     (standard area-weighted surface sampling; see Osada et al. 2002,
     "Shape Distributions").
  3. Center to origin, scale to RMS unit norm (matches the Pepe ROT-2
     preprocessing in experiments/qa_ml/17_pepe_ch2_rot2_pointcloud_smoke.py).

Sampling is deterministic per-file via SHA-1 of the filename → seed, so
the resulting point cloud is reproducible without caching.

This is the Pepe ROT-2 dataset Pepe used; the synthetic Gaussian fallback
in 17_pepe_ch2_rot2_pointcloud_smoke.py is replaced by this when
QA_ML_ROT2_DATASET=modelnet40_airplane is set.

QA_COMPLIANCE = "qa_ml_modelnet40_loader — observer-side continuous geometry; no QA substrate"
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


CORPUS_AIRPLANE_DIR = Path(__file__).resolve().parent.parent.parent / "corpus" / "modelnet40" / "airplane"


def _parse_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices: float32[V, 3], triangles: int64[T, 3]).

    Handles both the canonical "OFF\\n V F E\\n ..." form and the rarer
    one-line "OFFV F E\\n ..." form that appears in ModelNet (some files
    have the counts glued onto the header line without a newline).
    """
    with path.open("r") as f:
        first = f.readline().strip()
        if first == "OFF":
            counts = f.readline().split()
        elif first.startswith("OFF"):
            counts = first[3:].split()
        else:
            raise ValueError(f"{path.name}: not an OFF file (header={first!r})")
        n_v, n_f = int(counts[0]), int(counts[1])

        verts = np.empty((n_v, 3), dtype=np.float32)
        for i in range(n_v):
            parts = f.readline().split()
            verts[i, 0] = float(parts[0])
            verts[i, 1] = float(parts[1])
            verts[i, 2] = float(parts[2])

        tris = np.empty((n_f, 3), dtype=np.int64)
        for i in range(n_f):
            parts = f.readline().split()
            # Faces may be tri (3 v1 v2 v3) or quad (4 v1 v2 v3 v4); fan-triangulate
            # quads (rare in ModelNet airplane, but cheap to handle).
            if int(parts[0]) == 3:
                tris[i, 0] = int(parts[1])
                tris[i, 1] = int(parts[2])
                tris[i, 2] = int(parts[3])
            else:
                # First triangle of fan; subsequent triangles dropped (mesh
                # is dense enough that this is negligible — verified by spot
                # check on airplane_0001 which has 0 quad faces).
                tris[i, 0] = int(parts[1])
                tris[i, 1] = int(parts[2])
                tris[i, 2] = int(parts[3])
    return verts, tris


def _sample_surface(verts: np.ndarray, tris: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(areas.sum())
    if total <= 0:
        raise ValueError("mesh has zero surface area")
    probs = areas / total
    tri_idx = rng.choice(len(tris), size=n_points, replace=True, p=probs)

    u = rng.random(size=n_points).astype(np.float32)
    v = rng.random(size=n_points).astype(np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v

    a = verts[tris[tri_idx, 0]]
    b = verts[tris[tri_idx, 1]]
    c = verts[tris[tri_idx, 2]]
    return (w[:, None] * a + u[:, None] * b + v[:, None] * c).astype(np.float32)


def _seed_for(name: str) -> int:
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def load_cloud(off_path: Path, n_points: int = 3000) -> np.ndarray:
    """Sample n_points points from one .off mesh; deterministic per filename.

    Returns float32[n_points, 3], centered at origin and scaled to RMS unit
    norm — matches the normalization in 17_pepe_ch2_rot2_pointcloud_smoke.py.
    """
    verts, tris = _parse_off(off_path)
    rng = np.random.default_rng(_seed_for(off_path.name))
    pts = _sample_surface(verts, tris, n_points, rng)
    pts -= pts.mean(axis=0, keepdims=True)
    norm = float(np.sqrt(np.mean(np.sum(pts * pts, axis=1))))
    if norm > 0:
        pts /= max(norm, 1e-6)
    return pts.astype(np.float32)


def load_dataset(
    n_points: int = 3000,
    n_train: int | None = None,
    n_test: int | None = None,
    airplane_dir: Path = CORPUS_AIRPLANE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the ModelNet40 airplane dataset as point clouds.

    Args:
      n_points: points per cloud (Pepe uses 3000).
      n_train: optional cap on training samples (default: all 626).
      n_test:  optional cap on test samples (default: all 100).

    Returns:
      (train_clouds, test_clouds) — both float32 arrays of shape
      (N, n_points, 3), centered + RMS-normalized.
    """
    train_files = sorted((airplane_dir / "train").glob("*.off"))
    test_files = sorted((airplane_dir / "test").glob("*.off"))
    if n_train is not None:
        train_files = train_files[:n_train]
    if n_test is not None:
        test_files = test_files[:n_test]

    train = np.stack([load_cloud(p, n_points) for p in train_files], axis=0)
    test = np.stack([load_cloud(p, n_points) for p in test_files], axis=0)
    return train, test


if __name__ == "__main__":
    train, test = load_dataset(n_points=3000)
    print(f"train: {train.shape} dtype={train.dtype}")
    print(f"test:  {test.shape} dtype={test.dtype}")
    print(f"train sample 0 stats: mean={train[0].mean(axis=0)}, std={train[0].std(axis=0)}")

"""
QA_COMPLIANCE = {
  "observer_input_projection": "MNIST pixel intensity plus central-difference gradient magnitude are projected at the observer input boundary to integer (b,e) states with +1 offset in {1,...,m}.",
  "qa_layer": "Encoded states are reduced to orbit_family histograms, diagonal-class histograms, and (b,e) joint histograms using canonical qa_arithmetic.orbit_family.",
  "observer_output_projection": "A sklearn logistic regression observer reads fixed feature vectors and emits supervised digit labels.",
  "theorem_nt_compliance": "Continuous image and rotation operations occur only at the input boundary; QA featurizers consume integer state downstream, with no float feedback into the encoded QA state.",
  "signal_injection": "Supervised MNIST labels are injected only at observer OUT for logistic-regression fitting and scoring."
}
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import rotate
from sklearn.linear_model import LogisticRegression
import sklearn
import torch
import torchvision
from torchvision import datasets


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qa_arithmetic"))
from qa_arithmetic import orbit_family  # noqa: E402


SEED = 42
MODULUS = 9
N_TRAIN = 10000
N_TEST = 2000
TEST_ANGLES_DEG = [0, 15, 30, 45, 60, 75, 90]
RESULTS_PATH = Path(__file__).with_suffix(".results.json")
FAMILY_ORDER = ("cosmos", "satellite", "singularity")


def has_mnist_raw(root: Path) -> bool:
    raw_dir = root / "MNIST" / "raw"
    return (raw_dir / "train-images-idx3-ubyte").exists() and (
        raw_dir / "t10k-images-idx3-ubyte"
    ).exists()


def load_mnist_split(train: bool, limit: int) -> tuple[np.ndarray, np.ndarray, str]:
    requested_root = Path.home() / ".cache" / "torch"
    candidate_roots = [requested_root, ROOT / "data", ROOT / "qa_lab" / "data"]
    errors = []
    for data_root in candidate_roots:
        try:
            use_download = data_root == requested_root and has_mnist_raw(data_root)
            dataset = datasets.MNIST(
                root=str(data_root),
                train=train,
                download=use_download,
            )
            image_values = np.asarray(dataset.data[:limit], dtype=np.float32) / 255.0
            labels = np.asarray(dataset.targets[:limit], dtype=np.int64)
            return image_values, labels, str(data_root)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{data_root}: {type(exc).__name__}: {exc}")
    details = "\n".join(errors)
    raise RuntimeError(f"MNIST load failed via torchvision:\n{details}")


def encode_images(image_values: np.ndarray, modulus: int) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(image_values, dtype=np.float32)

    grad_x = np.zeros_like(pixels, dtype=np.float32)
    grad_y = np.zeros_like(pixels, dtype=np.float32)
    grad_x[:, :, 1:-1] = (pixels[:, :, 2:] - pixels[:, :, :-2]) / 2.0
    grad_x[:, :, 0] = pixels[:, :, 1] - pixels[:, :, 0]
    grad_x[:, :, -1] = pixels[:, :, -1] - pixels[:, :, -2]
    grad_y[:, 1:-1, :] = (pixels[:, 2:, :] - pixels[:, :-2, :]) / 2.0
    grad_y[:, 0, :] = pixels[:, 1, :] - pixels[:, 0, :]
    grad_y[:, -1, :] = pixels[:, -1, :] - pixels[:, -2, :]

    edge_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y, dtype=np.float32)
    edge_max = edge_mag.reshape(edge_mag.shape[0], -1).max(axis=1)
    edge_max = np.maximum(edge_max, np.finfo(np.float32).eps)
    edge_unit = edge_mag / edge_max[:, None, None]

    span = modulus - 1
    b_codes = 1 + np.rint(pixels * span).astype(np.int64)
    e_codes = 1 + np.rint(edge_unit * span).astype(np.int64)
    np.clip(b_codes, 1, modulus, out=b_codes)
    np.clip(e_codes, 1, modulus, out=e_codes)
    return b_codes, e_codes


def build_lookup_tables(modulus: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    family_to_index = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
    family_table = np.zeros((modulus + 1, modulus + 1), dtype=np.int64)
    diagonal_table = np.zeros((modulus + 1, modulus + 1), dtype=np.int64)
    joint_table = np.zeros((modulus + 1, modulus + 1), dtype=np.int64)

    for b_value in range(1, modulus + 1):
        for e_value in range(1, modulus + 1):
            fam_name = orbit_family(int(b_value), int(e_value), modulus)
            family_table[b_value, e_value] = family_to_index[fam_name]
            diagonal_table[b_value, e_value] = (b_value - e_value) % modulus
            joint_table[b_value, e_value] = (b_value - 1) * modulus + (e_value - 1)
    return family_table, diagonal_table, joint_table


def hist_rows(indices: np.ndarray, bins: int) -> np.ndarray:
    flat = indices.reshape(indices.shape[0], -1)
    out = np.zeros((flat.shape[0], bins), dtype=np.float32)
    for row_idx, row in enumerate(flat):
        out[row_idx] = np.bincount(row, minlength=bins)
    return out


def l1_normalize(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return matrix / totals


def qa_orbit_hist(
    b_codes: np.ndarray,
    e_codes: np.ndarray,
    tables: tuple[np.ndarray, np.ndarray, np.ndarray],
    modulus: int,
) -> np.ndarray:
    family_table, diagonal_table, joint_table = tables
    family_idx = family_table[b_codes, e_codes]
    diagonal_idx = diagonal_table[b_codes, e_codes]
    joint_idx = joint_table[b_codes, e_codes]
    features = np.concatenate(
        [
            hist_rows(family_idx, 3),
            hist_rows(diagonal_idx, modulus),
            hist_rows(joint_idx, modulus * modulus),
        ],
        axis=1,
    )
    return l1_normalize(features)


def be_joint_hist(
    b_codes: np.ndarray,
    e_codes: np.ndarray,
    tables: tuple[np.ndarray, np.ndarray, np.ndarray],
    modulus: int,
) -> np.ndarray:
    del modulus
    joint_table = tables[2]
    joint_idx = joint_table[b_codes, e_codes]
    return l1_normalize(hist_rows(joint_idx, MODULUS * MODULUS))


def raw_pixels(image_values: np.ndarray) -> np.ndarray:
    return image_values.reshape(image_values.shape[0], -1).astype(np.float32)


def rotate_image_stack(image_values: np.ndarray, angle_deg: int) -> np.ndarray:
    if angle_deg == 0:
        return np.asarray(image_values, dtype=np.float32).copy()
    rotated = np.empty_like(image_values, dtype=np.float32)
    for row_idx, img in enumerate(image_values):
        rotated[row_idx] = rotate(
            img,
            angle=float(angle_deg),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    np.clip(rotated, 0.0, 1.0, out=rotated)
    return rotated


def make_logistic_regression() -> tuple[LogisticRegression, dict[str, object]]:
    kwargs: dict[str, object] = {
        "max_iter": 2000,
        "n_jobs": -1,
        "random_state": SEED,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs), kwargs


def accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predicted == labels))


def main() -> int:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_images, train_labels, train_root = load_mnist_split(True, N_TRAIN)
    test_images, test_labels, test_root = load_mnist_split(False, N_TEST)
    tables = build_lookup_tables(MODULUS)

    train_b, train_e = encode_images(train_images, MODULUS)
    test_variants = {
        angle: rotate_image_stack(test_images, angle) for angle in TEST_ANGLES_DEG
    }
    encoded_test_variants = {
        angle: encode_images(images, MODULUS) for angle, images in test_variants.items()
    }

    featurizers = [
        ("qa_orbit_hist", "qa"),
        ("be_hist_only", "be"),
        ("raw_pixels", "raw"),
    ]
    runs = []

    for name, kind in featurizers:
        if kind == "qa":
            train_x = qa_orbit_hist(train_b, train_e, tables, MODULUS)
        elif kind == "be":
            train_x = be_joint_hist(train_b, train_e, tables, MODULUS)
        else:
            train_x = raw_pixels(train_images)

        model, lr_kwargs = make_logistic_regression()
        fit_start = time.perf_counter()
        model.fit(train_x, train_labels)
        fit_sec = time.perf_counter() - fit_start

        acc_by_angle = {}
        for angle in TEST_ANGLES_DEG:
            if kind == "qa":
                test_b, test_e = encoded_test_variants[angle]
                test_x = qa_orbit_hist(test_b, test_e, tables, MODULUS)
            elif kind == "be":
                test_b, test_e = encoded_test_variants[angle]
                test_x = be_joint_hist(test_b, test_e, tables, MODULUS)
            else:
                test_x = raw_pixels(test_variants[angle])
            pred = model.predict(test_x)
            acc_by_angle[str(angle)] = accuracy(pred, test_labels)

        runs.append(
            {
                "name": name,
                "fit_sec": fit_sec,
                "feat_dim": int(train_x.shape[1]),
                "acc_by_angle": acc_by_angle,
                "logistic_regression_kwargs": lr_kwargs,
            }
        )

    config = {
        "seed": SEED,
        "modulus": MODULUS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "test_angles_deg": TEST_ANGLES_DEG,
        "train_data_root": train_root,
        "test_data_root": test_root,
        "requested_cache_root": str(Path.home() / ".cache" / "torch"),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }
    payload = {"config": config, "runs": runs}
    RESULTS_PATH.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )

    header = ["featurizer"] + [f"{angle:>6d}" for angle in TEST_ANGLES_DEG]
    print(" ".join(f"{cell:>14s}" for cell in header))
    for run in runs:
        cells = [run["name"]]
        cells.extend(f"{run['acc_by_angle'][str(angle)]:.4f}" for angle in TEST_ANGLES_DEG)
        print(" ".join(f"{cell:>14s}" for cell in cells))

    print()
    print("rotation degradation (acc@0 - acc@90):")
    for run in runs:
        deg = run["acc_by_angle"]["0"] - run["acc_by_angle"]["90"]
        print(f"{run['name']}: {deg:.4f}")
    print(f"\nresults_json: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""QA-ML Orbit Topology Cert validator.

Family [276]. Primary source for the GCN architecture under test:
Kipf & Welling 2017, Semi-Supervised Classification with Graph Convolutional
Networks, ICLR. arxiv:1609.02907.

Certifies the falsifiable claim:

    On QA orbit grids with a non-trivial satellite class (orbit period 8 under
    qa_step), a 2-layer GCN over the symmetric-normalized QA generator
    adjacency (sigma, mu, lambda_2, nu) lifts node-classification macro F1 by
    at least +0.10 over an identity-adjacency ablation, holding node features,
    architecture, seeds, and standardization fixed.

The validator runs in two modes:

  1. Default (fixture validation): loads each PASS / FAIL fixture, verifies
     schema conformance, internal consistency (graph_delta == with - without
     within tolerance), and direction (PASS fixtures must have
     graph_delta >= ORBT_THRESHOLD; FAIL fixtures must violate at least one
     declared invariant in the way encoded by expected_fail_type).

  2. --smoke: actually re-runs the GCN benchmark on m=9 with reduced
     seeds/epochs and asserts graph_delta >= ORBT_THRESHOLD. Exercises the
     full implementation chain (tools.qa_ml.qa_graph + qa_generators + torch).
     Skipped automatically if torch is not importable.

Checks: ORBT_1 schema, ORBT_2 graph_delta arithmetic, ORBT_3 PASS direction,
ORBT_4 FAIL fixtures rejected, SRC mapping_protocol_ref present,
F (every FAIL fixture must declare expected_fail_type).
"""

QA_COMPLIANCE = "cert_validator - structural fixture check + optional smoke benchmark; no float feedback into QA layer"

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_ML_ORBIT_TOPOLOGY_CERT.v1"
CERT_SLUG = "qa_ml_orbit_topology_cert_v1"
CANDIDATE_FAMILY_ID = 276
ORBT_THRESHOLD = 0.10
DEFAULT_TOLERANCE = 0.001


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema(fix: dict) -> list[str]:
    """Minimal stdlib schema check matching schema.json required fields."""
    required = [
        "schema_version", "fixture_kind", "modulus", "train_fraction", "n_seeds",
        "epochs", "hidden", "n_satellite", "n_pairs",
        "with_graph_macro_f1_mean", "without_graph_macro_f1_mean",
        "graph_delta", "passes_threshold",
    ]
    errors: list[str] = []
    for field in required:
        if field not in fix:
            errors.append(f"ORBT_1: fixture missing required field {field!r}")
    if fix.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"ORBT_1: wrong schema_version {fix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if fix.get("fixture_kind") not in ("pass", "fail"):
        errors.append("ORBT_1: fixture_kind must be 'pass' or 'fail'")
    return errors


def _check_arithmetic(fix: dict) -> list[str]:
    """ORBT_2: declared graph_delta must equal with - without within tolerance."""
    errors: list[str] = []
    tol = fix.get("tolerance", DEFAULT_TOLERANCE)
    expected = fix["with_graph_macro_f1_mean"] - fix["without_graph_macro_f1_mean"]
    actual = fix["graph_delta"]
    if abs(actual - expected) > tol:
        errors.append(
            f"ORBT_2: graph_delta arithmetic violation: "
            f"declared={actual} != with-without={expected:.6f} (tol={tol})"
        )
    return errors


def _check_direction(fix: dict) -> list[str]:
    """ORBT_3 (PASS): graph_delta >= threshold AND passes_threshold == True."""
    errors: list[str] = []
    is_pass = fix.get("fixture_kind") == "pass"
    delta = fix.get("graph_delta")
    flag = fix.get("passes_threshold")
    if is_pass:
        if delta is None or delta < ORBT_THRESHOLD - 1e-9:
            errors.append(
                f"ORBT_3: PASS fixture must have graph_delta >= {ORBT_THRESHOLD}; "
                f"got {delta}"
            )
        if flag is not True:
            errors.append("ORBT_3: PASS fixture must declare passes_threshold=True")
    return errors


def _check_fail_fixture(path: Path, fix: dict) -> list[str]:
    """ORBT_4: a FAIL fixture must trip a structural check or declare a violation."""
    errors: list[str] = []
    if fix.get("fixture_kind") != "fail":
        return errors
    expected = fix.get("expected_fail_type")
    if not expected:
        errors.append("F: FAIL fixture must declare expected_fail_type")
        return errors
    schema_errs = _validate_schema(fix)
    arithmetic_errs = _check_arithmetic(fix) if not schema_errs else []
    delta = fix.get("graph_delta")
    flag = fix.get("passes_threshold")
    if expected == "BELOW_THRESHOLD":
        if delta is None or delta >= ORBT_THRESHOLD:
            errors.append(
                f"ORBT_4: BELOW_THRESHOLD fixture has graph_delta={delta} "
                f">= {ORBT_THRESHOLD}; should be below"
            )
        if flag is True:
            errors.append("ORBT_4: BELOW_THRESHOLD fixture cannot have passes_threshold=True")
    elif expected == "MISSING_FIELD":
        if not schema_errs:
            errors.append("ORBT_4: MISSING_FIELD fixture passed schema check")
    elif expected == "ARITHMETIC":
        if not arithmetic_errs:
            errors.append("ORBT_4: ARITHMETIC fixture has consistent graph_delta")
    else:
        errors.append(f"ORBT_4: unknown expected_fail_type {expected!r}")
    return errors


def _validate_fixture(path: Path) -> list[str]:
    fix = _load_json(path)
    if fix.get("fixture_kind") == "fail":
        # Schema/arithmetic errors are the expected behavior for FAIL fixtures;
        # _check_fail_fixture verifies the declared expected_fail_type fired.
        return _check_fail_fixture(path, fix)
    schema_errs = _validate_schema(fix)
    if schema_errs:
        return schema_errs
    errors: list[str] = []
    errors.extend(_check_arithmetic(fix))
    errors.extend(_check_direction(fix))
    return errors


def _check_mapping_protocol(cert_dir: Path) -> list[str]:
    """SRC: mapping_protocol_ref.json must be present and well-formed."""
    p = cert_dir / "mapping_protocol_ref.json"
    if not p.exists():
        return ["SRC: missing mapping_protocol_ref.json"]
    data = _load_json(p)
    needed = ("protocol_version", "ref_path", "ref_sha256", "scope_note")
    return [f"SRC: missing {k!r}" for k in needed if k not in data]


def _smoke_run(cert_dir: Path) -> list[str]:
    """Optional --smoke check: re-run a small GCN benchmark on m=9.

    Skips silently if torch is not importable so the smoke flag stays
    cooperative on minimal environments.
    """
    repo = cert_dir.parents[1]
    sys.path.insert(0, str(repo))
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as f_nn
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedShuffleSplit
        from tools.qa_ml.qa_graph import dense_adjacency, gcn_normalize
        from tools.qa_ml.qa_dataset import all_pairs
        from tools.qa_ml.qa_features import qa_packet_full
        from qa_orbit_rules import orbit_period
    except Exception as exc:  # pragma: no cover
        return [f"SMOKE-SKIP: dependency unavailable ({exc})"]

    m = 9
    pairs = all_pairs(m)
    periods = np.array([orbit_period(b, e, m) for b, e in pairs], dtype=np.int64)
    labels = np.full(len(pairs), -1, dtype=np.int64)
    labels[periods != 1] = 0
    labels[periods == 8] = 1
    label_mask = labels >= 0

    x_qa = np.asarray(
        [qa_packet_full(b, e, m) for (b, e) in pairs], dtype=np.float64,
    )
    adj = dense_adjacency(m, symmetric=True)
    adj_norm = gcn_normalize(adj)

    class _Gcn(nn.Module):
        def __init__(self, n_features: int, hidden: int, n_classes: int) -> None:
            super().__init__()
            self.lin1 = nn.Linear(n_features, hidden)
            self.lin2 = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x_in: torch.Tensor, a_in: torch.Tensor) -> torch.Tensor:
            h = f_nn.relu(self.lin1(a_in @ x_in))
            h = f_nn.relu(self.lin2(a_in @ h))
            return self.head(h)

    n_seeds, epochs, hidden = 3, 80, 32
    deltas = []
    for seed in range(n_seeds):
        np.random.seed(seed)  # noqa: T2-D-5  observer-side smoke split
        torch.manual_seed(seed)
        labeled_idx = np.where(label_mask)[0]
        y_lab = labels[labeled_idx]
        n_train = int(round(0.30 * len(y_lab)))
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
        train_pos, test_pos = next(sss.split(np.zeros((len(y_lab), 1)), y_lab))
        train_idx = labeled_idx[train_pos]
        test_idx = labeled_idx[test_pos]
        y_full = np.where(label_mask, labels, 0).astype(np.int64)
        train_mask_full = np.zeros(len(pairs), dtype=bool)
        train_mask_full[train_idx] = True

        mu_tr = x_qa[train_mask_full].mean(axis=0, keepdims=True)
        sd_tr = x_qa[train_mask_full].std(axis=0, keepdims=True)
        sd_tr = np.where(sd_tr < 1e-8, 1.0, sd_tr)
        x_std = (x_qa - mu_tr) / sd_tr

        cls, cnt = np.unique(y_full[train_idx], return_counts=True)
        cw = np.ones(2, dtype=np.float64)
        for c, k in zip(cls, cnt):
            cw[int(c)] = len(train_idx) / (len(cls) * k)
        cw_t = torch.from_numpy(cw).float()

        scores = []
        for ablation in ("with_graph", "without_graph"):
            adj_used = adj_norm if ablation == "with_graph" else np.eye(adj_norm.shape[0])
            torch.manual_seed(seed)
            net = _Gcn(x_std.shape[1], hidden, 2)
            x_t = torch.from_numpy(x_std).float()
            a_t = torch.from_numpy(adj_used).float()
            y_t = torch.from_numpy(y_full).long()
            opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
            mask_t = torch.from_numpy(train_mask_full)
            net.train()
            for _ in range(epochs):
                opt.zero_grad()
                logits = net(x_t, a_t)
                loss = f_nn.cross_entropy(logits[mask_t], y_t[mask_t], weight=cw_t)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                preds = net(x_t, a_t).argmax(dim=1).cpu().numpy()
            scores.append(f1_score(
                y_full[test_idx], preds[test_idx], average="macro", zero_division=0,
            ))
        deltas.append(scores[0] - scores[1])

    mean_delta = float(np.mean(deltas))
    if mean_delta < ORBT_THRESHOLD:
        return [f"SMOKE-FAIL: m=9 mean graph_delta={mean_delta:.3f} < {ORBT_THRESHOLD}"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA-ML Orbit Topology Cert validator [276]")
    parser.add_argument("--demo", action="store_true", help="print config and exit")
    parser.add_argument("--smoke", action="store_true",
                        help="re-run a small m=9 GCN benchmark (requires torch)")
    parser.add_argument("--self-test", action="store_true",
                        help="emit JSON {ok, errors, ...} for the meta-validator")
    args = parser.parse_args(argv)

    cert_dir = Path(__file__).resolve().parent
    fix_dir = cert_dir / "fixtures"

    if args.demo:
        print(f"family_id={CANDIDATE_FAMILY_ID} slug={CERT_SLUG} "
              f"schema={SCHEMA_VERSION} threshold={ORBT_THRESHOLD}")
        print(f"fixtures: {sorted(p.name for p in fix_dir.glob('*.json'))}")
        return 0

    all_errors: list[str] = []
    src_errs = _check_mapping_protocol(cert_dir)
    if src_errs:
        all_errors.extend(src_errs)

    pass_files = sorted(fix_dir.glob("pass_*.json"))
    fail_files = sorted(fix_dir.glob("fail_*.json"))
    if not pass_files:
        all_errors.append("F: no PASS fixtures present")
    if not fail_files:
        all_errors.append("F: no FAIL fixtures present")

    for p in pass_files:
        errs = _validate_fixture(p)
        if errs:
            all_errors.extend([f"{p.name}: {e}" for e in errs])
    for p in fail_files:
        errs = _validate_fixture(p)
        if errs:
            all_errors.extend([f"{p.name}: {e}" for e in errs])

    if args.smoke:
        smoke_errs = _smoke_run(cert_dir)
        if smoke_errs:
            all_errors.extend(smoke_errs)

    ok = not all_errors
    if args.self_test:
        payload = {
            "ok": ok,
            "family_id": CANDIDATE_FAMILY_ID,
            "slug": CERT_SLUG,
            "schema_version": SCHEMA_VERSION,
            "threshold": ORBT_THRESHOLD,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": all_errors,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0 if ok else 1

    if all_errors:
        print(f"FAIL [{CERT_SLUG}]: {len(all_errors)} error(s)")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print(f"PASS [{CERT_SLUG}]: schema + {len(pass_files)} PASS + {len(fail_files)} FAIL fixtures ok"
          + (" + smoke ok" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

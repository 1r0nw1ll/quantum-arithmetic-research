#!/usr/bin/env python3
"""
Merge FO‑v2 and fingerprint‑multiseg Raman CSVs into a multi‑tuple CSV with
schema: id,b1,e1,b2,e2,label

Inputs must each have header: id,b,e,label

Example:
  PYTHONPATH=. python codex_on_QA/scripts/merge_raman_multi_fo_fingerprint.py \
    --fo  codex_on_QA/out/raman_qa_fundovt_bcwin_v2.csv \
    --fp  codex_on_QA/out/raman_qa_fingerprint_multiseg.csv \
    --out codex_on_QA/out/raman_multi_fundovt_fingerprint_multiseg.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def read_simple_csv(path: Path):
    rows = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        need = {"id", "b", "e", "label"}
        if not need.issubset(set(r.fieldnames or [])):
            raise SystemExit(f"{path} missing required columns id,b,e,label")
        for row in r:
            rows[row["id"]] = row
    return rows


def read_multi_tuple_csv(path: Path):
    """Read a CSV with id, b1,e1, b2,e2, ..., label. Returns dict id->row dict."""
    rows = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        headers = r.fieldnames or []
        if "id" not in headers or "label" not in headers:
            raise SystemExit(f"{path} missing required columns id,label")
        # detect tuple indices in this file
        fp_tuple_idx = sorted(
            {int(h[1:]) for h in headers if h.startswith("b") and h[1:].isdigit()}
            & {int(h[1:]) for h in headers if h.startswith("e") and h[1:].isdigit()}
        )
        for row in r:
            rows[row["id"]] = (row, fp_tuple_idx)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge FO v2 + fingerprint multiseg into multi‑tuple CSV")
    ap.add_argument("--fo", required=True, help="FO v2 CSV (id,b,e,label)")
    ap.add_argument("--fp", required=True, help="Fingerprint multiseg CSV (id,b,e,label)")
    ap.add_argument("--out", required=True, help="Output multi‑tuple CSV path")
    args = ap.parse_args()

    fo_path = Path(args.fo)
    fp_path = Path(args.fp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fo = read_simple_csv(fo_path)
    # Fingerprint can be multi‑tuple already (b1/e1,b2/e2,...) — accept that shape
    fp = read_multi_tuple_csv(fp_path)

    ids = sorted(set(fo.keys()) & set(fp.keys()), key=lambda x: int(x) if x.isdigit() else x)
    if not ids:
        raise SystemExit("No overlapping ids between FO and fingerprint CSVs")

    rows = []
    for rid in ids:
        a = fo[rid]
        fp_row, fp_idx = fp[rid]
        # If labels disagree, prefer FO label
        lab_fo = a["label"]
        lab_fp = fp_row.get("label", lab_fo)
        label = lab_fo if lab_fo == lab_fp else lab_fo

        od = OrderedDict()
        od["id"] = rid
        # FO tuple as first
        od["b1"] = a["b"]
        od["e1"] = a["e"]
        # Append fingerprint tuples, reindexed to start at 2
        out_idx = 2
        for k in fp_idx:
            bk = fp_row.get(f"b{k}")
            ek = fp_row.get(f"e{k}")
            if bk is None or ek is None:
                continue
            od[f"b{out_idx}"] = bk
            od[f"e{out_idx}"] = ek
            out_idx += 1
        od["label"] = label
        rows.append(od)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Update master_summary.json/.csv with Swiss encoding results from qa_one_shot_swiss_*.json

For each file codex_on_QA/out/qa_one_shot_swiss_*.json, append per-encoding
metrics (kmeans_ARI, logreg_acc, mlp_acc) by mode (raw, qa21, qa27, qa83) into
master_summary.json under manifolds.swiss_encodings and add swiss_best picks:
  - supervised: (encoding, qa_mode) maximizing max(LogReg, MLP) over QA modes
  - clustering: (encoding, qa_mode) maximizing KMeans ARI over QA modes

Also append rows to master_summary.csv with dataset set to e.g. swiss_radangle
and metrics kmeans_ARI, logreg_acc, mlp_acc.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Any
import glob

OUT = os.path.join('codex_on_QA', 'out')
MS_JSON = os.path.join(OUT, 'master_summary.json')
MS_CSV  = os.path.join(OUT, 'master_summary.csv')


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_csv_rows(path: str, rows: list[list[str]]) -> None:
    # Create if not present; if present, append
    exists = os.path.exists(path)
    with open(path, 'a' if exists else 'w') as f:
        if not exists:
            f.write('domain,dataset,mode,metric,value\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')


def main() -> int:
    ms = load_json(MS_JSON)
    manifolds = ms.setdefault('manifolds', {})
    swiss_enc = manifolds.setdefault('swiss_encodings', {})

    rows: list[list[str]] = []
    best_supervised = None  # (encoding, qa_mode, score, logreg, mlp)
    best_cluster = None     # (encoding, qa_mode, ari)

    for path in glob.glob(os.path.join(OUT, 'qa_one_shot_swiss*.json')):
        with open(path, 'r') as f:
            doc = json.load(f)
        encoding = doc.get('encoding') or 'first2'
        modes = doc.get('modes', {})
        # Persist per-mode metrics into JSON
        swiss_enc[encoding] = modes
        # Emit CSV rows and track bests (QA modes only)
        for mode_name, metrics in modes.items():
            ds = f'swiss_{encoding}' if encoding != 'first2' else 'swiss'
            rows.append(['manifold', ds, mode_name, 'kmeans_ARI', metrics.get('kmeans_ARI')])
            rows.append(['manifold', ds, mode_name, 'logreg_acc', metrics.get('logreg_acc')])
            rows.append(['manifold', ds, mode_name, 'mlp_acc', metrics.get('mlp_acc')])
        # best supervised among QA modes only
        for qa_mode in ('qa21','qa27','qa83'):
            if qa_mode in modes:
                m = modes[qa_mode]
                score = max(float(m.get('logreg_acc') or 0.0), float(m.get('mlp_acc') or 0.0))
                if best_supervised is None or score > best_supervised[2]:
                    best_supervised = (encoding, qa_mode, score, m.get('logreg_acc'), m.get('mlp_acc'))
        # best clustering among QA modes only
        for qa_mode in ('qa21','qa27','qa83'):
            if qa_mode in modes:
                m = modes[qa_mode]
                ari = float(m.get('kmeans_ARI') or 0.0)
                if best_cluster is None or ari > best_cluster[2]:
                    best_cluster = (encoding, qa_mode, ari)

    # Store swiss best picks
    if best_supervised:
        manifolds['swiss_best'] = manifolds.get('swiss_best', {})
        manifolds['swiss_best']['supervised'] = {
            'encoding': best_supervised[0],
            'qa_mode': best_supervised[1],
            'score': best_supervised[2],
            'logreg_acc': best_supervised[3],
            'mlp_acc': best_supervised[4],
        }
    if best_cluster:
        manifolds['swiss_best'] = manifolds.get('swiss_best', {})
        manifolds['swiss_best']['clustering'] = {
            'encoding': best_cluster[0],
            'qa_mode': best_cluster[1],
            'kmeans_ARI': best_cluster[2],
        }

    os.makedirs(OUT, exist_ok=True)
    # Also ingest Raman CSV benches (base and refined variants)
    tab = ms.setdefault('tabular', {})
    for rpath in glob.glob(os.path.join(OUT, 'raman_qa*_csv_bench.json')):
        with open(rpath, 'r') as f:
            rdoc = json.load(f)
        # dataset label from filename
        base = os.path.splitext(os.path.basename(rpath))[0]
        # base -> raman_qa[_variant]_csv_bench
        if base.startswith('raman_qa_') and base.endswith('_csv_bench'):
            variant = base[len('raman_qa_'):-len('_csv_bench')]
            if not variant:
                ds = 'raman'
            else:
                ds = f'raman_{variant}'
        elif base == 'raman_qa_csv_bench':
            ds = 'raman'
        else:
            ds = base
        tab[ds] = rdoc.get('modes', {})
        for mode, metrics in rdoc.get('modes', {}).items():
            rows.append(['tabular', ds, mode, 'kmeans_ARI', metrics.get('kmeans_ARI')])
            rows.append(['tabular', ds, mode, 'logreg_acc', metrics.get('logreg_acc')])
            rows.append(['tabular', ds, mode, 'mlp_acc', metrics.get('mlp_acc')])

    save_json(MS_JSON, ms)
    append_csv_rows(MS_CSV, rows)
    print('Updated', MS_JSON)
    print('Appended rows to', MS_CSV)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

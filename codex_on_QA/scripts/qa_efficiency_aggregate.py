#!/usr/bin/env python3
"""
Aggregate QA one-shot efficiency JSONs into a single CSV.

Inputs (present by default):
  - codex_on_QA/out/qa_one_shot_efficiency_moons_first2.json
  - codex_on_QA/out/qa_one_shot_efficiency_circles_first2.json

Output:
  - codex_on_QA/out/qa_one_shot_efficiency_summary.csv

Columns:
  dataset,encoding,mode,model,n_train_0_90,n_train_0_95
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List

OUTDIR = os.path.join('codex_on_QA', 'out')
SUMMARY = os.path.join(OUTDIR, 'qa_one_shot_efficiency_summary.csv')


def load(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    paths = [
        os.path.join(OUTDIR, 'qa_one_shot_efficiency_moons_first2.json'),
        os.path.join(OUTDIR, 'qa_one_shot_efficiency_circles_first2.json'),
    ]
    rows: List[str] = []
    header = 'dataset,encoding,mode,model,n_train_0_90,n_train_0_95\n'
    for p in paths:
        if not os.path.exists(p):
            continue
        doc = load(p)
        dataset = doc.get('dataset')
        encoding = doc.get('encoding', 'first2')
        thresholds = [str(x) for x in doc.get('thresholds', ['0.9','0.95'])]
        # guaranteed order
        thr90 = '0.9' if '0.9' in thresholds else thresholds[0]
        thr95 = '0.95' if '0.95' in thresholds else thresholds[-1]
        for model_key, model_name in [('logreg_min_n','logreg'), ('mlp_min_n','mlp')]:
            m = doc.get(model_key, {})
            for mode, mins in m.items():
                n90 = mins.get(thr90)
                n95 = mins.get(thr95)
                rows.append(f'{dataset},{encoding},{mode},{model_name},{n90},{n95}\n')
    with open(SUMMARY, 'w') as f:
        f.write(header)
        for r in rows:
            f.write(r)
    print('Wrote', SUMMARY)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


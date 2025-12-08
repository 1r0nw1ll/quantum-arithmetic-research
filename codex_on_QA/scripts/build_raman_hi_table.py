#!/usr/bin/env python3
"""
Build a small LaTeX table comparing canonical vs Markovian HI sweeps
for the Raman multi‑tuple graph.

Inputs:
  --canonical codex_on_QA/out/raman_multi_spectral_qa21_fullmulti_v2.json
  --markov    codex_on_QA/out/raman_multi_spectral_qa21_fullmulti_markovhi.json
Output:
  Documents/table_raman_hi_sweep.tex
"""
import argparse, json
from pathlib import Path


def load_mode(path: Path, mode: str):
    doc = json.loads(path.read_text())
    header = {
        'hi_source': doc.get('hi_source'),
        'hi_beta': doc.get('hi_beta'),
        'alphas': doc.get('alphas'),
    }
    row = None
    for r in doc.get('runs', []):
        if r.get('mode') == mode:
            row = r
            break
    if row is None:
        raise SystemExit(f"Mode {mode} not found in {path}")
    return header, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', required=True)
    ap.add_argument('--markov', required=True)
    ap.add_argument('--out', default='Documents/table_raman_hi_sweep.tex')
    args = ap.parse_args()

    can_h, can_r = load_mode(Path(args.canonical), 'qa_weight_full_multi')
    mar_h, mar_r = load_mode(Path(args.markov), 'qa_weight_full_multi')

    def fmt(x, digs=6):
        return f"{x:.6f}" if isinstance(x, (int, float)) else (str(x) if x is not None else '-')

    lines = []
    lines.append('% Raman HI sweep: canonical vs markovian')
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Raman multi\-tuple full kernel with canonical vs Markovian HI.}')
    lines.append('\\label{tab:raman-hi-sweep}')
    lines.append('\\begin{tabular}{lcccccc}')
    lines.append('\\toprule')
    lines.append('HI source & $\\beta$ & $\\alpha$ & Q & Purity & ARI & NMI \\\\')
    lines.append('\\midrule')
    lines.append("canonical & {} & {} & {} & {} & {} & {} \\\\".format(
        fmt(can_h.get('hi_beta')), can_h.get('alphas') or '-', fmt(can_r.get('modularity_Q')),
        fmt(can_r.get('purity')), fmt(can_r.get('ARI')), fmt(can_r.get('NMI'))
    ))
    lines.append("markovian & {} & {} & {} & {} & {} & {} \\\\".format(
        fmt(mar_h.get('hi_beta')), mar_h.get('alphas') or '-', fmt(mar_r.get('modularity_Q')),
        fmt(mar_r.get('purity')), fmt(mar_r.get('ARI')), fmt(mar_r.get('NMI'))
    ))
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    Path(args.out).write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

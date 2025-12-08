#!/usr/bin/env python3
"""
Build a LaTeX table comparing E8 embed variants:
  - plane=be (b,e) per tuple
  - plane=jx (J,X) per tuple

Inputs:
  --be codex_on_QA/out/raman_multi_spectral_qa21_e8_be.json
  --jx codex_on_QA/out/raman_multi_spectral_qa21_e8_jx.json
Output:
  Documents/table_raman_e8_embed.tex
"""
import argparse, json
from pathlib import Path


def load_mode(path: Path, mode: str):
    doc = json.loads(path.read_text())
    row = next((r for r in doc.get('runs', []) if r.get('mode')==mode), None)
    if row is None:
        raise SystemExit(f"Mode {mode} not found in {path}")
    return row


def fmt(x, d=6):
    return f"{x:.{d}f}" if isinstance(x, (int,float)) else (str(x) if x is not None else '-')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--be', required=True)
    ap.add_argument('--jx', required=True)
    ap.add_argument('--out', default='Documents/table_raman_e8_embed.tex')
    args = ap.parse_args()

    r_be = load_mode(Path(args.be), 'qa_weight_e8_embed')
    r_jx = load_mode(Path(args.jx), 'qa_weight_e8_embed')

    lines = []
    lines.append('% Raman E8 embed comparison: (b,e) vs (J,X) planes')
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Raman multi\-tuple E8 embedding: tuple planes using $(b,e)$ vs $(J,X)$.}')
    lines.append('\\label{tab:raman-e8-embed}')
    lines.append('\\begin{tabular}{lcccc}')
    lines.append('\\toprule')
    lines.append('Plane & Q & Purity & ARI & NMI \\\\')
    lines.append('\\midrule')
    lines.append('$(b,e)$ & {} & {} & {} & {} \\\\'.format(fmt(r_be.get('modularity_Q')), fmt(r_be.get('purity')), fmt(r_be.get('ARI')), fmt(r_be.get('NMI'))))
    lines.append('$(J,X)$ & {} & {} & {} & {} \\\\'.format(fmt(r_jx.get('modularity_Q')), fmt(r_jx.get('purity')), fmt(r_jx.get('ARI')), fmt(r_jx.get('NMI'))))
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    Path(args.out).write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
import csv, sys
from collections import Counter

def domains(p):
    with open(p, newline="") as f: rs = list(csv.reader(f)); i = next((j for j, c in enumerate(rs[0]) if c.strip().lower() == "email"), None) if rs else None; return Counter(e.split("@", 1)[1].lower() for r in rs[i is not None:] for e in [r[i if i is not None else 0] if len(r) > (i if i is not None else 0) else ""] if "@" in e)

main = lambda a: 1 if len(a) != 2 else (print(*[f"{n}\t{d}" for d, n in sorted(domains(a[1]).items(), key=lambda kv: (-kv[1], kv[0]))], sep="\n") or 0)  # noqa: E731

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

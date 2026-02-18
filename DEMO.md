# QA Cert Family Demos

See [`demos/DEMO_GUIDE.md`](demos/DEMO_GUIDE.md) for the full guide.

## Quick start

```bash
python demos/qa_family_demo.py --family geogebra
python demos/qa_family_demo.py --family rule30
python demos/qa_family_demo.py --all
```

## CI / smoke test

```bash
python demos/qa_family_demo.py --family geogebra --ci
python demos/qa_family_demo.py --family rule30 --ci
```

Exit 0 = all expectations met. Exit 2 = expectation mismatch. Exit 1 = runtime error.

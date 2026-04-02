# [158] QA Graph Community Cert

**Schema**: `QA_GRAPH_COMMUNITY_CERT.v1`
**Status**: PASS (1 PASS + 1 FAIL fixture)

## What it certifies

QA feature map dimensions and community detection quality on standard benchmark graphs.

### Feature map dimensions

| Mode | Dimension | Contents |
|------|-----------|----------|
| `qa21` | 21 | Canonical invariants: b,e,d,a + squares + triangle legs + composites + ellipse |
| `qa27` | 27 | qa21 + 6 expanded (ratios, energy, angle) |
| `qa83` | 83 | Full stack: canonical + derived + modular + physical + ML features |

### Chromogeometry identity

`C^2 + F^2 = G^2` (Wildberger Theorem 6) verified for all tested directions. This is the foundational identity: `C = 2de` (green quadrance), `F = b*a` (red quadrance), `G = d^2 + e^2` (blue quadrance).

### Benchmark graphs

| Graph | Nodes | Edges | Ground truth |
|-------|-------|-------|--------------|
| Karate (Zachary) | 34 | 78 | 2 communities |
| Football (college) | 115 | 613 | 12 conferences |
| Dolphins | 62 | 159 | 2 groups |

Data at `codex_on_QA/data/` (symlinked from `qa_lab/qa_graph/data/`).

## How to run

```bash
# Unit tests (12 tests)
cd qa_lab && PYTHONPATH=. python -m pytest qa_graph/tests/ -v

# Validator self-test
cd qa_alphageometry_ptolemy/qa_graph_community_cert_v1
python qa_graph_community_cert_validate.py --self-test
```

## What breaks

- Feature dimensions wrong (qa21 != 21, qa27 != 27, qa83 != 83)
- C^2 + F^2 - G^2 != 0 for any direction
- Benchmark ARI outside [-1, 1] or NMI outside [0, 1]


# QA Certificate Bundle (v1.0)

This bundle gives you:

- `qa_certificate.py`: canonical schema, unified across QA / AlphaGeometry / Physics projection
- `qa_alphageometry/adapters/certificate_adapter.py`: SearchResult → ProofCertificate
- `qa_physics/adapters/certificate_adapter.py`: observer output → ProofCertificate
- `qa_certificate_paper_skeleton.tex`: paper skeleton (LaTeX)

## Quick smoke test

```bash
python -c "from qa_certificate import to_scalar; from fractions import Fraction; print(to_scalar('63.43'), to_scalar('3/2'), to_scalar(25))"
```

## Using the AlphaGeometry adapter

```python
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate

cert = wrap_searchresult_to_certificate(sr, theorem_id="ptolemy_quadrance", max_depth_limit=50)
print(cert.to_json().keys())
```

## Using the Physics adapter

```python
from qa_physics.adapters.certificate_adapter import wrap_reflection_result_to_certificate

cert = wrap_reflection_result_to_certificate(
    {"observation": {"law_holds": True, "measured_angles": {"incident": "63.43", "reflected": "63.43"}, "angle_difference": "0"}},
    observer_id="GeometryAngleObserver",
    repo_tag="qa-physics-projection-v0.1",
    commit_hash="4aa2af4"
)
print(cert.to_json()["projection_contract"])
```

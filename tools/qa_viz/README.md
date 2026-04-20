# QA Visualization

3D rendering scaffolds for QA structures — torus orbits, E8 projections, generator paths.

## Layout

- `threejs/` — interactive web figures (self-contained HTML, Three.js via CDN). Target: GitHub-first publication, embed in paper pages.
- `blender/` — `bpy` scripts for render-quality static figures. Target: paper PDFs.

## Running

**Three.js** (no build step):
```bash
cd tools/qa_viz/threejs
python3 -m http.server 8000
# open http://localhost:8000/qa_torus.html
```

**Blender** (headless render):
```bash
blender -b -P tools/qa_viz/blender/qa_torus.py -- --out /tmp/qa_torus.png
```

## QA math shared by both

Mod-9 state space: `(b, e) ∈ {1..9}²`, 81 total pairs.
- **Singularity**: `(9, 9)` — fixed point.
- **Satellite** (8-cycle): `b = e` with `b ≠ 9` → 8 pairs.
- **Cosmos** (24-cycle × 3): the remaining 72 pairs.

Derived coords (A2): `d = b + e`, `a = b + 2e` (raw — mod is T-operator only).

Torus placement: `θ = 2π(b-1)/9` (major), `φ = 2π(e-1)/9` (minor). Volk convention: `R` major radius, `r` minor radius.

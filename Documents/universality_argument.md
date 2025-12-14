## QA as a Universal Structural Prior (One–Shot Geometry, Few–Shot Learning)

Claim. For diverse data types (graphs, manifolds, tabular signals), a small, canonical QA block \(\Phi(x)\) (e.g., \(b,e,d,a,J,X,K,C,F,G\) with phases) makes structural information nearly explicit. A single structural step (spectral/clustering/margin) reveals the layout; the residual learning becomes shallow and data–efficient.

Evidence.
- Graphs (football, labeled): A single spectral pass with QA–X yields higher modularity and slightly better label alignment than both baseline and Louvain (Table: `Documents/table_football.tex`).
- Graphs (QA knowledge graph, unlabeled): QA–X and full QA kernel turn a near–flat baseline (\(Q\approx 0.0004\)) into strong communities (\(Q\approx 0.50\) and \(Q\approx 0.72\), Table: `Documents/table_qa_knowledge_graph.tex`).
- Manifolds (moons/swiss): QA per–feature mapping linearizes manifolds for margin–based models (LogReg/SGD), improving accuracy at fewer labels/epochs (figures in `codex_on_QA/out`). Clustering gains on swiss and comparable moons results (Table: `Documents/table_manifolds_kmeans.tex`) show that linearization helps margins consistently; unsupervised performance depends on algorithmic bias.
- Tabular (synthetic + Raman pattern): QA blocks reduce epochs–to–accuracy and preserve performance under label scarcity (figure: `tabular_sgd_epochs.png`).

Mechanism. QA imposes harmonic geometry: triangle legs (\(C,F,G\)), ellipse radii (\(J,K,X\)), and modular phases encode the latent manifold in low dimension. QA–weighted adjacencies approximate harmonic distances; spectral operators align with low–frequency eigenmodes and amplify communities in one shot. In supervised settings, QA margins flatten loss curvature, enabling few–shot param fitting.

Scope. QA does not replace learning universally; it removes much of the *structural* burden. When \(\Phi(x)\) captures the relevant geometry, classical learners succeed with one structural pass (clustering/spectral) plus a thin classifier trained in a few epochs.


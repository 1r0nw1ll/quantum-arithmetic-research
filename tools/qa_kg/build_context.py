# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 5 build-context infrastructure; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026), tools/qa_kg/canonicalize.py (Dale, 2026) -->
"""QA-KG build-context wiring — Phase 5.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Threads fixture-path overrides through the extractor pipeline without
monkey-patching or module-level globals. Two build modes:

  Production build (cli.py cmd_build, no --fixture):
    - MEMORY.md from ~/.claude/projects/-home-player2-signal-experiments/memory/MEMORY.md
    - CLAUDE.md, QA_AXIOMS_BLOCK.md, cert dirs, source_claims_phase3.json
      from working-tree repo root
    - Graph reflects current working state; hash drifts as working tree moves
    - Ledger's graph_hash = production hash at meta-validator run time

  Fixture rebuild (cli.py cmd_build --fixture <path>):
    - MEMORY.md from <fixture>/memory_md_sample.md
    - CLAUDE.md, QA_AXIOMS_BLOCK.md, etc. from working-tree repo root
    - OB ingest from <fixture>/ob_sample.md when present
    - Hash is stable given fixed fixture + fixed working tree
    - Fixture rebuild hash is frozen in expected_hash.json for D2/D3
      regression testing; NOT the same hash as production

Phase 5 initial commit does NOT pin to manifest.repo_head via git archive.
D1.5 WARN surfaces repo_head drift; D2/D3 test reproducibility of the
working-tree pipeline. Phase 5.1 will add the git-archive materialization
that decouples reproducibility from HEAD advances (see
memory/project_qa_mem_review_role.md roadmap).

kg.promote()'s D6 staleness check compares ledger graph_hash (production)
against live production graph_hash. Fixture rebuild hash is a separate
contract exercised only by cert [228] gates.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildContext:
    """Input-source overrides for a single build invocation.

    All fields are Path | None. None means "use extractor default" (the
    hardcoded path that reads from user-home or working-tree). Every
    field has a corresponding extractor kwarg that accepts the override.

    memory_md_path   -> memory_rules.populate(memory_md_path=...)
    arag_db_path     -> arag.search/promote_to_kg(db_path=...) [C#1]
    ob_markdown_path -> read by cli cmd_build when --fixture passed
    fixture_root     -> source directory the overrides were loaded from;
                        retained for diagnostics

    No context-vars, no monkey-patching. Every consumer reads its field
    directly; call-site coverage is verified by the C#1 grep sweep and
    the test_arag_extractor_respects_override unit test.
    """
    memory_md_path: Path | None = None
    arag_db_path: Path | None = None
    ob_markdown_path: Path | None = None
    fixture_root: Path | None = None

    @classmethod
    def production(cls) -> "BuildContext":
        """No overrides — all extractors use their hardcoded defaults."""
        return cls()

    @classmethod
    def from_fixture(cls, fixture_root: Path) -> "BuildContext":
        """Load the standard fixture layout from a corpus_snapshot_v<N> dir."""
        fixture_root = Path(fixture_root).resolve()
        mem = fixture_root / "memory_md_sample.md"
        ob = fixture_root / "ob_sample.md"
        return cls(
            memory_md_path=mem if mem.exists() else None,
            ob_markdown_path=ob if ob.exists() else None,
            fixture_root=fixture_root,
        )


def run_pipeline(kg, ctx: "BuildContext | None" = None) -> dict:
    """Execute the canonical QA-KG build pipeline against `kg`.

    Single source of truth for the build sequence used by:
      - cli.cmd_build (production and fixture builds)
      - qa_kg_determinism_cert_validate.py (D2 in-process test)

    Returns a stats dict with counts per stage. Extractor order is fixed:
    axioms → memory_rules → certs → source_claims → (ob if fixture) →
    edges. Changing this order changes the graph; see cert [228] D2/D3
    for the determinism contract.
    """
    from tools.qa_kg.extractors import axioms as x_axioms
    from tools.qa_kg.extractors import memory_rules as x_rules
    from tools.qa_kg.extractors import certs as x_certs
    from tools.qa_kg.extractors import edges as x_edges
    from tools.qa_kg.extractors import ob as x_ob
    from tools.qa_kg.extractors import source_claims as x_source_claims

    stats: dict = {}
    stats["axioms"] = len(x_axioms.populate(kg))
    stats["rules"] = len(x_rules.populate(
        kg,
        memory_md_path=ctx.memory_md_path if ctx else None,
    ))
    stats["certs"] = len(x_certs.populate(kg, run_validator=False))
    sc = x_source_claims.populate(kg)
    stats["source_claims"] = sc
    if ctx and ctx.ob_markdown_path is not None and ctx.ob_markdown_path.exists():
        text = ctx.ob_markdown_path.read_text(encoding="utf-8")
        stats["ob"] = x_ob.ingest_markdown(kg, text)
    stats["edges"] = x_edges.populate(kg)
    return stats

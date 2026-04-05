# Player5 Fresh Metal Pack

Purpose: move this repo onto `player5` and give Codex a consistent first-boot path
for steering docs, role rules, protocols, theorem context, and Open Brain access.

## What To Read First On player5

1. `AGENTS.md`
2. `MEMORY.md`
3. `CONSTITUTION.md`
4. `docs/specs/PROJECT_SPEC.md`
5. `docs/specs/VISION.md`
6. `CLAUDE.md`

## Minimal First Boot

From the cloned repo on `player5`:

```bash
cd /path/to/signal_experiments
bash player5_fresh_metal_pack/bootstrap_player5.sh
```

That script:

- verifies the expected project files exist
- creates `.venv_player5` if missing
- prints the key steering paths Codex should read first
- checks Open Brain connectivity with `qa_lab/open_brain_bootstrap.py`

## Install Open Brain Key

If the USB pack includes a key file, install it on `player5` with:

```bash
bash player5_fresh_metal_pack/install_open_brain_key.sh /path/to/open_brain_mcp_key.txt
```

The installer writes `~/.open_brain_mcp_key` with mode `600`.

## Clone From USB To Local Disk

If you do not want to work directly from the USB mount:

```bash
bash player5_fresh_metal_pack/clone_from_usb.sh \
  /path/to/usb/signal_experiments \
  /path/to/local/signal_experiments
```

This uses `rsync` and skips disposable virtualenv/cache paths that do not survive well
on exFAT anyway.

## Key Entry Points

- Roles and steering: `AGENTS.md`, `CLAUDE.md`, `MEMORY.md`, `CONSTITUTION.md`
- Full architecture: `docs/specs/PROJECT_SPEC.md`, `docs/specs/VISION.md`
- Cert health: `qa_alphageometry_ptolemy/qa_meta_validator.py`
- Cert docs: `docs/families/README.md`
- Open Brain bootstrap: `qa_lab/open_brain_bootstrap.py`
- Open Brain migration/ingest tools: `tools/open_brain_prepare_migration.py`, `tools/open_brain_ingest_queue_mcp.py`
- Theorem/discovery entry points: `qa_theorem_discovery_orchestrator.py`, `discover_theorems.sh`, `qa_resonance_theorem.md`, `QA_CONTROL_THEOREMS.md`

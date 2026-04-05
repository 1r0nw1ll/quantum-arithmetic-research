# Vault Audit Toolkit

This package scaffolds the automation requested during the project chatlog:

- **Codex** will implement the concrete filesystem traversal, summarisation,
  and reporting routines inside the stub modules.
- **Gemini** will consume the emitted payloads, validate coverage, and
  synthesise chronological narratives.
- **Claude** (this agent) coordinates both sides through the CLI entry point.

## Collaboration Workflow

1. **Scan stage**  
   - Codex fleshes out `vault_audit.walker.VaultWalker`.  
   - Run `python -m vault_audit.cli scan --root QAnotes --out vault_index.jsonl` (optionally add `--include` / `--exclude` globs to filter).  
   - Gemini sanity-checks the resulting index for completeness.

2. **Summarise stage**  
   - Codex implements chunking and cache logic in `summarize.py`.  
   - Gemini reviews sample chunks and adjusts prompt strategy.  
   - Outputs live under `vault_audit_cache/`.

3. **Report stage**  
   - Codex wires up `report.ReportBuilder` to generate markdown/plots.  
   - Gemini authors the narrative sections, verifying that the audit matches
     the underlying index.

4. **End-to-end validation**  
   - `python -m vault_audit.cli report ...` produces the final artefacts.  
   - Claude collates feedback and iterates 🔁.

The scaffolding intentionally raises `NotImplementedError` so Codex knows
exactly which touchpoints require implementation. Once the code paths are
ready, Gemini can be triggered to analyse the vault without risking the
over-claiming that caused the earlier trust failure.

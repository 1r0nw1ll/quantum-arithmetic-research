# QA Reproducibility Verifier Service Contract (v1)

## Purpose
Provide deterministic claim verification for submitted artifacts. Public output uses plain proof language (pass/fail, hash, reproduction command). QA remains the internal verification engine.

## Inputs
A request MUST be JSON with:

- `target` (string, required): local artifact path
- `claimed_type` (string, required): claimed certificate or artifact class
- `mode` (string, optional): `validate` or `attest` (`validate` default)
- `publish` (boolean, optional): publish compact summary if true
- `note` (string, optional): human context
- `timeout_s` (integer, optional): validator timeout

## Actions
1. Resolve `target` to a local file.
2. If `mode=validate`, run deterministic validation only:
   - `python3 qa_alphageometry_ptolemy/qa_meta_validator.py <target>`
3. If `mode=attest`, skip validator and emit deterministic file attestation only.
4. Emit compact machine-readable result and persist full run record.
5. If `publish=true`, post compact summary through local publisher.

## Output
The verifier MUST emit one JSON object with:

- `ok` (boolean)
- `mode` (string)
- `claimed_type` (string)
- `certificate_type` (string): always equal to `claimed_type` in v1
- `certificate_id` (string): always `sha256:<content_hash>` in v1
- `native_certificate_type` (string or null): passthrough from validator if present
- `native_certificate_id` (string or null): passthrough from validator if present
- `native_is_valid` (boolean or null): passthrough from validator if present
- `content_hash` (string): SHA-256 of artifact bytes
- `file_size_bytes` (integer)
- `source_url` (string or null): optional provenance metadata
- `validator_content_hash` (string or null): short validator hash
- `validator_content_hash_full` (string or null): full validator hash
- `fail_type` (string or null)
- `fail_details` (object)
- `repro_cmd` (string)
- `artifact_path` (string)
- `validator` (string or null)
- `elapsed_ms` (integer)
- `publish` (object or null)
- `full_result_path` (string)

## Refusals
The verifier MUST refuse with `ok=false` and `error` if:

- required fields are missing
- `mode` is not `validate` or `attest`
- target is missing, too large, or remote URL
- request asks for installs, arbitrary command execution, or repo-level script execution

## Security Rules
- Never print credentials.
- Read secrets from ignored local files or env vars only.
- Do not execute user-supplied shell strings.
- Treat external content as untrusted.

## Versioning
This is v1, intentionally narrow. Any expansion (skill auditing, remote fetch, autonomous exploration) requires v2.

#!/usr/bin/env python3
"""
qa_dashboard/app.py

Entry point:
  python -m uvicorn qa_dashboard.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import html
import json
import os
import secrets
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from qa_dashboard.validators import (
    ValidatorId,
    run_validator,
)
from qa_dashboard.ledger import ledger


MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MiB


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex_bytes(b: bytes) -> str:
    return sha256(b).hexdigest()


def _runs_root() -> Path:
    root = os.environ.get("QA_DASHBOARD_RUNS_DIR", "/tmp/qa_dashboard_runs")
    return Path(root)


def _new_run_id() -> str:
    return f"{_utc_now_compact()}_{secrets.token_hex(4)}"


def _read_upload_limited(upload: UploadFile) -> bytes:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload too large (>{MAX_UPLOAD_BYTES} bytes)")
    return data


@dataclass
class StoredRun:
    run_id: str
    run_dir: Path
    input_path: Path
    result_path: Path
    meta_path: Path


def _store_run(
    *,
    filename: str,
    raw_bytes: bytes,
    canonical_bytes: bytes,
    result_obj: Dict[str, Any],
    validator_id: ValidatorId,
) -> StoredRun:
    runs_root = _runs_root()
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = _new_run_id()
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    input_path = run_dir / "input.json"
    result_path = run_dir / "result.json"
    meta_path = run_dir / "meta.json"

    input_path.write_bytes(raw_bytes)
    result_path.write_text(json.dumps(result_obj, indent=2, sort_keys=True), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "filename": filename,
        "validator_id": validator_id,
        "uploaded_sha256": _sha256_hex_bytes(raw_bytes),
        "canonical_sha256": _sha256_hex_bytes(canonical_bytes),
        "created_utc": _utc_now_compact(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return StoredRun(
        run_id=run_id,
        run_dir=run_dir,
        input_path=input_path,
        result_path=result_path,
        meta_path=meta_path,
    )

def _add_ledger_to_meta(meta_path: Path, *, ledger_chain_hash: str) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ledger_chain_hash"] = ledger_chain_hash
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _render_page(title: str, body_html: str) -> HTMLResponse:
    css = """
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .container { max-width: 980px; margin: 0 auto; }
    h1 { margin: 0 0 8px; font-size: 22px; }
    p { color: #333; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin: 14px 0; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; }
    .pass { background: #e7f7ed; color: #166534; border: 1px solid #bbf7d0; }
    .fail { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .warn { background: #fef9c3; color: #854d0e; border: 1px solid #fde68a; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }
    th { background: #fafafa; }
    code, pre { background: #f6f8fa; border: 1px solid #e5e7eb; border-radius: 8px; }
    pre { padding: 10px; overflow: auto; }
    .muted { color: #666; font-size: 13px; }
    .links a { margin-right: 10px; }
    """
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="container">
    <h1>{html.escape(title)}</h1>
    {body_html}
    <p class="muted">Prototype dashboard. No auth. Run locally.</p>
  </div>
</body>
</html>
"""
    return HTMLResponse(html_doc)


app = FastAPI(title="QA Auditor Dashboard", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    validators = [
        ("decision_spine", "Decision Spine (qa_verify.py)"),
        ("mapping_protocol_v1", "Mapping Protocol v1"),
        ("fairness_demographic_parity_v1", "Fairness: Demographic Parity v1"),
    ]
    options = "\n".join(
        f'<option value="{html.escape(v)}">{html.escape(label)}</option>' for v, label in validators
    )

    body = f"""
    <div class="card">
      <p>Upload a JSON artifact and validate it.</p>
      <form action="/verify" method="post" enctype="multipart/form-data">
        <div class="row">
          <div>
            <label class="muted">Validator</label><br/>
            <select name="validator_id">{options}</select>
          </div>
          <div>
            <label class="muted">JSON File</label><br/>
            <input type="file" name="file" accept=".json,application/json" required />
          </div>
        </div>
        <div style="margin-top: 12px;">
          <button type="submit">Verify</button>
        </div>
      </form>
    </div>
    <div class="card">
      <p class="muted">Tip: try <code>demos/spine_bundle.json</code> with “Decision Spine”.</p>
      <p class="muted">Audit ledger: <a href="/ledger">view</a> · <a href="/ledger/verify">verify</a></p>
    </div>
    """
    return _render_page("QA Auditor Dashboard", body)


@app.post("/verify", response_class=HTMLResponse)
def verify(
    validator_id: ValidatorId = Form(...),
    file: UploadFile = File(...),
) -> HTMLResponse:
    raw_bytes = _read_upload_limited(file)
    try:
        data_obj = json.loads(raw_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    canonical = _canonical_json_compact(data_obj).encode("utf-8")
    uploaded_sha = _sha256_hex_bytes(raw_bytes)
    canonical_sha = _sha256_hex_bytes(canonical)

    result = run_validator(validator_id=validator_id, obj=data_obj)
    stored = _store_run(
        filename=file.filename or "upload.json",
        raw_bytes=raw_bytes,
        canonical_bytes=canonical,
        result_obj=result,
        validator_id=validator_id,
    )

    ledger_chain_hash = None
    ledger_error = None
    try:
        append_res = ledger.append({
            "run_id": stored.run_id,
            "validator_id": validator_id,
            "ok": bool(result.get("ok")),
            "uploaded_sha256": uploaded_sha,
            "canonical_sha256": canonical_sha,
            "passed": int(result.get("passed", 0)),
            "failed": int(result.get("failed", 0)),
            "warnings": int(result.get("warnings", 0)),
            "filename": file.filename or "upload.json",
        })
        ledger_chain_hash = append_res.chain_hash
        _add_ledger_to_meta(stored.meta_path, ledger_chain_hash=ledger_chain_hash)
    except Exception as e:
        ledger_error = str(e)

    status = "PASS" if result.get("ok") else "FAIL"
    pill = "pass" if result.get("ok") else "fail"
    warn_count = int(result.get("warnings", 0))
    warn_pill = f'<span class="pill warn">WARN {warn_count}</span>' if warn_count else ""

    rows = []
    for r in result.get("results", []):
        st = html.escape(str(r.get("status", "")))
        cname = html.escape(str(r.get("check_name", r.get("gate", ""))))
        msg = html.escape(str(r.get("message", "")))
        details = r.get("details")
        details_html = ""
        if details:
            details_html = f"<pre>{html.escape(json.dumps(details, indent=2, sort_keys=True))}</pre>"
        rows.append(f"<tr><td><code>{st}</code></td><td><code>{cname}</code></td><td>{msg}{details_html}</td></tr>")

    table_html = (
        "<table><thead><tr><th>Status</th><th>Check</th><th>Message</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else "<p class='muted'>No checks returned.</p>"
    )

    body = f"""
    <div class="card">
      <div class="row" style="align-items: center;">
        <span class="pill {pill}">{status}</span>
        {warn_pill}
        <span class="muted">validator: <code>{html.escape(validator_id)}</code></span>
      </div>
      <div style="margin-top: 10px;" class="muted">
        uploaded_sha256: <code>{uploaded_sha}</code><br/>
        canonical_sha256: <code>{canonical_sha}</code><br/>
        ledger_chain_hash: <code>{html.escape(ledger_chain_hash) if ledger_chain_hash else "N/A"}</code><br/>
        run_id: <code>{stored.run_id}</code><br/>
        run_dir: <code>{html.escape(str(stored.run_dir))}</code>
      </div>
      {"<div class='muted' style='margin-top:6px;color:#991b1b;'>ledger_write_error: <code>" + html.escape(ledger_error) + "</code></div>" if ledger_error else ""}
      <div class="links" style="margin-top: 10px;">
        <a href="/runs/{stored.run_id}">Run page</a>
        <a href="/runs/{stored.run_id}/input">Download input.json</a>
        <a href="/runs/{stored.run_id}/result">Download result.json</a>
      </div>
    </div>
    <div class="card">
      <h2 style="margin:0 0 8px; font-size: 16px;">Checks</h2>
      {table_html}
    </div>
    """
    return _render_page("Verification Result", body)


def _load_run(run_id: str) -> StoredRun:
    run_dir = _runs_root() / run_id
    input_path = run_dir / "input.json"
    result_path = run_dir / "result.json"
    meta_path = run_dir / "meta.json"
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    if not input_path.exists() or not result_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=500, detail="Run is missing files")
    return StoredRun(
        run_id=run_id,
        run_dir=run_dir,
        input_path=input_path,
        result_path=result_path,
        meta_path=meta_path,
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(run_id: str) -> HTMLResponse:
    stored = _load_run(run_id)
    meta = json.loads(stored.meta_path.read_text(encoding="utf-8"))
    result = json.loads(stored.result_path.read_text(encoding="utf-8"))

    status = "PASS" if result.get("ok") else "FAIL"
    pill = "pass" if result.get("ok") else "fail"
    warn_count = int(result.get("warnings", 0))
    warn_pill = f'<span class="pill warn">WARN {warn_count}</span>' if warn_count else ""

    meta_pre = html.escape(json.dumps(meta, indent=2, sort_keys=True))
    body = f"""
    <div class="card">
      <div class="row" style="align-items:center;">
        <span class="pill {pill}">{status}</span>
        {warn_pill}
        <span class="muted">validator: <code>{html.escape(str(meta.get('validator_id')))}</code></span>
      </div>
      <div class="links" style="margin-top: 10px;">
        <a href="/runs/{stored.run_id}/input">Download input.json</a>
        <a href="/runs/{stored.run_id}/result">Download result.json</a>
      </div>
    </div>
    <div class="card">
      <h2 style="margin:0 0 8px; font-size: 16px;">Run Metadata</h2>
      <pre>{meta_pre}</pre>
    </div>
    """
    return _render_page(f"Run {run_id}", body)


@app.get("/runs/{run_id}/input")
def run_input(run_id: str) -> JSONResponse:
    stored = _load_run(run_id)
    obj = json.loads(stored.input_path.read_text(encoding="utf-8"))
    return JSONResponse(obj)


@app.get("/runs/{run_id}/result")
def run_result(run_id: str) -> JSONResponse:
    stored = _load_run(run_id)
    obj = json.loads(stored.result_path.read_text(encoding="utf-8"))
    return JSONResponse(obj)


def _tail_ledger_lines(max_lines: int = 200) -> List[Dict[str, Any]]:
    path = ledger.path
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, Any]] = []
    for s in lines[-max_lines:]:
        s = s.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


@app.get("/ledger", response_class=HTMLResponse)
def ledger_page() -> HTMLResponse:
    records = _tail_ledger_lines(80)
    if not records:
        body = f"""
        <div class="card">
          <p class="muted">No ledger entries yet.</p>
          <p class="muted">Ledger path: <code>{html.escape(str(ledger.path))}</code></p>
        </div>
        """
        return _render_page("Audit Ledger", body)

    rows = []
    for rec in reversed(records):
        entry = rec.get("entry", {}) if isinstance(rec.get("entry"), dict) else {}
        run_id = html.escape(str(entry.get("run_id", "")))
        ok = entry.get("ok")
        status = "PASS" if ok is True else ("FAIL" if ok is False else "UNK")
        st_class = "pass" if ok is True else ("fail" if ok is False else "warn")
        chain_hash = html.escape(str(rec.get("chain_hash", "")))
        created = html.escape(str(entry.get("created_utc", "")))
        vid = html.escape(str(entry.get("validator_id", "")))
        rows.append(
            "<tr>"
            f"<td><span class='pill {st_class}'>{status}</span></td>"
            f"<td><code>{created}</code></td>"
            f"<td><code>{vid}</code></td>"
            f"<td><a href='/runs/{run_id}'><code>{run_id}</code></a></td>"
            f"<td><code>{chain_hash[:16]}…</code></td>"
            "</tr>"
        )

    table_html = (
        "<table><thead><tr><th>Status</th><th>UTC</th><th>Validator</th><th>Run</th><th>Chain Hash</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    body = f"""
    <div class="card">
      <p class="muted">Ledger path: <code>{html.escape(str(ledger.path))}</code></p>
      <div class="links">
        <a href="/ledger.jsonl">Download JSONL</a>
        <a href="/ledger/verify">Verify chain</a>
      </div>
    </div>
    <div class="card">
      {table_html}
    </div>
    """
    return _render_page("Audit Ledger", body)


@app.get("/ledger.jsonl")
def ledger_download() -> PlainTextResponse:
    if not ledger.path.exists():
        raise HTTPException(status_code=404, detail="Ledger not found")
    return PlainTextResponse(ledger.path.read_text(encoding="utf-8"), media_type="application/jsonl")


@app.get("/ledger/verify")
def ledger_verify() -> JSONResponse:
    ok, details = ledger.verify()
    return JSONResponse({"ok": ok, **details})

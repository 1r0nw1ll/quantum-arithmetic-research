"""
test_schemas.py — pytest suite for QA agent tool schemas.

Verifies:
  (a) additionalProperties=false on every schema
  (b) schema IDs match the ToolSpec definitions
  (c) validate_args accepts/rejects correctly
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qa_agent_security.schemas import (
    SCHEMAS,
    validate_args,
    SchemaValidationError,
)
from qa_agent_security import ToolSpec


# ---------------------------------------------------------------------------
# Schema registry invariants
# ---------------------------------------------------------------------------

class TestSchemaRegistry:
    def test_all_schemas_are_strict_no_additional_properties(self):
        for sid, sch in SCHEMAS.items():
            assert sch.get("type") == "object", f"{sid}: type != object"
            assert sch.get("additionalProperties") is False, \
                f"{sid}: additionalProperties is not False"

    def test_schema_ids_match_tool_specs(self):
        tools = [
            ToolSpec(name="run_shell", capability_scope="exec",
                     args_schema_id="SCHEMA.RUN_SHELL.v1"),
            ToolSpec(name="send_email", capability_scope="write_limited",
                     args_schema_id="SCHEMA.SEND_EMAIL.v1"),
            ToolSpec(name="http_fetch", capability_scope="network",
                     args_schema_id="SCHEMA.HTTP_FETCH.v1"),
        ]
        for t in tools:
            assert t.args_schema_id in SCHEMAS, \
                f"Tool {t.name} schema_id {t.args_schema_id!r} missing from registry"

    def test_all_schemas_have_required_field(self):
        for sid, sch in SCHEMAS.items():
            assert "required" in sch, f"{sid}: no 'required' field"
            assert len(sch["required"]) > 0, f"{sid}: empty 'required'"


# ---------------------------------------------------------------------------
# RUN_SHELL validation
# ---------------------------------------------------------------------------

class TestRunShell:
    SID = "SCHEMA.RUN_SHELL.v1"

    def test_accepts_minimal(self):
        validate_args(self.SID, {"command": "echo hi"})

    def test_accepts_full(self):
        validate_args(self.SID, {
            "command": "python3 test.py",
            "cwd": "/home/user",
            "env": {"PATH": "/usr/bin"},
            "timeout_ms": 30000,
            "stdin": "input data",
        })

    def test_rejects_additional_properties(self):
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {"command": "echo hi", "nope": 1})

    def test_rejects_missing_command(self):
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {"cwd": "/tmp"})

    def test_rejects_empty_command(self):
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {"command": ""})


# ---------------------------------------------------------------------------
# SEND_EMAIL validation
# ---------------------------------------------------------------------------

class TestSendEmail:
    SID = "SCHEMA.SEND_EMAIL.v1"

    def test_accepts_minimal(self):
        validate_args(self.SID, {
            "to": ["a@b.com"],
            "subject": "Hello",
            "body": "World",
        })

    def test_rejects_missing_body(self):
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {
                "to": ["a@b.com"],
                "subject": "Hello",
            })

    def test_rejects_additional_properties(self):
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {
                "to": ["a@b.com"],
                "subject": "Hello",
                "body": "x",
                "hidden_field": "gotcha",
            })


# ---------------------------------------------------------------------------
# HTTP_FETCH validation
# ---------------------------------------------------------------------------

class TestHttpFetch:
    SID = "SCHEMA.HTTP_FETCH.v1"

    def test_accepts_minimal(self):
        validate_args(self.SID, {"url": "https://example.com", "method": "GET"})

    def test_accepts_full(self):
        validate_args(self.SID, {
            "url": "https://api.github.com/repos",
            "method": "POST",
            "headers": {"Authorization": "Bearer token"},
            "body": '{"key": "value"}',
            "timeout_ms": 5000,
            "follow_redirects": True,
            "max_bytes": 1000000,
        })

    def test_rejects_missing_method(self):
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {"url": "https://example.com"})

    def test_rejects_invalid_method(self):
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {"url": "https://example.com", "method": "HACK"})

    def test_rejects_non_http_scheme(self):
        """Pattern enforcement: ^https?://"""
        with pytest.raises(SchemaValidationError):
            validate_args(self.SID, {"url": "file:///etc/passwd", "method": "GET"})


# ---------------------------------------------------------------------------
# Unknown schema
# ---------------------------------------------------------------------------

class TestUnknownSchema:
    def test_rejects_unknown_schema_id(self):
        with pytest.raises(SchemaValidationError, match="Unknown schema_id"):
            validate_args("SCHEMA.NONEXISTENT.v99", {"x": 1})

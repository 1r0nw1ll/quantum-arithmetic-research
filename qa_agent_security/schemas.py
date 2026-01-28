"""
schemas.py — Embedded strict JSON Schemas for QA agent tools.

Every tool's args must pass schema validation before the policy kernel
will mint a TOOL_CALL_CERT.v1.  additionalProperties=false on all schemas.

Requires only Python 3.10+ stdlib.  Falls back to minimal shape validator
if jsonschema is not installed.
"""
from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Embedded schemas (draft-07 / draft-2020-12 compatible)
# ---------------------------------------------------------------------------

SCHEMA_RUN_SHELL_V1: Dict[str, Any] = {
    "$id": "SCHEMA.RUN_SHELL.v1",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RUN_SHELL tool args",
    "type": "object",
    "additionalProperties": False,
    "required": ["command"],
    "properties": {
        "command": {"type": "string", "minLength": 1, "maxLength": 10000},
        "cwd": {"type": "string", "minLength": 1, "maxLength": 4096},
        "env": {
            "type": "object",
            "additionalProperties": {"type": "string", "maxLength": 10000},
            "maxProperties": 256,
        },
        "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 3600000},
        "stdin": {"type": "string", "maxLength": 1000000},
    },
}

SCHEMA_SEND_EMAIL_V1: Dict[str, Any] = {
    "$id": "SCHEMA.SEND_EMAIL.v1",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SEND_EMAIL tool args",
    "type": "object",
    "additionalProperties": False,
    "required": ["to", "subject", "body"],
    "properties": {
        "to": {
            "type": "array",
            "minItems": 1,
            "maxItems": 50,
            "items": {"type": "string", "format": "email", "maxLength": 320},
        },
        "cc": {
            "type": "array",
            "minItems": 0,
            "maxItems": 50,
            "items": {"type": "string", "format": "email", "maxLength": 320},
        },
        "bcc": {
            "type": "array",
            "minItems": 0,
            "maxItems": 50,
            "items": {"type": "string", "format": "email", "maxLength": 320},
        },
        "subject": {"type": "string", "minLength": 1, "maxLength": 998},
        "body": {"type": "string", "minLength": 1, "maxLength": 200000},
        "reply_to": {"type": "string", "format": "email", "maxLength": 320},
        "attachments": {
            "type": "array",
            "minItems": 0,
            "maxItems": 25,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["filename", "content_b64", "mime_type"],
                "properties": {
                    "filename": {"type": "string", "minLength": 1, "maxLength": 255},
                    "mime_type": {"type": "string", "minLength": 1, "maxLength": 255},
                    "content_b64": {"type": "string", "minLength": 1, "maxLength": 50000000},
                },
            },
        },
    },
}

SCHEMA_HTTP_FETCH_V1: Dict[str, Any] = {
    "$id": "SCHEMA.HTTP_FETCH.v1",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HTTP_FETCH tool args",
    "type": "object",
    "additionalProperties": False,
    "required": ["url", "method"],
    "properties": {
        "url": {
            "type": "string",
            "minLength": 1,
            "maxLength": 8192,
            "pattern": "^https?://",
        },
        "method": {
            "type": "string",
            "enum": ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        },
        "headers": {
            "type": "object",
            "additionalProperties": {"type": "string", "maxLength": 10000},
            "maxProperties": 200,
        },
        "body": {"type": "string", "maxLength": 5000000},
        "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 300000},
        "follow_redirects": {"type": "boolean"},
        "max_bytes": {"type": "integer", "minimum": 1, "maximum": 100000000},
    },
}

# ---------------------------------------------------------------------------
# Registry (single source of truth)
# ---------------------------------------------------------------------------

SCHEMAS: Dict[str, Dict[str, Any]] = {
    SCHEMA_RUN_SHELL_V1["$id"]: SCHEMA_RUN_SHELL_V1,
    SCHEMA_SEND_EMAIL_V1["$id"]: SCHEMA_SEND_EMAIL_V1,
    SCHEMA_HTTP_FETCH_V1["$id"]: SCHEMA_HTTP_FETCH_V1,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class SchemaValidationError(ValueError):
    """Raised when tool args fail schema validation."""
    pass


def get_schema(schema_id: str) -> Dict[str, Any]:
    """Look up a schema by $id.  Raises SchemaValidationError if unknown."""
    try:
        return SCHEMAS[schema_id]
    except KeyError as e:
        raise SchemaValidationError(f"Unknown schema_id: {schema_id}") from e


def _validate_with_jsonschema(schema: Dict[str, Any], instance: Dict[str, Any]) -> None:
    """Full validation via jsonschema package (if installed)."""
    import jsonschema  # type: ignore
    validator_cls = jsonschema.Draft202012Validator
    errors = sorted(validator_cls(schema).iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        parts = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            parts.append(f"{path}: {err.message}")
        raise SchemaValidationError(" | ".join(parts))


def _minimal_validate(schema: Dict[str, Any], instance: Dict[str, Any]) -> None:
    """
    Fallback validator (no external deps).
    Enforces: type=object, required fields, additionalProperties=false,
    basic type checks on required fields.
    """
    if schema.get("type") != "object":
        raise SchemaValidationError("Minimal validator only supports object schemas")
    if not isinstance(instance, dict):
        raise SchemaValidationError("Instance must be an object (dict)")

    # Required fields
    required = schema.get("required", [])
    for k in required:
        if k not in instance:
            raise SchemaValidationError(f"Missing required field: {k}")

    # Additional properties
    if schema.get("additionalProperties") is False:
        props = set((schema.get("properties") or {}).keys())
        extra = sorted(k for k in instance if k not in props)
        if extra:
            raise SchemaValidationError(f"Additional properties not allowed: {extra!r}")

    # Basic type checks on known properties
    properties = schema.get("properties", {})
    for k, v in instance.items():
        if k in properties:
            prop_schema = properties[k]
            prop_type = prop_schema.get("type")
            if prop_type == "string" and not isinstance(v, str):
                raise SchemaValidationError(f"Field {k!r}: expected string, got {type(v).__name__}")
            elif prop_type == "integer" and not isinstance(v, int):
                raise SchemaValidationError(f"Field {k!r}: expected integer, got {type(v).__name__}")
            elif prop_type == "boolean" and not isinstance(v, bool):
                raise SchemaValidationError(f"Field {k!r}: expected boolean, got {type(v).__name__}")
            elif prop_type == "array" and not isinstance(v, list):
                raise SchemaValidationError(f"Field {k!r}: expected array, got {type(v).__name__}")
            elif prop_type == "object" and not isinstance(v, dict):
                raise SchemaValidationError(f"Field {k!r}: expected object, got {type(v).__name__}")

            # String length checks
            if prop_type == "string" and isinstance(v, str):
                min_len = prop_schema.get("minLength")
                max_len = prop_schema.get("maxLength")
                if min_len is not None and len(v) < min_len:
                    raise SchemaValidationError(f"Field {k!r}: length {len(v)} < minLength {min_len}")
                if max_len is not None and len(v) > max_len:
                    raise SchemaValidationError(f"Field {k!r}: length {len(v)} > maxLength {max_len}")

            # Enum check
            if "enum" in prop_schema and v not in prop_schema["enum"]:
                raise SchemaValidationError(f"Field {k!r}: {v!r} not in enum {prop_schema['enum']}")

            # Pattern check (basic)
            if "pattern" in prop_schema and isinstance(v, str):
                import re
                if not re.search(prop_schema["pattern"], v):
                    raise SchemaValidationError(
                        f"Field {k!r}: {v!r} does not match pattern {prop_schema['pattern']!r}")


def validate_args(schema_id: str, args: Dict[str, Any]) -> None:
    """
    Validate tool args against an embedded schema.

    Tries jsonschema package first; falls back to minimal validator.
    Raises SchemaValidationError on failure.
    """
    schema = get_schema(schema_id)
    try:
        _validate_with_jsonschema(schema, args)
    except ModuleNotFoundError:
        _minimal_validate(schema, args)


# ---------------------------------------------------------------------------
# Export JSON schema files (for external tooling)
# ---------------------------------------------------------------------------

def export_schemas(output_dir: str) -> None:
    """Write each schema as a .json file to output_dir."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for sid, schema in SCHEMAS.items():
        # Convert Python False -> JSON false
        fname = sid.replace(".", "_").replace(" ", "_") + ".json"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(schema, f, indent=2, default=_bool_serializer)
        print(f"  Exported: {path}")


def _bool_serializer(obj):
    """Handle Python False -> JSON false in json.dump."""
    if isinstance(obj, bool):
        return obj
    raise TypeError(f"Not serializable: {obj!r}")


# Need json for export
import json

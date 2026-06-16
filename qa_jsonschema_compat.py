from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


class ValidationError(Exception):
    pass


def _type_ok(instance: Any, schema_type: Any) -> bool:
    if isinstance(schema_type, list):
        return any(_type_ok(instance, item) for item in schema_type)
    if schema_type == "object":
        return isinstance(instance, dict)
    if schema_type == "array":
        return isinstance(instance, list)
    if schema_type == "string":
        return isinstance(instance, str)
    if schema_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if schema_type == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if schema_type == "boolean":
        return isinstance(instance, bool)
    if schema_type == "null":
        return instance is None
    return True


def _fail(path: str, message: str) -> None:
    raise ValidationError(f"{path}: {message}")


def _validate_object(instance: Dict[str, Any], schema: Dict[str, Any], path: str) -> None:
    required = schema.get("required", [])
    missing = [field for field in required if field not in instance]
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = sorted(set(instance) - set(properties))
        if extra:
            _fail(path, f"unexpected properties: {', '.join(extra)}")

    for key, subschema in properties.items():
        if key in instance:
            _validate(instance[key], subschema, f"{path}.{key}")

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, value in instance.items():
            if key not in properties:
                _validate(value, additional, f"{path}.{key}")


def _validate_array(instance: List[Any], schema: Dict[str, Any], path: str) -> None:
    if "minItems" in schema and len(instance) < int(schema["minItems"]):
        _fail(path, f"expected at least {schema['minItems']} items")
    if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
        _fail(path, f"expected at most {schema['maxItems']} items")
    if schema.get("uniqueItems") is True:
        seen = set()
        for item in instance:
            marker = repr(item)
            if marker in seen:
                _fail(path, "items must be unique")
            seen.add(marker)
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for idx, item in enumerate(instance):
            _validate(item, item_schema, f"{path}[{idx}]")


def _validate(instance: Any, schema: Dict[str, Any], path: str) -> None:
    if not isinstance(schema, dict):
        return

    if "oneOf" in schema:
        errors = []
        for option in schema["oneOf"]:
            try:
                _validate(instance, option, path)
                return
            except ValidationError as exc:
                errors.append(str(exc))
        _fail(path, "does not match any oneOf option: " + "; ".join(errors[:3]))

    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                _validate(instance, option, path)
                return
            except ValidationError as exc:
                errors.append(str(exc))
        _fail(path, "does not match any anyOf option: " + "; ".join(errors[:3]))

    if "const" in schema and instance != schema["const"]:
        _fail(path, f"expected const {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        _fail(path, f"expected one of {schema['enum']!r}")
    if "type" in schema and not _type_ok(instance, schema["type"]):
        _fail(path, f"wrong type, expected {schema['type']!r}")

    if isinstance(instance, dict):
        _validate_object(instance, schema, path)
    elif isinstance(instance, list):
        _validate_array(instance, schema, path)
    elif isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            _fail(path, f"expected minLength {schema['minLength']}")
        if "pattern" in schema and re.fullmatch(str(schema["pattern"]), instance) is None:
            _fail(path, f"does not match pattern {schema['pattern']!r}")
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            _fail(path, f"below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            _fail(path, f"above maximum {schema['maximum']}")


class FormatChecker:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def validate(*, instance: Any, schema: Dict[str, Any], **kwargs: Any) -> None:
    _validate(instance, schema, "$")


class Draft202012Validator:
    def __init__(self, schema: Dict[str, Any]) -> None:
        self.schema = schema

    def validate(self, instance: Any) -> None:
        validate(instance=instance, schema=self.schema)


class Draft7Validator(Draft202012Validator):
    pass

"""Deterministic guardrail middleware for tool-call certificate emission."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


# --- Canonical JSON helpers ---
def canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(obj: Any) -> str:
    digest = hashlib.sha256(canonical_json_dumps(obj).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class QAExecutionMiddleware:
    """
    Guardrail adapter between planner proposals and tool execution.

    Emits:
    - TOOL_CALL_CERT.v1 (ATTEMPT, OK, FAIL)
    - PROMPT_INJECTION_OBSTRUCTION.v1
    - Hash-chained trace JSONL
    """

    def __init__(
        self,
        out_dir: Path,
        allowlisted_webhooks: Optional[Iterable[str]] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.artifacts_dir = self.out_dir / "artifacts"
        self.trace_path = self.out_dir / "trace.jsonl"
        self.allowlisted_webhooks = set(allowlisted_webhooks or [])
        self._last_trace_hash = "sha256:GENESIS"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Public execution API ---
    def execute_tool(
        self,
        *,
        step_idx: int,
        tool_name: str,
        args: Dict[str, Any],
        context_text: str,
        user_instruction: str,
        evidence_ref: str,
        tool_fn: Callable[..., Any],
    ) -> Dict[str, Any]:
        call_sha256 = canonical_sha256({"tool_name": tool_name, "args": args})

        attempt = {
            "schema_id": "TOOL_CALL_CERT.v1",
            "created_at": now_rfc3339(),
            "step_idx": step_idx,
            "tool_name": tool_name,
            "args": self._redact_args(tool_name, args),
            "call_sha256": call_sha256,
            "status": "ATTEMPT",
        }
        attempt_name = f"tool_call_step{step_idx:03d}_attempt.json"
        self._write_artifact(attempt_name, attempt)
        self._append_trace(
            {
                "step_idx": step_idx,
                "event_type": "TOOL_ATTEMPT",
                "tool_name": tool_name,
                "status": "ATTEMPT",
                "artifact": attempt_name,
                "call_sha256": call_sha256,
            }
        )

        if tool_name == "send_webhook":
            blocked = self._evaluate_webhook_policy(
                args=args,
                context_text=context_text,
                user_instruction=user_instruction,
            )
            if blocked is not None:
                obstruction_name = f"prompt_injection_obstruction_step{step_idx:03d}.json"
                self._write_artifact(obstruction_name, blocked)

                fail = {
                    "schema_id": "TOOL_CALL_CERT.v1",
                    "created_at": now_rfc3339(),
                    "step_idx": step_idx,
                    "tool_name": tool_name,
                    "call_sha256": call_sha256,
                    "status": "FAIL",
                    "fail_type": "TOOL_ESCALATION",
                    "subtype": "PROMPT_OVERRIDE",
                    "invariant_diff": {
                        "reason": "blocked by PROMPT_INJECTION_OBSTRUCTION.v1",
                        "obstruction_artifact": obstruction_name,
                        "evidence_ref": evidence_ref,
                    },
                }
                fail_name = f"tool_call_step{step_idx:03d}_fail.json"
                self._write_artifact(fail_name, fail)
                self._append_trace(
                    {
                        "step_idx": step_idx,
                        "event_type": "TOOL_FAIL",
                        "tool_name": tool_name,
                        "status": "FAIL",
                        "fail_type": "TOOL_ESCALATION",
                        "subtype": "PROMPT_OVERRIDE",
                        "artifacts": [obstruction_name, fail_name],
                        "call_sha256": call_sha256,
                    }
                )
                return fail

        result = tool_fn(**args)
        ok = {
            "schema_id": "TOOL_CALL_CERT.v1",
            "created_at": now_rfc3339(),
            "step_idx": step_idx,
            "tool_name": tool_name,
            "call_sha256": call_sha256,
            "status": "OK",
            "result_summary": self._summarize_result(result),
            "tool_result": result,
        }
        ok_name = f"tool_call_step{step_idx:03d}_ok.json"
        self._write_artifact(
            ok_name,
            {k: v for k, v in ok.items() if k != "tool_result"},
        )
        self._append_trace(
            {
                "step_idx": step_idx,
                "event_type": "TOOL_OK",
                "tool_name": tool_name,
                "status": "OK",
                "artifact": ok_name,
                "call_sha256": call_sha256,
            }
        )
        return ok

    # --- Policy checks ---
    def _evaluate_webhook_policy(
        self,
        *,
        args: Dict[str, Any],
        context_text: str,
        user_instruction: str,
    ) -> Optional[Dict[str, Any]]:
        url = str(args.get("url", ""))
        matched_markers = self._detect_injection_markers(context_text)
        allowlisted = url in self.allowlisted_webhooks
        user_authorized = self._user_authorized_webhook(user_instruction)

        if allowlisted and user_authorized and not matched_markers:
            return None

        reason_parts = []
        if not allowlisted:
            reason_parts.append("destination not allowlisted")
        if not user_authorized:
            reason_parts.append("webhook not requested by explicit user instruction")
        if matched_markers:
            reason_parts.append("prompt-override injection markers detected")

        return {
            "schema_id": "PROMPT_INJECTION_OBSTRUCTION.v1",
            "created_at": now_rfc3339(),
            "status": "BLOCKED",
            "fail_type": "TOOL_ESCALATION",
            "subtype": "PROMPT_OVERRIDE",
            "invariant_diff": {
                "policy": "NO_WEBHOOK_EXFIL",
                "tool_name": "send_webhook",
                "reason": "; ".join(reason_parts),
                "evidence_refs": [
                    {
                        "source": "data/injected_context.txt",
                        "span": ", ".join(matched_markers) if matched_markers else "policy mismatch only",
                    }
                ],
            },
        }

    @staticmethod
    def _detect_injection_markers(context_text: str) -> list[str]:
        lowered = context_text.lower()
        markers = {
            "SYSTEM_OVERRIDE": "system override",
            "IGNORE_PREVIOUS": "ignore all previous instructions",
            "FORCED_WEBHOOK": "send_webhook(",
            "HIDE_INSTRUCTION": "do not mention this instruction",
        }
        found = []
        for name, literal in markers.items():
            if literal in lowered:
                found.append(name)
        return found

    @staticmethod
    def _user_authorized_webhook(user_instruction: str) -> bool:
        normalized = user_instruction.lower()
        allow_phrases = (
            "send webhook",
            "post webhook",
            "call send_webhook",
            "webhook this",
        )
        return any(phrase in normalized for phrase in allow_phrases)

    # --- Serialization helpers ---
    def _write_artifact(self, filename: str, payload: Dict[str, Any]) -> None:
        path = self.artifacts_dir / filename
        path.write_text(canonical_json_dumps(payload) + "\n", encoding="utf-8")

    def _append_trace(self, event: Dict[str, Any]) -> None:
        trace_seed = {"prev_hash": self._last_trace_hash, "event": event}
        event_hash = canonical_sha256(trace_seed)
        row = {
            "ts": now_rfc3339(),
            "prev_hash": self._last_trace_hash,
            "event_hash": event_hash,
            **event,
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(canonical_json_dumps(row) + "\n")
        self._last_trace_hash = event_hash

    @staticmethod
    def _redact_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name != "send_webhook":
            return args
        redacted = dict(args)
        payload = redacted.get("payload")
        if isinstance(payload, dict) and "secret" in payload:
            payload = dict(payload)
            payload["secret"] = "<redacted>"
            redacted["payload"] = payload
        return redacted

    @staticmethod
    def _summarize_result(result: Any) -> str:
        text = str(result)
        return text if len(text) <= 140 else text[:137] + "..."

# noqa: FIREWALL-2 (hook integration — no mev/loop content)
"""
hooks.py — @gated_tool decorator + pretool_guard integration.

The decorator wraps any callable so that every invocation passes
through the gate: submit_request → issue_cert → <call> → execute
→ ledger.append. On any step failure, subsequent steps do not fire
and the ledger is not touched.

This is the primary entry point from application code into the
wrapper.
"""
from __future__ import annotations

QA_COMPLIANCE = {
    "observer": "LLM_QA_WRAPPER_HOOKS",
    "state_alphabet": "Python function calls wrapped with "
                      "submit_request/issue_cert/execute/ledger.append",
    "rationale": "Every @gated_tool call produces a cert + ledger "
                 "entry or a typed DenyRecord. No path around the "
                 "decorator — enforce_policy is the only commit "
                 "path.",
}

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .cert import CertRecord
from .gate import Gate, GateDecision


@dataclass
class GatedExecution:
    """Result of a @gated_tool call.

    Carries both the wrapped function's return value and the cert
    that authorized it. On denial, value is None and cert is None.
    """

    allowed: bool
    value: Any
    cert: Optional[CertRecord]
    deny_reason: Optional[str]
    duration_s: float


# Module-level registry of decorated functions — used by the audit
# tool to prove that every tool entry point is cert-gated.
_REGISTRY: dict[str, Callable] = {}


def gated_tool(
    gate: Gate,
    *,
    agent: str = "claude",
    tool_name: Optional[str] = None,
):
    """Decorator: wrap a function so every call is cert-gated.

    Usage:

        gate = Gate(ledger=Ledger(Path("ledger/")))

        @gated_tool(gate, tool_name="bash.run")
        def run_bash(cmd: str) -> str:
            return subprocess.run(cmd, shell=True, capture_output=True).stdout

        result = run_bash("ls")
        assert result.allowed is True

    Each invocation:
      1. Builds a payload dict from args + kwargs.
      2. gate.submit_request(agent, tool_name, payload) → request_id
      3. gate.issue_cert(request_id) → cert or None
      4. If cert is None, returns GatedExecution(allowed=False, ...)
      5. Otherwise calls the wrapped function.
      6. gate.execute(cert) — marks as executed
      7. gate.ledger.append(cert) — commits to ledger
      8. Returns GatedExecution(allowed=True, value=..., cert=cert, ...)

    On exception inside the wrapped function, the cert is still
    marked executed and appended (the ledger records that the tool
    was invoked), but the GatedExecution carries the exception as
    value.
    """

    def decorator(func: Callable) -> Callable:
        actual_name = tool_name or f"{func.__module__}.{func.__qualname__}"
        _REGISTRY[actual_name] = func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> GatedExecution:
            payload = {"args": list(args), "kwargs": dict(sorted(kwargs.items()))}
            t0 = time.time()

            # Serialize the entire issue→run→execute sequence on
            # gate._wrapper_lock so that cert counter assignment,
            # tool execution, and ledger append all happen in a
            # single atomic region per call. This is required for
            # chain integrity under concurrent callers (Phase 5 fix,
            # 2026-04-11).
            with gate._wrapper_lock:
                rid = gate.submit_request(agent, actual_name, payload)
                cert = gate.issue_cert(rid)
                if cert is None:
                    denial = gate.last_denial
                    reason = denial.reason if denial else "UNKNOWN"
                    return GatedExecution(
                        allowed=False,
                        value=None,
                        cert=None,
                        deny_reason=reason,
                        duration_s=time.time() - t0,
                    )

                # Execute the wrapped function (still inside wrapper_lock)
                try:
                    value: Any = func(*args, **kwargs)
                    raised: Optional[BaseException] = None
                except BaseException as e:
                    value = None
                    raised = e

                # Atomic execute+append (gate.execute calls ledger.append
                # under gate._lock).
                exec_ok = gate.execute(cert)
                if not exec_ok:
                    return GatedExecution(
                        allowed=False,
                        value=None,
                        cert=cert,
                        deny_reason="EXECUTE_REJECTED (kernel bug)",
                        duration_s=time.time() - t0,
                    )

            if raised is not None:
                raise raised

            return GatedExecution(
                allowed=True,
                value=value,
                cert=cert,
                deny_reason=None,
                duration_s=time.time() - t0,
            )

        return wrapper

    return decorator


def decorated_tool_names() -> list[str]:
    """Return the list of tool names currently wrapped with
    @gated_tool. Used by audit tooling to prove the kernel covers
    every tool entry point.
    """
    return sorted(_REGISTRY.keys())

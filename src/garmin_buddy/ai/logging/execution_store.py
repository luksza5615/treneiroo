from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any
from uuid import uuid4


_SENSITIVE_KEYS = {
    "api_key",
    "password",
    "secret",
    "token",
    "connection_string",
    "db_connection_string",
}


@dataclass(frozen=True)
class ExecutionArtifact:
    execution_id: str
    payload: dict[str, Any]


class ExecutionStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def append_execution(self, payload: dict[str, Any]) -> ExecutionArtifact:
        execution_id = str(uuid4())
        artifact_payload = _format_dates(_redact_sensitive(payload))
        artifact_payload.setdefault("execution_status", "succeeded")
        artifact_payload["execution_id"] = execution_id
        artifact_payload["created_at"] = datetime.now(timezone.utc).isoformat()

        path = self._base_dir / "training_review_executions.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact_payload) + "\n")

        return ExecutionArtifact(execution_id=execution_id, payload=artifact_payload)

    def append_failure(
        self, payload: dict[str, Any], error: Exception | BaseException
    ) -> ExecutionArtifact:
        failure_payload = dict(payload)
        failure_payload["execution_status"] = "failed"
        failure_payload["error"] = _serialize_exception(error)
        return self.append_execution(failure_payload)


def _format_dates(obj: Any) -> Any:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {key: _format_dates(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_format_dates(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(_format_dates(item) for item in obj)

    return obj


def _redact_sensitive(payload: dict[str, Any]) -> dict[str, Any]:
    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _redact_value(key, val) for key, val in value.items()}
        if isinstance(value, list):
            return [_redact(item) for item in value]
        return value

    def _redact_value(key: str, value: Any) -> Any:
        if key.lower() in _SENSITIVE_KEYS:
            return "***redacted***"
        return _redact(value)

    return {key: _redact_value(key, val) for key, val in payload.items()}


def _serialize_exception(error: Exception | BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }

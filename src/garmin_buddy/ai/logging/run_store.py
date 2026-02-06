from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
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
class RunArtifact:
    run_id: str
    payload: dict[str, Any]


class RunStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def append_run(self, payload: dict[str, Any]) -> RunArtifact:
        run_id = str(uuid4())
        artifact_payload = _redact_sensitive(payload)
        artifact_payload["run_id"] = run_id
        artifact_payload["created_at"] = datetime.now(timezone.utc).isoformat()

        path = self._base_dir / "training_review_runs.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact_payload) + "\n")

        return RunArtifact(run_id=run_id, payload=artifact_payload)


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

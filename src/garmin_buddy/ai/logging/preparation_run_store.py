from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from garmin_buddy.ai.logging.run_store import _redact_sensitive, _serialize_exception


@dataclass(frozen=True)
class PreparationRunArtifact:
    run_id: str
    payload: dict[str, Any]


class PreparationRunStore:
    """Persist stage traces because multi-step planning must be replayable and inspectable."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._strategy_dir = self._base_dir / "preparation_strategy_state"
        self._strategy_dir.mkdir(parents=True, exist_ok=True)

    def append_run(self, payload: dict[str, Any]) -> PreparationRunArtifact:
        run_id = str(uuid4())
        artifact_payload = _format_dates(dict(_redact_sensitive(payload)))
        artifact_payload.setdefault("run_status", "succeeded")
        artifact_payload["run_id"] = run_id
        artifact_payload["created_at"] = datetime.now(timezone.utc).isoformat()

        path = self._base_dir / "training_plan_preparation_runs.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact_payload) + "\n")

        return PreparationRunArtifact(run_id=run_id, payload=artifact_payload)

    def append_failure(
        self, payload: dict[str, Any], error: Exception | BaseException
    ) -> PreparationRunArtifact:
        failure_payload = dict(payload)
        failure_payload["run_status"] = "failed"
        failure_payload["error"] = _serialize_exception(error)
        return self.append_run(failure_payload)

    def save_strategy_state(self, strategy_id: str, payload: dict[str, Any]) -> None:
        path = self._strategy_dir / f"{strategy_id}.json"
        payload_to_save = _format_dates(dict(_redact_sensitive(payload)))
        payload_to_save["updated_at"] = datetime.now(timezone.utc).isoformat()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload_to_save, handle, indent=2)

    def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None:
        path = self._strategy_dir / f"{strategy_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def find_strategy_by_input_hash(self, input_hash: str) -> dict[str, Any] | None:
        for path in sorted(self._strategy_dir.glob("*.json"), reverse=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("strategy", {}).get("input_hash") == input_hash:
                return payload
        return None

    def find_lab_analysis(self, lab_fingerprint: str | None) -> dict[str, Any] | None:
        if lab_fingerprint is None:
            return None
        for path in sorted(self._strategy_dir.glob("*.json"), reverse=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("lab_fingerprint") == lab_fingerprint:
                return payload.get("lab_analysis")
        return None


def _format_dates(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _format_dates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_dates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_format_dates(item) for item in value)
    return value

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from garmin_buddy.ai.logging.preparation_execution_store import (
    PreparationExecutionStore,
)


def test_preparation_execution_store_persists_executions_and_strategy_state() -> None:
    temp_dir = _workspace_temp_dir("execution-store")
    store = PreparationExecutionStore(temp_dir)

    artifact = store.append_execution(
        {
            "api_key": "secret",
            "stage": "strategy",
            "total_input_tokens": 321,
            "total_output_tokens": 54,
        }
    )
    store.save_strategy_state("strategy-1", {"strategy": {"strategy_id": "strategy-1"}})

    executions_path = temp_dir / "training_plan_preparation_executions.jsonl"
    saved_line = json.loads(executions_path.read_text(encoding="utf-8").strip())
    strategy_state = store.load_strategy_state("strategy-1")

    assert artifact.execution_id == saved_line["execution_id"]
    assert saved_line["api_key"] == "***redacted***"
    assert saved_line["total_input_tokens"] == 321
    assert saved_line["total_output_tokens"] == 54
    assert strategy_state["strategy"]["strategy_id"] == "strategy-1"


def test_preparation_execution_store_persists_failure_artifacts() -> None:
    temp_dir = _workspace_temp_dir("execution-store-failure")
    store = PreparationExecutionStore(temp_dir)

    try:
        raise RuntimeError("503 Service Unavailable")
    except RuntimeError as exc:
        artifact = store.append_failure({"stage": "strategy_generation"}, exc)

    executions_path = temp_dir / "training_plan_preparation_executions.jsonl"
    saved_line = json.loads(executions_path.read_text(encoding="utf-8").strip())

    assert artifact.payload["execution_status"] == "failed"
    assert saved_line["stage"] == "strategy_generation"
    assert saved_line["error"]["type"] == "RuntimeError"


def _workspace_temp_dir(name: str) -> Path:
    path = Path("zignored") / "pytest-temp" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path

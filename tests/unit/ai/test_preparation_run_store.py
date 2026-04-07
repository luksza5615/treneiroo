from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from garmin_buddy.ai.logging.preparation_run_store import PreparationRunStore


def test_preparation_run_store_persists_runs_and_strategy_state() -> None:
    temp_dir = _workspace_temp_dir("run-store")
    store = PreparationRunStore(temp_dir)

    artifact = store.append_run({"api_key": "secret", "stage": "strategy"})
    store.save_strategy_state("strategy-1", {"strategy": {"strategy_id": "strategy-1"}})

    runs_path = temp_dir / "training_plan_preparation_runs.jsonl"
    saved_line = json.loads(runs_path.read_text(encoding="utf-8").strip())
    strategy_state = store.load_strategy_state("strategy-1")

    assert artifact.run_id == saved_line["run_id"]
    assert saved_line["api_key"] == "***redacted***"
    assert strategy_state["strategy"]["strategy_id"] == "strategy-1"


def _workspace_temp_dir(name: str) -> Path:
    path = Path("zignored") / "pytest-temp" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path

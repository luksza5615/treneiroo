from pathlib import Path

from garmin_buddy.ai.logging.run_store import RunStore


def test_run_store_redacts_sensitive_fields(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    artifact = store.append_run(
        {
            "prompt_version": "v1",
            "api_key": "secret",
            "nested": {"password": "hidden"},
            "tool_calls": [],
        }
    )

    assert artifact.run_id
    assert artifact.payload["api_key"] == "***redacted***"
    assert artifact.payload["nested"]["password"] == "***redacted***"
    assert (tmp_path / "training_review_runs.jsonl").exists()

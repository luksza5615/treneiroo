from pathlib import Path

from garmin_buddy.ai.logging.run_store import RunStore, _format_dates


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


def test_format_dates_recursively_converts_nested_datetimes() -> None:
    from datetime import datetime

    nested_payload = {
        "key1": {"key2": datetime(2021, 12, 1, 13, 45, 55), "key3": "test"},
        "items": [{"timestamp": datetime(2022, 1, 2, 3, 4, 5)}],
        "tuple_values": (datetime(2023, 2, 3, 4, 5, 6),),
        "key4": datetime(2025, 12, 1, 13, 45, 55),
    }

    formatted = _format_dates(nested_payload)

    assert isinstance(formatted["key1"]["key2"], str)
    assert isinstance(formatted["key1"]["key3"], str)
    assert isinstance(formatted["items"][0]["timestamp"], str)
    assert isinstance(formatted["tuple_values"][0], str)
    assert isinstance(formatted["key4"], str)


def test_format_dates_does_not_mutate_input_payload() -> None:
    from datetime import datetime

    original_dt = datetime(2021, 12, 1, 13, 45, 55)
    nested_payload = {"outer": {"inner": original_dt}, "items": [original_dt]}

    _ = _format_dates(nested_payload)

    assert isinstance(nested_payload["outer"]["inner"], datetime)
    assert isinstance(nested_payload["items"][0], datetime)


def test_append_run_formats_nested_datetime_values(tmp_path: Path) -> None:
    from datetime import datetime

    store = RunStore(tmp_path)
    payload = {
        "events": [{"started_at": datetime(2024, 3, 4, 5, 6, 7)}],
        "nested": {"ended_at": datetime(2024, 3, 4, 6, 6, 7)},
    }

    artifact = store.append_run(payload)

    assert isinstance(artifact.payload["events"][0]["started_at"], str)
    assert isinstance(artifact.payload["nested"]["ended_at"], str)
    assert isinstance(payload["events"][0]["started_at"], datetime)

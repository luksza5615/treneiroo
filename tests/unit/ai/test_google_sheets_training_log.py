from __future__ import annotations

from datetime import date

import pytest

from garmin_buddy.integrations.google_sheets_training_log import (
    GoogleSheetsSettings,
    GoogleSheetsTrainingLogClient,
    TrainingLogIntegrationError,
)


class _FakeSheetsService:
    def __init__(self, values: list[list[str]]) -> None:
        self._values = values

    def spreadsheets(self):
        return self

    def values(self):
        return self

    def get(self, spreadsheetId: str, range: str):
        return self

    def execute(self):
        return {"values": self._values}


def test_google_sheets_training_log_normalizes_rows() -> None:
    client = GoogleSheetsTrainingLogClient(
        GoogleSheetsSettings("sheet", "Plan!A:E", "{}"),
        sheets_service=_FakeSheetsService(
            [
                ["date", "session_type", "planned_focus", "planned_load", "notes"],
                ["2026-02-01", "easy", "aerobic", "low", "steady run"],
                ["2026-02-10", "long", "endurance", "medium", "long run"],
            ]
        ),
    )

    rows = client.list_sessions(date(2026, 2, 1), date(2026, 2, 5))

    assert rows == [
        {
            "date": "2026-02-01",
            "session_type": "easy",
            "planned_focus": "aerobic",
            "planned_load": "low",
            "notes": "steady run",
        }
    ]


def test_google_sheets_training_log_rejects_missing_columns() -> None:
    client = GoogleSheetsTrainingLogClient(
        GoogleSheetsSettings("sheet", "Plan!A:E", "{}"),
        sheets_service=_FakeSheetsService([["date", "notes"]]),
    )

    with pytest.raises(TrainingLogIntegrationError, match="Required Google Sheets column"):
        client.list_sessions()

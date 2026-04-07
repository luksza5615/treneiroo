from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from typing import Any

from google.oauth2 import service_account


class TrainingLogIntegrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GoogleSheetsSettings:
    spreadsheet_id: str
    worksheet_name: str
    service_account_info: str


class GoogleSheetsTrainingLogClient:
    """Read planned sessions from Sheets without introducing write-back complexity in v1."""

    def __init__(
        self,
        settings: GoogleSheetsSettings,
        *,
        column_mapping: dict[str, str] | None = None,
        sheets_service: Any | None = None,
    ) -> None:
        self._settings = settings
        self._column_mapping = column_mapping or {
            "date": "date",
            "session_type": "session_type",
            "planned_focus": "planned_focus",
            "planned_load": "planned_load",
            "notes": "notes",
        }
        self._sheets_service = sheets_service

    def list_sessions(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> list[dict[str, Any]]:
        rows = self._fetch_rows()
        return _normalize_rows(
            rows,
            column_mapping=self._column_mapping,
            start_date=start_date,
            end_date=end_date,
        )

    def _fetch_rows(self) -> list[list[str]]:
        service = self._sheets_service or self._build_service()
        response = (
            service.spreadsheets()
            .values()
            .get(
                spreadsheetId=self._settings.spreadsheet_id,
                range=self._settings.worksheet_name,
            )
            .execute()
        )
        values = response.get("values", [])
        if not values:
            raise TrainingLogIntegrationError("Google Sheets training log is empty.")
        return values

    def _build_service(self) -> Any:
        try:
            from googleapiclient.discovery import build
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise TrainingLogIntegrationError(
                "google-api-python-client is required for Google Sheets integration."
            ) from exc

        try:
            info = json.loads(self._settings.service_account_info)
        except json.JSONDecodeError as exc:
            raise TrainingLogIntegrationError(
                "GOOGLE_SERVICE_ACCOUNT_INFO must contain valid JSON."
            ) from exc

        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        return build("sheets", "v4", credentials=credentials, cache_discovery=False)


def _normalize_rows(
    rows: list[list[str]],
    *,
    column_mapping: dict[str, str],
    start_date: date | None,
    end_date: date | None,
) -> list[dict[str, Any]]:
    header = [str(value).strip().lower() for value in rows[0]]
    header_index = {name: index for index, name in enumerate(header)}
    normalized_rows: list[dict[str, Any]] = []

    required_fields = ("date", "session_type", "planned_focus", "planned_load", "notes")
    for field_name in required_fields:
        mapped = column_mapping[field_name].strip().lower()
        if mapped not in header_index:
            raise TrainingLogIntegrationError(
                f"Required Google Sheets column '{column_mapping[field_name]}' was not found."
            )

    for row in rows[1:]:
        if not row:
            continue

        payload = {
            field_name: _get_cell(row, header_index[column_mapping[field_name].strip().lower()])
            for field_name in required_fields
        }
        try:
            session_date = date.fromisoformat(payload["date"])
        except ValueError:
            continue

        if start_date and session_date < start_date:
            continue
        if end_date and session_date > end_date:
            continue

        normalized_rows.append(
            {
                "date": session_date.isoformat(),
                "session_type": payload["session_type"],
                "planned_focus": payload["planned_focus"],
                "planned_load": payload["planned_load"],
                "notes": payload["notes"],
            }
        )

    return normalized_rows


def _get_cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return str(row[index]).strip()

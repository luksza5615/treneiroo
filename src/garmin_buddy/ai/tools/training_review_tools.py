from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import pandas as pd

from garmin_buddy.database.db_service import ActivityRepository


class ToolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    payload: Any
    error: str | None = None


@dataclass
class ToolRegistry:
    repository: ActivityRepository
    max_tool_calls: int
    _cache: dict[tuple[str, tuple[tuple[str, Any], ...]], ToolResult] = field(
        default_factory=dict
    )
    _tool_calls: int = 0
    _call_log: list[dict[str, Any]] = field(default_factory=list)
    _tools: dict[str, Callable[[dict[str, Any]], ToolResult]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be a positive integer.")

        self._tools = {
            "get_training_summary": self._get_training_summary,
            "list_key_sessions": self._list_key_sessions,
            "get_activity": self._get_activity,
        }

    def available_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def remaining_budget(self) -> int:
        return max(self.max_tool_calls - self._tool_calls, 0)

    def get_call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            raise ToolValidationError(f"Tool '{name}' is not in the allowlist.")

        validated_args = self._validate_args(name, args)
        cache_key = (name, tuple(sorted(validated_args.items())))

        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            self._call_log.append(
                {
                    "name": name,
                    "args": validated_args,
                    "ok": cached_result.ok,
                    "error": cached_result.error,
                    "cached": True,
                }
            )
            return cached_result

        if self._tool_calls >= self.max_tool_calls:
            return ToolResult(
                name=name,
                ok=False,
                payload=None,
                error="Tool call budget exceeded.",
            )

        self._tool_calls += 1
        result = self._tools[name](validated_args)
        self._call_log.append(
            {
                "name": name,
                "args": validated_args,
                "ok": result.ok,
                "error": result.error,
                "cached": False,
            }
        )
        self._cache[cache_key] = result
        return result

    def _validate_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolValidationError("Tool arguments must be a dictionary.")

        if name in {"get_training_summary", "list_key_sessions"}:
            start_date = _parse_date_arg(args, "start_date")
            end_date = _parse_date_arg(args, "end_date")
            if start_date > end_date:
                raise ToolValidationError("start_date cannot be later than end_date.")
            athlete_id = _parse_optional_int(args.get("athlete_id"))
            validated = {
                "start_date": start_date,
                "end_date": end_date,
                "athlete_id": athlete_id,
            }
            if name == "list_key_sessions":
                n_value = args.get("n", 5)
                n = _parse_positive_int(n_value, "n")
                validated["n"] = n
            return validated

        if name == "get_activity":
            activity_id = _parse_positive_int(args.get("activity_id"), "activity_id")
            return {"activity_id": activity_id}

        raise ToolValidationError(f"Tool '{name}' validation is not implemented.")

    def _get_training_summary(self, args: dict[str, Any]) -> ToolResult:
        try:
            summary = self.repository.get_training_summary(
                args["start_date"],
                args["end_date"],
                athlete_id=args.get("athlete_id"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(
                name="get_training_summary",
                ok=False,
                payload=None,
                error=str(exc),
            )

        return ToolResult(name="get_training_summary", ok=True, payload=summary)

    def _list_key_sessions(self, args: dict[str, Any]) -> ToolResult:
        try:
            sessions_df = self.repository.list_key_sessions(
                args["start_date"],
                args["end_date"],
                athlete_id=args.get("athlete_id"),
                n=args["n"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(
                name="list_key_sessions",
                ok=False,
                payload=None,
                error=str(exc),
            )

        return ToolResult(
            name="list_key_sessions",
            ok=True,
            payload=_dataframe_to_records(sessions_df),
        )

    def _get_activity(self, args: dict[str, Any]) -> ToolResult:
        try:
            activity_df = self.repository.get_activity_by_id(args["activity_id"])
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(
                name="get_activity",
                ok=False,
                payload=None,
                error=str(exc),
            )

        return ToolResult(
            name="get_activity",
            ok=True,
            payload=_dataframe_to_records(activity_df),
        )


def _parse_date_arg(args: dict[str, Any], field_name: str) -> date:
    value = args.get(field_name)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ToolValidationError(
                f"{field_name} must be ISO format YYYY-MM-DD."
            ) from exc

    raise ToolValidationError(f"{field_name} must be a date or ISO string.")


# TODO
def _convert_date_for_json():
    pass


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError("athlete_id must be an integer.")
    return value


def _parse_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ToolValidationError(f"{field_name} must be a positive integer.")
    return value


def _dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    return df.where(pd.notna(df), None).to_dict(orient="records")

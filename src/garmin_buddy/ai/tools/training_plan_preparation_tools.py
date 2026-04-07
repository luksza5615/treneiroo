from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from garmin_buddy.database.db_service import ActivityRepository
from garmin_buddy.integrations.google_sheets_training_log import (
    TrainingLogIntegrationError,
)


class ToolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    payload: Any
    error: str | None = None


@dataclass
class PreparationToolRegistry:
    """Keep tool access narrow so the workflow remains explainable and testable."""

    repository: ActivityRepository
    max_tool_calls: int
    training_log_loader: Callable[[date, date], list[dict[str, Any]]] | None = None
    profile_loader: Callable[[], dict[str, Any]] | None = None
    lab_loader: Callable[[], dict[str, Any]] | None = None
    previous_artifact_loader: Callable[[str], dict[str, Any] | None] | None = None
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
            "get_runner_profile": self._get_runner_profile,
            "get_latest_lab_payload": self._get_latest_lab_payload,
            "get_training_log_rows": self._get_training_log_rows,
            "get_execution_summary": self._get_execution_summary,
            "list_executed_key_sessions": self._list_executed_key_sessions,
            "compare_planned_vs_executed": self._compare_planned_vs_executed,
            "get_previous_preparation_artifact": self._get_previous_preparation_artifact,
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
            result = self._cache[cache_key]
            self._call_log.append(
                {
                    "name": name,
                    "args": validated_args,
                    "ok": result.ok,
                    "error": result.error,
                    "cached": True,
                }
            )
            return result

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

        if name in {
            "get_training_log_rows",
            "get_execution_summary",
            "list_executed_key_sessions",
            "compare_planned_vs_executed",
        }:
            start_date = _parse_date_arg(args, "start_date")
            end_date = _parse_date_arg(args, "end_date")
            if start_date > end_date:
                raise ToolValidationError("start_date cannot be later than end_date.")
            validated: dict[str, Any] = {
                "start_date": start_date,
                "end_date": end_date,
                "athlete_id": _parse_optional_int(args.get("athlete_id")),
            }
            if name == "list_executed_key_sessions":
                validated["n"] = _parse_positive_int(args.get("n", 5), "n")
            return validated

        if name == "get_previous_preparation_artifact":
            strategy_id = args.get("strategy_id")
            if not isinstance(strategy_id, str) or not strategy_id.strip():
                raise ToolValidationError("strategy_id must be a non-empty string.")
            return {"strategy_id": strategy_id.strip()}

        return {}

    def _get_runner_profile(self, args: dict[str, Any]) -> ToolResult:
        if self.profile_loader is None:
            return ToolResult(
                name="get_runner_profile",
                ok=False,
                payload=None,
                error="profile_loader_not_configured",
            )
        return ToolResult(name="get_runner_profile", ok=True, payload=self.profile_loader())

    def _get_latest_lab_payload(self, args: dict[str, Any]) -> ToolResult:
        if self.lab_loader is None:
            return ToolResult(
                name="get_latest_lab_payload",
                ok=False,
                payload=None,
                error="lab_loader_not_configured",
            )
        return ToolResult(
            name="get_latest_lab_payload",
            ok=True,
            payload=self.lab_loader(),
        )

    def _get_training_log_rows(self, args: dict[str, Any]) -> ToolResult:
        if self.training_log_loader is None:
            return ToolResult(
                name="get_training_log_rows",
                ok=False,
                payload=None,
                error="training_log_loader_not_configured",
            )
        try:
            rows = self.training_log_loader(args["start_date"], args["end_date"])
        except TrainingLogIntegrationError as exc:
            return ToolResult(
                name="get_training_log_rows",
                ok=False,
                payload=None,
                error=str(exc),
            )
        return ToolResult(name="get_training_log_rows", ok=True, payload=rows)

    def _get_execution_summary(self, args: dict[str, Any]) -> ToolResult:
        try:
            payload = self.repository.get_execution_summary(
                args["start_date"],
                args["end_date"],
                athlete_id=args.get("athlete_id"),
            )
        except Exception as exc:  # pragma: no cover
            return ToolResult(
                name="get_execution_summary", ok=False, payload=None, error=str(exc)
            )
        return ToolResult(name="get_execution_summary", ok=True, payload=payload)

    def _list_executed_key_sessions(self, args: dict[str, Any]) -> ToolResult:
        try:
            sessions_df = self.repository.list_key_sessions(
                args["start_date"],
                args["end_date"],
                athlete_id=args.get("athlete_id"),
                n=args["n"],
            )
        except Exception as exc:  # pragma: no cover
            return ToolResult(
                name="list_executed_key_sessions",
                ok=False,
                payload=None,
                error=str(exc),
            )
        payload = sessions_df.where(sessions_df.notna(), None).to_dict(orient="records")
        return ToolResult(
            name="list_executed_key_sessions", ok=True, payload=payload
        )

    def _compare_planned_vs_executed(self, args: dict[str, Any]) -> ToolResult:
        rows_result = self._get_training_log_rows(args)
        if not rows_result.ok:
            return ToolResult(
                name="compare_planned_vs_executed",
                ok=False,
                payload=None,
                error=rows_result.error,
            )
        try:
            payload = self.repository.compare_planned_vs_executed(
                rows_result.payload,
                args["start_date"],
                args["end_date"],
                athlete_id=args.get("athlete_id"),
            )
        except Exception as exc:  # pragma: no cover
            return ToolResult(
                name="compare_planned_vs_executed",
                ok=False,
                payload=None,
                error=str(exc),
            )
        return ToolResult(
            name="compare_planned_vs_executed",
            ok=True,
            payload=payload,
        )

    def _get_previous_preparation_artifact(self, args: dict[str, Any]) -> ToolResult:
        if self.previous_artifact_loader is None:
            return ToolResult(
                name="get_previous_preparation_artifact",
                ok=False,
                payload=None,
                error="previous_artifact_loader_not_configured",
            )
        payload = self.previous_artifact_loader(args["strategy_id"])
        if payload is None:
            return ToolResult(
                name="get_previous_preparation_artifact",
                ok=False,
                payload=None,
                error="strategy_state_not_found",
            )
        return ToolResult(
            name="get_previous_preparation_artifact",
            ok=True,
            payload=payload,
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

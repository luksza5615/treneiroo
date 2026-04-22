from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import json
from typing import Any, Protocol

from garmin_buddy.ai.contracts.contracts import (
    TrainingReviewReport,
    build_fallback_training_review_report,
    parse_training_review_report,
)
from garmin_buddy.ai.logging.run_store import RunStore
from garmin_buddy.ai.tools.training_review_tools import ToolRegistry

_DEFAULT_MAX_TOOL_CALLS = 2
_PROMPT_VERSION = "training_review_v1"


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class TrainingReviewInputs:
    start_date: date
    end_date: date
    athlete_id: int | None = None
    include_key_sessions: bool = True
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS


@dataclass(frozen=True)
class TrainingReviewResult:
    report: TrainingReviewReport
    raw_response: str
    parse_ok: bool
    retry_count: int
    run_id: str | None = None


def run_training_review(
    *,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    inputs: TrainingReviewInputs,
    run_store: RunStore | None = None,
    model_name: str | None = None,
) -> TrainingReviewResult:
    current_stage = "get_training_summary"
    raw_response = ""

    try:
        summary_result = tool_registry.call_tool(
            "get_training_summary",
            {
                "start_date": inputs.start_date,
                "end_date": inputs.end_date,
                "athlete_id": inputs.athlete_id,
            },
        )
        if not summary_result.ok:
            fallback = build_fallback_training_review_report(
                inputs.start_date,
                inputs.end_date,
                error_reason=summary_result.error or "summary_tool_failed",
            )
            result = TrainingReviewResult(
                report=fallback,
                raw_response="",
                parse_ok=False,
                retry_count=0,
            )
        else:
            key_sessions_payload: list[dict[str, Any]] = []
            evidence_sessions: list[dict[str, Any]] = []
            missing_data: list[str] = []
            if inputs.include_key_sessions:
                current_stage = "list_key_sessions"
                key_sessions_result = tool_registry.call_tool(
                    "list_key_sessions",
                    {
                        "start_date": inputs.start_date,
                        "end_date": inputs.end_date,
                        "athlete_id": inputs.athlete_id,
                        "n": 5,
                    },
                )
                if key_sessions_result.ok:
                    key_sessions_payload = key_sessions_result.payload
                else:
                    missing_data.append("key_sessions_unavailable")

            current_stage = "fetch_evidence"
            evidence_sessions = _maybe_fetch_evidence(
                llm_client=llm_client,
                tool_registry=tool_registry,
                start_date=inputs.start_date,
                end_date=inputs.end_date,
                athlete_id=inputs.athlete_id,
                key_sessions=key_sessions_payload,
                missing_data=missing_data,
            )

            prompt = _build_prompt(
                start_date=inputs.start_date,
                end_date=inputs.end_date,
                athlete_id=inputs.athlete_id,
                training_summary=summary_result.payload,
                key_sessions=key_sessions_payload,
                evidence_sessions=evidence_sessions,
                missing_data=missing_data,
            )

            current_stage = "generate_report"
            raw_response = llm_client.generate(prompt)

            current_stage = "parse_or_repair"
            parse_ok, report = _parse_or_repair(
                llm_client=llm_client,
                start_date=inputs.start_date,
                end_date=inputs.end_date,
                raw_response=raw_response,
            )

            result = TrainingReviewResult(
                report=report,
                raw_response=raw_response,
                parse_ok=parse_ok,
                retry_count=0 if parse_ok else 1,
            )
    except Exception as exc:
        if run_store is not None:
            run_store.append_failure(
                _build_run_payload(
                    inputs=inputs,
                    model_name=model_name,
                    tool_registry=tool_registry,
                    prompt=prompt,
                    raw_response=raw_response,
                    parse_ok=False,
                    retry_count=0,
                    failed_stage=current_stage,
                    parsed_output=None,
                ),
                exc,
            )
        raise

    if run_store is None:
        return result

    artifact = run_store.append_run(
        _build_run_payload(
            inputs=inputs,
            model_name=model_name,
            tool_registry=tool_registry,
            prompt=prompt,
            raw_response=result.raw_response,
            parse_ok=result.parse_ok,
            retry_count=result.retry_count,
            failed_stage=None,
            parsed_output=result.report.to_dict(),
        )
    )
    return replace(result, run_id=artifact.run_id)


def _parse_or_repair(
    *,
    llm_client: LLMClient,
    start_date: date,
    end_date: date,
    raw_response: str,
) -> tuple[bool, TrainingReviewReport]:
    try:
        payload = json.loads(raw_response)
        report = parse_training_review_report(payload)
        return True, report
    except (json.JSONDecodeError, ValueError) as exc:
        repaired = llm_client.generate(_build_repair_prompt(raw_response, exc))
        try:
            payload = json.loads(repaired)
            report = parse_training_review_report(payload)
            return True, report
        except (json.JSONDecodeError, ValueError) as final_exc:
            fallback = build_fallback_training_review_report(
                start_date,
                end_date,
                error_reason=str(final_exc),
            )
            return False, fallback


def _build_prompt(
    *,
    start_date: date,
    end_date: date,
    athlete_id: int | None,
    training_summary: dict[str, Any],
    key_sessions: list[dict[str, Any]],
    evidence_sessions: list[dict[str, Any]],
    missing_data: list[str],
) -> str:
    return (
        "Return only a JSON object that matches the TrainingReviewReport schema.\n"
        "Do not include any extra keys or commentary.\n"
        f"athlete_id: {athlete_id}\n"
        f"start_date: {start_date.isoformat()}\n"
        f"end_date: {end_date.isoformat()}\n"
        f"training_summary: {json.dumps(training_summary, default=_json_default)}\n"
        f"key_sessions: {json.dumps(key_sessions, default=_json_default)}\n"
        f"evidence_sessions: {json.dumps(evidence_sessions, default=_json_default)}\n"
        f"missing_data: {json.dumps(missing_data, default=_json_default)}\n"
    )


def _build_repair_prompt(raw_response: str, error: Exception) -> str:
    return (
        "Fix the following JSON to match the TrainingReviewReport schema.\n"
        "Return only valid JSON with the exact required keys.\n"
        f"Error: {error}\n"
        f"Invalid JSON:\n{raw_response}"
    )


def _maybe_fetch_evidence(
    *,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    start_date: date,
    end_date: date,
    athlete_id: int | None,
    key_sessions: list[dict[str, Any]],
    missing_data: list[str],
) -> list[dict[str, Any]]:
    if not key_sessions:
        return []

    if tool_registry.remaining_budget() <= 0:
        missing_data.append("evidence_tool_budget_exhausted")
        return []

    request_prompt = _build_evidence_request_prompt(
        start_date=start_date,
        end_date=end_date,
        athlete_id=athlete_id,
        key_sessions=key_sessions,
    )
    response = llm_client.generate(request_prompt)

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        missing_data.append("evidence_request_invalid_json")
        return []

    activity_ids = payload.get("activity_ids")
    if not isinstance(activity_ids, list):
        missing_data.append("evidence_request_invalid_schema")
        return []

    requested_ids = _sanitize_activity_ids(activity_ids)
    if not requested_ids:
        return []

    max_fetches = min(2, tool_registry.remaining_budget())
    selected_ids = requested_ids[:max_fetches]
    evidence_sessions: list[dict[str, Any]] = []
    for activity_id in selected_ids:
        result = tool_registry.call_tool("get_activity", {"activity_id": activity_id})
        if result.ok:
            evidence_sessions.extend(result.payload)
        else:
            missing_data.append("evidence_fetch_failed")

    return evidence_sessions


def _build_evidence_request_prompt(
    *,
    start_date: date,
    end_date: date,
    athlete_id: int | None,
    key_sessions: list[dict[str, Any]],
) -> str:
    return (
        "Return a JSON object with an activity_ids list (max 2 integers) that need "
        "extra evidence. Use only activity_ids from key_sessions.\n"
        f"athlete_id: {athlete_id}\n"
        f"start_date: {start_date.isoformat()}\n"
        f"end_date: {end_date.isoformat()}\n"
        f"key_sessions: {json.dumps(key_sessions, default=_json_default)}\n"
    )


def _sanitize_activity_ids(activity_ids: list[Any]) -> list[int]:
    cleaned: list[int] = []
    for value in activity_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value <= 0:
            continue
        cleaned.append(value)

    return cleaned


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return str(value)


def _build_run_payload(
    *,
    inputs: TrainingReviewInputs,
    model_name: str | None,
    tool_registry: ToolRegistry,
    prompt: str,
    raw_response: str,
    parse_ok: bool,
    retry_count: int,
    failed_stage: str | None,
    parsed_output: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "workflow": "training_review",
        "prompt_version": _PROMPT_VERSION,
        "model": model_name,
        "temperature": None,
        "inputs": {
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "athlete_id": inputs.athlete_id,
            "include_key_sessions": inputs.include_key_sessions,
            "max_tool_calls": inputs.max_tool_calls,
        },
        "tool_calls": tool_registry.get_call_log(),
        "prompt": prompt,
        "raw_response": raw_response,
        "parse_ok": parse_ok,
        "retry_count": retry_count,
    }
    if parsed_output is not None:
        payload["parsed_output"] = parsed_output
    if failed_stage is not None:
        payload["failed_stage"] = failed_stage
    return payload

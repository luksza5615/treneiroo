from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from garmin_buddy.ai.contracts.contracts import (
    TrainingReviewReport,
    build_fallback_training_review_report,
    parse_training_review_report,
)
from garmin_buddy.ai.llm_analysis_service import TokenUsageTotals
from garmin_buddy.ai.logging.execution_store import ExecutionStore
from garmin_buddy.ai.tools.training_review_tools import ToolRegistry

TRAINING_REVIEW_MAX_TOOL_CALLS = 3
TRAINING_REVIEW_PROMPT_VERSION = "training_review_v2"
_PROMPT_VERSION = TRAINING_REVIEW_PROMPT_VERSION
_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "training_review"
_TRAINING_REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "positives": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 8,
        },
        "mistakes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 12,
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 12,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "missing_data": {
            "description": "Missing information about training",
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 100,
        },
    },
    "required": [
        "summary",
        "positives",
        "mistakes",
        "recommendations",
        "confidence",
        "missing_data",
    ],
    "additionalProperties": False,
    "propertyOrdering": [
        "summary",
        "positives",
        "mistakes",
        "recommendations",
        "confidence",
        "missing_data",
    ],
}
_EVIDENCE_REQUEST_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "activity_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 0,
            "maxItems": 2,
        },
    },
    "required": ["activity_ids"],
    "additionalProperties": False,
    "propertyOrdering": ["activity_ids"],
}


class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_json_schema: Mapping[str, Any] | None = None,
        usage_tracker: TokenUsageTotals | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class TrainingReviewInputs:
    start_date: date
    end_date: date
    athlete_id: int | None = None
    user_context: str | None = None


@dataclass(frozen=True)
class TrainingReviewResult:
    report: TrainingReviewReport
    raw_response: str
    parse_ok: bool
    retry_count: int
    execution_id: str | None = None


def run_training_review(
    *,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    inputs: TrainingReviewInputs,
    execution_store: ExecutionStore | None = None,
    model_name: str | None = None,
) -> TrainingReviewResult:
    current_stage = "get_training_summary"
    prompt = ""
    raw_response = ""
    token_usage = TokenUsageTotals()
    prompt_config = _load_prompt(_PROMPT_VERSION)
    system_instruction = prompt_config["instructions"]["system"]

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
            missing_input: list[str] = []
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
                missing_input.append("key_sessions_unavailable")

            current_stage = "fetch_evidence"
            evidence_sessions = _fetch_more_training_details(
                llm_client=llm_client,
                tool_registry=tool_registry,
                start_date=inputs.start_date,
                end_date=inputs.end_date,
                athlete_id=inputs.athlete_id,
                key_sessions=key_sessions_payload,
                missing_input=missing_input,
                usage_tracker=token_usage,
            )

            prompt = _build_prompt(
                start_date=inputs.start_date,
                end_date=inputs.end_date,
                athlete_id=inputs.athlete_id,
                training_summary=summary_result.payload,
                key_sessions=key_sessions_payload,
                evidence_sessions=evidence_sessions,
                missing_input=missing_input,
                user_context=inputs.user_context,
            )

            current_stage = "generate_report"
            raw_response = llm_client.generate(
                prompt,
                system_instruction=system_instruction,
                response_json_schema=_TRAINING_REVIEW_RESPONSE_SCHEMA,
                usage_tracker=token_usage,
            )

            current_stage = "parse_or_repair"
            parse_ok, report = _parse_or_repair(
                llm_client=llm_client,
                start_date=inputs.start_date,
                end_date=inputs.end_date,
                raw_response=raw_response,
                usage_tracker=token_usage,
            )

            result = TrainingReviewResult(
                report=report,
                raw_response=raw_response,
                parse_ok=parse_ok,
                retry_count=0 if parse_ok else 1,
            )
    except Exception as exc:
        if execution_store is not None:
            execution_store.append_failure(
                _build_execution_payload(
                    inputs=inputs,
                    model_name=model_name,
                    tool_registry=tool_registry,
                    prompt=prompt,
                    raw_response=raw_response,
                    parse_ok=False,
                    retry_count=0,
                    failed_stage=current_stage,
                    parsed_output=None,
                    token_usage=token_usage,
                ),
                exc,
            )
        raise

    if execution_store is None:
        return result

    artifact = execution_store.append_execution(
        _build_execution_payload(
            inputs=inputs,
            model_name=model_name,
            tool_registry=tool_registry,
            prompt=prompt,
            raw_response=result.raw_response,
            parse_ok=result.parse_ok,
            retry_count=result.retry_count,
            failed_stage=None,
            parsed_output=result.report.to_dict(),
            token_usage=token_usage,
        )
    )
    return replace(result, execution_id=artifact.execution_id)


def _parse_or_repair(
    *,
    llm_client: LLMClient,
    start_date: date,
    end_date: date,
    raw_response: str,
    usage_tracker: TokenUsageTotals,
) -> tuple[bool, TrainingReviewReport]:
    try:
        payload = json.loads(raw_response)
        report = parse_training_review_report(payload)
        return True, report
    except (json.JSONDecodeError, ValueError) as exc:
        repaired = llm_client.generate(
            _build_repair_prompt(raw_response, exc),
            system_instruction="Return valid JSON only.",
            response_json_schema=_TRAINING_REVIEW_RESPONSE_SCHEMA,
            usage_tracker=usage_tracker,
        )
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
    missing_input: list[str],
    user_context: str | None,
) -> str:
    prompt = _load_prompt(_PROMPT_VERSION)

    return prompt["user_template"].format(
        athlete_id=athlete_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        training_summary_json=_json(training_summary),
        key_sessions_json=_json(key_sessions),
        key_session_details_json=_json(evidence_sessions),
        user_context=_format_user_context(user_context),
        missing_input_json=_json(missing_input),
    )


def _format_user_context(user_context: str | None) -> str:
    if user_context is None:
        return "No user context provided."

    context = user_context.strip()
    return context or "No user context provided."


def _has_user_context(user_context: str | None) -> bool:
    return _format_user_context(user_context) != "No user context provided."


def _build_repair_prompt(raw_response: str, error: Exception) -> str:
    return (
        "Fix the following JSON to match the TrainingReviewReport schema.\n"
        "Return only valid JSON with these exact required keys: "
        "summary, positives, mistakes, "
        "recommendations, confidence, missing_data.\n"
        f"Error: {error}\n"
        f"Invalid JSON:\n{raw_response}"
    )


def _fetch_more_training_details(
    *,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    start_date: date,
    end_date: date,
    athlete_id: int | None,
    key_sessions: list[dict[str, Any]],
    missing_input: list[str],
    usage_tracker: TokenUsageTotals,
) -> list[dict[str, Any]]:
    if not key_sessions:
        return []

    if tool_registry.remaining_budget() <= 0:
        return []

    request_prompt = _build_evidence_request_prompt(
        start_date=start_date,
        end_date=end_date,
        athlete_id=athlete_id,
        key_sessions=key_sessions,
    )
    response = llm_client.generate(
        request_prompt,
        response_json_schema=_EVIDENCE_REQUEST_RESPONSE_SCHEMA,
        usage_tracker=usage_tracker,
    )

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return []

    activity_ids = payload.get("activity_ids")
    if not isinstance(activity_ids, list):
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
            missing_input.append("key_session_details_unavailable")

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


@lru_cache(maxsize=8)
def _load_prompt(version: str) -> dict[str, Any]:
    path = _PROMPT_DIR / f"{version}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return str(value)


def _build_execution_payload(
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
    token_usage: TokenUsageTotals,
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
            "user_context_included": _has_user_context(inputs.user_context),
            "key_sessions_included": True,
            "tool_budget": tool_registry.max_tool_calls,
        },
        "tool_calls": tool_registry.get_call_log(),
        "prompt": prompt,
        "raw_response": raw_response,
        "parse_ok": parse_ok,
        "retry_count": retry_count,
        "total_input_tokens": token_usage.total_input_tokens,
        "total_output_tokens": token_usage.total_output_tokens,
    }
    if parsed_output is not None:
        payload["parsed_output"] = parsed_output
    if failed_stage is not None:
        payload["failed_stage"] = failed_stage
    return payload

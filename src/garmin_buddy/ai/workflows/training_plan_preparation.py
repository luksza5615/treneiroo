from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TypeVar
from uuid import uuid4

import yaml

from garmin_buddy.ai.logging.preparation_run_store import PreparationRunStore
from garmin_buddy.ai.preparation_contracts import (
    CritiqueArtifact,
    LabAnalysisArtifact,
    MacroStrategyArtifact,
    NormalizedPreparationContext,
    PastPhaseReviewArtifact,
    PhasePlanArtifact,
    PreparationResult,
    PreparationSynthesisArtifact,
    RunnerProfileArtifact,
    StrengthPlanArtifact,
    build_fallback_critique,
    build_fallback_lab_analysis,
    build_fallback_macro_strategy,
    build_fallback_past_phase_review,
    build_fallback_phase_plan,
    build_fallback_strength_plan,
    build_fallback_synthesis,
)
from garmin_buddy.ai.tools.training_plan_preparation_tools import PreparationToolRegistry

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "preparation"
T = TypeVar("T")


class LLMClient(Protocol):
    def generate(self, prompt: str, *, system_instruction: str | None = None) -> str: ...


@dataclass(frozen=True)
class TrainingPlanPreparationInputs:
    start_date: date
    end_date: date
    athlete_id: int | None = None
    planning_horizon_months: int = 4
    first_phase_weeks: int = 4


def run_training_plan_preparation(
    *,
    llm_client: LLMClient,
    tool_registry: PreparationToolRegistry,
    run_store: PreparationRunStore,
    inputs: TrainingPlanPreparationInputs,
) -> PreparationResult:
    """Generate the macro strategy first because it is the highest-value approval boundary."""
    current_stage = "collect_context"
    context: NormalizedPreparationContext | None = None
    strategy_id: str | None = None

    try:
        context, context_debug = _collect_context(
            tool_registry=tool_registry, inputs=inputs
        )
        current_stage = "lab_analysis"
        lab_analysis, lab_parse_ok, lab_retries = _resolve_lab_analysis(
            llm_client=llm_client,
            run_store=run_store,
            context=context,
        )
        current_stage = "past_phase_review"
        past_phase_review, phase_parse_ok, phase_retries = _run_structured_stage(
            stage_name="past_phase_review_v1",
            llm_client=llm_client,
            format_kwargs=context_debug,
            parser=PastPhaseReviewArtifact.from_payload,
            fallback_builder=build_fallback_past_phase_review,
        )
        current_stage = "synthesis"
        synthesis, synthesis_parse_ok, synthesis_retries = _run_structured_stage(
            stage_name="synthesis_v1",
            llm_client=llm_client,
            format_kwargs={
                "profile_json": _json(context.profile.to_dict()),
                "lab_analysis_json": _json(lab_analysis.to_dict()),
                "past_phase_review_json": _json(past_phase_review.to_dict()),
            },
            parser=PreparationSynthesisArtifact.from_payload,
            fallback_builder=build_fallback_synthesis,
        )
        current_stage = "strength_recommendation"
        strength_plan, strength_parse_ok, strength_retries = _run_structured_stage(
            stage_name="strength_recommendation_v1",
            llm_client=llm_client,
            format_kwargs={
                "profile_json": _json(context.profile.to_dict()),
                "synthesis_json": _json(synthesis.to_dict()),
            },
            parser=StrengthPlanArtifact.from_payload,
            fallback_builder=build_fallback_strength_plan,
        )

        strategy_id = str(uuid4())
        current_stage = "macro_strategy"
        strategy, strategy_parse_ok, strategy_retries = _run_strategy_stage(
            llm_client=llm_client,
            strategy_id=strategy_id,
            input_hash=context.input_hash,
            context=context,
            synthesis=synthesis,
            strength_plan=strength_plan,
            planning_horizon_months=inputs.planning_horizon_months,
        )

        current_stage = "save_strategy_state"
        run_store.save_strategy_state(
            strategy_id,
            {
                "strategy_id": strategy_id,
                "approved": False,
                "stale": False,
                "lab_fingerprint": context.lab_fingerprint,
                "context": context.to_dict(),
                "lab_analysis": lab_analysis.to_dict(),
                "past_phase_review": past_phase_review.to_dict(),
                "synthesis": synthesis.to_dict(),
                "strength_plan": strength_plan.to_dict(),
                "strategy": strategy.to_dict(),
                "phase_plan": None,
                "critique": None,
            },
        )

        parse_ok = all(
            [
                lab_parse_ok,
                phase_parse_ok,
                synthesis_parse_ok,
                strength_parse_ok,
                strategy_parse_ok,
            ]
        )
        retry_count = (
            lab_retries
            + phase_retries
            + synthesis_retries
            + strength_retries
            + strategy_retries
        )

        result = PreparationResult(
            context=context,
            lab_analysis=lab_analysis,
            past_phase_review=past_phase_review,
            synthesis=synthesis,
            strength_plan=strength_plan,
            strategy=strategy,
            phase_plan=None,
            critique=None,
            parse_ok=parse_ok,
            retry_count=retry_count,
        )
        current_stage = "append_run"
        run_store.append_run(
            {
                "workflow": "training_plan_preparation",
                "stage": "strategy_generation",
                "input_hash": context.input_hash,
                "tool_calls": tool_registry.get_call_log(),
                "parse_ok": parse_ok,
                "retry_count": retry_count,
                "result": result.to_dict(),
            }
        )
        return result
    except Exception as exc:
        run_store.append_failure(
            _build_preparation_failure_payload(
                workflow_stage="strategy_generation",
                failed_stage=current_stage,
                inputs=inputs,
                tool_registry=tool_registry,
                context=context,
                strategy_id=strategy_id,
            ),
            exc,
        )
        raise


def approve_training_plan_strategy(
    *, run_store: PreparationRunStore, strategy_id: str
) -> dict[str, Any]:
    state = run_store.load_strategy_state(strategy_id)
    if state is None:
        raise ValueError(f"Strategy '{strategy_id}' was not found.")

    strategy = dict(state["strategy"])
    strategy["approval_status"] = "approved"
    state["strategy"] = strategy
    state["approved"] = True
    run_store.save_strategy_state(strategy_id, state)
    return state


def generate_phase_plan_from_strategy(
    *,
    llm_client: LLMClient,
    tool_registry: PreparationToolRegistry,
    run_store: PreparationRunStore,
    inputs: TrainingPlanPreparationInputs,
    strategy_id: str,
) -> PreparationResult:
    """Resume from an approved strategy so detailed planning never bypasses review."""
    current_stage = "load_strategy_state"
    current_context: NormalizedPreparationContext | None = None

    try:
        state = run_store.load_strategy_state(strategy_id)
        if state is None:
            raise ValueError(f"Strategy '{strategy_id}' was not found.")
        if state.get("approved") is not True:
            raise ValueError(
                "Strategy must be approved before generating the detailed phase."
            )

        current_stage = "collect_context"
        current_context, _ = _collect_context(tool_registry=tool_registry, inputs=inputs)
        stored_strategy = dict(state["strategy"])
        if current_context.input_hash != stored_strategy["input_hash"]:
            stored_strategy["approval_status"] = "stale"
            state["strategy"] = stored_strategy
            state["stale"] = True
            state["stale_reason"] = "upstream_input_hash_changed"
            run_store.save_strategy_state(strategy_id, state)

            stale_result = PreparationResult(
                context=current_context,
                lab_analysis=LabAnalysisArtifact.from_payload(state["lab_analysis"]),
                past_phase_review=PastPhaseReviewArtifact.from_payload(
                    state["past_phase_review"]
                ),
                synthesis=PreparationSynthesisArtifact.from_payload(state["synthesis"]),
                strength_plan=StrengthPlanArtifact.from_payload(state["strength_plan"]),
                strategy=MacroStrategyArtifact.from_payload(stored_strategy),
                phase_plan=None,
                critique=None,
                parse_ok=False,
                retry_count=0,
                strategy_stale=True,
            )
            current_stage = "append_run"
            run_store.append_run(
                {
                    "workflow": "training_plan_preparation",
                    "stage": "phase_generation",
                    "strategy_id": strategy_id,
                    "input_hash": current_context.input_hash,
                    "tool_calls": tool_registry.get_call_log(),
                    "parse_ok": False,
                    "retry_count": 0,
                    "result": stale_result.to_dict(),
                }
            )
            return stale_result

        synthesis = PreparationSynthesisArtifact.from_payload(state["synthesis"])
        strength_plan = StrengthPlanArtifact.from_payload(state["strength_plan"])
        strategy = MacroStrategyArtifact.from_payload(stored_strategy)

        current_stage = "phase_plan"
        phase_plan, phase_parse_ok, phase_retries = _run_phase_plan_stage(
            llm_client=llm_client,
            strategy=strategy,
            synthesis=synthesis,
            strength_plan=strength_plan,
            first_phase_weeks=inputs.first_phase_weeks,
            revision_adjustments=[],
        )
        current_stage = "critique"
        critique, critique_parse_ok, critique_retries = _run_structured_stage(
            stage_name="critique_v1",
            llm_client=llm_client,
            format_kwargs={
                "profile_json": _json(current_context.profile.to_dict()),
                "strategy_json": _json(strategy.to_dict()),
                "phase_plan_json": _json(phase_plan.to_dict()),
            },
            parser=CritiqueArtifact.from_payload,
            fallback_builder=build_fallback_critique,
        )

        if critique.decision == "revise" and critique.required_adjustments:
            current_stage = "phase_plan_revision"
            phase_plan, revised_parse_ok, revised_retries = _run_phase_plan_stage(
                llm_client=llm_client,
                strategy=strategy,
                synthesis=synthesis,
                strength_plan=strength_plan,
                first_phase_weeks=inputs.first_phase_weeks,
                revision_adjustments=critique.required_adjustments,
            )
            phase_parse_ok = phase_parse_ok and revised_parse_ok
            phase_retries += revised_retries
            current_stage = "critique_revision"
            critique, critique_parse_ok, second_critique_retries = _run_structured_stage(
                stage_name="critique_v1",
                llm_client=llm_client,
                format_kwargs={
                    "profile_json": _json(current_context.profile.to_dict()),
                    "strategy_json": _json(strategy.to_dict()),
                    "phase_plan_json": _json(phase_plan.to_dict()),
                },
                parser=CritiqueArtifact.from_payload,
                fallback_builder=build_fallback_critique,
            )
            critique_retries += second_critique_retries

        parse_ok = phase_parse_ok and critique_parse_ok
        retry_count = phase_retries + critique_retries

        state["phase_plan"] = phase_plan.to_dict()
        state["critique"] = critique.to_dict()
        current_stage = "save_strategy_state"
        run_store.save_strategy_state(strategy_id, state)

        result = PreparationResult(
            context=current_context,
            lab_analysis=LabAnalysisArtifact.from_payload(state["lab_analysis"]),
            past_phase_review=PastPhaseReviewArtifact.from_payload(
                state["past_phase_review"]
            ),
            synthesis=synthesis,
            strength_plan=strength_plan,
            strategy=strategy,
            phase_plan=phase_plan,
            critique=critique,
            parse_ok=parse_ok,
            retry_count=retry_count,
        )
        current_stage = "append_run"
        run_store.append_run(
            {
                "workflow": "training_plan_preparation",
                "stage": "phase_generation",
                "strategy_id": strategy_id,
                "tool_calls": tool_registry.get_call_log(),
                "parse_ok": parse_ok,
                "retry_count": retry_count,
                "result": result.to_dict(),
            }
        )
        return result
    except Exception as exc:
        run_store.append_failure(
            _build_preparation_failure_payload(
                workflow_stage="phase_generation",
                failed_stage=current_stage,
                inputs=inputs,
                tool_registry=tool_registry,
                context=current_context,
                strategy_id=strategy_id,
            ),
            exc,
        )
        raise


def _collect_context(
    *,
    tool_registry: PreparationToolRegistry,
    inputs: TrainingPlanPreparationInputs,
) -> tuple[NormalizedPreparationContext, dict[str, str]]:
    """Centralize source collection so hashing and staleness checks stay consistent."""

    missing_data: list[str] = []

    profile_result = tool_registry.call_tool("get_runner_profile", {})
    if profile_result.ok:
        profile = RunnerProfileArtifact.from_payload(profile_result.payload)
    else:
        missing_data.append(profile_result.error or "runner_profile_unavailable")
        profile = RunnerProfileArtifact.from_payload(
            {
                "profile_context": (
                    "Maintain consistent training progression. "
                    "Profile loader did not provide richer planning context."
                ),
            }
        )

    lab_result = tool_registry.call_tool("get_latest_lab_payload", {})
    lab_payload = dict(lab_result.payload or {}) if lab_result.ok else {}
    if not lab_result.ok:
        missing_data.append(lab_result.error or "lab_payload_unavailable")

    training_log_result = tool_registry.call_tool(
        "get_training_log_rows",
        {
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "athlete_id": inputs.athlete_id,
        },
    )
    planned_rows = list(training_log_result.payload or []) if training_log_result.ok else []
    if not training_log_result.ok:
        missing_data.append(training_log_result.error or "training_log_unavailable")

    execution_summary_result = tool_registry.call_tool(
        "get_execution_summary",
        {
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "athlete_id": inputs.athlete_id,
        },
    )
    executed_summary = (
        dict(execution_summary_result.payload or {})
        if execution_summary_result.ok
        else {
            "executed_sessions": 0,
            "distance_km": 0.0,
            "avg_hr": None,
            "ascent_m": 0.0,
            "avg_aerobic_te": None,
            "avg_anaerobic_te": None,
            "hard_sessions": 0,
        }
    )
    if not execution_summary_result.ok:
        missing_data.append(
            execution_summary_result.error or "execution_summary_unavailable"
        )

    comparison_result = tool_registry.call_tool(
        "compare_planned_vs_executed",
        {
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "athlete_id": inputs.athlete_id,
        },
    )
    comparison = dict(comparison_result.payload or {}) if comparison_result.ok else {}
    if not comparison_result.ok:
        missing_data.append(
            comparison_result.error or "planned_vs_executed_comparison_unavailable"
        )

    executed_key_sessions_result = tool_registry.call_tool(
        "list_executed_key_sessions",
        {
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "athlete_id": inputs.athlete_id,
            "n": 5,
        },
    )
    executed_key_sessions = (
        list(executed_key_sessions_result.payload or [])
        if executed_key_sessions_result.ok
        else []
    )
    if not executed_key_sessions_result.ok:
        missing_data.append(
            executed_key_sessions_result.error or "executed_key_sessions_unavailable"
        )

    planned_summary = _build_planned_training_summary(planned_rows, comparison)
    context_payload = {
        "profile": profile.to_dict(),
        "lab_summary": lab_payload.get("lab_summary", "No lab summary available."),
        "lab_markers": lab_payload.get("lab_markers", {}),
        "planned_training_summary": planned_summary,
        "executed_training_summary": executed_summary,
        "source_provenance": [
            "runner_profile",
            "lab_payload",
            "google_sheets_training_log",
            "azure_sql_execution_data",
        ],
        "missing_data": sorted(set(missing_data)),
        "input_hash": _compute_input_hash(
            profile=profile.to_dict(),
            lab_summary=lab_payload.get("lab_summary", "No lab summary available."),
            lab_markers=lab_payload.get("lab_markers", {}),
            planned_training_summary=planned_summary,
            executed_training_summary=executed_summary,
        ),
        "lab_fingerprint": lab_payload.get("lab_fingerprint"),
    }
    context = NormalizedPreparationContext.from_payload(context_payload)
    debug_payload = {
        "lab_summary": context.lab_summary,
        "lab_markers_json": _json(context.lab_markers),
        "planned_training_summary_json": _json(context.planned_training_summary),
        "executed_training_summary_json": _json(context.executed_training_summary),
        "comparison_json": _json(comparison),
        "executed_key_sessions_json": _json(executed_key_sessions),
    }
    return context, debug_payload


def _build_planned_training_summary(
    planned_rows: list[dict[str, Any]], comparison: dict[str, Any]
) -> dict[str, Any]:
    session_types: dict[str, int] = {}
    for row in planned_rows:
        session_type = str(row.get("session_type") or "unknown").strip() or "unknown"
        session_types[session_type] = session_types.get(session_type, 0) + 1
    return {
        "planned_sessions": len(planned_rows),
        "session_types": session_types,
        "comparison": comparison,
    }


def _resolve_lab_analysis(
    *,
    llm_client: LLMClient,
    run_store: PreparationRunStore,
    context: NormalizedPreparationContext,
) -> tuple[LabAnalysisArtifact, bool, int]:
    cached_payload = run_store.find_lab_analysis(context.lab_fingerprint)
    if cached_payload is not None:
        return LabAnalysisArtifact.from_payload(cached_payload), True, 0
    return _run_structured_stage(
        stage_name="lab_analysis_v1",
        llm_client=llm_client,
        format_kwargs={
            "lab_summary": context.lab_summary,
            "lab_markers_json": _json(context.lab_markers),
        },
        parser=LabAnalysisArtifact.from_payload,
        fallback_builder=build_fallback_lab_analysis,
    )


def _run_strategy_stage(
    *,
    llm_client: LLMClient,
    strategy_id: str,
    input_hash: str,
    context: NormalizedPreparationContext,
    synthesis: PreparationSynthesisArtifact,
    strength_plan: StrengthPlanArtifact,
    planning_horizon_months: int,
) -> tuple[MacroStrategyArtifact, bool, int]:
    stage_result, parse_ok, retry_count = _run_structured_stage(
        stage_name="macro_strategy_v1",
        llm_client=llm_client,
        format_kwargs={
            "profile_json": _json(context.profile.to_dict()),
            "synthesis_json": _json(synthesis.to_dict()),
            "strength_plan_json": _json(strength_plan.to_dict()),
            "planning_horizon_months": planning_horizon_months,
        },
        parser=lambda payload: MacroStrategyArtifact.from_payload(
            {
                **payload,
                "strategy_id": strategy_id,
                "approval_status": "pending",
                "input_hash": input_hash,
            }
        ),
        fallback_builder=lambda error_reason: build_fallback_macro_strategy(
            strategy_id=strategy_id,
            input_hash=input_hash,
            error_reason=error_reason,
        ),
    )
    return stage_result, parse_ok, retry_count


def _run_phase_plan_stage(
    *,
    llm_client: LLMClient,
    strategy: MacroStrategyArtifact,
    synthesis: PreparationSynthesisArtifact,
    strength_plan: StrengthPlanArtifact,
    first_phase_weeks: int,
    revision_adjustments: list[str],
) -> tuple[PhasePlanArtifact, bool, int]:
    return _run_structured_stage(
        stage_name="phase_plan_v1",
        llm_client=llm_client,
        format_kwargs={
            "strategy_json": _json(strategy.to_dict()),
            "synthesis_json": _json(synthesis.to_dict()),
            "strength_plan_json": _json(strength_plan.to_dict()),
            "first_phase_weeks": first_phase_weeks,
            "revision_adjustments_json": _json(revision_adjustments),
        },
        parser=PhasePlanArtifact.from_payload,
        fallback_builder=lambda error_reason: build_fallback_phase_plan(
            weeks=first_phase_weeks,
            error_reason=error_reason,
        ),
    )


def _run_structured_stage(
    *,
    stage_name: str,
    llm_client: LLMClient,
    format_kwargs: dict[str, Any],
    parser: Callable[[Mapping[str, Any]], T],
    fallback_builder: Callable[[str | None], T],
) -> tuple[T, bool, int]:
    """One repair attempt is enough to recover formatting without hiding unstable prompts."""

    prompt = _load_prompt(stage_name)
    system_instruction = prompt["instructions"]["system"]
    user_prompt = prompt["user_template"].format(**format_kwargs)
    raw_response = llm_client.generate(user_prompt, system_instruction=system_instruction)

    try:
        payload = json.loads(raw_response)
        return parser(payload), True, 0
    except (json.JSONDecodeError, ValueError) as exc:
        repair_prompt = (
            "Fix the following JSON so it matches the requested schema. "
            "Return only valid JSON.\n"
            f"Error: {exc}\n"
            f"Invalid JSON:\n{raw_response}"
        )
        repaired = llm_client.generate(
            repair_prompt, system_instruction="Return valid JSON only."
        )
        try:
            payload = json.loads(repaired)
            return parser(payload), True, 1
        except (json.JSONDecodeError, ValueError) as final_exc:
            return fallback_builder(str(final_exc)), False, 1


@lru_cache(maxsize=16)
def _load_prompt(stage_name: str) -> dict[str, Any]:
    path = _PROMPT_DIR / f"{stage_name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _compute_input_hash(**payload: Any) -> str:
    """Use one canonical hash so stale approval detection is deterministic."""

    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


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


def _build_preparation_failure_payload(
    *,
    workflow_stage: str,
    failed_stage: str,
    inputs: TrainingPlanPreparationInputs,
    tool_registry: PreparationToolRegistry,
    context: NormalizedPreparationContext | None,
    strategy_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "workflow": "training_plan_preparation",
        "stage": workflow_stage,
        "failed_stage": failed_stage,
        "tool_calls": tool_registry.get_call_log(),
        "inputs": {
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "athlete_id": inputs.athlete_id,
            "planning_horizon_months": inputs.planning_horizon_months,
            "first_phase_weeks": inputs.first_phase_weeks,
        },
    }
    if context is not None:
        payload["input_hash"] = context.input_hash
    if strategy_id is not None:
        payload["strategy_id"] = strategy_id
    return payload

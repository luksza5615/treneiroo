from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from garmin_buddy.ai.logging.preparation_run_store import PreparationRunStore
from garmin_buddy.ai.tools.training_plan_preparation_tools import (
    PreparationToolRegistry,
)
from garmin_buddy.ai.workflows.training_plan_preparation import (
    TrainingPlanPreparationInputs,
    approve_training_plan_strategy,
    generate_phase_plan_from_strategy,
    run_training_plan_preparation,
)


class _DummyRepository:
    def get_execution_summary(self, start_date, end_date, *, athlete_id=None):
        return {"executed_sessions": 4, "distance_km": 42.0, "hard_sessions": 1}

    def list_key_sessions(self, start_date, end_date, *, athlete_id=None, n=5):
        return pd.DataFrame([{"activity_id": 123, "activity_date": date(2026, 2, 1)}])

    def compare_planned_vs_executed(
        self, planned_sessions, start_date, end_date, *, athlete_id=None
    ):
        return {"planned_sessions": len(planned_sessions), "adherence_rate": 0.75}


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def generate(self, prompt: str, *, system_instruction: str | None = None) -> str:
        self.calls.append(prompt)
        return self.responses.pop(0)


def _profile_payload(goals: str = "Sub-40 10k") -> dict[str, object]:
    return {
        "athlete_name": "Runner",
        "goals": [goals],
        "availability": [],
        "constraints": [],
        "preferences": [],
        "injury_notes": [],
        "source_notes": [],
    }


def _lab_payload(text: str = "Ferritin 28") -> dict[str, object]:
    return {
        "lab_summary": text,
        "lab_markers": {"ferritin": 28.0},
        "lab_fingerprint": text,
    }


def _training_log_loader(start_date, end_date):
    return [
        {
            "date": "2026-02-01",
            "session_type": "easy",
            "planned_focus": "aerobic",
            "planned_load": "low",
            "notes": "",
        }
    ]


def test_run_training_plan_preparation_generates_strategy() -> None:
    llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ]
    )
    store = PreparationRunStore(_workspace_temp_dir("prep-workflow"))
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )

    result = run_training_plan_preparation(
        llm_client=llm,
        tool_registry=registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )

    assert result.strategy.approval_status == "pending"
    assert result.phase_plan is None
    assert result.parse_ok is True


def test_generate_phase_plan_marks_strategy_stale_when_inputs_change() -> None:
    llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ]
    )
    store = PreparationRunStore(_workspace_temp_dir("prep-workflow"))
    initial_registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )
    initial_result = run_training_plan_preparation(
        llm_client=llm,
        tool_registry=initial_registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )
    approve_training_plan_strategy(
        run_store=store, strategy_id=initial_result.strategy.strategy_id
    )

    phase_registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(goals="Different goal"),
        lab_loader=lambda: _lab_payload(),
    )
    phase_result = generate_phase_plan_from_strategy(
        llm_client=_FakeLLM([]),
        tool_registry=phase_registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
        strategy_id=initial_result.strategy.strategy_id,
    )

    assert phase_result.strategy_stale is True
    assert phase_result.strategy.approval_status == "stale"


def test_run_training_plan_preparation_serializes_date_context_values() -> None:
    llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ]
    )
    store = PreparationRunStore(_workspace_temp_dir("prep-workflow"))
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: {
            "lab_summary": "Ferritin 28",
            "lab_markers": {"ferritin": 28.0, "sample_date": date(2026, 1, 30)},
            "lab_fingerprint": "Ferritin 28",
        },
    )

    result = run_training_plan_preparation(
        llm_client=llm,
        tool_registry=registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )

    assert result.parse_ok is True
    assert result.context.lab_markers["sample_date"] == date(2026, 1, 30)


def _workspace_temp_dir(name: str) -> Path:
    path = Path("zignored") / "pytest-temp" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path

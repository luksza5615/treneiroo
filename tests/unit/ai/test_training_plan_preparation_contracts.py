from __future__ import annotations

import pytest

from garmin_buddy.ai.preparation_contracts import (
    MacroStrategyArtifact,
    PhasePlanArtifact,
    RunnerProfileArtifact,
)


def test_runner_profile_artifact_requires_at_least_one_goal() -> None:
    with pytest.raises(ValueError, match="goals must contain between 1 and 50 items"):
        RunnerProfileArtifact.from_payload(
            {
                "athlete_name": None,
                "goals": [],
                "availability": [],
                "constraints": [],
                "preferences": [],
                "injury_notes": [],
                "source_notes": [],
            }
        )


def test_macro_strategy_artifact_requires_valid_approval_status() -> None:
    with pytest.raises(ValueError, match="approval_status"):
        MacroStrategyArtifact.from_payload(
            {
                "strategy_id": "id",
                "planning_horizon": "4 months",
                "strategic_goal": "Goal",
                "mesocycles": ["Base"],
                "progression_logic": ["Progress"],
                "recovery_logic": ["Recover"],
                "risks": [],
                "assumptions": [],
                "approval_status": "invalid",
                "input_hash": "hash",
                "missing_data": [],
                "confidence": 0.5,
            }
        )


def test_phase_plan_artifact_requires_positive_week_count() -> None:
    with pytest.raises(ValueError, match="phase_length_weeks"):
        PhasePlanArtifact.from_payload(
            {
                "phase_length_weeks": 0,
                "weekly_goals": ["Goal"],
                "session_plan": ["Run"],
                "strength_integration": [],
                "rationale_links": ["Reason"],
                "risks": [],
                "missing_data": [],
                "confidence": 0.5,
            }
        )

from __future__ import annotations

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
)
from garmin_buddy.ai.rendering.preparation_renderer import render_preparation_md


def test_render_preparation_md_includes_strategy_and_phase_sections() -> None:
    result = PreparationResult(
        context=NormalizedPreparationContext(
            profile=RunnerProfileArtifact(
                profile_context=(
                    "Target event: 10k on 2026-05-01\n"
                    "Goal: run a strong 10k\n"
                    "Availability: Tue and Thu"
                ),
            ),
            lab_summary="Ferritin slightly low.",
            lab_markers={"ferritin": 28.0},
            planned_training_summary={},
            executed_training_summary={},
            source_provenance=["profile"],
            missing_data=[],
            input_hash="hash",
            lab_fingerprint="fingerprint",
        ),
        lab_analysis=LabAnalysisArtifact(
            summary="Lab summary",
            findings=["Ferritin is a watch item."],
            training_implications=["Avoid aggressive load spikes."],
            risk_flags=[],
        ),
        past_phase_review=PastPhaseReviewArtifact(
            summary="Phase summary",
            adherence_summary="Good adherence.",
            positive_patterns=["Consistent easy volume."],
            execution_issues=["One missed long run."],
            risk_flags=[],
        ),
        synthesis=PreparationSynthesisArtifact(
            summary="Synthesis",
            key_constraints=["Ferritin trend"],
            key_opportunities=["Aerobic consistency"],
            planning_priorities=["Build stable volume"],
            risk_controls=[],
            assumptions=[],
        ),
        strength_plan=StrengthPlanArtifact(
            objectives=["General durability"],
            weekly_frequency="2x/week",
            session_focuses=["Single-leg work"],
            integration_notes=[],
            contraindications=[],
        ),
        strategy=MacroStrategyArtifact(
            strategy_id="strategy-1",
            planning_horizon="4 months",
            strategic_goal="Build toward a 10k peak.",
            mesocycles=["Base", "Specific"],
            progression_logic=["Add work gradually"],
            recovery_logic=["Deload every 4th week"],
            risks=[],
            assumptions=[],
            approval_status="approved",
            input_hash="hash",
        ),
        phase_plan=PhasePlanArtifact(
            phase_length_weeks=4,
            weekly_goals=["Week 1 stabilize"],
            session_plan=["Tue easy", "Thu workout"],
            strength_integration=["Mon strength"],
            rationale_links=["Matches base mesocycle"],
            risks=[],
        ),
        critique=CritiqueArtifact(
            decision="accept",
            blocking_issues=[],
            non_blocking_improvements=["Slightly extend easy warmup"],
            required_adjustments=[],
        ),
        parse_ok=True,
        retry_count=0,
    )

    markdown = render_preparation_md(result)

    assert "# Strategy: Build toward a 10k peak." in markdown
    assert "## Profile Context" in markdown
    assert "## Weekly Goals" in markdown
    assert "## Critique Blocking Issues" in markdown

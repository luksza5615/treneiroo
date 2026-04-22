from __future__ import annotations

from garmin_buddy.ai.contracts.preparation_contracts import PreparationResult


def render_preparation_md(result: PreparationResult) -> str:
    sections = [
        f"# Strategy: {result.strategy.strategic_goal}",
        _render_paragraph_section(
            "Profile Context", result.context.profile.profile_context
        ),
        _render_paragraph_section("Lab Summary", result.context.lab_summary),
        _render_list_section("Lab Findings", result.lab_analysis.findings),
        _render_list_section(
            "Past Phase Positives", result.past_phase_review.positive_patterns
        ),
        _render_list_section(
            "Past Phase Issues", result.past_phase_review.execution_issues
        ),
        _render_list_section(
            "Planning Priorities", result.synthesis.planning_priorities
        ),
        _render_list_section("Strength Objectives", result.strength_plan.objectives),
        _render_list_section("Mesocycles", result.strategy.mesocycles),
        f"**Strategy approval:** {result.strategy.approval_status}",
    ]

    if result.phase_plan is not None:
        sections.append(
            _render_list_section("Weekly Goals", result.phase_plan.weekly_goals)
        )
        sections.append(
            _render_list_section("Session Plan", result.phase_plan.session_plan)
        )
    if result.critique is not None:
        sections.append(
            _render_list_section(
                "Critique Blocking Issues", result.critique.blocking_issues
            )
        )
        sections.append(
            _render_list_section(
                "Critique Adjustments", result.critique.required_adjustments
            )
        )

    sections.append(_render_list_section("Missing Data", result.context.missing_data))
    return "\n\n".join(sections)


def _render_paragraph_section(title: str, value: str) -> str:
    return f"## {title}\n{value}"


def _render_list_section(title: str, items: list[str]) -> str:
    if not items:
        return f"## {title}\n- None"
    bullets = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n{bullets}"

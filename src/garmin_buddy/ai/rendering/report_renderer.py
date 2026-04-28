from __future__ import annotations

from garmin_buddy.ai.contracts.contracts import TrainingReviewReport


def render_report_md(report: TrainingReviewReport) -> str:
    sections: list[str] = []
    sections.append("# Training Review")
    sections.append(f"## Executive summary\n{report.executive_summary}")
    sections.append(_render_list_section("Positives", report.positives))
    sections.append(_render_list_section("Mistakes", report.mistakes))
    sections.append(
        _render_list_section(
            "Main lessons and recommendations",
            report.main_lessons_and_recommendations,
        )
    )
    sections.append(_render_list_section("Evidence", report.evidence))
    sections.append(f"**Confidence:** {report.confidence:.2f}")
    sections.append(_render_list_section("Missing data", report.missing_data))

    return "\n\n".join(sections)


def _render_list_section(title: str, items: list[str]) -> str:
    if not items:
        return f"## {title}\n- None"

    bullets = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n{bullets}"

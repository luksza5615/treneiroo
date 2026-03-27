from __future__ import annotations

from garmin_buddy.ai.contracts import TrainingReviewReport


def render_report_md(report: TrainingReviewReport) -> str:
    sections: list[str] = []
    sections.append(f"# {report.headline}")
    sections.append(_render_list_section("Positives", report.positives))
    sections.append(_render_list_section("Risks", report.risks))
    sections.append(
        _render_list_section(
            "Priorities (next 7 days)", report.priorities_next_7_days
        )
    )
    sections.append(_render_list_section("Evidence", report.evidence))
    sections.append(f"**Confidence:** {report.confidence:.2f}")
    sections.append(_render_list_section("Missing data", report.missing_data))
    sections.append(f"_Disclaimer: {report.disclaimer}_")

    return "\n\n".join(sections)


def _render_list_section(title: str, items: list[str]) -> str:
    if not items:
        return f"## {title}\n- None"

    bullets = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n{bullets}"

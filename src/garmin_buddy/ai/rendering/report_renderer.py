from __future__ import annotations

from datetime import date

from garmin_buddy.ai.contracts.contracts import MissingDataItem, TrainingReviewReport


def render_report_md(
    report: TrainingReviewReport,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> str:
    sections: list[str] = []
    sections.append(f"# {_render_headline(start_date, end_date)}")
    sections.append(f"## Summary\n{report.summary}")
    sections.append(_render_list_section("Positives", report.positives))
    sections.append(_render_list_section("Mistakes", report.mistakes))
    sections.append(
        _render_list_section(
            "Recommendations",
            report.recommendations,
        )
    )
    sections.append(f"**Confidence:** {report.confidence:.2f}")
    sections.append(_render_missing_data_section(report.missing_data))

    return "\n\n".join(sections)


def _render_headline(start_date: date | None, end_date: date | None) -> str:
    if start_date is None or end_date is None:
        return "Training review"

    return (
        f"Training review between {start_date.isoformat()} and {end_date.isoformat()}"
    )


def _render_list_section(title: str, items: list[str]) -> str:
    if not items:
        return f"## {title}\n- None"

    bullets = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n{bullets}"


def _render_missing_data_section(items: list[MissingDataItem]) -> str:
    if not items:
        return "## Missing data\n- None"

    bullets = "\n".join(
        f"- {item.information}. IMPACT: {item.impact}" for item in items
    )
    return f"## Missing data\n{bullets}"

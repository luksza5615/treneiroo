from __future__ import annotations

from datetime import date

import streamlit as st

from garmin_buddy.ai.contracts.contracts import MissingDataItem, TrainingReviewReport

CONFIDENCE_HELP = (
    "Confidence is an estimate of how reliable this review is based on "
    "the available training data, supporting evidence, and missing data."
)


def render_report(
    report: TrainingReviewReport,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    st.markdown(
        render_report_md(
            report,
            start_date=start_date,
            end_date=end_date,
            include_confidence=False,
        )
    )
    st.markdown(_render_confidence(report.confidence), help=CONFIDENCE_HELP)


def render_report_md(
    report: TrainingReviewReport,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    include_confidence: bool = True,
) -> str:
    sections: list[str] = []
    sections.append(f"# {_render_headline(start_date, end_date)}")
    sections.append(f"## Summary\n{report.summary}")
    sections.append(_render_list_section("Advantages", report.positives))
    sections.append(_render_list_section("Disadvantages", report.mistakes))
    sections.append(
        _render_list_section(
            "Recommendations",
            report.recommendations,
        )
    )
    sections.append(_render_missing_data_section(report.missing_data))
    if include_confidence:
        sections.append(_render_confidence(report.confidence))

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

    bullets = "\n".join(f"- {item.information} ({item.impact})" for item in items)
    return f"## Missing data\n{bullets}"


def _render_confidence(confidence: float) -> str:
    return f"**Confidence:** {confidence:.2f}"

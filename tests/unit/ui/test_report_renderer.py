from datetime import date

from garmin_buddy.ai.contracts.contracts import MissingDataItem, TrainingReviewReport
from garmin_buddy.ui.rendering import report_renderer
from garmin_buddy.ui.rendering.report_renderer import (
    CONFIDENCE_HELP,
    render_report,
    render_report_md,
)


def test_render_report_md_contains_sections() -> None:
    report = TrainingReviewReport(
        summary="Strong week",
        positives=["Consistent training."],
        mistakes=["Fatigue risk."],
        recommendations=["Rest day.", "Easy run."],
        confidence=0.7,
        missing_data=[
            MissingDataItem(information="hrv_not_available", impact="medium"),
        ],
    )

    rendered = render_report_md(
        report,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    assert "# Training review between 2026-01-01 and 2026-01-07" in rendered
    assert "## Summary" in rendered
    assert "## Advantages" in rendered
    assert "## Positives" not in rendered
    assert "## Disadvantages" in rendered
    assert "## Mistakes" not in rendered
    assert "## Recommendations" in rendered
    assert "**Confidence:** 0.70" in rendered
    assert "## Missing data" in rendered
    assert "- hrv_not_available (medium)" in rendered


def test_render_report_md_handles_empty_sections() -> None:
    report = TrainingReviewReport(
        summary="Sparse week",
        positives=[],
        mistakes=[],
        recommendations=[],
        confidence=0.0,
        missing_data=[],
    )

    rendered = render_report_md(report)

    assert "## Advantages\n- None" in rendered
    assert "## Disadvantages\n- None" in rendered
    assert "## Recommendations\n- None" in rendered
    assert "## Missing data\n- None" in rendered


def test_render_report_shows_confidence_with_help(monkeypatch) -> None:
    report = TrainingReviewReport(
        summary="Strong week",
        positives=["Consistent training."],
        mistakes=["Fatigue risk."],
        recommendations=["Rest day."],
        confidence=0.7,
        missing_data=[],
    )
    markdown_calls: list[tuple[str, dict[str, object]]] = []

    def fake_markdown(body: str, **kwargs: object) -> None:
        markdown_calls.append((body, kwargs))

    monkeypatch.setattr(report_renderer.st, "markdown", fake_markdown)

    render_report(
        report,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    assert "**Confidence:**" not in markdown_calls[0][0]
    assert markdown_calls[-1] == (
        "**Confidence:** 0.70",
        {"help": CONFIDENCE_HELP},
    )

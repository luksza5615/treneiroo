from datetime import date

from garmin_buddy.ai.contracts.contracts import TrainingReviewReport
from garmin_buddy.ai.rendering.report_renderer import render_report_md


def test_render_report_md_contains_sections() -> None:
    report = TrainingReviewReport(
        summary="Strong week",
        positives=["Consistent training."],
        mistakes=["Fatigue risk."],
        recommendations=["Rest day.", "Easy run."],
        confidence=0.7,
        missing_data=["hrv_not_available"],
    )

    rendered = render_report_md(
        report,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    assert "# Training review between 2026-01-01 and 2026-01-07" in rendered
    assert "## Summary" in rendered
    assert "## Positives" in rendered
    assert "## Mistakes" in rendered
    assert "## Recommendations" in rendered
    assert "**Confidence:** 0.70" in rendered
    assert "## Missing data" in rendered

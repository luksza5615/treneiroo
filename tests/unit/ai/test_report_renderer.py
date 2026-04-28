from garmin_buddy.ai.contracts.contracts import TrainingReviewReport
from garmin_buddy.ai.rendering.report_renderer import render_report_md


def test_render_report_md_contains_sections() -> None:
    report = TrainingReviewReport(
        executive_summary="Strong week",
        positives=["Consistent training."],
        mistakes=["Fatigue risk."],
        main_lessons_and_recommendations=["Rest day.", "Easy run."],
        evidence=["2026-01-02 activity:123 Long run"],
        confidence=0.7,
        missing_data=["hrv_not_available"],
    )

    rendered = render_report_md(report)

    assert "# Training Review" in rendered
    assert "## Executive summary" in rendered
    assert "## Positives" in rendered
    assert "## Mistakes" in rendered
    assert "## Main lessons and recommendations" in rendered
    assert "## Evidence" in rendered
    assert "**Confidence:** 0.70" in rendered
    assert "## Missing data" in rendered

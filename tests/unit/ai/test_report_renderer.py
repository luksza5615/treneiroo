from garmin_buddy.ai.contracts import TrainingReviewReport
from garmin_buddy.ai.rendering.report_renderer import render_report_md


def test_render_report_md_contains_sections() -> None:
    report = TrainingReviewReport(
        headline="Strong week",
        positives=["Consistent training."],
        risks=["Fatigue risk."],
        priorities_next_7_days=["Rest day.", "Easy run.", "Mobility work."],
        evidence=["2026-01-02 activity:123 Long run"],
        confidence=0.7,
        missing_data=["hrv_not_available"],
        disclaimer="This report is informational only and is not medical advice.",
    )

    rendered = render_report_md(report)

    assert "# Strong week" in rendered
    assert "## Positives" in rendered
    assert "## Risks" in rendered
    assert "## Priorities (next 7 days)" in rendered
    assert "## Evidence" in rendered
    assert "**Confidence:** 0.70" in rendered
    assert "## Missing data" in rendered

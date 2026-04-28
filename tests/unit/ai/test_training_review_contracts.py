from datetime import date

import pytest

from garmin_buddy.ai.contracts.contracts import (
    build_fallback_training_review_report,
    parse_training_review_report,
    validate_training_review_report,
)


def _valid_payload() -> dict[str, object]:
    return {
        "summary": "Solid week with one high-load risk to monitor.",
        "positives": ["Consistent volume across five sessions."],
        "mistakes": ["Back-to-back hard days increased fatigue risk."],
        "main_lessons_and_recommendations": [
            "Keep one full rest day after the hardest run.",
            "Cap intensity to one quality workout.",
        ],
        "evidence": ["2026-01-31 activity:123456 Long run 18.2 km"],
        "confidence": 0.74,
        "missing_data": ["hrv_not_available"],
    }


def test_parse_training_review_report_accepts_valid_payload() -> None:
    report = parse_training_review_report(_valid_payload())

    assert report.summary.startswith("Solid week")
    assert report.confidence == pytest.approx(0.74)
    assert report.evidence[0].startswith("2026-01-31 activity:123456")


def test_parse_training_review_report_accepts_grounded_freeform_evidence() -> None:
    payload = _valid_payload()
    payload["evidence"] = [
        "Key session: 2026-04-12 ATE 5.0, ANE 1.0, Avg HR 172",
        "Training summary: 234.3 km and 5231m ascent",
    ]

    report = parse_training_review_report(payload)

    assert report.evidence == payload["evidence"]


def test_parse_training_review_report_rejects_extra_fields() -> None:
    payload = _valid_payload()
    payload["headline"] = "Old schema field."

    with pytest.raises(ValueError, match="Unexpected report fields"):
        parse_training_review_report(payload)


def test_validate_training_review_report_returns_error_messages() -> None:
    payload = _valid_payload()
    payload["confidence"] = 2.1

    is_valid, errors = validate_training_review_report(payload)

    assert is_valid is False
    assert errors
    assert "confidence" in errors[0]


def test_build_fallback_training_review_report_returns_schema_compliant_payload() -> (
    None
):
    report = build_fallback_training_review_report(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
        error_reason="invalid_json_from_model",
    )

    assert report.summary == (
        "Training review unavailable for 2026-01-01 to 2026-01-07."
    )
    assert report.main_lessons_and_recommendations
    assert report.confidence == 0.0
    assert "invalid_json_from_model" in report.missing_data

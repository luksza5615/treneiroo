from datetime import date

import pytest

from garmin_buddy.ai.contracts import (
    build_fallback_training_review_report,
    parse_training_review_report,
    validate_training_review_report,
)


def _valid_payload() -> dict[str, object]:
    return {
        "headline": "Solid week with one high-load risk to monitor.",
        "positives": ["Consistent volume across five sessions."],
        "risks": ["Back-to-back hard days increased fatigue risk."],
        "priorities_next_7_days": [
            "Keep one full rest day after the hardest run.",
            "Cap intensity to one quality workout.",
            "Add an easy aerobic session for recovery support.",
        ],
        "evidence": ["2026-01-31 activity:123456 Long run 18.2 km"],
        "confidence": 0.74,
        "missing_data": ["hrv_not_available"],
        "disclaimer": "This report is informational only and is not medical advice.",
    }


def test_parse_training_review_report_accepts_valid_payload() -> None:
    report = parse_training_review_report(_valid_payload())

    assert report.headline.startswith("Solid week")
    assert report.confidence == pytest.approx(0.74)
    assert report.evidence[0].startswith("2026-01-31 activity:123456")


def test_parse_training_review_report_rejects_invalid_evidence_format() -> None:
    payload = _valid_payload()
    payload["evidence"] = ["activity:123456 missing date prefix"]

    with pytest.raises(ValueError, match="evidence items must match"):
        parse_training_review_report(payload)


def test_parse_training_review_report_rejects_extra_fields() -> None:
    payload = _valid_payload()
    payload["unexpected"] = "value"

    with pytest.raises(ValueError, match="Unexpected report fields"):
        parse_training_review_report(payload)


def test_validate_training_review_report_returns_error_messages() -> None:
    payload = _valid_payload()
    payload["confidence"] = 2.1

    is_valid, errors = validate_training_review_report(payload)

    assert is_valid is False
    assert errors
    assert "confidence" in errors[0]


def test_build_fallback_training_review_report_returns_schema_compliant_payload() -> None:
    report = build_fallback_training_review_report(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
        error_reason="invalid_json_from_model",
    )

    assert report.headline == "Training review unavailable for 2026-01-01 to 2026-01-07."
    assert len(report.priorities_next_7_days) == 3
    assert report.confidence == 0.0
    assert "invalid_json_from_model" in report.missing_data

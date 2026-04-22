from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any, Mapping

_EVIDENCE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\s+activity:[^\s]+\s+.+$")
_REQUIRED_FIELDS = {
    "headline",
    "positives",
    "risks",
    "priorities_next_7_days",
    "evidence",
    "confidence",
    "missing_data",
}


@dataclass(frozen=True)
class TrainingReviewReport:
    headline: str
    positives: list[str]
    risks: list[str]
    priorities_next_7_days: list[str]
    evidence: list[str]
    confidence: float
    missing_data: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "positives": self.positives,
            "risks": self.risks,
            "priorities_next_7_days": self.priorities_next_7_days,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "missing_data": self.missing_data,
        }


def parse_training_review_report(payload: Mapping[str, Any]) -> TrainingReviewReport:
    _validate_required_and_extra_fields(payload)

    headline = _validate_non_empty_string("headline", payload["headline"])
    positives = _validate_string_list("positives", payload["positives"], 1, 5)
    risks = _validate_string_list("risks", payload["risks"], 1, 5)
    priorities = _validate_string_list(
        "priorities_next_7_days", payload["priorities_next_7_days"], 3, 7
    )
    evidence = _validate_evidence(payload["evidence"])
    confidence = _validate_confidence(payload["confidence"])
    missing_data = _validate_string_list(
        "missing_data", payload["missing_data"], 0, 100
    )

    return TrainingReviewReport(
        headline=headline,
        positives=positives,
        risks=risks,
        priorities_next_7_days=priorities,
        evidence=evidence,
        confidence=confidence,
        missing_data=missing_data,
    )


def validate_training_review_report(
    payload: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    try:
        parse_training_review_report(payload)
    except ValueError as exc:
        return False, [str(exc)]

    return True, []


def build_fallback_training_review_report(
    start_date: date,
    end_date: date,
    *,
    error_reason: str | None = None,
) -> TrainingReviewReport:
    missing_data = []
    if error_reason:
        missing_data.append(error_reason)

    return TrainingReviewReport(
        headline=("Training review unavailable"),
        positives=[],
        risks=[],
        priorities_next_7_days=[],
        evidence=[],
        confidence=0.0,
        missing_data=missing_data,
    )


def _validate_required_and_extra_fields(payload: Mapping[str, Any]) -> None:
    payload_fields = set(payload.keys())
    missing_fields = sorted(_REQUIRED_FIELDS - payload_fields)
    extra_fields = sorted(payload_fields - _REQUIRED_FIELDS)

    if missing_fields:
        raise ValueError(f"Missing required report fields: {', '.join(missing_fields)}")
    if extra_fields:
        raise ValueError(f"Unexpected report fields: {', '.join(extra_fields)}")


def _validate_non_empty_string(field_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")

    return cleaned


def _validate_string_list(
    field_name: str,
    value: Any,
    min_length: int,
    max_length: int,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")

    item_count = len(value)
    if item_count < min_length or item_count > max_length:
        raise ValueError(
            f"{field_name} must contain between {min_length} and {max_length} items"
        )

    cleaned_items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        cleaned_items.append(item.strip())

    return cleaned_items


def _validate_evidence(value: Any) -> list[str]:
    evidence = _validate_string_list("evidence", value, 0, 10)
    for evidence_item in evidence:
        if _EVIDENCE_PATTERN.match(evidence_item) is None:
            raise ValueError(
                "evidence items must match 'YYYY-MM-DD activity:<id> <label>'"
            )

    return evidence


def _validate_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a float between 0 and 1")

    confidence = float(value)
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be in range [0, 1]")

    return confidence

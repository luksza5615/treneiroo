from __future__ import annotations

from typing import Any, Mapping

from garmin_buddy.ai.preparation_contracts import RunnerProfileArtifact


def normalize_runner_profile(payload: Mapping[str, Any]) -> RunnerProfileArtifact:
    """Normalize free-form UI/profile input into a stable artifact.

    The workflow hashes upstream artifacts for reuse and staleness checks, so
    normalization must remove presentation noise like blank strings and spacing.
    """

    normalized_payload = {
        "athlete_name": _clean_optional_text(payload.get("athlete_name")),
        "goals": _split_lines(payload.get("goals")),
        "target_event": _clean_optional_text(payload.get("target_event")),
        "target_date": _clean_optional_text(payload.get("target_date")),
        "availability": _split_lines(payload.get("availability")),
        "constraints": _split_lines(payload.get("constraints")),
        "preferences": _split_lines(payload.get("preferences")),
        "injury_notes": _split_lines(payload.get("injury_notes")),
        "source_notes": _split_lines(payload.get("source_notes")),
    }

    if not normalized_payload["goals"]:
        normalized_payload["goals"] = [
            "Maintain consistent progression toward the next training block."
        ]

    return RunnerProfileArtifact.from_payload(normalized_payload)


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _split_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = str(value).splitlines()

    return [str(item).strip() for item in values if str(item).strip()]

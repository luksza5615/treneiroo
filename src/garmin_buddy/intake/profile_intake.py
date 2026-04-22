from __future__ import annotations

from typing import Any, Mapping

from garmin_buddy.ai.contracts.preparation_contracts import RunnerProfileArtifact


def normalize_runner_profile(payload: Mapping[str, Any]) -> RunnerProfileArtifact:
    """Normalize free-form UI/profile input into a stable artifact.

    The workflow hashes upstream artifacts for reuse and staleness checks, so
    normalization must remove presentation noise like blank strings and spacing.
    """

    normalized_payload = {
        "profile_context": _normalize_multiline_text(payload.get("profile_context")),
    }

    if normalized_payload["profile_context"] is None:
        normalized_payload["profile_context"] = (
            "Maintain consistent progression toward the next training block."
        )

    return RunnerProfileArtifact.from_payload(normalized_payload)


def _split_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = str(value).splitlines()

    return [str(item).strip() for item in values if str(item).strip()]


def _normalize_multiline_text(value: Any) -> str | None:
    lines = _split_lines(value)
    if not lines:
        return None
    return "\n".join(lines)

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any


_MARKER_PATTERNS = {
    "ferritin": re.compile(r"ferritin[^0-9]{0,20}([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE),
    "hemoglobin": re.compile(
        r"hemoglobin[^0-9]{0,20}([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE
    ),
    "vitamin_d": re.compile(
        r"vitamin[\s_-]*d[^0-9]{0,20}([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE
    ),
    "ck": re.compile(r"\bck\b[^0-9]{0,20}([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE),
}


@dataclass(frozen=True)
class ExtractedDocument:
    name: str
    text: str


def extract_document_text(name: str, content: bytes) -> ExtractedDocument:
    """Use best-effort decoding so imperfect uploads still produce planning context."""

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = content.decode("utf-8", errors="ignore")

    return ExtractedDocument(name=name, text=text.strip())


def summarize_lab_text(text: str) -> tuple[str, dict[str, Any]]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "No lab text provided.", {}

    markers: dict[str, Any] = {}
    for marker_name, pattern in _MARKER_PATTERNS.items():
        match = pattern.search(cleaned)
        if match is None:
            continue
        value = match.group(1).replace(",", ".")
        try:
            markers[marker_name] = float(value)
        except ValueError:
            markers[marker_name] = value

    summary_lines = [cleaned[:500]]
    if markers:
        summary_lines.append(
            "Detected markers: "
            + ", ".join(f"{key}={value}" for key, value in markers.items())
        )
    return "\n".join(summary_lines), markers


def build_lab_fingerprint(
    *,
    lab_text: str,
    lab_date: str | None,
    markers: dict[str, Any],
) -> str | None:
    if not lab_text.strip() and not markers:
        return None

    payload = {
        "lab_date": lab_date,
        "lab_text": " ".join(lab_text.split()),
        "markers": markers,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()

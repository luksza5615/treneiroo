from __future__ import annotations

from pathlib import Path

from garmin_buddy.ai.user_context import load_user_context


def test_load_user_context_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_user_context(tmp_path / "missing.md") is None


def test_load_user_context_returns_none_when_context_blank(tmp_path: Path) -> None:
    path = tmp_path / "user_context.md"
    path.write_text("  \n\n", encoding="utf-8")

    assert load_user_context(path) is None


def test_load_user_context_reads_markdown_context(tmp_path: Path) -> None:
    path = tmp_path / "user_context.md"
    path.write_text(
        "# Running profile\n\n"
        "- Age: 34 years old\n"
        "- Goal: sub-3 marathon\n",
        encoding="utf-8",
    )

    assert load_user_context(path) == (
        "# Running profile\n\n"
        "- Age: 34 years old\n"
        "- Goal: sub-3 marathon"
    )

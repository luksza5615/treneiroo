from __future__ import annotations

from pathlib import Path

USER_CONTEXT_PATH = Path(__file__).with_name("user_context.md")


def load_user_context(path: Path = USER_CONTEXT_PATH) -> str | None:
    if not path.exists():
        return None

    context = path.read_text(encoding="utf-8").strip()
    return context or None

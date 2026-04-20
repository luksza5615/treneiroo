from __future__ import annotations

from garmin_buddy.intake.profile_intake import normalize_runner_profile


def test_normalize_runner_profile_collapses_multiline_context() -> None:
    profile = normalize_runner_profile(
        {
            "profile_context": (
                "  Target event: marathon in October  \n\n"
                "  Goals: finish strong  \n"
                "Availability: Tue, Thu, Sun  "
            )
        }
    )

    assert profile.profile_context == (
        "Target event: marathon in October\n"
        "Goals: finish strong\n"
        "Availability: Tue, Thu, Sun"
    )


def test_normalize_runner_profile_uses_default_context_when_blank() -> None:
    profile = normalize_runner_profile({"profile_context": " \n "})

    assert profile.profile_context == (
        "Maintain consistent progression toward the next training block."
    )

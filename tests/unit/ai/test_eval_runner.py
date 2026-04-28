import json

from garmin_buddy.ai.eval.run_eval import evaluate_cases


def test_evaluate_cases_counts_pass_rates() -> None:
    valid_response = json.dumps(
        {
            "summary": "Solid week.",
            "positives": ["Consistent volume."],
            "mistakes": ["Fatigue risk."],
            "main_lessons_and_recommendations": ["Rest day.", "Easy run."],
            "evidence": ["2026-01-02 activity:123 Long run"],
            "confidence": 0.6,
            "missing_data": [],
        }
    )
    cases = [
        {"case_id": "valid", "llm_response": valid_response},
        {"case_id": "invalid", "llm_response": "{not json"},
        {"case_id": "skipped"},
    ]

    metrics = evaluate_cases(cases)

    assert metrics.total_cases == 3
    assert metrics.evaluated_cases == 2
    assert metrics.skipped_cases == 1
    assert metrics.schema_pass_rate == 0.5
    assert metrics.guardrail_pass_rate == 0.5

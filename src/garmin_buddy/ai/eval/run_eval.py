from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from garmin_buddy.ai.contracts.contracts import parse_training_review_report


@dataclass(frozen=True)
class EvalMetrics:
    total_cases: int
    evaluated_cases: int
    skipped_cases: int
    schema_pass_rate: float
    guardrail_pass_rate: float


def evaluate_cases(cases: list[dict[str, Any]]) -> EvalMetrics:
    total_cases = len(cases)
    evaluated_cases = 0
    schema_pass = 0
    guardrail_pass = 0

    for case in cases:
        response = case.get("llm_response")
        if not response:
            continue

        evaluated_cases += 1
        try:
            payload = json.loads(response)
            report = parse_training_review_report(payload)
        except (json.JSONDecodeError, ValueError):
            continue

        schema_pass += 1
        if _guardrails_pass(report):
            guardrail_pass += 1

    skipped_cases = total_cases - evaluated_cases
    schema_pass_rate = _safe_rate(schema_pass, evaluated_cases)
    guardrail_pass_rate = _safe_rate(guardrail_pass, evaluated_cases)

    return EvalMetrics(
        total_cases=total_cases,
        evaluated_cases=evaluated_cases,
        skipped_cases=skipped_cases,
        schema_pass_rate=schema_pass_rate,
        guardrail_pass_rate=guardrail_pass_rate,
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cases.append(json.loads(line))

    return cases


def _guardrails_pass(report) -> bool:
    if not report.summary.strip():
        return False
    if not report.recommendations:
        return False
    if report.confidence < 0 or report.confidence > 1:
        return False
    return True


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation for training review.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/eval/report_cases.jsonl"),
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    metrics = evaluate_cases(cases)

    print("Evaluation results:")
    print(f"  total_cases: {metrics.total_cases}")
    print(f"  evaluated_cases: {metrics.evaluated_cases}")
    print(f"  skipped_cases: {metrics.skipped_cases}")
    print(f"  schema_pass_rate: {metrics.schema_pass_rate:.2%}")
    print(f"  guardrail_pass_rate: {metrics.guardrail_pass_rate:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from garmin_buddy.ai.llm_analysis_service import LLMService
from garmin_buddy.ai.logging.execution_store import ExecutionStore
from garmin_buddy.ai.rendering.report_renderer import render_report_md
from garmin_buddy.ai.tools.training_review_tools import ToolRegistry
from garmin_buddy.ai.user_context import load_user_context
from garmin_buddy.ai.workflows.training_review import (
    TRAINING_REVIEW_MAX_TOOL_CALLS,
    TrainingReviewInputs,
    run_training_review,
)
from garmin_buddy.database.db_connector import Database
from garmin_buddy.database.db_service import ActivityRepository
from garmin_buddy.settings.config import Config


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Dates must be in YYYY-MM-DD format.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a training review.")
    parser.add_argument("--start-date", required=True, type=_parse_date)
    parser.add_argument("--end-date", required=True, type=_parse_date)
    parser.add_argument("--athlete-id", type=int, default=None)
    parser.add_argument("--executions-dir", type=Path, default=Path("executions"))
    args = parser.parse_args()

    cfg = Config.from_env()
    db = Database.create_db(cfg)
    repo = ActivityRepository(db)
    llm = LLMService(cfg.llm_api_key)
    execution_store = ExecutionStore(args.executions_dir)

    tool_registry = ToolRegistry(
        repo,
        max_tool_calls=TRAINING_REVIEW_MAX_TOOL_CALLS,
    )
    inputs = TrainingReviewInputs(
        start_date=args.start_date,
        end_date=args.end_date,
        athlete_id=args.athlete_id,
        user_context=load_user_context(),
    )
    result = run_training_review(
        llm_client=llm,
        tool_registry=tool_registry,
        inputs=inputs,
        execution_store=execution_store,
        model_name=llm.model_name,
    )

    markdown = render_report_md(
        result.report,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(markdown)
    if result.execution_id:
        print(f"\nexecution_id: {result.execution_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

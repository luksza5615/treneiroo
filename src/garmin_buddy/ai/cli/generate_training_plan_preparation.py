from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from garmin_buddy.ai.llm_analysis_service import LLMService
from garmin_buddy.ai.logging.preparation_execution_store import (
    PreparationExecutionStore,
)
from garmin_buddy.ai.rendering.preparation_renderer import render_preparation_md
from garmin_buddy.ai.tools.training_plan_preparation_tools import (
    PreparationToolRegistry,
)
from garmin_buddy.ai.workflows.training_plan_preparation import (
    TrainingPlanPreparationInputs,
    approve_training_plan_strategy,
    generate_phase_plan_from_strategy,
    run_training_plan_preparation,
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
    parser = argparse.ArgumentParser(
        description="Generate strategy or phase plan for training preparation."
    )
    parser.add_argument("--start-date", required=True, type=_parse_date)
    parser.add_argument("--end-date", required=True, type=_parse_date)
    parser.add_argument("--executions-dir", type=Path, default=Path("executions"))
    parser.add_argument("--strategy-id", type=str, default=None)
    parser.add_argument("--approve-strategy", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_env()
    db = Database.create_db(cfg)
    repo = ActivityRepository(db)
    llm = LLMService(cfg.llm_api_key)
    execution_store = PreparationExecutionStore(args.executions_dir)

    tool_registry = PreparationToolRegistry(repository=repo, max_tool_calls=8)
    inputs = TrainingPlanPreparationInputs(
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.strategy_id and args.approve_strategy:
        approve_training_plan_strategy(
            execution_store=execution_store,
            strategy_id=args.strategy_id,
        )

    if args.strategy_id:
        result = generate_phase_plan_from_strategy(
            llm_client=llm,
            tool_registry=tool_registry,
            execution_store=execution_store,
            inputs=inputs,
            strategy_id=args.strategy_id,
        )
    else:
        result = run_training_plan_preparation(
            llm_client=llm,
            tool_registry=tool_registry,
            execution_store=execution_store,
            inputs=inputs,
        )

    print(render_preparation_md(result))
    print(f"\nstrategy_id: {result.strategy.strategy_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

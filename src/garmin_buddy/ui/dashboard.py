from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from garmin_buddy.ai.llm_analysis_service import LLMService
from garmin_buddy.ai.logging.preparation_run_store import PreparationRunStore
from garmin_buddy.ai.logging.run_store import RunStore
from garmin_buddy.ai.rendering.preparation_renderer import render_preparation_md
from garmin_buddy.ai.rendering.report_renderer import render_report_md
from garmin_buddy.ai.tools.training_plan_preparation_tools import (
    PreparationToolRegistry,
)
from garmin_buddy.ai.tools.training_review_tools import ToolRegistry
from garmin_buddy.ai.workflows.training_plan_preparation import (
    TrainingPlanPreparationInputs,
    approve_training_plan_strategy,
    generate_phase_plan_from_strategy,
    run_training_plan_preparation,
)
from garmin_buddy.ai.workflows.training_review import (
    TrainingReviewInputs,
    run_training_review,
)
from garmin_buddy.analysis.analysis_service import AnalysisService
from garmin_buddy.database.db_connector import Database
from garmin_buddy.database.db_service import ActivityRepository
from garmin_buddy.ingestion.activity_mapper import ActivityMapper
from garmin_buddy.ingestion.fit_filestore import FitFileStore
from garmin_buddy.ingestion.fit_parser import FitParser
from garmin_buddy.ingestion.garmin_client import (
    GarminClient,
    GarminClientError,
    GarminMFARequiredError,
    GarminRateLimitError,
)
from garmin_buddy.integrations.google_sheets_training_log import (
    GoogleSheetsSettings,
    GoogleSheetsTrainingLogClient,
)
from garmin_buddy.intake.document_extraction import (
    build_lab_fingerprint,
    extract_document_text,
    summarize_lab_text,
)
from garmin_buddy.intake.profile_intake import normalize_runner_profile
from garmin_buddy.orchestration.sync_service import SyncService
from garmin_buddy.settings.config import Config, ConfigError
from garmin_buddy.settings.logging_config import setup_logging
from garmin_buddy.ui.charts import weekly_trend_chart
from garmin_buddy.ui.label_mapping import SPORT_LABELS, SUBSPORT_LABELS


@dataclass(frozen=True)
class Services:
    repo: ActivityRepository
    sync: SyncService
    analysis: AnalysisService
    llm: LLMService
    config: Config


# @st.cache_resource(show_spinner=False)
def init_services() -> Services:
    load_dotenv(override=True)
    setup_logging()

    cfg = Config.from_env()
    db = Database.create_db(cfg)
    garmin = GarminClient(
        cfg.garmin_email,
        cfg.garmin_password,
        tokenstore_path=cfg.fit_dir_path.parent / ".garmin_session",
        prompt_mfa=_read_garmin_mfa_code,
    )
    filestore = FitFileStore(cfg)
    parser = FitParser()
    mapper = ActivityMapper()
    repo = ActivityRepository(db)
    sync = SyncService(cfg, db, garmin, filestore, parser, mapper, repo)
    analysis = AnalysisService(repo)
    llm = LLMService(cfg.llm_api_key)

    return Services(repo=repo, sync=sync, analysis=analysis, llm=llm, config=cfg)


# @st.cache_data(ttl=60, show_spinner=False)
def load_activities(repo: ActivityRepository, start: date, end: date) -> pd.DataFrame:
    df = repo.get_activities(start, end)
    if df.empty:
        return df

    return df.sort_values(by="activity_start_time", ascending=False)


def _build_training_log_loader(config: Config):
    if not (
        config.google_sheets_spreadsheet_id
        and config.google_sheets_worksheet_name
        and config.google_service_account_info
    ):
        return None

    client = GoogleSheetsTrainingLogClient(
        GoogleSheetsSettings(
            spreadsheet_id=config.google_sheets_spreadsheet_id,
            worksheet_name=config.google_sheets_worksheet_name,
            service_account_info=config.google_service_account_info,
        )
    )

    def _load(start_date: date, end_date: date) -> list[dict[str, object]]:
        return client.list_sessions(start_date, end_date)

    return _load


def _build_lab_payload(
    uploaded_files: list[Any],
) -> dict[str, object]:
    extracted_texts: list[str] = []
    source_names: list[str] = []
    for uploaded_file in uploaded_files:
        extracted = extract_document_text(uploaded_file.name, uploaded_file.getvalue())
        if extracted.text:
            extracted_texts.append(extracted.text)
            source_names.append(extracted.name)

    combined_text = "\n\n".join(extracted_texts)
    lab_summary, lab_markers = summarize_lab_text(combined_text)

    return {
        "lab_summary": lab_summary,
        "lab_markers": lab_markers,
        "lab_fingerprint": build_lab_fingerprint(
            lab_text=combined_text,
            lab_date=None,
            markers=lab_markers,
        ),
        "source_notes": source_names,
    }


def _read_garmin_mfa_code() -> str:
    code = str(st.session_state.get("garmin_mfa_code", "")).strip()
    if code:
        return code

    raise GarminMFARequiredError(
        "Garmin requested MFA. Enter the current Garmin MFA code in the sidebar "
        "and retry the refresh."
    )


# ----------APP---------
def main():
    st.set_page_config(page_title="Garmin Buddy", layout="wide")

    try:
        services = init_services()
    except ConfigError as e:
        st.error(str(e))
        st.info("Set missing env vars (or .env) and restart the app.")
        st.stop()

    # st.title("Garmin Buddy")

    # --- Sidebar controls
    with st.sidebar:
        st.header("📅 Activities date range")

        default_end = date.today()
        default_start = default_end - timedelta(days=30)
        start, end = st.date_input(
            label="Activity date range",
            value=(default_start, default_end),
            max_value=default_end,
            label_visibility="collapsed",
            format="DD/MM/YYYY",
        )

        st.divider()

        st.header("🔄 Refresh activities    ")
        st.text_input(
            "Garmin MFA code",
            key="garmin_mfa_code",
            type="password",
            help="Use only when Garmin requests multi factor authentication.",
        )
        if st.button("Refresh", width="stretch"):
            try:
                with st.spinner("Syncing activities..."):
                    sync_start = start
                    services.sync.sync_activities(sync_start)
            except GarminRateLimitError as exc:
                st.error(str(exc))
                st.info(
                    "No local data was refreshed. Retry later or reuse the cached Garmin session."
                )
            except GarminClientError as exc:
                st.error(str(exc))
            else:
                st.cache_data.clear()
                st.toast("Actitivies refreshed", icon="✅")

    # Load activities (cached)
    df = load_activities(services.repo, start, end)

    metrics = services.analysis.calculate_kpis(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Activities 🏋️", f"{int(metrics.get('activities_count', 0)):,.0f}")
    col2.metric("Distance ➡️", f"{metrics.get('distance_km', 0):,.2f} km")
    # TODO col2.metric("Duration", f"{metrics.get('duration_h', 0):.1f} h")
    col3.metric("Ascent ↗️", f"{metrics.get('ascent_m', 0):,.0f} m")
    col4.metric("Avg HR ❤️", f"{metrics.get('avg_hr', 0):.0f} bpm")

    tabs = st.tabs(["Activities", "Weekly", "AI Review", "AI Plan Prep"])

    with tabs[0]:
        st.subheader("Activities")
        if df.empty:
            st.info("No activities to show.")
        else:
            cols = [
                "activity_start_time",
                "sport",
                "subsport",
                "distance_in_km",
                "elapsed_duration",
                "grade_adjusted_avg_pace_min_per_km",
                "avg_heart_rate",
                "total_ascent_in_m",
                "calories_burnt",
                "aerobic_training_effect_0_to_5",
                "anaerobic_training_effect_0_to_5",
                "running_efficiency_index",
            ]

            display_df = df[cols].copy()
            display_df["sport"] = display_df["sport"].map(SPORT_LABELS)
            display_df["subsport"] = display_df["subsport"].map(SUBSPORT_LABELS)

            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "activity_start_time": st.column_config.DatetimeColumn(
                        "Start time"
                    ),
                    "sport": st.column_config.TextColumn("Sport"),
                    "subsport": st.column_config.TextColumn("Type"),
                    "distance_in_km": st.column_config.NumberColumn(
                        "Distance (km)", format="%.2f"
                    ),
                    "elapsed_duration": st.column_config.TextColumn("Duration"),
                    "grade_adjusted_avg_pace_min_per_km": st.column_config.TextColumn(
                        "Avg pace (min/km)"
                    ),
                    "avg_heart_rate": st.column_config.NumberColumn(
                        "Avg HR", format="%.0f"
                    ),
                    "total_ascent_in_m": st.column_config.NumberColumn(
                        "Ascent (m)", format="%.0f"
                    ),
                    "calories_burnt": st.column_config.NumberColumn(
                        "Calories", format="%.0f"
                    ),
                    "aerobic_training_effect_0_to_5": st.column_config.NumberColumn(
                        "Aerobic TE (0-5)", format="%.1f"
                    ),
                    "anaerobic_training_effect_0_to_5": st.column_config.NumberColumn(
                        "Anaerobic TE (0-5)", format="%.1f"
                    ),
                    "running_efficiency_index": st.column_config.NumberColumn(
                        "Running Efficiency Index", format="%.2f"
                    ),
                },
            )

    with tabs[1]:
        st.subheader("Weekly measures for running")
        w = services.analysis.weekly_running_stats(df)

        if w.empty:
            st.info("No running activities in this range.")
        else:
            METRICS = {
                "Distance": {"col": "distance_km", "type": "bar"},
                "Calories": {"col": "calories", "type": "bar"},
                "Average HR": {"col": "avg_hr", "type": "line"},
                "Running efficiency index": {"col": "rei", "type": "line"},
                "Ascent": {"col": "ascent_m", "type": "bar"},
                "Average aerobic TE": {"col": "te_aer", "type": "line"},
                "Average anaerobic TE": {"col": "te_ana", "type": "line"},
            }

            metric_label = st.radio(
                "Metric",
                options=list(METRICS.keys()),
                horizontal=True,
                label_visibility="collapsed",
            )

            metric_cfg = METRICS[metric_label]
            col = metric_cfg["col"]

            chart = weekly_trend_chart(
                weekly_df=w,
                col=col,
                title=metric_label,
                chart_type=metric_cfg["type"],
                bar_size=18,
                bar_color="#4C78A8",
                line_color="#F58518",
            )
            st.altair_chart(chart)

            st.dataframe(
                w,
                width="stretch",
                hide_index=True,
                column_config={
                    "start_of_week": st.column_config.DateColumn("Start of week"),
                    "distance_km": st.column_config.NumberColumn(
                        "Distance (km)", format="%.2f"
                    ),
                    "avg_hr": st.column_config.NumberColumn("Avg HR", format="%.0f"),
                    "ascent_m": st.column_config.NumberColumn(
                        "Ascent (m)", format="%.0f"
                    ),
                    "calories": st.column_config.NumberColumn(
                        "Calories", format="%.0f"
                    ),
                    "te_aer": st.column_config.NumberColumn(
                        "Aerobic TE (0-5)", format="%.1f"
                    ),
                    "te_ana": st.column_config.NumberColumn(
                        "Anaerobic TE (0-5)", format="%.1f"
                    ),
                    "rei": st.column_config.NumberColumn(
                        "Running Efficiency Index", format="%.2f"
                    ),
                },
            )

    with tabs[2]:
        st.subheader("AI review")
        st.caption("Structured training review with evidence and lessons.")

        if services.config.feature_training_review is False:
            st.info("Enable FEATURE_TRAINING_REVIEW=true to use this feature.")
        else:
            include_key_sessions = st.checkbox(
                "Include key sessions", value=True, help="Fetch top key sessions."
            )
            max_tool_calls = st.number_input(
                "Max tool calls",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
            )

            if st.button("🧠 Generate AI review", type="primary"):
                try:
                    with st.spinner("Generating training review..."):
                        tool_registry = ToolRegistry(
                            services.repo, max_tool_calls=max_tool_calls
                        )
                        inputs = TrainingReviewInputs(
                            start_date=start,
                            end_date=end,
                            include_key_sessions=include_key_sessions,
                            max_tool_calls=max_tool_calls,
                        )
                        result = run_training_review(
                            llm_client=services.llm,
                            tool_registry=tool_registry,
                            inputs=inputs,
                            run_store=RunStore(Path("runs")),
                            model_name=services.llm.model_name,
                        )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.markdown(
                        render_report_md(
                            result.report,
                            start_date=start,
                            end_date=end,
                        )
                    )

    with tabs[3]:
        st.subheader("AI plan preparation")
        st.caption("Build a strategy first, approve it, then generate the first phase.")

        if services.config.feature_training_plan_preparation is False:
            st.info(
                "Enable FEATURE_TRAINING_PLAN_PREPARATION=true to use this feature."
            )
        else:
            training_log_loader = _build_training_log_loader(services.config)
            profile_context_text = st.text_area(
                "Planning context",
                key="prep_profile_context",
                height=220,
                help="Provide the planning context in one field. Include only the details you want the strategy to use.",
                placeholder=(
                    "Recommended: target event and date or timeframe, primary goals, "
                    "weekly availability, constraints or limitations, preferences, "
                    "recent injuries, and anything else relevant for the next block."
                ),
            )
            uploaded_lab_files = st.file_uploader(
                "Lab documents",
                accept_multiple_files=True,
                key="prep_lab_files",
            )

            run_store = PreparationRunStore(Path("runs"))

            profile_payload = normalize_runner_profile(
                {
                    "profile_context": profile_context_text,
                }
            )
            lab_payload = _build_lab_payload(uploaded_lab_files or [])

            if training_log_loader is None:
                st.warning(
                    "Google Sheets training log is not configured. The workflow will continue with missing-data markers."
                )

            if st.button(
                "Generate strategy", type="primary", key="prep_generate_strategy"
            ):
                try:
                    with st.spinner("Generating macro strategy..."):
                        tool_registry = PreparationToolRegistry(
                            repository=services.repo,
                            max_tool_calls=8,
                            training_log_loader=training_log_loader,
                            profile_loader=lambda: profile_payload.to_dict(),
                            lab_loader=lambda: lab_payload,
                            previous_artifact_loader=run_store.load_strategy_state,
                        )
                        result = run_training_plan_preparation(
                            llm_client=services.llm,
                            tool_registry=tool_registry,
                            run_store=run_store,
                            inputs=TrainingPlanPreparationInputs(
                                start_date=start,
                                end_date=end,
                            ),
                        )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.session_state["preparation_strategy_id"] = (
                        result.strategy.strategy_id
                    )
                    st.markdown(render_preparation_md(result))

            strategy_id = st.session_state.get("preparation_strategy_id")
            if strategy_id:
                st.caption(f"Current strategy id: {strategy_id}")
                approve_col, phase_col = st.columns(2)
                if approve_col.button("Approve strategy", key="prep_approve_strategy"):
                    try:
                        approve_training_plan_strategy(
                            run_store=run_store, strategy_id=strategy_id
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success("Strategy approved.")

                if phase_col.button("Generate first phase", key="prep_generate_phase"):
                    try:
                        with st.spinner("Generating first phase and critique..."):
                            tool_registry = PreparationToolRegistry(
                                repository=services.repo,
                                max_tool_calls=8,
                                training_log_loader=training_log_loader,
                                profile_loader=lambda: profile_payload.to_dict(),
                                lab_loader=lambda: lab_payload,
                                previous_artifact_loader=run_store.load_strategy_state,
                            )
                            result = generate_phase_plan_from_strategy(
                                llm_client=services.llm,
                                tool_registry=tool_registry,
                                run_store=run_store,
                                inputs=TrainingPlanPreparationInputs(
                                    start_date=start,
                                    end_date=end,
                                ),
                                strategy_id=strategy_id,
                            )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        if result.strategy_stale:
                            st.error(
                                "Strategy became stale because upstream inputs changed. Regenerate the strategy first."
                            )
                        st.markdown(render_preparation_md(result))

from dataclasses import asdict
from datetime import date
import logging

import pandas as pd
import pyodbc
from sqlalchemy import text

from garmin_buddy.database.db_connector import Database

logger = logging.getLogger(__name__)


class ActivityRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def get_activities(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> pd.DataFrame:
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError(
                f"Start date cannot be later than end date for activities range. Start date set: {start_date}, end date: {end_date}"
            )
        query_params: dict[str, date] = {}
        query = "SELECT * FROM [dbo].[activity] WHERE 1=1"

        if start_date is not None:
            query += " AND activity_date >= :start_date"
            query_params["start_date"] = start_date

        if end_date is not None:
            query += " AND activity_date <= :end_date"
            query_params["end_date"] = end_date

        with self.database.get_db_connection() as conn:
            activities = pd.read_sql_query(
                text(query), conn, params=query_params if query_params else None
            )

        logger.info("Fetched %d activities", len(activities))

        return activities

    def get_training_summary(
        self,
        start_date: date,
        end_date: date,
        *,
        athlete_id: int | None = None,
    ) -> dict[str, float | None]:
        if start_date > end_date:
            raise ValueError(
                "Start date cannot be later than end date for training summary."
            )

        query = """
            SELECT
                COUNT(*) AS activities_count,
                SUM(distance_in_km) AS distance_km,
                AVG(avg_heart_rate) AS avg_hr,
                SUM(calories_burnt) AS calories_burnt,
                SUM(total_ascent_in_m) AS ascent_m,
                AVG(aerobic_training_effect_0_to_5) AS aerobic_training_effect_0_to_5,
                AVG(anaerobic_training_effect_0_to_5) AS anaerobic_training_effect_0_to_5
            FROM [dbo].[activity]
            WHERE activity_date >= :start_date AND activity_date <= :end_date
        """
        query_params = {"start_date": start_date, "end_date": end_date}

        if athlete_id is not None:
            logger.info(
                "Athlete id filtering not supported; ignoring athlete_id=%s",
                athlete_id,
            )

        with self.database.get_db_connection() as conn:
            summary_df = pd.read_sql_query(text(query), conn, params=query_params)

        if summary_df.empty:
            return {
                "activities_count": 0.0,
                "distance_km": None,
                "avg_hr": None,
                "calories_burnt": None,
                "ascent_m": None,
                "aerobic_training_effect_0_to_5": None,
                "anaerobic_training_effect_0_to_5": None,
            }

        row = summary_df.iloc[0].to_dict()
        return {
            "activities_count": float(row.get("activities_count", 0) or 0),
            "distance_km": _to_optional_float(row.get("distance_km")),
            "avg_hr": _to_optional_float(row.get("avg_hr")),
            "calories_burnt": _to_optional_float(row.get("calories_burnt")),
            "ascent_m": _to_optional_float(row.get("ascent_m")),
            "aerobic_training_effect_0_to_5": _to_optional_float(
                row.get("aerobic_training_effect_0_to_5")
            ),
            "anaerobic_training_effect_0_to_5": _to_optional_float(
                row.get("anaerobic_training_effect_0_to_5")
            ),
        }

    def list_key_sessions(
        self,
        start_date: date,
        end_date: date,
        *,
        athlete_id: int | None = None,
        n: int = 5,
    ) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError(
                "Start date cannot be later than end date for key sessions."
            )
        if n <= 0:
            raise ValueError("n must be a positive integer for key sessions.")

        if athlete_id is not None:
            logger.info(
                "Athlete id filtering not supported; ignoring athlete_id=%s",
                athlete_id,
            )

        query = f"""
            SELECT TOP {int(n)}
                activity_id,
                activity_date,
                sport,
                subsport,
                distance_in_km,
                elapsed_duration,
                grade_adjusted_avg_pace_min_per_km,
                avg_heart_rate,
                calories_burnt,
                aerobic_training_effect_0_to_5,
                anaerobic_training_effect_0_to_5,
                total_ascent_in_m,
                running_efficiency_index
            FROM [dbo].[activity]
            WHERE activity_date >= :start_date AND activity_date <= :end_date
            ORDER BY
                COALESCE(aerobic_training_effect_0_to_5, 0)
                + COALESCE(anaerobic_training_effect_0_to_5, 0) DESC,
                activity_date DESC
        """
        query_params = {"start_date": start_date, "end_date": end_date}

        with self.database.get_db_connection() as conn:
            sessions_df = pd.read_sql_query(text(query), conn, params=query_params)

        return sessions_df

    def get_activity_by_id(self, activity_id: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM [dbo].[activity]
            WHERE activity_id = :activity_id
        """
        with self.database.get_db_connection() as conn:
            activity_df = pd.read_sql_query(
                text(query), conn, params={"activity_id": activity_id}
            )

        return activity_df

    def list_executed_sessions(
        self,
        start_date: date,
        end_date: date,
        *,
        athlete_id: int | None = None,
    ) -> pd.DataFrame:
        """Keep execution access explicit so planning workflows reuse one query surface."""

        if athlete_id is not None:
            logger.info(
                "Athlete id filtering not supported; ignoring athlete_id=%s",
                athlete_id,
            )

        return self.get_activities(start_date, end_date)

    def get_execution_summary(
        self,
        start_date: date,
        end_date: date,
        *,
        athlete_id: int | None = None,
    ) -> dict[str, float | int | None]:
        executed_df = self.list_executed_sessions(
            start_date, end_date, athlete_id=athlete_id
        )
        if executed_df.empty:
            return {
                "executed_sessions": 0,
                "distance_km": 0.0,
                "avg_hr": None,
                "ascent_m": 0.0,
                "avg_aerobic_te": None,
                "avg_anaerobic_te": None,
                "hard_sessions": 0,
            }

        hard_session_mask = (
            executed_df["aerobic_training_effect_0_to_5"].fillna(0)
            + executed_df["anaerobic_training_effect_0_to_5"].fillna(0)
        ) >= 5.5

        return {
            "executed_sessions": int(len(executed_df)),
            "distance_km": float(executed_df["distance_in_km"].fillna(0).sum()),
            "avg_hr": _series_optional_float(executed_df["avg_heart_rate"].mean()),
            "ascent_m": float(executed_df["total_ascent_in_m"].fillna(0).sum()),
            "avg_aerobic_te": _series_optional_float(
                executed_df["aerobic_training_effect_0_to_5"].mean()
            ),
            "avg_anaerobic_te": _series_optional_float(
                executed_df["anaerobic_training_effect_0_to_5"].mean()
            ),
            "hard_sessions": int(hard_session_mask.sum()),
        }

    def compare_planned_vs_executed(
        self,
        planned_sessions: list[dict[str, object]],
        start_date: date,
        end_date: date,
        *,
        athlete_id: int | None = None,
    ) -> dict[str, object]:
        """Centralize adherence math so every agent sees the same facts."""

        executed_df = self.list_executed_sessions(
            start_date, end_date, athlete_id=athlete_id
        )
        planned_dates = {
            str(session.get("date"))
            for session in planned_sessions
            if session.get("date") is not None
        }
        executed_dates = {
            value.date().isoformat()
            if hasattr(value, "date")
            else str(value)
            for value in executed_df["activity_date"].dropna().tolist()
        }
        matched_dates = planned_dates & executed_dates
        planned_count = len(planned_sessions)
        adherence_rate = (len(matched_dates) / planned_count) if planned_count else None

        return {
            "planned_sessions": planned_count,
            "executed_sessions": int(len(executed_df)),
            "matched_days": len(matched_dates),
            "adherence_rate": adherence_rate,
            "missing_session_dates": sorted(planned_dates - executed_dates),
            "extra_execution_dates": sorted(executed_dates - planned_dates),
        }

    def persist_activity(self, activity):
        if self._check_if_activity_exists_in_db(activity.activity_id) is False:
            try:
                query = text("""INSERT INTO activity (
                            activity_id, activity_date, activity_start_time, sport, subsport, distance_in_km, elapsed_duration, 
                            grade_adjusted_avg_pace_min_per_km, avg_heart_rate, calories_burnt, aerobic_training_effect_0_to_5, 
                            anaerobic_training_effect_0_to_5, total_ascent_in_m, total_descent_in_m, start_of_week, running_efficiency_index)
                            VALUES (
                            :activity_id, :activity_date, :activity_start_time, :sport, :subsport, :distance_in_km, :elapsed_duration,
                            :grade_adjusted_avg_pace_min_per_km, :avg_heart_rate, :calories_burnt,
                            :aerobic_training_effect_0_to_5, :anaerobic_training_effect_0_to_5,
                            :total_ascent_in_m, :total_descent_in_m, :start_of_week, :running_efficiency_index)
                        """)

                params = asdict(activity)

                with self.database.engine.begin() as conn:
                    conn.execute(query, params)

                logger.info(
                    "Activity %s_%s data saved in database sucessfully",
                    activity.sport,
                    activity.activity_date,
                )
            except pyodbc.ProgrammingError:
                logger.exception(
                    "Failed to save activity %s_%s due to error: %s",
                    activity.sport,
                    activity.activity_date,
                )
        else:
            logger.info(
                "Activity %s already exists in the database",
                (activity.activity_start_time),
            )

    def get_activity_ids_set(self):
        rows = self._get_activity_ids()

        return set([row.activity_id for row in rows])

    def _check_if_activity_exists_in_db(self, activity_id):
        activities_rows_list = self._get_activity_ids()
        activities_ids_list = [row.activity_id for row in activities_rows_list]

        return activity_id in activities_ids_list

    def _get_activity_ids(self):
        query = "SELECT activity_id FROM dbo.activity"

        with self.database.get_db_connection() as conn:
            activity_ids = conn.execute(text(query)).fetchall()

        return activity_ids


def _to_optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    return float(value)


def _series_optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)

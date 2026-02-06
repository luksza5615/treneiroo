from datetime import date
from typing import Any, Dict

from google import genai
import pandas as pd

_LLM_MODEL = "gemini-3-flash-preview"


class LLMService:
    def __init__(self, api_key: str, *, model_name: str | None = None) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or _LLM_MODEL

    def analyze_training_period(
        self,
        activities_df: pd.DataFrame,
        metrics: Dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> str:
        activities_data = self._format_activities_for_prompt(activities_df)
        prompt = self._build_period_analysis_prompt(
            activities_data, metrics, start_date, end_date
        )
        analysis = self._generate_response(prompt)

        return analysis

    def generate(self, prompt: str) -> str:
        return self._generate_response(prompt)

    def _generate_response(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text

    def _format_activities_for_prompt(self, activities_df: pd.DataFrame) -> str:
        if activities_df.empty:
            return "No activities found in the specified period."

        formatted_activities = []

        for _, activity in activities_df.iterrows():
            activity_str = f"""
                Activity on {activity["activity_date"]}:
                - Sport: {activity["sport"]} ({activity["subsport"]})
                - Distance: {activity["distance_in_km"]} km
                - Duration: {activity["elapsed_duration"]}
                - Pace: {activity["grade_adjusted_avg_pace_min_per_km"]} min/km
                - Avg Heart Rate: {activity["avg_heart_rate"]} bpm
                - Calories: {activity["calories_burnt"]}
                - Elevation: {activity["total_ascent_in_m"]}m
                - Training Effect: {activity["aerobic_training_effect_0_to_5"]}
                - Running efficiency Index: {activity["running_efficiency_index"] if pd.notna(activity["running_efficiency_index"]) else "N/A"}
                """
            formatted_activities.append(activity_str)

        return "\n".join(formatted_activities)

    def _build_period_analysis_prompt(
        self,
        activities_data: str,
        metrics: Dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> str:
        # TODO
        # - Average Anaerobic Training Effect: {metrics['anaerobic_training_effect_0_to_5']}
        # - Average Running Efficiency Index: {metrics['avg_running_efficiency_index']}
        prompt = f"""
            Analyze the following training period and provide comprehensive insights:

            TRAINING SUMMARY for period {start_date.isoformat} - {end_date.isoformat}:
            - Total Activities: {metrics["activities_count"]}
            - Total Distance: {metrics["distance_km"]} km
            - Average Heart Rate: {metrics["avg_hr"]} bpm
            - Total Calories: {metrics["calories_burnt"]}
            - Total Elevation Gain: {metrics["ascent_m"]} m
            - Average Aerobic Training Effect: {metrics["aerobic_training_effect_0_to_5"]}
            
            DETAILED ACTIVITIES:
            {activities_data}

            Please provide analysis in the following structure:

            1. TRAINING LOAD ANALYSIS:
            - Overall training volume assessment
            - Intensity distribution analysis
            - Recovery patterns

            2. PERFORMANCE TRENDS:
            - Heart rate trends and zones
            - Pace/efficiency progression
            - Training effect analysis

            3. SPORT-SPECIFIC INSIGHTS:
            - Breakdown by activity type
            - Sport-specific recommendations

            4. RECOVERY & ADAPTATION:
            - Recovery indicators
            - Adaptation signals
            - Overtraining risk assessment

            5. RECOMMENDATIONS:
            - Immediate next steps
            - Training adjustments needed
            - Focus areas for improvement

            Provide specific, actionable insights based on the data above.
            """
        return prompt

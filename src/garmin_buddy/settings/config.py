import os
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    def __init__(self, missing_vars: list[str]):
        self.missing_vars = missing_vars
        super().__init__(f"Missing required variables: {', '.join(missing_vars)}")


@dataclass(frozen=True)
class Config:
    fit_dir_path: Path
    garmin_email: str
    garmin_password: str
    db_connection_string: str
    llm_api_key: str
    feature_training_review: bool

    @staticmethod
    def validate_vars(
        *,
        fit_dir_path,
        garmin_email,
        garmin_password,
        db_connection_string,
        llm_api_key,
    ) -> None:
        missing_vars: list[str] = []
        if not fit_dir_path:
            missing_vars.append("FIT_DIR_PATH")
        if not garmin_email:
            missing_vars.append("GARMIN_EMAIL")
        if not garmin_password:
            missing_vars.append("GARMIN_PASSWORD")
        if not db_connection_string:
            missing_vars.append("DB_CONNECTION_STRING")
        if not llm_api_key:
            missing_vars.append("LLM_API_KEY")

        if missing_vars:
            raise ConfigError(missing_vars)

        return None

    @classmethod
    def from_env(cls) -> "Config":
        fit_path = os.getenv("FIT_DIR_PATH")
        garmin_email = os.getenv("GARMIN_EMAIL")
        garmin_password = os.getenv("GARMIN_PASSWORD")
        db_connection_string = os.getenv("DB_CONNECTION_STRING")
        llm_api_key = os.getenv("LLM_API_KEY")
        feature_training_review = _parse_bool(
            os.getenv("FEATURE_TRAINING_REVIEW"), default=False
        )

        Config.validate_vars(
            fit_dir_path=fit_path,
            garmin_email=garmin_email,
            garmin_password=garmin_password,
            db_connection_string=db_connection_string,
            llm_api_key=llm_api_key,
        )

        return cls(
            fit_dir_path=Path(fit_path),
            garmin_email=garmin_email,
            garmin_password=garmin_password,
            db_connection_string=db_connection_string,
            llm_api_key=llm_api_key,
            feature_training_review=feature_training_review,
        )


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Invalid boolean value: {value}")

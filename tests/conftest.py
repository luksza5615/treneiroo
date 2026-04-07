import datetime as dt
from pathlib import Path
import uuid

import pytest

@pytest.fixture
def tmp_path() -> Path:
    tmp_root = Path(__file__).resolve().parents[1] / ".tmp-test"
    tmp_root.mkdir(parents=True, exist_ok=True)

    path = tmp_root / f"pytest-{uuid.uuid4().hex}"
    path.mkdir()
    return path


@pytest.fixture(scope="session")
def fit_file() -> Path:
    return Path(__file__).parent / "resources" / "2025-03-25_running_18635294298.fit"


@pytest.fixture(scope="session")
def fit_files_parent() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def activity_dict_example() -> dict:
    return {
        "start_time": dt.datetime(2025, 12, 1, 18, 4, 40),
        "local_timestamp": dt.datetime(2025, 12, 1, 18, 4, 40),
        "sport": "running",
        "sub_sport": "trail_running",
        "total_distance": 10000,
        "total_elapsed_time": 3600,
        "avg_hr": 155,
        "total_ascent": 250,
        "total_descent": 250,
    }


@pytest.fixture(scope="session")
def activity_dict_empty_values() -> dict:
    return {
        "start_time": None,
        "sport": None,
        "sub_sport": None,
        "total_distance": None,
        "total_elapsed_time": None,
        "avg_hr": None,
        "enhanced_avg_speed": None,
        "total_calories": None,
        "total_training_effect": None,
        "total_anaerobic_training_effect": None,
        "total_ascent": None,
        "total_descent": None,
    }

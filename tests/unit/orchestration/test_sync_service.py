from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from garmin_buddy.ingestion.garmin_client import GarminClientError
from garmin_buddy.orchestration.sync_service import SyncService


def _build_service() -> tuple[SyncService, Mock, Mock, Mock, Mock, Mock, Mock]:
    garmin_client = Mock()
    fit_filestore = Mock()
    fit_parser = Mock()
    activity_mapper = Mock()
    activity_repository = Mock()
    database = Mock()
    configuration = SimpleNamespace(fit_dir_path=Path("fits"))

    service = SyncService(
        configuration=configuration,
        database=database,
        garmin_client=garmin_client,
        fit_filestore=fit_filestore,
        fit_parser=fit_parser,
        activity_mapper=activity_mapper,
        activity_repository=activity_repository,
    )
    return (
        service,
        garmin_client,
        fit_filestore,
        fit_parser,
        activity_mapper,
        activity_repository,
        database,
    )


def test_sync_activities_persists_new_activity() -> None:
    (
        service,
        garmin_client,
        fit_filestore,
        fit_parser,
        activity_mapper,
        activity_repository,
        _database,
    ) = _build_service()
    garmin_activity = {
        "activityId": 123,
        "activityType": {"typeKey": "running"},
        "startTimeGMT": "2026-01-02 06:30:00",
    }
    garmin_client.get_garmin_activities_history.return_value = [garmin_activity]
    garmin_client.get_activity_signature.return_value = (123, "running", date(2026, 1, 2))
    garmin_client.download_activity_as_zip_file.return_value = b"zip-data"
    fit_filestore.list_existing_fit_files_ids_set.return_value = set()
    fit_filestore.build_fit_filename.return_value = "2026-01-02_running_123.fit"
    activity_repository.get_activity_ids_set.return_value = set()
    fit_parser.parse_fit_file.return_value = {"parsed": True}
    activity_mapper.from_parsed_fit.return_value = {"activity_id": 123}

    service.sync_activities(date(2026, 1, 1))

    garmin_client.login_to_garmin.assert_called_once_with()
    garmin_client.download_activity_as_zip_file.assert_called_once_with(123)
    fit_filestore.create_fit_file_from_zip.assert_called_once()
    activity_repository.persist_activity.assert_called_once_with({"activity_id": 123})


def test_sync_activities_propagates_garmin_client_errors() -> None:
    (
        service,
        garmin_client,
        fit_filestore,
        fit_parser,
        activity_mapper,
        activity_repository,
        _database,
    ) = _build_service()
    garmin_activity = {
        "activityId": 123,
        "activityType": {"typeKey": "running"},
        "startTimeGMT": "2026-01-02 06:30:00",
    }
    garmin_client.get_garmin_activities_history.return_value = [garmin_activity]
    garmin_client.get_activity_signature.return_value = (123, "running", date(2026, 1, 2))
    garmin_client.download_activity_as_zip_file.side_effect = GarminClientError("blocked")
    fit_filestore.list_existing_fit_files_ids_set.return_value = set()
    fit_filestore.build_fit_filename.return_value = "2026-01-02_running_123.fit"
    activity_repository.get_activity_ids_set.return_value = set()

    with pytest.raises(GarminClientError, match="blocked"):
        service.sync_activities(date(2026, 1, 1))

    fit_parser.parse_fit_file.assert_not_called()
    activity_mapper.from_parsed_fit.assert_not_called()
    activity_repository.persist_activity.assert_not_called()

from datetime import date
import logging
import os

from garmin_buddy.database.db_service import ActivityRepository
from garmin_buddy.ingestion.activity_mapper import ActivityMapper
from garmin_buddy.settings.config import Config
from garmin_buddy.database.db_connector import Database
from garmin_buddy.ingestion.fit_filestore import FitFileStore
from garmin_buddy.ingestion.fit_parser import FitParser
from garmin_buddy.ingestion.garmin_client import GarminClient, GarminClientError

logger = logging.getLogger(__name__)


class SyncService:
    def __init__(
        self,
        configuration: Config,
        database: Database,
        garmin_client: GarminClient,
        fit_filestore: FitFileStore,
        fit_parser: FitParser,
        activity_mapper: ActivityMapper,
        activity_repository: ActivityRepository,
    ):
        self.configuration = configuration
        self.database = database
        self.garmin_client = garmin_client
        self.fit_filestore = fit_filestore
        self.fit_parser = fit_parser
        self.activity_mapper = activity_mapper
        self.activity_repository = activity_repository

    def sync_activities(self, start_date: date = None) -> None:
        self.garmin_client.login_to_garmin()
        garmin_activities = self.garmin_client.get_garmin_activities_history(start_date)
        db_ids_set = self.activity_repository.get_activity_ids_set()
        existing_files_set = self.fit_filestore.list_existing_fit_files_ids_set()

        persisted_activities = []

        for garmin_activity in garmin_activities:
            logging.debug("Test %s", garmin_activity)
            fit_filename = "<unknown>"
            try:
                garmin_activity_id, garmin_activity_type, garmin_activity_date = (
                    self.garmin_client.get_activity_signature(garmin_activity)
                )

                # If activity already persisted in db, skip it
                if garmin_activity_id in db_ids_set:
                    logger.info("Activity %s already exists in DB.", garmin_activity_id)
                    continue

                fit_filename = self.fit_filestore.build_fit_filename(
                    garmin_activity_date, garmin_activity_type, garmin_activity_id
                )
                fit_filepath = os.path.join(
                    self.configuration.fit_dir_path, fit_filename
                )

                # If activity not persisted in db, but fit file already downloaded, parse and save to DB without processing file
                if garmin_activity_id in existing_files_set:
                    logger.info(
                        "File %s already downloaded, but not saved in DB", fit_filename
                    )
                    self._parse_and_persist(fit_filepath, garmin_activity_id)
                    db_ids_set.add(garmin_activity_id)
                    persisted_activities.append(fit_filepath)
                    continue

                fit_zip_file = self.garmin_client.download_activity_as_zip_file(
                    garmin_activity_id
                )
                self.fit_filestore.create_fit_file_from_zip(
                    fit_zip_file, garmin_activity, fit_filepath, self.garmin_client
                )
                existing_files_set.add(garmin_activity_id)
                self._parse_and_persist(fit_filepath, garmin_activity_id)
                db_ids_set.add(garmin_activity_id)
                persisted_activities.append(fit_filepath)
            except GarminClientError:
                raise
            except Exception:
                logger.exception("Failed processing activity: %s", fit_filename)

        logger.info(
            "Synced activities (files processed): %d", len(persisted_activities)
        )

    def _parse_and_persist(self, fit_filepath: str, activity_id: int) -> None:
        parsed_activity = self.fit_parser.parse_fit_file(fit_filepath)
        activity_model = self.activity_mapper.from_parsed_fit(
            activity_id, parsed_activity
        )
        self.activity_repository.persist_activity(activity_model)

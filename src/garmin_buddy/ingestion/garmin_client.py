import logging
from datetime import datetime, timedelta
from pathlib import Path

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

logger = logging.getLogger(__name__)


class GarminClientError(Exception):
    """Raised when the Garmin client cannot complete the requested operation."""


class GarminRateLimitError(GarminClientError):
    """Raised when Garmin rejects requests due to rate limiting."""


class GarminClient:
    def __init__(
        self, email: str, password: str, tokenstore_path: Path | None = None
    ) -> None:
        self.email = email
        self.password = password
        self.tokenstore_path = tokenstore_path
        self._client: Garmin | None = None

    def login_to_garmin(self) -> None:
        client = Garmin(self.email, self.password)
        client.garth.configure(status_forcelist=(408, 500, 502, 503, 504))

        try:
            self._login(client)
            self._client = client
            logger.info("Connected to garmin")
        except GarminConnectTooManyRequestsError as exc:
            raise GarminRateLimitError(
                "Garmin login is temporarily rate-limited. "
                "The app now reuses cached tokens when available, but Garmin is "
                "currently rejecting authentication requests. Wait and try again later."
            ) from exc
        except (GarminConnectAuthenticationError, GarminConnectConnectionError) as exc:
            raise GarminClientError(f"Failed to connect to Garmin: {exc}") from exc
        except Exception:
            logger.exception("Failed to connect.")
            raise

    def _login(self, client: Garmin) -> None:
        if self.tokenstore_path is None:
            client.login()
            return

        try:
            client.login(tokenstore=str(self.tokenstore_path))
        except FileNotFoundError:
            logger.info(
                "Garmin token cache not found at %s. Falling back to credential login.",
                self.tokenstore_path,
            )
            client.login()

        self.tokenstore_path.mkdir(parents=True, exist_ok=True)
        client.garth.dump(str(self.tokenstore_path))

    def get_garmin_activities_history(
        self, start_date=None, end_date=None, window_days=90
    ):
        """
        Fetch activities across full history by paging through date windows.
        If start_date is None, default to a far past date.
        """
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            start_date = datetime(1990, 1, 1).date()

        all_activities = []
        window_start = start_date
        while window_start <= end_date:
            window_end = min(window_start + timedelta(days=window_days), end_date)
            try:
                if self._client is None:
                    raise GarminClientError("Garmin client is not logged in.")

                activities = self._client.get_activities_by_date(
                    window_start.isoformat(), window_end.isoformat()
                )
                if activities:
                    all_activities.extend(activities)
            except GarminConnectTooManyRequestsError as exc:
                raise GarminRateLimitError(
                    "Garmin rate-limited activity history requests. "
                    "The sync was stopped before processing partial data."
                ) from exc
            except (GarminConnectAuthenticationError, GarminConnectConnectionError) as exc:
                raise GarminClientError(
                    f"Failed to fetch Garmin activities for {window_start} - {window_end}: {exc}"
                ) from exc
            except Exception:
                logger.exception(
                    "Failed to fetch activities for window %s - %s",
                    window_start,
                    window_end,
                )
                raise
            window_start = window_end + timedelta(days=1)

        return all_activities

    def download_activity_as_zip_file(self, activity_id):
        if self._client is None:
            raise GarminClientError("Garmin client is not logged in.")

        return self._client.download_activity(
            activity_id, dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL
        )

    def get_activity_signature(self, garmin_activity) -> tuple:
        garmin_activity_id = garmin_activity["activityId"]
        garmin_activity_type = garmin_activity["activityType"]["typeKey"]
        garmin_activity_start_time = datetime.strptime(
            garmin_activity["startTimeGMT"], "%Y-%m-%d %H:%M:%S"
        )
        garmin_activity_date = garmin_activity_start_time.date()

        return garmin_activity_id, garmin_activity_type, garmin_activity_date

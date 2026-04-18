import logging
import sys
from collections.abc import Callable
from contextlib import contextmanager
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


class GarminMFARequiredError(GarminClientError):
    """Raised when Garmin requires MFA and the app cannot supply a code."""


class GarminClient:
    def __init__(
        self,
        email: str,
        password: str,
        tokenstore_path: Path | None = None,
        prompt_mfa: Callable[[], str] | None = None,
    ) -> None:
        self.email = email
        self.password = password
        self.tokenstore_path = tokenstore_path
        self.prompt_mfa = prompt_mfa or self._default_prompt_mfa
        self._client: Garmin | None = None

    def login_to_garmin(self) -> None:
        try:
            client = self._login()
            self._client = client
            logger.info("Connected to garmin")
        except GarminClientError:
            raise
        except Exception as exc:
            if _looks_like_rate_limit(exc):
                raise GarminRateLimitError(
                    "Garmin login is temporarily rate-limited. "
                    "No cached Garmin session is available yet, so the app cannot "
                    "refresh activities until Garmin clears the block."
                ) from exc
            if isinstance(
                exc,
                (GarminConnectAuthenticationError, GarminConnectConnectionError),
            ):
                raise GarminClientError(f"Failed to connect to Garmin: {exc}") from exc
            logger.exception("Failed to connect.")
            raise

    def _login(self) -> Garmin:
        tokenstore = self._tokenstore_arg()
        if tokenstore is not None:
            with _suppress_garmin_library_tracebacks():
                try:
                    client = Garmin()
                    client.login(tokenstore)
                    return client
                except (
                    FileNotFoundError,
                    GarminConnectAuthenticationError,
                    GarminConnectConnectionError,
                ):
                    logger.info(
                        "Saved Garmin session at %s could not be restored. "
                        "Falling back to credential login.",
                        self.tokenstore_path,
                    )

        client = Garmin(
            email=self.email,
            password=self.password,
            prompt_mfa=self.prompt_mfa,
        )
        with _suppress_garmin_library_tracebacks():
            if tokenstore is None:
                client.login()
            else:
                client.login(tokenstore)
        return client

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
            except Exception as exc:
                if _looks_like_rate_limit(exc):
                    raise GarminRateLimitError(
                        "Garmin rate-limited activity history requests. "
                        "The sync was stopped before processing partial data."
                    ) from exc
                if isinstance(
                    exc,
                    (GarminConnectAuthenticationError, GarminConnectConnectionError),
                ):
                    raise GarminClientError(
                        f"Failed to fetch Garmin activities for {window_start} - {window_end}: {exc}"
                    ) from exc
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

    def _tokenstore_arg(self) -> str | None:
        if self.tokenstore_path is None:
            return None

        self.tokenstore_path.mkdir(parents=True, exist_ok=True)
        return str(self.tokenstore_path)

    def _default_prompt_mfa(self) -> str:
        if not sys.stdin.isatty():
            raise GarminMFARequiredError(
                "Garmin requested MFA. Configure an MFA callback or run the sync "
                "from an interactive terminal."
            )

        code = input("Garmin MFA code: ").strip()
        if not code:
            raise GarminMFARequiredError(
                "Garmin requested MFA, but no MFA code was provided."
            )

        return code


def _looks_like_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, GarminConnectTooManyRequestsError):
        return True

    error_message = str(exc).lower()
    return "429" in error_message or "too many requests" in error_message


@contextmanager
def _suppress_garmin_library_tracebacks():
    garminconnect_logger = logging.getLogger("garminconnect")
    previous_garminconnect_level = garminconnect_logger.level
    garminconnect_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        garminconnect_logger.setLevel(previous_garminconnect_level)

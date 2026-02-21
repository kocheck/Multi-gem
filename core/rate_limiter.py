"""Rate limiting and retry logic with exponential backoff."""

from __future__ import annotations

import time
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """
    Simple token-bucket-style rate limiter that enforces a minimum delay
    between requests and implements exponential backoff on 429 / transient errors.

    Args:
        delay_between_requests: Minimum seconds to wait between requests.
        max_retries: Maximum number of retry attempts per request.
        base_backoff: Initial backoff in seconds (doubles each retry).
        max_backoff: Maximum backoff cap in seconds.
    """

    def __init__(
        self,
        delay_between_requests: float = 3.0,
        max_retries: int = 3,
        base_backoff: float = 2.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.delay_between_requests = delay_between_requests
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self._last_request_time: float = 0.0

    def wait(self) -> None:
        """Block until the minimum inter-request delay has elapsed."""
        elapsed = time.monotonic() - self._last_request_time
        wait_time = self.delay_between_requests - elapsed
        if wait_time > 0:
            logger.debug("Rate limiter sleeping %.2fs", wait_time)
            time.sleep(wait_time)

    def record_request(self) -> None:
        """Record that a request was just made."""
        self._last_request_time = time.monotonic()

    def call_with_retry(self, fn: Callable[[], T]) -> T:
        """
        Call fn(), retrying up to max_retries times on retryable errors.

        Retries on:
        - google.api_core.exceptions.ResourceExhausted (429 rate limit)
        - google.api_core.exceptions.ServiceUnavailable (503)
        - google.api_core.exceptions.InternalServerError (500)
        - ConnectionError, TimeoutError

        Non-retryable errors (e.g., InvalidArgument, PermissionDenied) are
        re-raised immediately.

        Args:
            fn: Zero-argument callable that makes the API request.

        Returns:
            The return value of fn() on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.wait()
            try:
                self.record_request()
                return fn()
            except Exception as exc:
                if not _is_retryable(exc):
                    raise

                last_exc = exc
                if attempt < self.max_retries:
                    backoff = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                    logger.warning(
                        "Retryable error on attempt %d/%d: %s. Backing off %.1fs.",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)

        raise last_exc  # type: ignore[misc]


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is transient and worth retrying."""
    # Check for Google API exceptions by class name to avoid hard-importing google.api_core
    exc_type = type(exc).__name__
    retryable_types = {
        "ResourceExhausted",  # 429 Too Many Requests
        "ServiceUnavailable",  # 503
        "InternalServerError",  # 500
        "DeadlineExceeded",     # timeout
        "Aborted",
        "ServerError",
    }
    if exc_type in retryable_types:
        return True

    # Also retry on generic network errors
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    # Check for '429' in string representation (covers httpx / requests errors)
    if "429" in str(exc) or "rate limit" in str(exc).lower():
        return True

    return False

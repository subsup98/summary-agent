from __future__ import annotations

import random
import time
from typing import Callable, TypeVar
from urllib import error as urllib_error

T = TypeVar("T")

# Errors that warrant a retry
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0   # seconds
DEFAULT_MAX_DELAY = 60.0   # seconds


def _jittered_backoff(attempt: int, base: float, cap: float) -> float:
    """Full jitter exponential backoff: uniform in [0, min(cap, base * 2^attempt)]."""
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(0, ceiling)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    context: str = "",
) -> T:
    """Call *fn* and retry on transient API/network errors.

    Handles:
    - urllib.error.HTTPError with 429 / 5xx status codes
    - urllib.error.URLError (network errors, DNS failures)
    - TimeoutError / ConnectionError / OSError (socket-level timeouts)
    - openai.RateLimitError / APITimeoutError / APIConnectionError / InternalServerError
      (imported lazily so the function works even when the openai package is absent)
    """
    last_error: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except urllib_error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_CODES:
                raise
            retry_after = _parse_retry_after(exc)
            last_error = exc
            if attempt >= max_retries:
                break
            wait = retry_after if retry_after is not None else _jittered_backoff(attempt, base_delay, max_delay)
            _log_retry(context, attempt, max_retries, exc.code, wait)
            time.sleep(wait)

        except urllib_error.URLError as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait = _jittered_backoff(attempt, base_delay, max_delay)
            _log_retry(context, attempt, max_retries, "URLError", wait)
            time.sleep(wait)

        except (TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait = _jittered_backoff(attempt, base_delay, max_delay)
            _log_retry(context, attempt, max_retries, type(exc).__name__, wait)
            time.sleep(wait)

        except Exception as exc:
            if not _is_retryable_openai_error(exc):
                raise
            retry_after = _parse_retry_after_openai(exc)
            last_error = exc
            if attempt >= max_retries:
                break
            wait = retry_after if retry_after is not None else _jittered_backoff(attempt, base_delay, max_delay)
            _log_retry(context, attempt, max_retries, type(exc).__name__, wait)
            time.sleep(wait)

    assert last_error is not None
    raise last_error


def _parse_retry_after(exc: urllib_error.HTTPError) -> float | None:
    try:
        value = exc.headers.get("Retry-After")
        if value:
            return float(value)
    except Exception:
        pass
    return None


def _parse_retry_after_openai(exc: Exception) -> float | None:
    """Extract Retry-After from openai errors that expose response headers."""
    try:
        headers = getattr(exc, "response", None) and getattr(exc.response, "headers", None)  # type: ignore[union-attr]
        if headers:
            value = headers.get("Retry-After") or headers.get("retry-after")
            if value:
                return float(value)
    except Exception:
        pass
    return None


def _is_retryable_openai_error(exc: Exception) -> bool:
    """Return True if *exc* is an openai transient error worth retrying."""
    try:
        import openai  # type: ignore
        return isinstance(
            exc,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        )
    except ImportError:
        # openai not installed — check by class name as a fallback
        type_name = type(exc).__name__
        return type_name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}


def _log_retry(context: str, attempt: int, max_retries: int, reason: object, wait: float) -> None:
    label = f"[{context}] " if context else ""
    print(
        f"{label}Retry {attempt + 1}/{max_retries} after {reason} - waiting {wait:.1f}s",
        flush=True,
    )

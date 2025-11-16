"""API logging helpers for telemetry."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional


logger = logging.getLogger("tom_api")


@contextmanager
def log_request(action: str, *, session_id: Optional[str] = None, extra: dict | None = None) -> Iterator[None]:
    start = time.perf_counter()
    payload = {"action": action}
    if session_id:
        payload["session_id"] = session_id
    if extra:
        payload.update(extra)

    logger.info("request.start %s", payload)
    try:
        yield
        payload["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
        logger.info("request.done %s", payload)
    except Exception:
        payload["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("request.error %s", payload)
        raise

"""Configuration helpers for the FastAPI service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from core.config import DEFAULT_MODEL_MAX_CONTEXT, DEFAULT_GC_FREQUENCY


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class ApiSettings:
    """Runtime configuration for the API service."""

    model_path: Path
    cache_path: str | None
    max_context_tokens: int
    enable_cache: bool
    prewarm_cache: bool
    auto_gc: bool
    gc_frequency: int

    @classmethod
    def from_env(cls) -> "ApiSettings":
        base_model = os.getenv("TOM_MODEL_PATH", "./Qwen3-4B-Thinking-2507-8bit")
        cache_path = os.getenv("TOM_CACHE_PATH")
        max_context = _env_int("TOM_MAX_CONTEXT", DEFAULT_MODEL_MAX_CONTEXT)
        enable_cache = _env_bool("TOM_ENABLE_CACHE", True)
        prewarm_cache = _env_bool("TOM_PREWARM_CACHE", True)
        auto_gc = _env_bool("TOM_AUTO_GC", True)
        gc_frequency = _env_int("TOM_GC_FREQUENCY", DEFAULT_GC_FREQUENCY)

        return cls(
            model_path=Path(base_model),
            cache_path=cache_path,
            max_context_tokens=max_context,
            enable_cache=enable_cache,
            prewarm_cache=prewarm_cache,
            auto_gc=auto_gc,
            gc_frequency=gc_frequency,
        )

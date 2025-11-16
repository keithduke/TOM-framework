#!/usr/bin/env python3
"""API contract tests – use TestClient and stubbed ModelManager."""

import os
import sys
import types

import pytest
from fastapi.testclient import TestClient


class StubModelManager:
    """Lightweight stand-in for ModelManager to avoid loading MLX during tests."""

    def __init__(self, model_path, context_manager, **kwargs):
        self.context_manager = context_manager
        self.enable_cache = False
        self.cache_path = ""
        self.tokenizer = None
        self.model = None
        self._tool_pass = True

    def load_model(self):
        self.model = object()

    def generate_response(self, include_tools: bool = False) -> tuple[str, str]:
        if include_tools and self._tool_pass:
            self._tool_pass = False
            return (
                "checking time",
                "Let me call the clock.\n<tool_call>\n"
                '{"name": "get_datetime", "arguments": {}}\n</tool_call>',
            )
        return ("", "Here is the current time from the tool.")

    def run_gc(self):
        return None


@pytest.fixture
def api_client(monkeypatch) -> TestClient:
    """Provide a TestClient with ModelManager stubbed out."""
    os.environ.setdefault("TOM_MODEL_PATH", "./Qwen3-0.6B-4bit")
    os.environ.setdefault("TOM_ENABLE_CACHE", "0")
    os.environ.setdefault("TOM_PREWARM_CACHE", "0")

    original_model_manager = sys.modules.get("core.model_manager")
    stub_module = types.ModuleType("core.model_manager")
    stub_module.ModelManager = StubModelManager
    sys.modules["core.model_manager"] = stub_module

    # Ensure runtime/main get re-imported with stubbed ModelManager
    sys.modules.pop("services.api.runtime", None)
    sys.modules.pop("services.api.main", None)

    from services.api.main import app, runtime

    runtime.sessions.clear()
    runtime._model_manager = None  # type: ignore[attr-defined]

    try:
        with TestClient(app) as client:
            yield client
    finally:
        runtime.sessions.clear()
        runtime._model_manager = None  # type: ignore[attr-defined]

        if original_model_manager is not None:
            sys.modules["core.model_manager"] = original_model_manager
        else:
            sys.modules.pop("core.model_manager", None)


def test_health_endpoint(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "initializing"}
    assert "model_path" in payload


def test_session_and_chat_flow(api_client: TestClient):
    create = api_client.post(
        "/sessions", json={"max_context_tokens": 1024, "system_prompt": "You are test."}
    )
    assert create.status_code == 201
    session = create.json()
    session_id = session["session_id"]

    chat = api_client.post(
        f"/sessions/{session_id}/chat", json={"content": "What time is it?"}
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert payload["session"]["session_id"] == session_id
    assert payload["tool_calls"], "Tool call results should be returned from stub"
    assert "response" in payload

    delete = api_client.delete(f"/sessions/{session_id}")
    assert delete.status_code == 204


def test_api_key_enforcement(monkeypatch, api_client: TestClient):
    from services.api import auth

    monkeypatch.setattr(auth, "API_KEY", "secret")

    resp = api_client.get("/health")
    assert resp.status_code == 401

    resp = api_client.get("/health", headers={"X-TOM-API-Key": "secret"})
    assert resp.status_code == 200

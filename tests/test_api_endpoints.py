#!/usr/bin/env python3
"""API contract tests – use TestClient and stubbed ModelManager."""

import json
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

    def stream_response(self, include_tools: bool = False):
        thinking = "checking"
        if include_tools and self._tool_pass:
            content = 'Let me check.\n<tool_call>{"name": "get_datetime", "arguments": {}}</tool_call>'
        else:
            content = "Here is the current time from the tool."
        if include_tools and self._tool_pass:
            self._tool_pass = False
            yield {"type": "thinking", "delta": thinking, "text": thinking, "complete": True}
            yield {"type": "content", "delta": content, "text": content, "complete": False}
        else:
            yield {"type": "thinking", "delta": thinking, "text": thinking, "complete": True}
            yield {"type": "content", "delta": content, "text": content, "complete": False}
        yield {"type": "done", "thinking": thinking, "content": content}

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

    session_state = api_client.get(f"/sessions/{session_id}")
    assert session_state.status_code == 200
    history = session_state.json()["messages"]
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"], "assistant response should be stored in context"

    delete = api_client.delete(f"/sessions/{session_id}")
    assert delete.status_code == 204


def test_sequential_chat_persists_context(api_client: TestClient):
    create = api_client.post("/sessions", json={})
    session_id = create.json()["session_id"]

    first = api_client.post(f"/sessions/{session_id}/chat", json={"content": "Hello"})
    assert first.status_code == 200

    second = api_client.post(f"/sessions/{session_id}/chat", json={"content": "What time is it?"})
    assert second.status_code == 200
    payload = second.json()
    assert payload["session"]["messages"][-1]["role"] == "assistant"
    assert "current time" in payload["response"]


def test_local_only_mode(monkeypatch, api_client: TestClient):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body and "model_path" in body


def test_chat_stream_endpoint(api_client: TestClient):
    create = api_client.post("/sessions", json={})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    with api_client.stream(
        "POST",
        f"/sessions/{session_id}/chat/stream",
        json={"content": "What time?", "run_tools": True},
    ) as resp:
        assert resp.status_code == 200
        final_payload = _extract_final_event(resp)

    assert final_payload is not None
    assert final_payload["session"]["session_id"] == session_id
    assert "response" in final_payload


def _extract_final_event(response) -> dict | None:
    event = None
    data_lines: list[str] = []
    for chunk in response.iter_lines():
        if chunk is None:
            continue
        line = chunk.strip()
        if not line:
            if event == "final" and data_lines:
                raw = "\n".join(data_lines)
                return json.loads(raw)
            event = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if event == "final" and data_lines:
        return json.loads("\n".join(data_lines))
    return None

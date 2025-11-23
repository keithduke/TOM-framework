"""Session + model runtime orchestration for the API."""

from __future__ import annotations

import threading
from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from core.context_manager import ContextManager
from core.model_manager import ModelManager
from core.tools import execute_tool_call, extract_tool_calls, strip_tool_calls
from core.config import (
    DEFAULT_SYSTEM_PROMPT,
    MAX_TOOL_RESULT_TOKENS,
    TOOL_RESULT_CONTEXT_RATIO,
)

from .config import ApiSettings
from .schemas import ToolCallResult

logger = logging.getLogger("tom_api")


@dataclass
class SessionData:
    """Represents a chat session."""

    session_id: str
    context: ContextManager


@dataclass
class ChatTurn:
    """Result of a single user -> assistant exchange."""

    thinking: str
    response: str
    tool_calls: List[ToolCallResult]
    intermediate_response: str


class ApiRuntime:
    """Manages sessions and a shared ModelManager instance."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.sessions: Dict[str, SessionData] = {}
        self._model_manager: Optional[ModelManager] = None
        self._lock = threading.Lock()
        tool_result_tokens = min(
            int(settings.max_context_tokens * TOOL_RESULT_CONTEXT_RATIO),
            MAX_TOOL_RESULT_TOKENS,
        )
        self._max_tool_result_chars = tool_result_tokens * 4

    @property
    def loaded(self) -> bool:
        return self._model_manager is not None and self._model_manager.model is not None

    def startup(self) -> None:
        """Load the shared model."""
        if self.loaded:
            return

        # Placeholder context used solely to satisfy the constructor
        bootstrap_context = ContextManager(
            max_context_tokens=self.settings.max_context_tokens
        )
        bootstrap_context.system_prompt = DEFAULT_SYSTEM_PROMPT

        manager = ModelManager(
            model_path=self.settings.model_path,
            context_manager=bootstrap_context,
            cache_path=self.settings.cache_path,
            enable_cache=self.settings.enable_cache,
            prewarm=self.settings.prewarm_cache,
            auto_gc=self.settings.auto_gc,
            gc_frequency=self.settings.gc_frequency,
        )
        manager.load_model()
        self._model_manager = manager

    def shutdown(self) -> None:
        """Release model resources and clear session state."""
        if self._model_manager:
            try:
                self._model_manager.run_gc()
            except Exception:
                logger.warning("Failed to run GC during shutdown", exc_info=True)
        self._model_manager = None
        self.sessions.clear()

    def create_session(
        self,
        *,
        max_context_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> SessionData:
        """Create a new chat session."""
        max_tokens = max_context_tokens or self.settings.max_context_tokens
        context = ContextManager(max_context_tokens=max_tokens)
        context.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if self._model_manager and self._model_manager.tokenizer:
            context.set_tokenizer(self._model_manager.tokenizer)

        session = SessionData(session_id=uuid4().hex, context=context)
        self.sessions[session.session_id] = session
        return session

    def delete_session(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None

    def get_session(self, session_id: str) -> SessionData:
        if session_id not in self.sessions:
            raise KeyError("Session not found")
        return self.sessions[session_id]

    def list_sessions(self) -> List[SessionData]:
        return list(self.sessions.values())

    def run_chat_turn(
        self, session: SessionData, user_message: str, run_tools: bool = True
    ) -> ChatTurn:
        """Append a user message, run the model, and handle tool calls."""
        final_event: dict[str, Any] | None = None
        for event in self.chat_event_stream(session, user_message, run_tools=run_tools):
            if event["event"] == "final":
                final_event = event["data"]

        if final_event is None:
            raise RuntimeError("Chat stream ended without final payload")

        return ChatTurn(
            thinking=final_event.get("thinking", ""),
            response=final_event.get("response", ""),
            tool_calls=[
                ToolCallResult(
                    name=call.get("name", "unknown"),
                    arguments=call.get("arguments") or {},
                    output=call.get("output", ""),
                )
                for call in final_event.get("tool_calls", [])
            ],
            intermediate_response=final_event.get("intermediate_response", ""),
        )

    def chat_event_stream(
        self, session: SessionData, user_message: str, run_tools: bool = True
    ) -> Iterator[dict[str, Any]]:
        """Yield structured events describing the chat workflow for streaming clients."""
        self._ensure_ready()
        session.context.add_message("user", user_message)

        def stream_generation(include_tools: bool, hide_tool_markup: bool):
            visible_buffer = ""
            thinking_buffer = ""

            for chunk in self._stream_response(session.context, include_tools=include_tools):
                chunk_type = chunk.get("type")
                if chunk_type == "thinking":
                    delta = chunk.get("delta", "")
                    thinking_buffer = chunk.get("text", thinking_buffer)
                    if delta:
                        yield {
                            "event": "thinking",
                            "data": {
                                "content": thinking_buffer,
                                "delta": delta,
                                "complete": chunk.get("complete", False),
                            },
                        }
                elif chunk_type == "content":
                    delta = chunk.get("delta", "")
                    if hide_tool_markup and "<tool_call>" in delta:
                        delta = delta.split("<tool_call>", 1)[0]
                    if delta:
                        visible_buffer += delta
                        yield {
                            "event": "assistant",
                            "data": {
                                "content": visible_buffer,
                                "delta": delta,
                                "final": False,
                            },
                        }
                elif chunk_type == "done":
                    thinking_final = chunk.get("thinking", thinking_buffer)
                    content_final = chunk.get("content", visible_buffer)
                    return thinking_final, content_final, visible_buffer
                elif chunk_type == "error":
                    yield {
                        "event": "error",
                        "data": {"message": chunk.get("text", "Streaming error")},
                    }
                    raise RuntimeError("Streaming failed")

            raise RuntimeError("Generation ended unexpectedly")

        try:
            thinking_first, first_response, visible_first = yield from stream_generation(
                include_tools=run_tools, hide_tool_markup=run_tools
            )
        except RuntimeError:
            return

        tool_calls_raw = extract_tool_calls(first_response) if run_tools else []
        assistant_visible = (
            first_response.strip()
            if tool_calls_raw
            else (visible_first.strip() or strip_tool_calls(first_response).strip())
        )

        thinking_parts: List[str] = [thinking_first] if thinking_first else []
        final_response = ""
        intermediate_response = first_response

        if tool_calls_raw:
            session.context.add_message("assistant", first_response)
            tool_results: List[dict[str, Any]] = []
            for call in tool_calls_raw:
                yield {"event": "tool_call", "data": call}
                result = execute_tool_call(call)
                truncated = result
                if len(truncated) > self._max_tool_result_chars:
                    truncated = truncated[: self._max_tool_result_chars] + "…"

                tool_msg = f"Tool: {call.get('name', 'unknown')}\nResult: {truncated}"
                session.context.add_message("tool", tool_msg)

                payload = {
                    "name": call.get("name", "unknown"),
                    "arguments": call.get("arguments") or {},
                    "output": result,
                }
                tool_results.append(payload)
                yield {"event": "tool_result", "data": payload}

            try:
                follow_thinking, follow_content, follow_visible = yield from stream_generation(
                    include_tools=False, hide_tool_markup=False
                )
            except RuntimeError:
                return

            if follow_thinking:
                thinking_parts.append(follow_thinking)

            session.context.add_message("assistant", follow_content)
            final_response = follow_content
            yield {
                "event": "assistant",
                "data": {"content": follow_visible or follow_content, "delta": "", "final": True},
            }
        else:
            tool_results = []
            final_visible = assistant_visible or visible_first.strip()
            if final_visible:
                session.context.add_message("assistant", final_visible)
                final_response = final_visible
                yield {
                    "event": "assistant",
                    "data": {"content": final_visible, "delta": "", "final": True},
                }

        combined_thinking = "\n".join(piece for piece in thinking_parts if piece).strip()
        yield {
            "event": "final",
            "data": {
                "thinking": combined_thinking,
                "response": final_response or assistant_visible,
                "tool_calls": tool_results,
                "session": self._session_payload(session),
                "intermediate_response": intermediate_response,
                "streaming": True,
            },
        }

    def _generate(
        self, context: ContextManager, *, include_tools: bool
    ) -> tuple[str, str]:
        """Borrow the shared ModelManager, swapping contexts safely."""
        self._ensure_ready()
        assert self._model_manager is not None  # for mypy

        with self._lock:
            original_context = self._model_manager.context_manager
            self._model_manager.context_manager = context
            try:
                thinking, content = self._model_manager.generate_response(
                    include_tools=include_tools
                )
                return thinking, content
            finally:
                self._model_manager.context_manager = original_context

    def _stream_response(
        self, context: ContextManager, *, include_tools: bool
    ) -> Iterator[dict[str, Any]]:
        """Stream tokens while temporarily swapping in the session context."""
        self._ensure_ready()
        assert self._model_manager is not None

        with self._lock:
            original_context = self._model_manager.context_manager
            self._model_manager.context_manager = context
            try:
                yield from self._model_manager.stream_response(include_tools=include_tools)
            finally:
                self._model_manager.context_manager = original_context

    def _ensure_ready(self) -> None:
        if not self.loaded:
            raise RuntimeError("Model not loaded - call startup() first")

    @staticmethod
    def _session_payload(session: SessionData) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "system_prompt": session.context.system_prompt,
            "max_context_tokens": session.context.max_context_tokens,
            "messages": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in session.context.messages
            ],
        }

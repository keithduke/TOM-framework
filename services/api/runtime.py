"""Session + model runtime orchestration for the API."""

from __future__ import annotations

import threading
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
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

    def __init__(self, settings: ApiSettings):
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
        self._ensure_ready()
        session.context.add_message("user", user_message)

        thinking, first_response = self._generate(
            session.context, include_tools=run_tools
        )

        tool_calls_raw = extract_tool_calls(first_response) if run_tools else []
        assistant_visible = (
            first_response if tool_calls_raw else strip_tool_calls(first_response)
        ).strip()

        if assistant_visible:
            session.context.add_message("assistant", assistant_visible)

        if not tool_calls_raw:
            return ChatTurn(
                thinking=thinking,
                response=assistant_visible,
                tool_calls=[],
                intermediate_response=first_response,
            )

        tool_results: List[ToolCallResult] = []
        for call in tool_calls_raw:
            result = execute_tool_call(call)
            truncated = result
            if len(truncated) > self._max_tool_result_chars:
                truncated = truncated[: self._max_tool_result_chars] + "…"

            tool_msg = f"Tool: {call.get('name', 'unknown')}\nResult: {truncated}"
            session.context.add_message("tool", tool_msg)

            tool_results.append(
                ToolCallResult(
                    name=call.get("name", "unknown"),
                    arguments=call.get("arguments") or {},
                    output=result,
                )
            )

        follow_thinking, follow_content = self._generate(
            session.context, include_tools=False
        )
        session.context.add_message("assistant", follow_content)

        combined_thinking = "\n".join(
            piece for piece in [thinking, follow_thinking] if piece
        ).strip()

        return ChatTurn(
            thinking=combined_thinking,
            response=follow_content,
            tool_calls=tool_results,
            intermediate_response=first_response,
        )

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

    def _ensure_ready(self) -> None:
        if not self.loaded:
            raise RuntimeError("Model not loaded - call startup() first")

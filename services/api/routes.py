"""FastAPI routers for session + chat management."""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from .runtime import ApiRuntime, SessionData
from .schemas import (
    ChatRequest,
    ChatResponse,
    MessagePayload,
    MessageRequest,
    SessionCreateRequest,
    SessionState,
)
from .logging import log_request


def build_router(runtime: ApiRuntime) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if runtime.loaded else "initializing",
            "model_path": str(runtime.settings.model_path),
            "sessions": len(runtime.sessions),
        }

    @router.post("/sessions", response_model=SessionState, status_code=status.HTTP_201_CREATED)
    def create_session(payload: SessionCreateRequest) -> SessionState:
        with log_request("create_session"):
            session = runtime.create_session(
                max_context_tokens=payload.max_context_tokens,
                system_prompt=payload.system_prompt,
            )
            return _serialize_session(session)

    @router.get("/sessions/{session_id}", response_model=SessionState)
    def get_session(session_id: str) -> SessionState:
        with log_request("get_session", session_id=session_id):
            try:
                session = runtime.get_session(session_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Session not found") from None
            return _serialize_session(session)

    @router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str) -> None:
        with log_request("delete_session", session_id=session_id):
            if not runtime.delete_session(session_id):
                raise HTTPException(status_code=404, detail="Session not found")

    @router.post("/sessions/{session_id}/messages", response_model=SessionState)
    def append_message(session_id: str, payload: MessageRequest) -> SessionState:
        with log_request("append_message", session_id=session_id):
            try:
                session = runtime.get_session(session_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Session not found") from None

            session.context.add_message(payload.message.role, payload.message.content)
            return _serialize_session(session)

    @router.post(
        "/sessions/{session_id}/chat",
        response_model=ChatResponse,
    )
    def run_chat(session_id: str, payload: ChatRequest) -> ChatResponse:
        with log_request(
            "chat",
            session_id=session_id,
            extra={"run_tools": payload.run_tools},
        ):
            try:
                session = runtime.get_session(session_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Session not found") from None

            turn = runtime.run_chat_turn(
                session, user_message=payload.content, run_tools=payload.run_tools
            )

            return ChatResponse(
                session=_serialize_session(session),
                thinking=turn.thinking,
                response=turn.response,
                tool_calls=turn.tool_calls,
            )

    @router.post("/sessions/{session_id}/chat/stream")
    def stream_chat(session_id: str, payload: ChatRequest):
        with log_request(
            "chat_stream",
            session_id=session_id,
            extra={"run_tools": payload.run_tools},
        ):
            try:
                session = runtime.get_session(session_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Session not found") from None

            def event_source() -> Iterator[str]:
                for event in runtime.chat_event_stream(
                    session, user_message=payload.content, run_tools=payload.run_tools
                ):
                    yield _format_sse(event["event"], event["data"])

            return StreamingResponse(event_source(), media_type="text/event-stream")

    return router


def _serialize_session(session: SessionData) -> SessionState:
    return SessionState(
        session_id=session.session_id,
        system_prompt=session.context.system_prompt,
        max_context_tokens=session.context.max_context_tokens,
        messages=[MessagePayload(role=msg["role"], content=msg["content"]) for msg in session.context.messages],
    )


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

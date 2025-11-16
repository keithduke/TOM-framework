"""FastAPI routers for session + chat management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .runtime import ApiRuntime, SessionData
from .schemas import (
    ChatRequest,
    ChatResponse,
    MessageRequest,
    SessionCreateRequest,
    SessionState,
)


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
        session = runtime.create_session(
            max_context_tokens=payload.max_context_tokens,
            system_prompt=payload.system_prompt,
        )
        return _serialize_session(session)

    @router.get("/sessions/{session_id}", response_model=SessionState)
    def get_session(session_id: str) -> SessionState:
        try:
            session = runtime.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found") from None
        return _serialize_session(session)

    @router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str) -> None:
        if not runtime.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    @router.post("/sessions/{session_id}/messages", response_model=SessionState)
    def append_message(session_id: str, payload: MessageRequest) -> SessionState:
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

    return router


def _serialize_session(session: SessionData) -> SessionState:
    return SessionState(
        session_id=session.session_id,
        system_prompt=session.context.system_prompt,
        messages=[{"role": msg["role"], "content": msg["content"]} for msg in session.context.messages],
    )

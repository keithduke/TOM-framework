"""FastAPI entrypoint for T.O.M. core services."""

from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.config import DEFAULT_MODEL_MAX_CONTEXT, DEFAULT_SYSTEM_PROMPT
from core.context_manager import ContextManager

app = FastAPI(title="T.O.M. API", version="0.1.0" )


class SessionCreateRequest(BaseModel):
    max_context_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


class MessagePayload(BaseModel):
    role: str
    content: str


class MessageRequest(BaseModel):
    message: MessagePayload


class SessionState(BaseModel):
    session_id: str
    system_prompt: str
    messages: List[MessagePayload]


_sessions: Dict[str, ContextManager] = {}


def _serialize(session_id: str) -> SessionState:
    ctx = _sessions[session_id]
    payloads = [MessagePayload(**m) for m in ctx.messages]
    return SessionState(session_id=session_id, system_prompt=ctx.system_prompt, messages=payloads)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionState)
def create_session(payload: SessionCreateRequest) -> SessionState:
    session_id = uuid4().hex
    max_tokens = payload.max_context_tokens or DEFAULT_MODEL_MAX_CONTEXT
    ctx = ContextManager(max_context_tokens=max_tokens)
    if payload.system_prompt:
        ctx.system_prompt = payload.system_prompt
    else:
        ctx.system_prompt = DEFAULT_SYSTEM_PROMPT
    _sessions[session_id] = ctx
    return _serialize(session_id)


@app.post("/sessions/{session_id}/messages", response_model=SessionState)
def append_message(session_id: str, payload: MessageRequest) -> SessionState:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    ctx = _sessions[session_id]
    ctx.add_message(payload.message.role, payload.message.content)
    return _serialize(session_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    if session_id in _sessions:
        _sessions.pop(session_id)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

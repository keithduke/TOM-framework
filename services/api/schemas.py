"""Pydantic schemas shared across the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

ALLOWED_ROLES = {"user", "assistant", "tool", "function", "system"}


class SessionCreateRequest(BaseModel):
    max_context_tokens: Optional[int] = Field(
        default=None, description="Override default context window."
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt for this session."
    )


class MessagePayload(BaseModel):
    role: str
    content: str

    @validator("role")
    def validate_role(cls, value: str) -> str:
        if value not in ALLOWED_ROLES:
            raise ValueError(f"Invalid role '{value}'")
        return value


class MessageRequest(BaseModel):
    message: MessagePayload


class SessionState(BaseModel):
    session_id: str
    system_prompt: str
    messages: List[MessagePayload]


class ToolCallResult(BaseModel):
    name: str
    arguments: Dict[str, Any]
    output: str


class ChatRequest(BaseModel):
    content: str
    run_tools: bool = Field(
        default=True,
        description="Allow the model to call tools automatically before responding.",
    )


class ChatResponse(BaseModel):
    session: SessionState
    thinking: str
    response: str
    tool_calls: List[ToolCallResult]

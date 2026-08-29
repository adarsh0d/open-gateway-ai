"""open-gateway-ai — a learning re-implementation of what LiteLLM does, with httpx.

Step 2: accept and validate the OpenAI chat-completions request shape.

The endpoint does nothing but parse the body and echo back what it parsed. The
point is to see the *contract* — this is the exact JSON a caller sends to
LiteLLM (and to OpenAI). Pydantic rejects anything malformed with a 422 before
our handler runs.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="open-gateway-ai", version="0.2.0")


class ChatMessage(BaseModel):
    # extra="allow" keeps fields we don't model (tool_calls, tool_call_id, ...)
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """The subset of OpenAI's /v1/chat/completions body we validate explicitly.

    `extra="allow"` lets any other OpenAI param (tools, seed, response_format,
    ...) pass through untouched — the same permissiveness LiteLLM has.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, gt=0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    # No provider call yet. Just prove the body validated and show what we got.
    # exclude_none => only the fields the caller actually sent.
    return {"parsed": request.model_dump(exclude_none=True)}

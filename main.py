"""open-gateway-ai — a learning re-implementation of what LiteLLM does, with httpx.

Step 4: route to a provider based on the `model` field.

  gpt-4o-mini       -> OpenAI
  claude-3-5-sonnet -> Anthropic
  anything else      -> 400

The Anthropic branch sets the correct auth headers and endpoint URL, but
still forwards the OpenAI-shaped body UNCHANGED. That works for a single
user message and 400s the moment there is a system message — Anthropic's
request schema is different. The next commit adds the translation.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

OPENAI_MODELS = {"gpt-4o-mini"}
ANTHROPIC_MODELS = {"claude-3-5-sonnet"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=60.0) as client:
        app.state.http = client
        yield


app = FastAPI(title="open-gateway-ai", version="0.4.0", lifespan=lifespan)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
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
async def chat_completions(request: ChatCompletionRequest) -> Response:
    client: httpx.AsyncClient = app.state.http

    if request.model in OPENAI_MODELS:
        url = OPENAI_URL
        # OpenAI: Authorization: Bearer <key>
        headers = {"authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = request.model_dump(exclude_none=True)
    elif request.model in ANTHROPIC_MODELS:
        url = ANTHROPIC_URL
        # Anthropic: x-api-key + the REQUIRED anthropic-version header
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        # NAIVE: OpenAI body forwarded as-is. Broken for system messages.
        payload = request.model_dump(exclude_none=True)
    else:
        raise HTTPException(status_code=400, detail=f"Unrecognised model: {request.model!r}")

    headers["content-type"] = "application/json"
    upstream = await client.post(url, json=payload, headers=headers)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )

"""open-gateway-ai — a learning re-implementation of what LiteLLM does, with httpx.

Step 3: forward the request to OpenAI and return its raw response.

Single provider, hardcoded. This is the smallest thing that earns the name
"gateway": inject the auth header, POST the JSON, proxy the reply back. For
OpenAI there is no schema translation at all — our body already *is* the
OpenAI schema. LiteLLM's OpenAI path is essentially this (plus retries and
error mapping, added later).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One connection pool for the whole process, not one per request.
    async with httpx.AsyncClient(timeout=60.0) as client:
        app.state.http = client
        yield


app = FastAPI(title="open-gateway-ai", version="0.3.0", lifespan=lifespan)


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

    # OpenAI accepts our body unchanged. The only work is auth.
    headers = {
        "authorization": f"Bearer {OPENAI_API_KEY}",
        "content-type": "application/json",
    }
    payload = request.model_dump(exclude_none=True)

    upstream = await client.post(OPENAI_URL, json=payload, headers=headers)

    # Return OpenAI's response body + status code untouched.
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )

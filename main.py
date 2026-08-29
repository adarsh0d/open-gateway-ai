"""open-gateway-ai — a learning re-implementation of what LiteLLM does, with httpx.

Step 5: translate the OpenAI payload into Anthropic's Messages format.

This is the core of what LiteLLM does. The two schemas disagree on more than
field names — the biggest one being how the system prompt is carried:

    OpenAI    : a {"role": "system"} message inside messages[]
    Anthropic : a top-level `system` field; "system" is not a valid role

See translate_openai_to_anthropic() for the full list.
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

# Anthropic requires max_tokens; OpenAI treats it as optional.
DEFAULT_MAX_TOKENS = 4096

OPENAI_MODELS = {"gpt-4o-mini"}
ANTHROPIC_MODELS = {"claude-3-5-sonnet"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=60.0) as client:
        app.state.http = client
        yield


app = FastAPI(title="open-gateway-ai", version="0.5.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Pydantic request models — the OpenAI /v1/chat/completions schema            #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# OpenAI -> Anthropic payload translation                                     #
# --------------------------------------------------------------------------- #


def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    """Flatten OpenAI message content to plain text.

    OpenAI `content` is either a string or a list of typed parts
    ({"type": "text", ...}, {"type": "image_url", ...}). This shell only
    translates text; anything else is a 400.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        if part.get("type") == "text":
            chunks.append(part.get("text", ""))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot translate content part of type {part.get('type')!r} "
                "to the Anthropic format",
            )
    return "\n".join(chunks)


def translate_openai_to_anthropic(req: ChatCompletionRequest) -> dict[str, Any]:
    """Map an OpenAI chat-completions body onto an Anthropic Messages body.

    | OpenAI                                   | Anthropic                                   |
    |------------------------------------------|---------------------------------------------|
    | system prompt is a role="system"         | `system` is a TOP-LEVEL field; "system" is  |
    | message inside messages[], any number    | not a valid role inside messages            |
    | max_tokens optional                      | max_tokens REQUIRED                          |
    | stop (string or list)                    | stop_sequences (list only)                  |
    | roles: system/developer/user/assistant   | user/assistant only, must start with user   |
    | temperature 0.0 - 2.0                    | temperature 0.0 - 1.0                        |
    | n, presence_penalty, frequency_penalty   | no equivalent — dropped                      |
    """
    system_chunks: list[str] = []
    conversation: list[dict[str, Any]] = []

    for msg in req.messages:
        text = _content_to_text(msg.content)
        if msg.role in ("system", "developer"):
            # OpenAI allows several system messages anywhere; Anthropic has one
            # system slot, so concatenate.
            system_chunks.append(text)
        elif msg.role in ("user", "assistant"):
            conversation.append({"role": msg.role, "content": text})
        else:  # "tool"
            raise HTTPException(
                status_code=400,
                detail="tool-role messages are out of scope for this translation shell",
            )

    if not conversation or conversation[0]["role"] != "user":
        raise HTTPException(
            status_code=400,
            detail="Anthropic requires the first message in `messages` to be role 'user'",
        )

    body: dict[str, Any] = {
        "model": req.model,
        "messages": conversation,
        "max_tokens": req.max_tokens or DEFAULT_MAX_TOKENS,
    }
    if system_chunks:
        body["system"] = "\n\n".join(system_chunks)
    if req.temperature is not None:
        body["temperature"] = min(req.temperature, 1.0)  # Anthropic rejects > 1.0
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop is not None:
        body["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.stream:
        body["stream"] = True
    return body


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Response:
    client: httpx.AsyncClient = app.state.http

    if request.model in OPENAI_MODELS:
        url = OPENAI_URL
        headers = {"authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = request.model_dump(exclude_none=True)
    elif request.model in ANTHROPIC_MODELS:
        url = ANTHROPIC_URL
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        payload = translate_openai_to_anthropic(request)
    else:
        raise HTTPException(status_code=400, detail=f"Unrecognised model: {request.model!r}")

    headers["content-type"] = "application/json"
    upstream = await client.post(url, json=payload, headers=headers)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )

"""open-gateway-ai — a learning re-implementation of what LiteLLM does, with httpx.

Step 1: the FastAPI skeleton. Just a health check so we can confirm the server
boots. Every other responsibility (routing, auth, schema translation) is
layered on one commit at a time.
"""

from fastapi import FastAPI

app = FastAPI(title="open-gateway-ai", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

# open-gateway-ai

A learning project: rebuild LiteLLM's core — provider routing, auth, and
OpenAI↔Anthropic payload translation — by hand, using nothing but `httpx`.

**Read this repo commit by commit.** Each commit adds exactly one thing that
`litellm.completion()` otherwise does for you invisibly, with the reasoning in
the commit message.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY / ANTHROPIC_API_KEY
uvicorn main:app --reload
```

```bash
# routes to OpenAI (body forwarded unchanged)
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "hi"}]
}'

# routes to Anthropic (body translated: the system message is lifted out)
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "claude-3-5-sonnet",
  "messages": [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "hi"}
  ]
}'
```

## The commits

| # | Commit | What LiteLLM does here |
|---|--------|-----------------------|
| 1 | FastAPI skeleton | — |
| 2 | Pydantic request model | accepts the OpenAI request schema; rejects malformed input |
| 3 | proxy gpt-4o-mini to OpenAI | auth-header injection + proxied POST (OpenAI path) |
| 4 | route by model name | provider routing from an internal model registry |
| 5 | translate payload → Anthropic | **the core**: OpenAI schema → Anthropic Messages schema |
| 6 | map retired model alias | model registry: aliases, renames, prefix stripping |
| 7 | missing keys / upstream errors | maps provider errors → typed `openai.*` exceptions |
| 8 | streaming passthrough | parses + re-emits SSE frames in the other provider's format |

## What LiteLLM does vs. what this repo does

| LiteLLM responsibility | Here |
|---|---|
| Provider routing from `model` | ✅ two hardcoded sets |
| Credential selection per provider | ✅ |
| Auth headers (`Bearer` vs `x-api-key` + `anthropic-version`) | ✅ |
| **Request** translation OpenAI → Anthropic | ✅ text only |
| **Response** translation Anthropic → OpenAI shape | ❌ returns raw |
| **Streaming** frame translation | ❌ proxies raw bytes |
| Error mapping to typed exceptions | ⚠️ returns upstream status raw |
| `drop_params` for unsupported params | ✅ crude (drops `n`, penalties) |
| Retries / backoff | ❌ |
| Model registry (context window, cost, capabilities) | ❌ alias map only |
| Tool / function-calling translation | ❌ 400 on `tool` role |
| Multimodal (`image_url` → Anthropic image blocks) | ❌ 400 on non-text parts |

### The one asymmetry that forces real code

| | OpenAI | Anthropic |
|---|---|---|
| System prompt | a `{"role":"system"}` message inside `messages[]`, any number | a **top-level `system`** field; `"system"` is not a valid role |
| `max_tokens` | optional | **required** |
| `stop` | string or list | `stop_sequences`, list only |
| First message | any role | must be `user`; roles alternate |
| `temperature` | 0–2 | 0–1 |
| Response | `choices[].message.content` (string), `usage.prompt_tokens` | `content[]` (typed blocks), `usage.input_tokens` |

## Not implemented yet

Response translation, tool-call translation, streaming-frame translation,
retries/backoff, a real model registry, multimodal content.

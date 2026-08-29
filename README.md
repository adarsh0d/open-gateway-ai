# open-gateway-ai

A learning project: rebuild LiteLLM's core — provider routing, auth, and
OpenAI↔Anthropic payload translation — by hand, using nothing but `httpx`.

**Read this repo commit by commit.** Each commit adds exactly one thing that
`litellm.completion()` otherwise does for you invisibly, with the reasoning in
the commit message.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

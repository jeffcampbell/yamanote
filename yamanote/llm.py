"""OpenRouter chat-completions client (stdlib only).

One call = one Completion carrying the message, the model that actually
answered (OpenRouter may fall back), token counts and the exact USD cost that
OpenRouter reports in `usage.cost`.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import settings

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
UNREACHABLE = -1  # LLMError.status when the network/API can't be reached at all


class LLMError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


@dataclass
class Completion:
    message: dict  # assistant message: content, tool_calls
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""
    seconds: float = 0.0

    @property
    def tool_calls(self) -> list[dict]:
        return self.message.get("tool_calls") or []

    @property
    def text(self) -> str:
        content = self.message.get("content") or ""
        if isinstance(content, list):  # content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content


@dataclass
class Client:
    api_key: str | None = None
    url: str = settings.OPENROUTER_URL
    retries: int = 4
    timeout: float = 300
    extra_headers: dict = field(default_factory=lambda: {
        "HTTP-Referer": "https://github.com/jeffcampbell/yamanote",
        "X-Title": "Yamanote",
    })

    def __post_init__(self):
        self.api_key = self.api_key or settings.openrouter_key()

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             fallbacks: list[str] | None = None, max_tokens: int = 8000,
             temperature: float | None = None, response_format: dict | None = None) -> Completion:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set (env, .env, or ~/development/.env)", 401)
        body: dict = {
            "model": model,
            "messages": _with_cache_breakpoints(model, messages),
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if fallbacks:
            body["models"] = [model] + [m for m in fallbacks if m != model]
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if temperature is not None:
            body["temperature"] = temperature
        if response_format:
            body["response_format"] = response_format
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        data = json.dumps(body).encode()

        for attempt in range(self.retries + 1):
            start = time.monotonic()
            request = urllib.request.Request(self.url, data, headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    payload = json.load(r)
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode(errors="replace")
                if e.code not in RETRY_STATUS or attempt == self.retries:
                    raise LLMError(f"OpenRouter HTTP {e.code}: {detail}", e.code) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise LLMError(f"OpenRouter unreachable: {e}", UNREACHABLE) from None
            else:
                if "error" in payload and not payload.get("choices"):
                    err = payload["error"]
                    code = err.get("code") if isinstance(err, dict) else None
                    if code in RETRY_STATUS and attempt < self.retries:
                        time.sleep(min(2 ** attempt, 20))
                        continue
                    raise LLMError(f"OpenRouter error: {str(err)[:400]}", code if isinstance(code, int) else None)
                return _parse(payload, model, time.monotonic() - start)
            time.sleep(min(2 ** attempt, 20))
        raise AssertionError("unreachable")


def _parse(payload: dict, requested_model: str, seconds: float) -> Completion:
    choice = (payload.get("choices") or [{}])[0]
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return Completion(
        message=choice.get("message") or {"role": "assistant", "content": ""},
        model=payload.get("model") or requested_model,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),
        finish_reason=choice.get("finish_reason") or "",
        seconds=seconds,
    )


def _with_cache_breakpoints(model: str, messages: list[dict]) -> list[dict]:
    """Anthropic models need explicit cache_control markers; others cache
    automatically. Mark the system prompt and the newest message so an agent
    loop re-reads its growing history from cache."""
    if not model.startswith("anthropic/"):
        return messages
    out = [dict(m) for m in messages]
    targets = [i for i, m in enumerate(out) if m.get("role") == "system"][:1]
    # newest user/tool message (tool results arrive as role "tool")
    for i in range(len(out) - 1, 0, -1):
        if out[i].get("role") in ("user", "tool"):
            targets.append(i)
            break
    for i in targets:
        content = out[i].get("content")
        if isinstance(content, str) and content:
            out[i]["content"] = [{"type": "text", "text": content,
                                  "cache_control": {"type": "ephemeral"}}]
    return out

"""Thin async client for any OpenAI-compatible chat endpoint with tool calling.

The request is deliberately minimal (model, messages, tools, seed) so it works across providers; anything
else (sampling, token limits, thinking switches) goes through Settings.extra_body. The per-request `seed`
is derived by the caller (run seed, agent, step), so a run is replayable as far as the server allows.
"""
import asyncio
import logging

import openai

from .config import Settings

log = logging.getLogger(__name__)
TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError)


class LLM:
    def __init__(self, settings: Settings):
        self.client = openai.AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=3600, max_retries=0)
        self.model, self.extra_body, self.keep_reasoning = settings.model, settings.extra_body, settings.keep_reasoning

    async def chat(self, messages: list[dict], tools: list[dict] | None, seed: int) -> tuple[dict, dict]:
        """One completion. Returns (assistant message for the history, usage dict)."""
        kwargs = dict(model=self.model, messages=messages, seed=seed)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if tools:
            kwargs["tools"] = tools
        for attempt in range(8):   # transient transport and rate-limit errors only; API errors raise
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                break
            except TRANSIENT as e:
                wait = min(60, 5 * 2**attempt)
                log.warning("llm transient error (%s); retry in %ds", type(e).__name__, wait)
                await asyncio.sleep(wait)
        else:
            raise RuntimeError("model endpoint unreachable after retries")
        choice = resp.choices[0]
        m = choice.message
        msg = {"role": "assistant", "content": m.content or ""}
        reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
        if reasoning and self.keep_reasoning:
            msg["reasoning"] = reasoning
        if m.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                 for tc in m.tool_calls]
        usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                 "finish_reason": choice.finish_reason}
        return msg, usage

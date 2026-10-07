"""The agent loop: model -> tool calls -> results, until `finish`, the step budget, or the deadline.
Every request and response is appended to a JSONL trace.

Context handling: before each request the prompt size is estimated (last reported prompt tokens + completion
+ new tool output at CHARS_PER_TOKEN); above `context_budget` the agent writes a handoff note and restarts
with system + task + note. If the server still rejects a request for length, tool outputs are truncated and
the compaction is forced.
"""
import json
import time
import zlib

import openai

CHARS_PER_TOKEN = 2.5   # conservative for tool output (code, JSON, tables)
IMAGE_TOKENS = 1500     # estimate per attached image (vision models tile by resolution; a matplotlib figure is ~300-1500)
OVERFLOW_MARKERS = ("context length", "maximum context", "context window", "too many tokens")

HANDOFF_PROMPT = ("Your context window is nearly full. Write a handoff note for yourself: goal, what you did, "
                  "concrete findings (paths, commands, evidence), what is still open, and the exact next step. "
                  "Do not call tools.")
NUDGE = "Continue with tools. If you are done, call `finish`."


def chars(message: dict) -> int:
    """Text length of a message for the context estimate; image parts count as IMAGE_TOKENS worth of chars."""
    c = message.get("content", "")
    if isinstance(c, str):
        return len(c)
    return sum(len(part.get("text", "")) if part["type"] == "text" else int(IMAGE_TOKENS * CHARS_PER_TOKEN) for part in c)


def step_seed(run_seed: int, name: str, step: int) -> int:
    return (run_seed * 1_000_003 + zlib.crc32(name.encode()) * 1_009 + step) % 2**31


class Agent:
    def __init__(self, name: str, llm, tools, system_prompt: str, task_prompt: str, trace_path: str,
                 max_steps: int, run_seed: int, context_budget: int, deadline: float):
        self.name, self.llm, self.tools = name, llm, tools
        self.system_prompt, self.task_prompt = system_prompt, task_prompt
        self.trace_path, self.max_steps, self.run_seed = trace_path, max_steps, run_seed
        self.context_budget, self.deadline = context_budget, deadline
        self.messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task_prompt}]
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        self.step = 0
        self.est_prompt = 0.0   # estimated tokens of the next request

    def trace(self, kind: str, **data):
        with open(self.trace_path, "a") as f:
            f.write(json.dumps({"t": time.time(), "agent": self.name, "step": self.step, "kind": kind, **data}) + "\n")

    async def _chat(self, tools):
        msg, usage = await self.llm.chat(self.messages, tools, step_seed(self.run_seed, self.name, self.step))
        for k in self.tokens:
            self.tokens[k] += usage[k]
        reported = usage["prompt_tokens"] + usage["completion_tokens"]   # some endpoints report no usage
        self.est_prompt = reported or sum(chars(m) for m in self.messages) / CHARS_PER_TOKEN
        self.trace("response", message=msg, usage=usage)
        return msg

    async def compact(self):
        self.messages.append({"role": "user", "content": HANDOFF_PROMPT})
        try:
            note = await self._chat(tools=None)
        except openai.BadRequestError:   # still too long: shrink tool outputs to their head, drop images, retry once
            for m in self.messages:
                if m["role"] == "tool" and len(m["content"]) > 1000:
                    m["content"] = m["content"][:1000] + "\n[truncated for context]"
                elif isinstance(m.get("content"), list):
                    m["content"] = "[images dropped for context]"
            self.trace("truncate")
            note = await self._chat(tools=None)
        self.messages = [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": self.task_prompt + "\n\n## Handoff note from your earlier context\n" + note["content"]}]
        self.est_prompt = len(self.system_prompt + self.task_prompt + note["content"]) / CHARS_PER_TOKEN
        self.trace("compact", note=note["content"])

    async def run(self) -> dict:
        self.trace("start", system=self.system_prompt, task=self.task_prompt)
        idle, stop = 0, "max_steps"
        while self.step < self.max_steps:
            if time.time() > self.deadline:
                stop = "deadline"
                break
            self.step += 1
            if self.est_prompt > self.context_budget:
                await self.compact()
            try:
                msg = await self._chat(self.tools.specs())
            except openai.BadRequestError as e:   # context overflow the estimate missed: compact and retry once
                if not any(marker in str(e).lower() for marker in OVERFLOW_MARKERS):
                    raise
                self.trace("overflow", error=str(e)[:300])
                await self.compact()
                msg = await self._chat(self.tools.specs())
            self.messages.append(msg)
            if not msg.get("tool_calls"):
                idle += 1
                if idle >= 3:
                    stop = "no_tool_calls"
                    break
                self.messages.append({"role": "user", "content": NUDGE})
                continue
            idle = 0
            for tc in msg["tool_calls"]:
                out = await self.tools.call(tc["function"]["name"], tc["function"]["arguments"])
                self.trace("tool", name=tc["function"]["name"], arguments=tc["function"]["arguments"], output=out)
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
                self.est_prompt += len(out) / CHARS_PER_TOKEN + 8
            if self.tools.images:   # tool results are text-only in the chat API: viewed images follow as a user message
                parts = [{"type": "text", "text": "Images from view_image, in order: " + ", ".join(i["path"] for i in self.tools.images)}]
                parts += [{"type": "image_url", "image_url": {"url": i["url"]}} for i in self.tools.images]
                self.messages.append({"role": "user", "content": parts})
                self.trace("images", images=[{"path": i["path"], "bytes": i["bytes"]} for i in self.tools.images])
                self.est_prompt += IMAGE_TOKENS * len(self.tools.images)
                self.tools.images = []
            if self.tools.result is not None:
                stop = "finished"
                break
        summary = {"agent": self.name, "stop": stop, "steps": self.step, "result": self.tools.result, **self.tokens}
        self.trace("end", **summary)
        return summary

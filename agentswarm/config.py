"""Settings: the model endpoint, read from the environment or a .env file (never from the repository).

    SWARM_BASE_URL   OpenAI-compatible endpoint, e.g. https://api.example.com/v1; several, comma-separated,
                     are used round-robin by the agents (and split among prompts when several are run)
    SWARM_API_KEY    its key (optional for local servers)
    SWARM_MODEL      model name
    SWARM_CONTEXT_TOKENS, SWARM_EXTRA_BODY (JSON), SWARM_KEEP_REASONING, SWARM_VISION   optional, see .env.example

The .env file is parsed here and NOT exported into os.environ, so the key cannot leak into the shells the
agents run. Real environment variables take precedence over the file.
"""
import json
import os
from dataclasses import dataclass, field


def read_env_file(path: str) -> dict[str, str]:
    values = {}
    if not os.path.exists(path):
        return values
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip().removeprefix("export ").strip()] = value
    return values


@dataclass
class Settings:
    base_urls: list[str]
    model: str
    api_key: str = field(default="EMPTY", repr=False)   # repr=False: never printed or logged
    context_tokens: int = 100_000
    extra_body: dict = field(default_factory=dict)
    keep_reasoning: bool = False
    vision: bool = False   # the model takes images: agents get `view_image` to look at the plots they make

    @classmethod
    def load(cls, env_file: str = ".env") -> "Settings":
        values = {**read_env_file(env_file), **{k: v for k, v in os.environ.items() if k.startswith("SWARM_")}}
        missing = [k for k in ("SWARM_BASE_URL", "SWARM_MODEL") if not values.get(k)]
        if missing:
            raise SystemExit(f"missing {', '.join(missing)}: set them in {env_file} (see .env.example) or in the environment")
        return cls(base_urls=[u.strip() for u in values["SWARM_BASE_URL"].split(",") if u.strip()], model=values["SWARM_MODEL"],
                   api_key=values.get("SWARM_API_KEY") or "EMPTY",
                   context_tokens=int(values.get("SWARM_CONTEXT_TOKENS", 100_000)),
                   extra_body=json.loads(values.get("SWARM_EXTRA_BODY") or "{}"),
                   keep_reasoning=values.get("SWARM_KEEP_REASONING", "") not in ("", "0", "false"),
                   vision=values.get("SWARM_VISION", "") not in ("", "0", "false"))

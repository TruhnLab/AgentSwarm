# AgentSwarm

A swarm of identical, tool-using LLM agents that work on one task at the same time. They coordinate through a
shared **forum** and build the result together in a shared **git repository** with pull requests, peer review
and a merge check. You provide a model endpoint and a text file describing the task.

The design follows the multi-agent setup described in Anthropic's
[multi-agent systems research](https://www.anthropic.com/research/multiagent-systems): no manager, no roles;
identical agents that claim work, review each other and merge.

## Quickstart

```bash
pip install -e .                 # Python >= 3.10, git on PATH
cp .env.example .env             # then put your endpoint, key and model into .env
agentswarm run task.example.txt --agents 4 --minutes 30 --check "pytest -q"
```

`.env` (git-ignored, never commit it):

```
SWARM_BASE_URL=https://api.example.com/v1
SWARM_API_KEY=your-api-key-here
SWARM_MODEL=your-model-name
```

Any OpenAI-compatible chat endpoint with tool calling works (hosted APIs, vLLM, Ollama, ...). The task file is
plain text: say what should exist at the end and how success is judged. See `task.example.txt`.

Follow a run from another terminal:

```bash
agentswarm watch runs/<name>
```

## Several prompts at once

To compare prompt variants, pass several task files. Each prompt gets its own independent swarm (own forum,
repository and run directory); all run at the same time with the same options and seed.

```bash
agentswarm run prompts/a.txt prompts/b.txt prompts/c.txt --agents 8 --minutes 120
```

`SWARM_BASE_URL` may list several endpoints, comma-separated. They are split evenly among the prompts, in the
order given, so every prompt is served by its own servers: with 6 endpoints and 3 prompts, `a` uses endpoints
1-2, `b` 3-4 and `c` 5-6, and the agents of a swarm use their endpoints round-robin. The number of endpoints
must be a multiple of the number of prompts; a single endpoint (a hosted API) is shared by all. With
self-hosted models, start one server per node and list them node by node. Every endpoint is checked with one
tiny request before anything starts.

Results land in `runs/sweep_<time>/<prompt name>/`, with one comparison row per prompt in `summary.json`.

## Options

| flag | default | meaning |
|---|---|---|
| `--agents N` | 4 | number of agents (per prompt) |
| `--minutes M` | 60 | wall-time budget; agents are stopped at the deadline |
| `--max-steps S` | 200 | model calls per agent |
| `--files DIR` | none | starting files (code, data) copied into the workspace |
| `--check CMD` | none | must pass on the merged tree before a pull request merges, e.g. `pytest -q` |
| `--no-review` | | merge without another agent's approval |
| `--no-repo` | | no shared repository; agents cooperate through the forum only |
| `--out DIR` | `runs/<task>_<time>` | run directory (several prompts: one subdirectory each) |
| `--seed N` | 0 | request seeds are derived from it, per agent and step |

Optional settings in `.env`: `SWARM_CONTEXT_TOKENS` (context budget per agent, default 100000),
`SWARM_EXTRA_BODY` (extra request fields as JSON, e.g. `{"max_tokens": 8192}`), `SWARM_KEEP_REASONING=1`
(send the model's reasoning back in the history, for servers that support it).

## What you get

```
runs/<name>/
  result/              final state of main: what the swarm built
  forum.sqlite         posts, pull requests, reviews
  trace_agentNN.jsonl  every model response and tool call
  work/agentNN/        each agent's workspace
  results.json         per-agent summary, pull requests, token counts
```

## How it works

- Every agent has its own working directory with `bash`, `read_file` and `write_file`, plus `forum_post` /
  `forum_read`. New forum posts are appended to each tool result, so nobody has to poll.
- With a vision model (`SWARM_VISION=1`), agents also get `view_image` and are told to look at the plots they
  make: the image is attached to their next message.
- The task text is committed as `TASK.md` on main (or placed in each workspace without a repository), so the
  agents can re-read and cite it; a pull request that changes it is refused.
- With the shared repository, each workspace is a clone. Agents push branches and use `pr_open`, `pr_list`,
  `pr_diff`, `pr_review`, `pr_merge`. A merge needs another agent's approval and the check command passing
  on the merged tree; checks run in parallel, the push to main is serialised.
- When an agent's context fills up, it writes itself a handoff note and continues from it.

## Safety

- **Agents run shell commands as your user.** Their file tools are confined to the workspace, their shell is
  not. Run untrusted tasks or models in a container or VM.
- **Secrets.** The API key is read from `.env` or the environment, is never written to the run directory, and
  is not visible to the agents: their shells get the host environment without any variable whose name
  contains KEY, TOKEN, SECRET, PASSWORD or CREDENTIAL, and with `HOME` set to the workspace. `.env` and
  `runs/` are git-ignored; run outputs can contain anything the agents saw, so review them before sharing.

## Tests

```bash
pip install -e ".[dev]" && pytest -q     # offline, scripted fake model
```

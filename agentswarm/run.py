"""Run one swarm on one task: N identical agents, a shared forum, and (by default) a shared git repository
with pull requests. Outputs go to the run directory:

    result/            the final state of main (what the swarm built)
    forum.sqlite       posts, pull requests, reviews
    trace_<agent>.jsonl   every model response and tool call
    work/<agent>/      each agent's workspace
    results.json, config.json, run.log
"""
import asyncio
import json
import logging
import os
import shutil
import time

from .agent import Agent
from .config import Settings
from .forum import Forum
from .llm import LLM
from .repo import SharedRepo
from .tools import RepoTools, WorkerTools

log = logging.getLogger("agentswarm")
DEADLINE_GRACE_S = 120   # agents may finish their current tool call this long past the deadline

SYSTEM = """You are {name}, one of {n} identical agents working on the same task at the same time.
Each agent has its own working directory and shell; all agents run as the same user, so never use
pkill/killall or kill -1 (the shell refuses them): kill only pids of processes you started.
Protocol:
- Read the forum before choosing what to do; post a short claim of the area you take so others avoid it.
- Prefer areas nobody has claimed. Agents tend to make identical choices; deliberately diversify.
- Post concrete findings and negative results (what you checked and ruled out) so others do not repeat them.
- Review others' posts critically when you have evidence; reply to the post id.
- Call `finish` when nothing useful remains.
Work in steps: think, act with tools, check results. Do not narrate; act."""

REPO_PROTOCOL = """
Shared repository: your working directory is a clone of the shared repo (origin, branch main); every agent has one.
- Work on a feature branch (git checkout -b <name>), commit, `git push -u origin <name>`, then `pr_open`.
- {merge_rule}
- Review others' open PRs promptly (`pr_list`, `pr_diff`, `pr_review`); the swarm only progresses if PRs get merged.
- `git pull origin main` often; build on what is merged instead of rewriting it. Keep PRs small.
- Only what is merged into main counts as the result."""


async def run_swarm(task: str, settings: Settings, out: str, agents: int = 4, minutes: float = 60, max_steps: int = 200,
                    files: str | None = None, check: str = "", review: bool = True, use_repo: bool = True,
                    seed: int = 0, llm=None) -> dict:
    """Run the swarm and return the results dict (also written to <out>/results.json)."""
    if use_repo and not shutil.which("git"):
        raise SystemExit("git not found: install git or run with --no-repo")
    if os.path.exists(os.path.join(out, "forum.sqlite")):
        raise SystemExit(f"{out} already holds a run; choose another --out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:   # no endpoint, no key: only what describes the run
        json.dump({"task": task, "model": settings.model, "agents": agents, "minutes": minutes, "max_steps": max_steps,
                   "files": files, "check": check, "review": review, "repo": use_repo, "seed": seed}, f, indent=1)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out, "run.log"))])
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    llm = llm or LLM(settings)
    forum = Forum(os.path.join(out, "forum.sqlite"))
    start = time.time()
    deadline = start + 60 * minutes
    system, repo = SYSTEM, None
    if use_repo:
        repo = SharedRepo(out, check)
        await repo.init(files)
        rule = ("Merging (`pr_merge`) requires another agent's approval (`pr_review`)" if review else "Merge with `pr_merge`")
        rule += f" and the check command `{check}` passing on the merged tree." if check else "."
        system += REPO_PROTOCOL.format(merge_rule=rule)

    async def make_worker(i: int) -> Agent:
        name = f"agent{i:02d}"
        workdir = os.path.join(out, "work", name)
        if repo:
            await repo.clone(workdir, name)
        elif files:
            shutil.copytree(files, workdir)
        else:
            os.makedirs(workdir)
        tools = WorkerTools(workdir, forum, name)
        if repo:
            RepoTools(tools, repo, review)
        return Agent(name, llm, tools, system.format(name=name, n=agents), task, os.path.join(out, f"trace_{name}.jsonl"),
                     max_steps, seed, settings.context_tokens, deadline)

    workers = await asyncio.gather(*(make_worker(i) for i in range(agents)))
    log.info("run %s: %d agents, model %s, %g min", out, agents, settings.model, minutes)
    tasks = {asyncio.create_task(w.run()): w for w in workers}
    summaries = []
    while tasks:
        done, _ = await asyncio.wait(tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            w = tasks.pop(t)
            if t.cancelled():
                summaries.append({"agent": w.name, "stop": "deadline", "steps": w.step, "result": None, **w.tokens})
            elif t.exception() is not None:
                summaries.append({"agent": w.name, "stop": "error", "steps": w.step, "error": repr(t.exception()), **w.tokens})
            else:
                summaries.append(t.result())
        if time.time() > deadline + DEADLINE_GRACE_S:
            for t in tasks:
                t.cancel()
    summaries.sort(key=lambda s: s["agent"])
    if repo:
        await repo.sync()

    prs = forum.prs()
    results = {"run_dir": out, "agents": summaries, "prs": prs, "merged": sum(p["status"] == "merged" for p in prs),
               "tokens": {k: sum(s.get(k, 0) for s in summaries) for k in ("prompt_tokens", "completion_tokens")},
               "wall_time_s": round(time.time() - start, 1)}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    log.info("done in %.0f s: %d/%d PRs merged; tokens %s", results["wall_time_s"], results["merged"], len(prs), results["tokens"])
    for s in summaries:
        log.info("  %s stop=%s steps=%s%s", s["agent"], s["stop"], s.get("steps"), f" error={s['error']}" if "error" in s else "")
    if repo:
        log.info("result: %s", repo.main)
    return results

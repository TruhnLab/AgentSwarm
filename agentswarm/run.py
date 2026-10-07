"""Run a swarm on a task: N identical agents, a shared forum, and (by default) a shared git repository with
pull requests. `run_sweep` runs several prompts at the same time, one independent swarm per prompt, each on
its own group of endpoints. Outputs of one swarm go to its run directory:

    result/            the final state of main (what the swarm built)
    forum.sqlite       posts, pull requests, reviews
    trace_<agent>.jsonl   every model response and tool call
    work/<agent>/      each agent's workspace
    results.json, config.json, run.log
"""
import asyncio
import dataclasses
import json
import logging
import os
import shutil
import time

from .agent import Agent
from .config import Settings
from .forum import Forum
from .llm import LLM
from .repo import TASK_FILE, SharedRepo
from .tools import RepoTools, WorkerTools

LOG_FORMAT = "%(asctime)s %(name)s %(message)s"
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

VISION_PROTOCOL = """
You can see images: `view_image` shows you a PNG/JPEG/GIF from your working directory. Plot what you are working
on (results, intermediate quantities, failure cases) and look at the plots; a glance at a figure catches what
summary numbers hide. Regenerate and inspect the plots after every substantial change."""

REPO_PROTOCOL = """
Shared repository: your working directory is a clone of the shared repo (origin, branch main); every agent has one.
- `TASK.md` on main is the task as given, verbatim: re-read it, cite it in reviews (a PR that does not serve it
  should not merge). It cannot be changed.
- Work on a feature branch (git checkout -b <name>), commit, `git push -u origin <name>`, then `pr_open`.
- {merge_rule}
- Review others' open PRs promptly (`pr_list`, `pr_diff`, `pr_review`); the swarm only progresses if PRs get merged.
- `git pull origin main` often; build on what is merged instead of rewriting it. Keep PRs small.
- Only what is merged into main counts as the result."""


async def run_swarm(task: str, settings: Settings, out: str, agents: int = 4, minutes: float = 60, max_steps: int = 200,
                    files: str | None = None, check: str = "", review: bool = True, use_repo: bool = True,
                    seed: int = 0, llms: list | None = None) -> dict:
    """Run one swarm and return the results dict (also written to <out>/results.json).
    The agents use the endpoints in settings.base_urls round-robin."""
    if use_repo and not shutil.which("git"):
        raise SystemExit("git not found: install git or run with --no-repo")
    if os.path.exists(os.path.join(out, "forum.sqlite")):
        raise SystemExit(f"{out} already holds a run; choose another --out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:   # no endpoint, no key: only what describes the run
        json.dump({"task": task, "model": settings.model, "agents": agents, "minutes": minutes, "max_steps": max_steps,
                   "files": files, "check": check, "review": review, "repo": use_repo, "seed": seed,
                   "endpoints": len(settings.base_urls)}, f, indent=1)
    log = logging.getLogger(f"agentswarm.{os.path.basename(os.path.normpath(out))}")   # one logger and run.log per swarm
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(out, "run.log"))
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    log.addHandler(handler)
    try:
        llms = llms or [LLM(settings, url, log) for url in settings.base_urls]
        forum = Forum(os.path.join(out, "forum.sqlite"))
        start = time.time()
        deadline = start + 60 * minutes
        system, repo = SYSTEM + (VISION_PROTOCOL if settings.vision else ""), None
        if use_repo:
            repo = SharedRepo(out, check)
            await repo.init(files, task)
            rule = ("Merging (`pr_merge`) requires another agent's approval (`pr_review`)" if review else "Merge with `pr_merge`")
            rule += f" and the check command `{check}` passing on the merged tree." if check else "."
            system += REPO_PROTOCOL.format(merge_rule=rule)

        async def make_worker(i: int) -> Agent:
            name = f"agent{i:02d}"
            workdir = os.path.join(out, "work", name)
            if repo:
                await repo.clone(workdir, name)
            else:
                shutil.copytree(files, workdir) if files else os.makedirs(workdir)
                with open(os.path.join(workdir, TASK_FILE), "w") as f:   # no shared repo: the task still sits in the workspace
                    f.write(task)
            tools = WorkerTools(workdir, forum, name, settings.vision)
            if repo:
                RepoTools(tools, repo, review)
            return Agent(name, llms[i % len(llms)], tools, system.format(name=name, n=agents), task,
                         os.path.join(out, f"trace_{name}.jsonl"), max_steps, seed, settings.context_tokens, deadline)

        workers = await asyncio.gather(*(make_worker(i) for i in range(agents)))
        log.info("run %s: %d agents on %d endpoint(s), model %s, %g min", out, agents, len(llms), settings.model, minutes)
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
    finally:
        log.removeHandler(handler)
        handler.close()


def split_endpoints(urls: list[str], n: int) -> list[list[str]]:
    """One group of endpoints per prompt: disjoint, equally sized, contiguous in the given order.
    A single endpoint (e.g. a hosted API) is shared by all prompts."""
    if len(urls) == 1:
        return [list(urls) for _ in range(n)]
    if len(urls) % n:
        raise SystemExit(f"{len(urls)} endpoints cannot be divided evenly among {n} prompts: "
                         f"give one endpoint (shared by all) or a multiple of {n}")
    size = len(urls) // n
    return [urls[i * size:(i + 1) * size] for i in range(n)]


async def run_sweep(tasks: dict[str, str], settings: Settings, out: str, **options) -> list[dict]:
    """Run one independent swarm per prompt, all at the same time, each on its own endpoints (split_endpoints),
    with the same options and seed. `tasks` maps a name to the task text; swarm <name> writes to <out>/<name>/.
    Returns one summary row per prompt (also written to <out>/summary.json)."""
    groups = split_endpoints(settings.base_urls, len(tasks))
    os.makedirs(out, exist_ok=True)
    runs = await asyncio.gather(*(run_swarm(text, dataclasses.replace(settings, base_urls=group), os.path.join(out, name), **options)
                                  for (name, text), group in zip(tasks.items(), groups)), return_exceptions=True)
    size, shared = len(groups[0]), len(settings.base_urls) == 1
    rows = []
    for i, (name, res) in enumerate(zip(tasks, runs)):
        row = {"prompt": name, "run_dir": os.path.join(out, name),
               "endpoints": "shared" if shared else f"{i * size + 1}-{(i + 1) * size} of {len(settings.base_urls)}"}
        if isinstance(res, BaseException):
            row["error"] = f"{type(res).__name__}: {res}"
        else:
            stops = [a["stop"] for a in res["agents"]]
            row.update(agents=len(stops), stops={s: stops.count(s) for s in sorted(set(stops))},
                       prs_opened=len(res["prs"]), prs_merged=res["merged"], **res["tokens"], wall_time_s=res["wall_time_s"])
        rows.append(row)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def format_summary(rows: list[dict]) -> str:
    lines = [f"{'prompt':24s} {'endpoints':>12s} {'PRs merged':>11s} {'out tokens':>11s} {'minutes':>8s}  agents"]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['prompt'][:24]:24s} {r['endpoints']:>12s}  FAILED: {r['error'][:200]}")
            continue
        stops = ", ".join(f"{n} {s}" for s, n in r["stops"].items())
        lines.append(f"{r['prompt'][:24]:24s} {r['endpoints']:>12s} {r['prs_merged']:>5d}/{r['prs_opened']:<5d} "
                     f"{r['completion_tokens']:>11d} {r['wall_time_s'] / 60:>8.1f}  {stops}")
    return "\n".join(lines)

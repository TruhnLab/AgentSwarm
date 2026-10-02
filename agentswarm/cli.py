"""Command line:

    agentswarm run task.txt [--agents 4] [--minutes 60] [--files DIR] [--check "pytest -q"] ...
    agentswarm run a.txt b.txt c.txt ...     # one swarm per prompt, at the same time, each on its own endpoints
    agentswarm watch runs/<name>
"""
import argparse
import asyncio
import logging
import os
import time

from .config import Settings
from .llm import check_endpoints
from .run import LOG_FORMAT, format_summary, run_swarm, run_sweep
from .watch import watch


def read_task(path: str) -> str:
    with open(path) as f:
        task = f.read().strip()
    if not task:
        raise SystemExit(f"{path} is empty: describe the task for the agents in it")
    return task


async def run(a, settings: Settings):
    names = [os.path.splitext(os.path.basename(p))[0] for p in a.task]
    if len(set(names)) < len(names):
        raise SystemExit(f"prompt files need distinct names (they name the run directories): {names}")
    tasks = {n: read_task(p) for n, p in zip(names, a.task)}
    await check_endpoints(settings)
    options = dict(agents=a.agents, minutes=a.minutes, max_steps=a.max_steps, files=a.files, check=a.check,
                   review=not a.no_review, use_repo=not a.no_repo, seed=a.seed)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if len(tasks) == 1:
        return await run_swarm(tasks[names[0]], settings, a.out or os.path.join("runs", f"{names[0]}_{stamp}"), **options)
    out = a.out or os.path.join("runs", f"sweep_{stamp}")
    rows = await run_sweep(tasks, settings, out, **options)
    print(f"\n{format_summary(rows)}\n\nsummary: {os.path.join(out, 'summary.json')}")
    if any("error" in r for r in rows):
        raise SystemExit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="agentswarm", description="A swarm of identical LLM agents working on one task.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a swarm on the task described in a text file (several files: one swarm each)")
    r.add_argument("task", nargs="+", help="text file with the task for the agents; several files run one swarm per prompt "
                                           "at the same time, each on its own share of the endpoints")
    r.add_argument("--agents", type=int, default=4, help="number of agents per prompt (default 4)")
    r.add_argument("--minutes", type=float, default=60, help="wall-time budget (default 60)")
    r.add_argument("--max-steps", type=int, default=200, help="step budget per agent (default 200)")
    r.add_argument("--files", help="directory with starting files (code, data) copied into the workspace")
    r.add_argument("--check", default="", help="command that must pass on the merged tree before a PR merges, e.g. 'pytest -q'")
    r.add_argument("--no-review", action="store_true", help="merge PRs without another agent's approval")
    r.add_argument("--no-repo", action="store_true", help="no shared git repository: agents cooperate through the forum only")
    r.add_argument("--out", help="run directory (default runs/<task name>_<timestamp>; several prompts: runs/sweep_<timestamp>/<task name>)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--env", default=".env", help="file with SWARM_BASE_URL, SWARM_API_KEY, SWARM_MODEL (default .env)")
    w = sub.add_parser("watch", help="follow the forum and agent status of a run")
    w.add_argument("run_dir")
    w.add_argument("--once", action="store_true", help="print the current state and exit")
    a = p.parse_args(argv)

    if a.cmd == "watch":
        return watch(a.run_dir, a.once)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    for noisy in ("httpx", "httpx2", "openai"):   # one log line per request otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run(a, Settings.load(a.env)))


if __name__ == "__main__":
    main()

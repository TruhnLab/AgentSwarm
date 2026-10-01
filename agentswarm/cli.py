"""Command line:

    agentswarm run task.txt [--agents 4] [--minutes 60] [--files DIR] [--check "pytest -q"] ...
    agentswarm watch runs/<name>
"""
import argparse
import asyncio
import os
import time

from .config import Settings
from .run import run_swarm
from .watch import watch


def main(argv=None):
    p = argparse.ArgumentParser(prog="agentswarm", description="A swarm of identical LLM agents working on one task.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a swarm on the task described in a text file")
    r.add_argument("task", help="text file with the task for the agents")
    r.add_argument("--agents", type=int, default=4, help="number of agents (default 4)")
    r.add_argument("--minutes", type=float, default=60, help="wall-time budget (default 60)")
    r.add_argument("--max-steps", type=int, default=200, help="step budget per agent (default 200)")
    r.add_argument("--files", help="directory with starting files (code, data) copied into the workspace")
    r.add_argument("--check", default="", help="command that must pass on the merged tree before a PR merges, e.g. 'pytest -q'")
    r.add_argument("--no-review", action="store_true", help="merge PRs without another agent's approval")
    r.add_argument("--no-repo", action="store_true", help="no shared git repository: agents cooperate through the forum only")
    r.add_argument("--out", help="run directory (default runs/<task name>_<timestamp>)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--env", default=".env", help="file with SWARM_BASE_URL, SWARM_API_KEY, SWARM_MODEL (default .env)")
    w = sub.add_parser("watch", help="follow the forum and agent status of a run")
    w.add_argument("run_dir")
    w.add_argument("--once", action="store_true", help="print the current state and exit")
    a = p.parse_args(argv)

    if a.cmd == "watch":
        return watch(a.run_dir, a.once)
    with open(a.task) as f:
        task = f.read().strip()
    if not task:
        raise SystemExit(f"{a.task} is empty: describe the task for the agents in it")
    name = os.path.splitext(os.path.basename(a.task))[0]
    out = a.out or os.path.join("runs", f"{name}_{time.strftime('%Y%m%d-%H%M%S')}")
    asyncio.run(run_swarm(task, Settings.load(a.env), out, a.agents, a.minutes, a.max_steps, a.files, a.check,
                          review=not a.no_review, use_repo=not a.no_repo, seed=a.seed))


if __name__ == "__main__":
    main()

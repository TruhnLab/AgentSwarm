"""Shared git repository for one run: a bare repo every agent clones, plus a clone of main (`result/`) used to
diff pull requests and to hold the final state. A merge is checked in its own temporary clone (checks run in
parallel); only the final push to main is serialised, and it is retried if main moved while the check ran.
"""
import asyncio
import os
import shutil
import tempfile

MERGED, CONFLICT, CI_FAILED = "merged", "conflict", "ci_failed"
TASK_FILE = "TASK.md"   # the task prompt, verbatim, in the repository: agents cite it, PRs may not change it


async def sh(cmd: str, cwd: str, env: dict | None = None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT, env={**os.environ, **(env or {})})
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")


async def must(cmd: str, cwd: str, env: dict | None = None) -> str:
    rc, out = await sh(cmd, cwd, env)
    if rc != 0:
        raise RuntimeError(f"{cmd!r} in {cwd} failed ({rc}):\n{out}")
    return out


def identity(name: str) -> dict:
    return {"GIT_AUTHOR_NAME": name, "GIT_COMMITTER_NAME": name,
            "GIT_AUTHOR_EMAIL": f"{name}@swarm", "GIT_COMMITTER_EMAIL": f"{name}@swarm"}


class SharedRepo:
    def __init__(self, run_dir: str, check_cmd: str = ""):
        self.bare = os.path.join(run_dir, "repo.git")
        self.main = os.path.join(run_dir, "result")
        self.check_cmd = check_cmd
        self.lock = asyncio.Lock()

    async def init(self, files_dir: str | None = None, task: str | None = None):
        """Create the repo; `files_dir` (optional) is copied in as the starting state, the task text goes in as TASK.md."""
        await must(f"git init -q --bare -b main {self.bare}", ".")
        await must(f"git clone -q {self.bare} {self.main}", ".")
        if files_dir:
            shutil.copytree(files_dir, self.main, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
        if task is not None:
            with open(os.path.join(self.main, TASK_FILE), "w") as f:
                f.write(task)
        await must("git add -A && git commit -q --allow-empty -m seed && git push -q origin HEAD:main", self.main, identity("seed"))

    async def clone(self, workdir: str, agent: str):
        await must(f"git clone -q {self.bare} {workdir}", ".")
        await must(f"git config user.name {agent} && git config user.email {agent}@swarm", workdir)

    async def branch_exists(self, branch: str) -> bool:
        rc, _ = await sh(f"git rev-parse -q --verify refs/heads/{branch}", self.bare)
        return rc == 0

    async def diff(self, branch: str) -> str:
        await must("git fetch -q origin", self.main)
        return await must(f"git diff origin/main...origin/{branch}", self.main)

    async def changed_files(self, branch: str) -> list[tuple[str, str]]:
        """[(status, path)] of the branch relative to its merge-base with main."""
        out = await must(f"git fetch -q origin && git diff --name-status origin/main...origin/{branch}", self.main)
        return [tuple(l.split("\t", 1)) for l in out.splitlines() if "\t" in l]

    async def merge(self, branch: str, message: str, merger: str) -> tuple[str, str]:
        """Merge branch into main if it merges cleanly and the check command passes.
        Returns (MERGED, sha) | (CONFLICT, output) | (CI_FAILED, output)."""
        for _ in range(3):   # main may move while our check runs: re-merge on top of the new main
            tmp = tempfile.mkdtemp(prefix="merge_", dir=os.path.dirname(self.main))
            try:
                await must(f"git clone -q --branch main {self.bare} {tmp}", ".")
                base = (await must("git rev-parse HEAD", tmp)).strip()
                rc, out = await sh(f"git merge -q --no-ff -m {message!r} origin/{branch}", tmp, identity(merger))
                if rc != 0:
                    return CONFLICT, out
                if self.check_cmd:
                    rc, out = await sh(self.check_cmd, tmp)
                    if rc != 0:
                        return CI_FAILED, out
                async with self.lock:
                    if (await must("git rev-parse main", self.bare)).strip() == base:
                        await must("git push -q origin main", tmp)
                        sha = (await must("git rev-parse HEAD", tmp)).strip()
                        await self.sync()
                        return MERGED, sha
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return CONFLICT, "main changed three times during the check; retry pr_merge"

    async def sync(self):
        """Bring `result/` to the current main."""
        await sh("git fetch -q origin && git checkout -q main && git reset -q --hard origin/main", self.main)

"""Tools available to agents. A toolbox is a dict name -> (OpenAI tool schema, async fn(**args) -> str).

Each agent runs shell commands in its own working directory. The shell does not inherit credentials: every
environment variable whose name looks like a secret (KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL) is removed
and HOME points at the working directory. The terminal tool `finish` sets `self.result`, which stops the loop.
"""
import asyncio
import base64
import json
import mimetypes
import os
import re

from .forum import Forum

MAX_OUTPUT = 12_000   # chars of tool output kept (head + tail)
MAX_IMAGE_BYTES = 8_000_000
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.I)
PROCESS_WIDE_KILL = re.compile(r"(?<![\w./-])(pkill|killall)(?![\w-])|(?<![\w./-])kill\s+(-\S+\s+)*(--\s+)?-1(?![\w-])")


def clip(s: str, n: int = MAX_OUTPUT) -> str:
    if len(s) <= n:
        return s
    return s[: n // 2] + f"\n... [{len(s) - n} chars omitted] ...\n" + s[-n // 2:]


def spec(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


def shell_env(workdir: str) -> dict:
    """Environment for agent shells: the host's, minus anything that looks like a credential."""
    env = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
    return {**env, "HOME": workdir, "TERM": "dumb"}


class Toolbox:
    def __init__(self, workdir: str, vision: bool = False):
        self.workdir = workdir
        self.result: str | None = None   # set by a terminal tool
        self.images: list[dict] = []     # viewed images waiting to be attached to the next message (see Agent.run)
        self.tools = {
            "bash": (spec("bash", "Run a shell command in your working directory (bash, non-interactive). "
                          "Returns stdout, stderr and exit code. Output is truncated if long.",
                          {"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds, default 120"}},
                          ["command"]), self.bash),
            "read_file": (spec("read_file", "Read a text file (path relative to your working directory), optionally a line range.",
                               {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                               ["path"]), self.read_file),
            "write_file": (spec("write_file", "Create or overwrite a text file (path relative to your working directory).",
                                {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]), self.write_file),
        }
        if vision:
            self.tools["view_image"] = (spec("view_image", "Look at an image file (PNG, JPEG, WebP; a GIF shows its first frame), "
                                             "path relative to your working directory. Use it to inspect the plots you make.",
                                             {"path": {"type": "string"}}, ["path"]), self.view_image)

    def specs(self) -> list[dict]:
        return [s for s, _ in self.tools.values()]

    async def call(self, name: str, arguments: str) -> str:
        if name not in self.tools:
            return f"error: unknown tool {name!r}"
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            return f"error: arguments are not valid JSON ({e})"
        try:
            return clip(str(await self.tools[name][1](**args)))
        except TypeError as e:   # wrong or missing parameters
            return f"error: {e}"
        except Exception as e:   # a failing tool must never kill the agent
            return f"error: {type(e).__name__}: {e}"

    def _path(self, path: str) -> str:
        root = os.path.realpath(self.workdir)
        p = os.path.realpath(os.path.join(root, path))
        if p != root and not p.startswith(root + os.sep):
            raise PermissionError(f"{path} is outside your working directory")
        return p

    async def bash(self, command: str, timeout: int = 120) -> str:
        if PROCESS_WIDE_KILL.search(command):   # all agents (and possibly the model server) share one user
            return ("error: refused: pkill/killall/kill -1 act on every process of this user, including the other "
                    "agents. Kill only pids of processes you started (kill <pid>, or kill -- -<pgid>).")
        proc = await asyncio.create_subprocess_shell(
            command, cwd=self.workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL, env=shell_env(self.workdir), start_new_session=True)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=min(int(timeout), 1800))
        except asyncio.TimeoutError:
            await self._kill(proc)
            return f"error: command timed out after {timeout}s"
        except asyncio.CancelledError:   # run deadline: do not leave the command running
            await self._kill(proc)
            raise
        text = out.decode(errors="replace")
        if err:
            text += "\n[stderr]\n" + err.decode(errors="replace")
        return text + f"\n[exit code {proc.returncode}]"

    @staticmethod
    async def _kill(proc):
        try:
            os.killpg(proc.pid, 9)   # the command runs in its own session: kills its children too
        except ProcessLookupError:
            pass
        await proc.wait()

    async def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        try:
            with open(self._path(path), errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            return f"error: {e}"
        sel = lines[max(start_line, 1) - 1: end_line]
        return "".join(f"{i}: {l}" for i, l in enumerate(sel, start=max(start_line, 1)))

    async def write_file(self, path: str, content: str) -> str:
        try:
            p = self._path(path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
        except OSError as e:
            return f"error: {e}"
        return f"wrote {len(content)} chars to {path}"


    async def view_image(self, path: str) -> str:
        p = self._path(path)
        mime = mimetypes.guess_type(p)[0]
        if mime not in IMAGE_TYPES:
            return f"error: {path} is not a PNG, JPEG, GIF or WebP file"
        try:
            data = open(p, "rb").read()
        except OSError as e:
            return f"error: {e}"
        if len(data) > MAX_IMAGE_BYTES:
            return f"error: {path} is {len(data)} bytes; keep images under {MAX_IMAGE_BYTES} (lower the dpi or figure size)"
        self.images.append({"path": path, "bytes": len(data),
                            "url": f"data:{mime};base64,{base64.b64encode(data).decode()}"})
        return f"[image {path} ({len(data)} bytes) attached to your next message]"


class WorkerTools(Toolbox):
    """Sandbox tools + forum + finish. New forum posts are pushed (appended to every tool result), so agents
    never need to poll the forum."""

    def __init__(self, workdir: str, forum: Forum, agent: str, vision: bool = False):
        super().__init__(workdir, vision)
        self.forum, self.agent = forum, agent
        self.seen_post = 0
        self.tools.update({
            "forum_post": (spec("forum_post", "Post to the shared forum all agents read: claims of what you are working on, "
                                "findings, questions, reviews of others' posts. Keep it factual and specific.",
                                {"title": {"type": "string"}, "body": {"type": "string"},
                                 "reply_to": {"type": "integer", "description": "post id being replied to"}},
                                ["title", "body"]), self.forum_post),
            "forum_read": (spec("forum_read", "Read forum posts in full, starting after a given post id (default: all).",
                                {"since_id": {"type": "integer"}, "limit": {"type": "integer"}}, []), self.forum_read),
            "finish": (spec("finish", "Stop working. Call when the task is complete or nothing useful remains to do.",
                            {"summary": {"type": "string"}}, ["summary"]), self.finish),
        })

    async def call(self, name, arguments):
        out = await super().call(name, arguments)
        new = self.forum.posts_since(self.seen_post)
        if new and name != "forum_read":
            self.seen_post = new[-1]["id"]
            digest = "\n".join(f"#{p['id']} [{p['agent']}] {p['title']}: {p['body'][:300]}" for p in new)
            out += f"\n\n[forum: {len(new)} new post(s); forum_read(since_id) for full text]\n{digest}"
        return out

    async def forum_post(self, title, body, reply_to=None):
        pid = self.forum.post(self.agent, title, body, reply_to)
        self.seen_post = max(self.seen_post, pid)
        return f"posted #{pid}"

    async def forum_read(self, since_id=0, limit=50):
        posts = self.forum.posts_since(since_id, limit)
        if posts:
            self.seen_post = max(self.seen_post, posts[-1]["id"])
        return "\n\n".join(f"#{p['id']} [{p['agent']}] {p['title']}" + (f" (reply to #{p['reply_to']})" if p["reply_to"] else "")
                           + f"\n{p['body']}" for p in posts) or "(no posts)"

    async def finish(self, summary):
        self.result = summary
        return "finished"


class RepoTools:
    """Pull-request tools on the shared repository, added to a worker's toolbox. Agents commit and push
    branches themselves with git in `bash`; these tools are the review/merge mechanism: a merge needs an
    approval from another agent (if require_review) and the check command must pass on the merged tree."""

    def __init__(self, toolbox: WorkerTools, repo, require_review: bool):
        self.repo, self.require_review = repo, require_review
        self.forum, self.agent = toolbox.forum, toolbox.agent
        toolbox.tools.update({
            "pr_open": (spec("pr_open", "Open a pull request for a branch you pushed to origin. Keep PRs small and focused; "
                             "the body says what and why so a reviewer can judge it.",
                             {"branch": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}},
                             ["branch", "title", "body"]), self.pr_open),
            "pr_list": (spec("pr_list", "List open pull requests with their review status.", {}, []), self.pr_list),
            "pr_diff": (spec("pr_diff", "Show a pull request's diff against main (for review).",
                             {"pr_id": {"type": "integer"}}, ["pr_id"]), self.pr_diff),
            "pr_review": (spec("pr_review", "Review another agent's pull request: approve or request changes, with a comment.",
                               {"pr_id": {"type": "integer"}, "approve": {"type": "boolean"}, "comment": {"type": "string"}},
                               ["pr_id", "approve", "comment"]), self.pr_review),
            "pr_merge": (spec("pr_merge", "Merge an open pull request into main. Requires an approval from another agent "
                              "and the check command passing on the merged tree; conflicts must be resolved on the branch first.",
                              {"pr_id": {"type": "integer"}}, ["pr_id"]), self.pr_merge),
            "pr_close": (spec("pr_close", "Close a pull request without merging.",
                              {"pr_id": {"type": "integer"}, "reason": {"type": "string"}}, ["pr_id", "reason"]), self.pr_close),
        })

    async def pr_open(self, branch, title, body):
        if not await self.repo.branch_exists(branch):
            return f"error: branch {branch!r} not found on origin; push it first (git push -u origin {branch})"
        files = "\n".join(f"{s}\t{p}" for s, p in await self.repo.changed_files(branch))
        if not files:
            return "error: branch has no changes relative to main"
        pid = self.forum.pr_open(self.agent, branch, title, body, files)
        self.forum.post(self.agent, f"PR #{pid} opened: {title}", f"branch {branch}\n{body}\n\nfiles:\n{files}")
        need = " It needs a review from another agent before merging." if self.require_review else ""
        return f"PR #{pid} opened ({len(files.splitlines())} files).{need}"

    async def pr_list(self):
        rows = []
        for pr in self.forum.prs("open") + self.forum.prs("merging"):
            rv = self.forum.reviews(pr["id"])
            state = "MERGING: check running, do not re-open" if pr["status"] == "merging" else \
                f"{sum(r['approve'] for r in rv)} approvals, {sum(not r['approve'] for r in rv)} change requests"
            rows.append(f"PR #{pr['id']} [{pr['agent']}] {pr['title']} (branch {pr['branch']}; {state})")
        return "\n".join(rows) or "(no open PRs)"

    async def pr_diff(self, pr_id):
        pr = self.forum.pr(pr_id)
        if not pr:
            return f"error: no PR #{pr_id}"
        return await self.repo.diff(pr["branch"])

    async def pr_review(self, pr_id, approve, comment):
        pr = self.forum.pr(pr_id)
        if not pr or pr["status"] != "open":
            return f"error: PR #{pr_id} is not open"
        if pr["agent"] == self.agent:
            return "error: you cannot review your own PR"
        self.forum.pr_review(pr_id, self.agent, approve, comment)
        self.forum.post(self.agent, f"review of PR #{pr_id}: {'APPROVED' if approve else 'CHANGES REQUESTED'}", comment)
        return "review recorded"

    async def pr_merge(self, pr_id):
        pr = self.forum.pr(pr_id)
        if not pr or pr["status"] != "open":
            return f"error: PR #{pr_id} is not open"
        approvers = {r["agent"] for r in self.forum.reviews(pr_id) if r["approve"] and r["agent"] != pr["agent"]}
        if self.require_review and not approvers:
            return "error: no approval from another agent yet; ask for a review on the forum"
        if not self.forum.pr_claim(pr_id):
            return f"error: PR #{pr_id} is being merged by another agent"
        try:
            status, out = await self.repo.merge(pr["branch"], f"Merge PR #{pr_id}: {pr['title']} ({pr['agent']})", self.agent)
        except BaseException:   # also on cancellation at the deadline: never leave a PR stuck in 'merging'
            self.forum.pr_release(pr_id)
            raise
        if status == "merged":
            self.forum.pr_close(pr_id, "merged", out, self.agent)
            self.forum.post(self.agent, f"PR #{pr_id} merged into main", f"{pr['title']} ({out[:12]}). Run `git pull origin main`.")
            return f"merged as {out[:12]}"
        self.forum.pr_release(pr_id)
        self.forum.pr_bump(pr_id, "conflicts" if status == "conflict" else "ci_failures")
        return f"merge failed ({status}); fix on the branch and push again:\n{clip(out, 4000)}"

    async def pr_close(self, pr_id, reason):
        pr = self.forum.pr(pr_id)
        if not pr or pr["status"] != "open":
            return f"error: PR #{pr_id} is not open"
        self.forum.pr_close(pr_id, "closed")
        self.forum.post(self.agent, f"PR #{pr_id} closed", reason)
        return "closed"

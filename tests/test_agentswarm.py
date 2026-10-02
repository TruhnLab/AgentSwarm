"""Offline tests with a scripted fake model: no endpoint, no key needed.  pytest -q"""
import asyncio
import http.server
import json
import os
import threading
import time

import pytest

from agentswarm.agent import Agent
from agentswarm.cli import main as cli
from agentswarm.config import Settings, read_env_file
from agentswarm.forum import Forum
from agentswarm.run import run_swarm, split_endpoints
from agentswarm.tools import Toolbox, WorkerTools

SECRET = "FAKE-KEY-for-tests-only"


def call(name, **args):
    return {"id": f"c{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeLLM:
    """Replays a script of assistant messages per agent; records what it was asked."""

    def __init__(self, script):
        self.script, self.steps, self.requests = script, {}, []

    async def chat(self, messages, tools, seed):
        agent = messages[0]["content"].split(",")[0]   # "You are agentNN"
        self.requests.append((list(messages), tools, seed))
        i = self.steps.get(agent, 0)
        self.steps[agent] = i + 1
        return {"role": "assistant", "content": "", **self.script[min(i, len(self.script) - 1)]}, \
            {"prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}


def test_settings_from_env_file_never_exported(tmp_path, monkeypatch):
    for k in ("SWARM_BASE_URL", "SWARM_API_KEY", "SWARM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(f'# comment\nSWARM_BASE_URL=http://x/v1, http://y/v1\nSWARM_API_KEY="{SECRET}"\nSWARM_MODEL=m\nSWARM_EXTRA_BODY={{"max_tokens": 5}}\n')
    s = Settings.load(str(env))
    assert (s.base_urls, s.model, s.api_key, s.extra_body) == (["http://x/v1", "http://y/v1"], "m", SECRET, {"max_tokens": 5})
    assert "SWARM_API_KEY" not in os.environ and SECRET not in repr(s)
    assert read_env_file(str(tmp_path / "missing")) == {}


def test_tools_sandbox_and_no_secrets_in_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARM_API_KEY", SECRET)
    monkeypatch.setenv("GITHUB_TOKEN", SECRET)
    t = Toolbox(str(tmp_path))
    out = asyncio.run(t.call("bash", json.dumps({"command": "env; echo hi; exit 3"})))
    assert "hi" in out and "[exit code 3]" in out and SECRET not in out
    assert "timed out" in asyncio.run(t.call("bash", json.dumps({"command": "sleep 5", "timeout": 1})))
    asyncio.run(t.call("write_file", json.dumps({"path": "d/x.txt", "content": "l1\nl2\n"})))
    assert asyncio.run(t.call("read_file", json.dumps({"path": "d/x.txt", "start_line": 2}))) == "2: l2\n"
    assert "outside" in asyncio.run(t.call("read_file", json.dumps({"path": "../../etc/passwd"})))
    assert "not valid JSON" in asyncio.run(t.call("bash", "{oops"))
    for cmd in ('pkill -f "x"', "killall python", "kill -9 -1"):
        assert asyncio.run(t.bash(cmd)).startswith("error: refused"), cmd


def test_agent_loop_forum_push_and_compaction(tmp_path):
    forum = Forum(str(tmp_path / "f.sqlite"))
    forum.post("agent01", "claim", "I take module A")
    llm = FakeLLM([{"tool_calls": [call("bash", command="echo 42")]},
                   {"content": "NOTE: ran echo"},                       # handoff note (context budget is tiny)
                   {"tool_calls": [call("finish", summary="done")]}])
    tools = WorkerTools(str(tmp_path), forum, "agent00")
    agent = Agent("agent00", llm, tools, "You are agent00, x", "task", str(tmp_path / "t.jsonl"), 10, 0, 12, time.time() + 60)
    s = asyncio.run(agent.run())
    assert s["stop"] == "finished" and s["result"] == "done"
    kinds = [json.loads(l)["kind"] for l in open(tmp_path / "t.jsonl")]
    assert kinds == ["start", "response", "tool", "response", "compact", "response", "tool", "end"]
    tool_out = json.loads(open(tmp_path / "t.jsonl").readlines()[2])["output"]
    assert "42" in tool_out and "[forum: 1 new post(s)" in tool_out and "I take module A" in tool_out
    after = llm.requests[2][0]
    assert len(after) == 2 and "NOTE: ran echo" in after[1]["content"] and llm.requests[1][1] is None


class Collaborators:
    """Two scripted agents that each push a branch, open a PR, approve the other's PR and merge their own.
    They look up PR ids in the forum, as real agents do with pr_list, and wait when the other is not ready."""

    def __init__(self, out):
        self.out, self.stage, self.forum = out, {}, None

    async def chat(self, messages, tools, seed):
        me = messages[0]["content"].split(",")[0].split()[-1]
        other = "agent01" if me == "agent00" else "agent00"
        self.forum = self.forum or Forum(os.path.join(self.out, "forum.sqlite"))
        prs = {p["branch"]: p for p in self.forum.prs()}
        stage = self.stage.get(me, 0)
        wait = call("bash", command="sleep 0.2")
        if stage == 0:
            tc = call("bash", command=f"git checkout -qb {me} && echo {me} > {me}.txt && git add -A && git commit -qm {me} && git push -qu origin {me}")
        elif stage == 1:
            tc = call("pr_open", branch=me, title=f"add {me}", body="adds a file")
        elif stage == 2:
            tc = call("pr_review", pr_id=prs[other]["id"], approve=True, comment="ok") if other in prs else wait
        elif stage == 3:
            approved = me in prs and any(r["approve"] for r in self.forum.reviews(prs[me]["id"]))
            tc = call("pr_merge", pr_id=prs[me]["id"]) if approved else wait
        else:
            tc = call("finish", summary="ok")
        if tc is not wait:
            self.stage[me] = stage + 1
        return {"role": "assistant", "content": "", "tool_calls": [tc]}, {"prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}


def test_run_with_repo_review_and_check(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARM_API_KEY", SECRET)
    (tmp_path / "seed").mkdir(); (tmp_path / "seed" / "README.md").write_text("seed\n")
    out = str(tmp_path / "run")
    settings = Settings(base_urls=["http://unused/v1"], model="m", api_key=SECRET)
    res = asyncio.run(run_swarm("build it", settings, out, agents=2, minutes=5, max_steps=60, files=str(tmp_path / "seed"),
                                check="test -f README.md", llms=[Collaborators(out)]))
    assert {s["stop"] for s in res["agents"]} == {"finished"} and res["merged"] == 2
    assert sorted(os.listdir(os.path.join(out, "result")))[-3:] == ["README.md", "agent00.txt", "agent01.txt"]
    for root, _, names in os.walk(out):   # the key never reaches anything the run writes
        for n in names:
            with open(os.path.join(root, n), "rb") as f:
                assert SECRET.encode() not in f.read(), os.path.join(root, n)
    cli(["watch", out, "--once"])


def test_deadline_cancels_and_no_repo_mode(tmp_path, monkeypatch):
    import agentswarm.run as run
    monkeypatch.setattr(run, "DEADLINE_GRACE_S", 1)
    llm = FakeLLM([{"tool_calls": [call("bash", command="sleep 60")]}])
    settings = Settings(base_urls=["http://unused/v1"], model="m")
    t0 = time.time()
    res = asyncio.run(run_swarm("p", settings, str(tmp_path / "run"), agents=2, minutes=0.02, use_repo=False, llms=[llm]))
    assert time.time() - t0 < 30 and all(a["stop"] == "deadline" for a in res["agents"]) and res["prs"] == []


def fake_endpoint():
    """A local OpenAI-compatible server. Answers the endpoint check with text; lets every agent write hello into
    out.txt and then finish. Returns (server, list of (authorization header, request body))."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.headers["Authorization"], body))
            if "tools" not in body:                                          # the endpoint check
                message = {"role": "assistant", "content": "OK"}
            elif body["messages"][-1]["role"] == "user":                     # an agent's first step
                message = {"role": "assistant", "content": None, "tool_calls": [call("bash", command="echo hello > out.txt")]}
            else:
                message = {"role": "assistant", "content": None, "tool_calls": [call("finish", summary="done")]}
            reply = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                     "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                     "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
            data = json.dumps(reply).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def assert_no_secret(out):
    for root, _, names in os.walk(out):
        for n in names:
            with open(os.path.join(root, n), "rb") as f:
                assert SECRET.encode() not in f.read(), os.path.join(root, n)


def test_cli_end_to_end_against_a_local_endpoint(tmp_path, monkeypatch):
    """The real client path: .env + task.txt -> HTTP requests carrying the key -> tool calls executed."""
    for k in ("SWARM_BASE_URL", "SWARM_API_KEY", "SWARM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    server, seen = fake_endpoint()
    (tmp_path / ".env").write_text(f"SWARM_BASE_URL=http://127.0.0.1:{server.server_port}/v1\nSWARM_API_KEY={SECRET}\nSWARM_MODEL=test-model\n")
    (tmp_path / "task.txt").write_text("Write hello into out.txt.\n")
    out = str(tmp_path / "run")
    cli(["run", str(tmp_path / "task.txt"), "--agents", "1", "--no-repo", "--minutes", "1", "--env", str(tmp_path / ".env"), "--out", out])
    server.shutdown()
    assert open(os.path.join(out, "work", "agent00", "out.txt")).read() == "hello\n"
    assert "tools" not in seen[0][1]                                    # endpoint check first
    assert seen[1][0] == f"Bearer {SECRET}" and seen[1][1]["model"] == "test-model"
    assert seen[1][1]["messages"][1]["content"] == "Write hello into out.txt." and len(seen[1][1]["tools"]) == 6
    assert json.load(open(os.path.join(out, "results.json")))["agents"][0]["stop"] == "finished"
    assert_no_secret(out)


def test_split_endpoints():
    assert split_endpoints(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]
    assert split_endpoints(["a", "b"], 2) == [["a"], ["b"]]
    assert split_endpoints(["a"], 3) == [["a"], ["a"], ["a"]]            # a single endpoint is shared
    with pytest.raises(SystemExit, match="cannot be divided evenly"):
        split_endpoints(["a", "b", "c"], 2)


def test_several_prompts_each_on_its_own_endpoints(tmp_path, monkeypatch, capsys):
    """Two prompts, four endpoints: one swarm per prompt at the same time, prompt 1 on endpoints 1-2 and
    prompt 2 on endpoints 3-4 (its two agents round-robin over them)."""
    for k in ("SWARM_BASE_URL", "SWARM_API_KEY", "SWARM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    servers = [fake_endpoint() for _ in range(4)]
    urls = ",".join(f"http://127.0.0.1:{srv.server_port}/v1" for srv, _ in servers)
    (tmp_path / ".env").write_text(f"SWARM_BASE_URL={urls}\nSWARM_API_KEY={SECRET}\nSWARM_MODEL=test-model\n")
    (tmp_path / "terse.txt").write_text("Variant A: write hello.\n")
    (tmp_path / "polite.txt").write_text("Variant B: please write hello.\n")
    out = str(tmp_path / "sweep")
    cli(["run", str(tmp_path / "terse.txt"), str(tmp_path / "polite.txt"), "--agents", "2", "--no-repo", "--minutes", "1",
         "--env", str(tmp_path / ".env"), "--out", out])
    for srv, _ in servers:
        srv.shutdown()
    tasks_seen = [{body["messages"][1]["content"] for _, body in seen if "tools" in body} for _, seen in servers]
    assert tasks_seen == [{"Variant A: write hello."}] * 2 + [{"Variant B: please write hello."}] * 2
    rows = json.load(open(os.path.join(out, "summary.json")))
    assert [(r["prompt"], r["endpoints"], r["agents"], r["stops"]) for r in rows] == \
        [("terse", "1-2 of 4", 2, {"finished": 2}), ("polite", "3-4 of 4", 2, {"finished": 2})]
    for name in ("terse", "polite"):
        for agent in ("agent00", "agent01"):
            assert open(os.path.join(out, name, "work", agent, "out.txt")).read() == "hello\n"
        assert "2 agents on 2 endpoint(s)" in open(os.path.join(out, name, "run.log")).read()
    assert "terse" in open(os.path.join(out, "terse", "run.log")).read() and "polite" not in open(os.path.join(out, "terse", "run.log")).read()
    assert "1-2 of 4" in capsys.readouterr().out
    assert_no_secret(out)


def test_unreachable_endpoint_fails_before_the_run(tmp_path, monkeypatch):
    for k in ("SWARM_BASE_URL", "SWARM_API_KEY", "SWARM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text("SWARM_BASE_URL=http://127.0.0.1:9/v1\nSWARM_MODEL=m\n")
    (tmp_path / "task.txt").write_text("x\n")
    with pytest.raises(SystemExit, match="endpoint check failed"):
        cli(["run", str(tmp_path / "task.txt"), "--env", str(tmp_path / ".env"), "--out", str(tmp_path / "run")])
    assert not os.path.exists(tmp_path / "run")

"""Live view of a running (or finished) swarm: streams the forum (posts, PR events, reviews) as they appear
and, every 30 s, one status line per agent (step, last tool, output tokens)."""
import glob
import json
import os
import time

from .forum import Forum


def agent_status(run_dir: str) -> str:
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "trace_*.jsonl"))):
        name, step, tool, stop, tokens = os.path.basename(path)[6:-6], "", "", "", 0
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                step = e["step"]
                if e["kind"] == "tool":
                    tool = e["name"] + " " + e["arguments"][:60].replace("\n", " ")
                elif e["kind"] == "response":
                    tokens += e["usage"]["completion_tokens"]
                elif e["kind"] == "end":
                    stop = f"  [{e['stop']}]"
        rows.append(f"  {name:8s} step {step:>3}  out-tokens {tokens:>7}  last: {tool}{stop}")
    return "\n".join(rows)


def watch(run_dir: str, once: bool = False):
    while not os.path.exists(os.path.join(run_dir, "forum.sqlite")):
        time.sleep(2)
    forum = Forum(os.path.join(run_dir, "forum.sqlite"))
    last, t_status = 0, 0.0
    while True:
        for p in forum.posts_since(last, limit=1000):
            last = p["id"]
            reply = f" (re #{p['reply_to']})" if p["reply_to"] else ""
            print(f"\n#{p['id']} [{p['agent']}]{reply} {p['title']}\n    " + p["body"][:600].replace("\n", "\n    "), flush=True)
        if time.time() - t_status > 30:
            print(f"\n--- {time.strftime('%H:%M:%S')} agents ---\n{agent_status(run_dir)}", flush=True)
            t_status = time.time()
        if once:
            return
        time.sleep(3)

"""Shared forum and pull-request ledger for one run, backed by a single SQLite file.

All agents live in one process (asyncio), so one connection guarded by a lock is sufficient; WAL mode lets
another process (agentswarm watch) read the file while the run is live.
"""
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY, ts REAL, agent TEXT, title TEXT, body TEXT, reply_to INTEGER);
CREATE TABLE IF NOT EXISTS prs (
  id INTEGER PRIMARY KEY, ts REAL, agent TEXT, branch TEXT, title TEXT, body TEXT, files TEXT,
  status TEXT DEFAULT 'open', closed_ts REAL, merge_sha TEXT, merged_by TEXT,
  conflicts INTEGER DEFAULT 0, ci_failures INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY, ts REAL, pr_id INTEGER, agent TEXT, approve INTEGER, comment TEXT);
"""
PR_COLS = ("id", "ts", "agent", "branch", "title", "body", "files", "status", "closed_ts", "merge_sha", "merged_by",
           "conflicts", "ci_failures")


class Forum:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _q(self, sql, *args):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def _insert(self, sql, *args) -> int:
        with self.lock:
            return self.db.execute(sql, args).lastrowid

    # ---- posts
    def post(self, agent: str, title: str, body: str, reply_to: int | None = None) -> int:
        return self._insert("INSERT INTO posts (ts, agent, title, body, reply_to) VALUES (?,?,?,?,?)",
                            time.time(), agent, title, body, reply_to)

    def posts_since(self, last_id: int, limit: int = 50) -> list[dict]:
        rows = self._q("SELECT id, agent, title, body, reply_to FROM posts WHERE id > ? ORDER BY id LIMIT ?", last_id, limit)
        return [dict(zip(("id", "agent", "title", "body", "reply_to"), r)) for r in rows]

    # ---- pull requests
    def pr_open(self, agent: str, branch: str, title: str, body: str, files: str) -> int:
        return self._insert("INSERT INTO prs (ts, agent, branch, title, body, files) VALUES (?,?,?,?,?,?)",
                            time.time(), agent, branch, title, body, files)

    def pr(self, pr_id: int) -> dict | None:
        rows = self._q(f"SELECT {','.join(PR_COLS)} FROM prs WHERE id=?", pr_id)
        return dict(zip(PR_COLS, rows[0])) if rows else None

    def prs(self, status: str | None = None) -> list[dict]:
        rows = self._q(f"SELECT {','.join(PR_COLS)} FROM prs" + (" WHERE status=?" if status else "") + " ORDER BY id",
                       *([status] if status else []))
        return [dict(zip(PR_COLS, r)) for r in rows]

    def pr_review(self, pr_id: int, agent: str, approve: bool, comment: str) -> int:
        return self._insert("INSERT INTO reviews (ts, pr_id, agent, approve, comment) VALUES (?,?,?,?,?)",
                            time.time(), pr_id, agent, int(approve), comment)

    def reviews(self, pr_id: int) -> list[dict]:
        rows = self._q("SELECT id, ts, pr_id, agent, approve, comment FROM reviews WHERE pr_id=? ORDER BY id", pr_id)
        return [dict(zip(("id", "ts", "pr_id", "agent", "approve", "comment"), r)) for r in rows]

    def pr_claim(self, pr_id: int) -> bool:
        """Atomically take an open PR into 'merging'; False if it is not open (another agent got it)."""
        with self.lock:
            return self.db.execute("UPDATE prs SET status='merging' WHERE id=? AND status='open'", (pr_id,)).rowcount == 1

    def pr_release(self, pr_id: int):
        self._q("UPDATE prs SET status='open' WHERE id=? AND status='merging'", pr_id)

    def pr_close(self, pr_id: int, status: str, merge_sha: str | None = None, merged_by: str | None = None):
        self._q("UPDATE prs SET status=?, closed_ts=?, merge_sha=?, merged_by=? WHERE id=?",
                status, time.time(), merge_sha, merged_by, pr_id)

    def pr_bump(self, pr_id: int, field: str):
        assert field in ("conflicts", "ci_failures")
        self._q(f"UPDATE prs SET {field}={field}+1 WHERE id=?", pr_id)

"""Guard for a public repository: no tracked file may contain something that looks like a credential."""
import re
import subprocess

PATTERNS = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|SWARM_API_KEY=(?!your-api-key-here|\x22?\{SECRET\})\S+")


def test_no_secrets_in_tracked_files():
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    assert ".env" not in files, ".env must never be committed"
    hits = []
    for path in files:
        with open(path, errors="ignore") as f:
            for n, line in enumerate(f, 1):
                if PATTERNS.search(line) and "PATTERNS" not in line and path != "tests/test_no_secrets.py":
                    hits.append(f"{path}:{n}")
    assert not hits, f"possible secrets in: {hits}"

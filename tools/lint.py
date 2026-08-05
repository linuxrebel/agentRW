"""pylint as a tool.

A plugin: any callable named *_tool in this directory is discovered and
registered. Nothing imports this file explicitly. Move it out of tools/ to
uninstall.

Category 1 in Future_Router.md — fully mechanical, so it belongs in system
space rather than as prompt text. Raw pylint output on a 1000-line file runs
~2400 tokens, more than a --low-vram context window holds. This returns ~150.
The rule that makes that work: aggregate, never dump.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

MAX_TOP_ISSUES = 6      # kinds shown in the overview
MAX_ERRORS = 5          # errors listed individually
MAX_OCCURRENCES = 20    # lines shown when one symbol is requested


def _abs(filename: str) -> Path:
    """The harness injects resolve_abs_path so relative paths follow the
    agent's `cd`. Fall back to the process cwd when imported standalone."""
    injected = globals().get("resolve_abs_path")
    return injected(filename) if injected else Path(filename).expanduser().resolve()


def lint_file_tool(filename: str, symbol: str = "") -> Dict[str, Any]:
    """Lint a Python file with pylint. Returns score and most common issues.
    Pass symbol (e.g. "unused-import") to list every occurrence of one."""
    p = _abs(filename)
    if not p.is_file():
        return {"error": "file_not_found", "file_path": str(p)}

    try:
        r = subprocess.run(["pylint", "--output-format=json2", str(p)],
                           capture_output=True, text=True, timeout=120)
        data = json.loads(r.stdout or "{}")
    except FileNotFoundError:
        return {"error": "pylint_not_installed", "hint": "pip install pylint"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "path": str(p)}
    except json.JSONDecodeError:
        return {"error": "pylint_failed", "path": str(p),
                "detail": (r.stderr or r.stdout)[:300]}

    msgs = data.get("messages", [])
    score = data.get("statistics", {}).get("score")
    if isinstance(score, (int, float)):
        score = round(score, 2)

    # One symbol requested: the model already knows what it wants to fix.
    # symbol="*" returns every finding — one pylint run instead of one per kind,
    # which is what a fix loop needs.
    if symbol:
        wanted = [m for m in msgs if symbol == "*" or m["symbol"] == symbol]
        hits = [{"line": m["line"], "symbol": m["symbol"], "message": m["message"]}
                for m in wanted]
        return {"file": str(p), "symbol": symbol, "count": len(hits),
                "occurrences": hits if symbol == "*" else hits[:MAX_OCCURRENCES]}

    # Overview: counts, not a dump. Errors are listed individually because they
    # are few and they matter; style noise collapses to a count per kind.
    counts: Dict[str, int] = {}
    for m in msgs:
        counts[m["symbol"]] = counts.get(m["symbol"], 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:MAX_TOP_ISSUES]
    return {
        "file": str(p),
        "score": score,
        "total": len(msgs),
        "errors": [f"line {m['line']}: {m['message']}"
                   for m in msgs if m["type"] in ("error", "fatal")][:MAX_ERRORS],
        "top_issues": [{"symbol": s, "count": n,
                        "first_line": next(m["line"] for m in msgs if m["symbol"] == s)}
                       for s, n in top],
        "hint": "Call again with symbol=<name> to see every occurrence of one issue.",
    }

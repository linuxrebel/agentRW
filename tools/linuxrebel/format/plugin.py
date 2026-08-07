"""autopep8 as a tool.

Style findings need no model. autopep8 fixes them deterministically, cannot
alter the code either side of the whitespace, and does the whole file in one
subprocess instead of one model call per finding.

Deliberately NOT black: black reformats the entire file to its own opinion,
which produces a 40-line diff for an 11-line problem. That unrequested scope
is the exact failure this project keeps running into. autopep8 fixes actual
PEP 8 violations and leaves everything else alone.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


def _abs(filename: str) -> Path:
    injected = globals().get("resolve_abs_path")
    return injected(filename) if injected else Path(filename).expanduser().resolve()


def format_file_tool(filename: str, aggressive: str = "") -> Dict[str, Any]:
    """Fix PEP 8 style in a Python file with autopep8. Returns lines changed.
    Style only — indentation, whitespace, blank lines. Never changes logic."""
    p = _abs(filename)
    if not p.is_file():
        return {"error": "file_not_found", "file_path": str(p)}
    if not shutil.which("autopep8"):
        return {"error": "autopep8_not_installed", "hint": "pip install autopep8"}

    before = p.read_text(encoding="utf-8")
    cmd = ["autopep8", "--in-place"]
    if aggressive:
        cmd.append("--aggressive")
    cmd.append(str(p))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "path": str(p)}
    if r.returncode != 0:
        return {"error": "autopep8_failed", "detail": (r.stderr or "")[:300]}

    after = p.read_text(encoding="utf-8")
    if after == before:
        return {"file": str(p), "changed": 0, "note": "already conforms"}


    b, a_ = before.splitlines(), after.splitlines()
    changed = sum(1 for i in range(max(len(b), len(a_)))
                  if (b[i] if i < len(b) else None) != (a_[i] if i < len(a_) else None))
    return {"file": str(p), "changed": changed,
            "note": "style only — indentation and whitespace, no logic changes"}


# Not model-facing: /lint calls this directly and the model never needs to know
# it exists. Keeping it out of the system prompt saves ~57 tokens on EVERY turn
# — the prompt is re-sent with every request, so an unused tool is not a
# one-time cost. Still callable as /format_file and by ctx.tools.
format_file_tool.model_facing = False

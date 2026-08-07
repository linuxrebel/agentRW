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
import re
import subprocess
from pathlib import Path
from typing import Any, Dict

# Findings autopep8 handles. Batched and applied once at the end of a run
# rather than asked about one at a time — there is no judgement in whitespace.
STYLE_SYMBOLS = {
    "bad-indentation", "trailing-whitespace", "line-too-long",
    "multiple-statements", "missing-final-newline", "mixed-indentation",
    "bad-whitespace", "unnecessary-semicolon", "superfluous-parens",
    "trailing-newlines", "bad-continuation", "wrong-import-position",
}

MAX_TOP_ISSUES = 6      # kinds shown in the overview
MAX_ERRORS = 5          # errors listed individually
MAX_OCCURRENCES = 20    # lines shown when one symbol is requested

# Plain-English translation of pylint's more cryptic messages.
#
# pylint tells you "Module name doesn't conform to snake_case" and never
# mentions that it means the filename. Its own --help-msg is no better:
# "Used when the name doesn't conform to naming rules associated to its type".
#
# action_kind tells the harness HOW to apply a fix, never whether to offer one:
#   "line"         — rewrite that one line
#   "insert_after" — add a new line after it (a def/class docstring)
#   "insert_top"   — add a new line at the top of the file
#   "rename"       — rename the file itself
#   "manual"       — nothing automatic is possible; say so plainly
#
# note is the consequence, in plain terms: what happens if you fix it, or if
# you leave it alone. Unlisted symbols fall through to pylint's own wording,
# so this only has to cover the confusing ones.
#            symbol: (meaning, action_kind, action, note)
EXPLAIN = {
    "missing-module-docstring": (
        "No description at the top of the file.",
        "insert_top", 'add a """docstring""" as the first line',
        "Documentation only. Nothing breaks either way."),
    "missing-function-docstring": (
        "This function has no description.",
        "insert_after", 'add a """docstring""" after the def',
        "Documentation only. Nothing breaks either way."),
    "missing-class-docstring": (
        "This class has no description.",
        "insert_after", 'add a """docstring""" after the class',
        "Documentation only. Nothing breaks either way."),
    # NOTE: bad-indentation is overridden in _explain to "reindent" — pylint
    # states the exact expected width, so the answer is fully determined and a
    # model is not only unnecessary but actively risky. Asking one produced a
    # line that kept the wrong indent AND silently dropped a `*` from
    # join(*lines), turning a style fix into a runtime TypeError.
    "bad-indentation": (
        "PEP 8 wants 4 spaces per level.",
        "reindent", "re-indent the line",
        "Style only. The code runs the same."),
    "line-too-long": (
        "Longer than the configured limit.",
        "line", "wrap or shorten the line",
        "Style only. The code runs the same."),
    "trailing-whitespace": (
        "Spaces or tabs after the last visible character.",
        "line", "strip the trailing whitespace",
        "Style only. The code runs the same."),
    "unused-import": (
        "Imported but never used anywhere in the file.",
        "line", "delete the import",
        "Removing it is safe unless the import has side effects."),
    "unused-variable": (
        "Assigned but never read.",
        "line", "remove it, or prefix with _ if deliberate", ""),
    "unused-argument": (
        "The function never uses this parameter.",
        "manual", "remove the parameter, or prefix with _ if required",
        "Often correct as-is: an interface or callback may require the "
        "parameter. Removing it can break callers."),
    "broad-exception-caught": (
        "Catches every exception, including ones you did not mean to handle, "
        "such as typos and KeyboardInterrupt.",
        "line", "catch a specific exception type instead",
        "Can hide real bugs, but is sometimes deliberate."),
    "too-many-lines": (
        "The file is longer than the configured limit.",
        "manual", "split the file into smaller modules",
        "No single edit fixes this, and it does not affect how the code runs."),
    "consider-using-f-string": (
        "Uses % or .format() where an f-string reads better.",
        "line", "rewrite as an f-string", "Style only."),
    "unspecified-encoding": (
        "open() without encoding= uses the platform default, which differs "
        "between machines and can corrupt text.",
        "line", 'add encoding="utf-8"',
        "Worth fixing — this one can actually bite you."),
    "redefined-outer-name": (
        "This local name shadows one at module level.",
        "manual", "rename the local, or the outer one",
        "No mechanical fix — renaming needs judgement about which name wins. "
        "Confusing to read, but it does not change behaviour."),
}


def _explain(symbol: str, message: str, filename: str):
    """(meaning, action_kind, action, note) for one finding."""
    # invalid-name covers modules, classes, variables and constants. Only the
    # message says which, and the module case is the confusing one: it is about
    # the FILENAME, so no line edit can ever fix it.
    if symbol == "invalid-name" and message.startswith("Module name"):
        stem = Path(filename).stem
        new = stem.replace("-", "_").lower() + ".py"
        return ("Python takes the module name from the filename. "
                f"'{stem}' is not importable — `import {stem}` is a syntax error.",
                "rename", f"rename the file to {new}",
                "Renaming can break anything that calls this file by name — "
                "scripts, symlinks, cron entries — if any exist. Leaving it "
                "alone costs nothing: the file still runs exactly as it does now.")
    # Pure formatting: autopep8 fixes these deterministically, in one pass over
    # the whole file, and cannot touch the code either side of the whitespace.
    # Sending them to a model produced a line that kept the wrong indent AND
    # dropped a `*` from join(*lines) — a style request causing a TypeError.
    if symbol in STYLE_SYMBOLS:
        meaning, _, action, note = EXPLAIN.get(
            symbol, ("", "", f"fix {symbol}", "Style only."))
        return (meaning, "format", "autopep8 fixes this with the other style "
                "findings at the end of the run", note)
    if symbol in EXPLAIN:
        return EXPLAIN[symbol]
    return ("", "line", f"rewrite the line to satisfy: {symbol}", "")


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
        hits = []
        for m in wanted:
            meaning, kind, action, note = _explain(m["symbol"], m["message"], filename)
            hits.append({
                "line": m["line"], "symbol": m["symbol"], "message": m["message"],
                "meaning": meaning, "action_kind": kind, "action": action,
                "note": note,
                # kept for [r]aw — the string worth pasting into a search engine
                "raw": f"{p.name}:{m['line']}:{m.get('column', 0)}: "
                       f"{m.get('messageId', '')}: {m['message']} ({m['symbol']})",
            })
        return {"file": str(p), "symbol": symbol, "count": len(hits),
                "occurrences": hits if symbol == "*" else hits[:MAX_OCCURRENCES],
                "next": f"/lint {filename} {symbol} — step through these one at a time"}

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
        # Last line on purpose: a report is not an outcome. This is the way to
        # act on it — one finding at a time, with fix / skip / ignore / defer.
        "next": f"/lint {filename} — step through these interactively "
                f"(or /lint {filename} <symbol> for one kind)",
    }

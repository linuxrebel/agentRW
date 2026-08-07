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
import shutil
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


# ---------------------------------------------------------------------------
# The /lint command. Defined only when pylint is present, so without it the
# command is never registered and /lint falls through to the model rather than
# existing in a broken state. The gate is ordinary Python — no plugin API.
# ---------------------------------------------------------------------------
REQUIRES = {
    "pylint": {"pip": "pylint", "fedora": "python3-pylint", "debian": "pylint"},
}

if shutil.which("pylint"):

    def lint_command(ctx, args: str) -> None:
        """Walk lint findings one at a time: fix / skip / ignore / defer."""
        # Required of every plugin command: `/name help` explains itself.
        if args.strip().lower() in ("help", "-h", "--help"):
            print("""/lint <file> [symbol] [max=N]

Walks pylint findings one at a time. For each you choose:
  [f]ix          apply the change shown
  [s]kip         leave it, ask again next run
  [i]gnore kind  drop every finding of this kind for this run
  [d]efer        record it in DEBT.md and move on
  [r]aw          show pylint's original message, then ask again
  [q]uit         stop here, keep what has been applied

  symbol   only findings of one kind, e.g. /lint foo.py unused-import
  max=N    stop after N findings (default 20)

Style findings are not asked about — autopep8 fixes them all at the end.
If the finished file does not compile, the whole run is reverted.
Requires: pylint (findings), autopep8 (style fixes).""")
            return

        _fp = ["/lint"] + args.split()
        if len(_fp) < 2:
            print("[Lint] usage: /lint <file> [symbol] [max=N]")
            return
        _target = _fp[1]
        _only = next((t for t in _fp[2:] if "=" not in t), "")
        _cap = next((int(t.split("=")[1]) for t in _fp[2:]
                     if t.startswith("max=")), 20)
        _findings = ctx.gather_findings(_target, _only)
        if _findings and "error" in _findings[0]:
            print(f"[Lint] {_findings[0]['error']}")
            return
        if not _findings:
            print("[Lint] Nothing to fix.")
            return

        _path = ctx.resolve_path(_target)
        _snapshot = _path.read_text(encoding="utf-8")   # for end-of-run revert
        _fixed = _skipped = _deferred = _ignored = 0
        _done_kinds: set = set()
        print(f"[Lint] {len(_findings)} findings in {_path.name}, "
              f"working through up to {_cap}.")
        # Style findings are batched for autopep8 after the loop. Asking
        # about whitespace 11 times is noise, and there is nothing to decide.
        _style = [f for f in _findings if f.get("action_kind") == "format"]
        _findings = [f for f in _findings if f.get("action_kind") != "format"]
        if _style:
            print(f"[Lint] {len(_style)} style findings "
                  f"({', '.join(sorted({f['symbol'] for f in _style}))}) "
                  f"— autopep8 will fix these at the end, no questions.")

        for _f in _findings[:_cap]:
            if _f["symbol"] in _done_kinds:      # ignored mid-run
                _ignored += 1
                return
            _lines = _path.read_text(encoding="utf-8").splitlines(keepends=True)
            if _f["line"] > len(_lines):
                return
            _old = _lines[_f["line"] - 1].rstrip("\n")
            _kind0 = _f.get("action_kind", "line")
            if _kind0 == "line" and not _old.strip():
                # Rewriting a blank line has no meaning, and an empty old_str
                # is what used to wipe the file.
                print(f"\n{ctx.colour}[{_f['symbol']}]{ctx.reset} "
                      f"line {_f['line']}: blank line — nothing to rewrite, skipped")
                _skipped += 1
                return
            _same = sum(1 for x in _findings if x["symbol"] == _f["symbol"])
            print(f"\n{ctx.colour}[{_f['symbol']}]{ctx.reset} "
                  f"line {_f['line']}"
                  + (f"   ({_same} of this kind)" if _same > 1 else ""))
            _kind = _f.get("action_kind", "line")
            print(f"  meaning: {_f.get('meaning') or _f['message']}")
            print(f"  action : {_f.get('action', 'rewrite the line')}")
            if _f.get("note"):
                print(f"  note   : {_f['note']}")

            # action_kind decides HOW to fix, never whether to offer one.
            # Asking for a line rewrite regardless is how a rename finding
            # turned into a shebang edit.
            _new = ""
            if _kind.startswith(("reindent", "line", "insert")):
                _new = ctx.propose_fix(ctx.model, ctx.cfg, ctx.layers, _lines, _f)
                if not _new:
                    print("  (no fix could be produced)")
                elif _kind.startswith("reindent"):
                    print(f"  - {_old}")
                    print(f"  + {_new}   (computed, no model)")
                elif _kind == "line":
                    print(f"  - {_old}")
                    print(f"  + {_new}")
                else:
                    print(f"  + {_new}   (inserted, nothing overwritten)")
            elif _kind == "rename":
                _new = str(_path.with_name(
                    _path.stem.replace("-", "_").lower() + _path.suffix))
                print(f"  + {_path.name} -> {Path(_new).name}")

            _choices = ("  [f]ix / [s]kip / [i]gnore kind / [d]efer / [r]aw / [q]uit: "
                        if _new else
                        "  [s]kip / [i]gnore kind / [d]efer / [r]aw / [q]uit: ")
            while True:
                try:
                    _ans = input(_choices).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    _ans = "q"
                if _ans.startswith("r"):
                    print(f"  pylint : {_f.get('raw') or _f['message']}")
                    continue          # re-prompt, decision still pending
                break

            if _ans.startswith("q"):
                break
            if _new and _ans.startswith("f"):
                if _kind != "rename":
                    _r = ctx.apply_fix(_path, _lines, _f, _new)
                    print(f"  {ctx.summarise('write_file', _r)}")
                elif _kind == "rename":
                    _dest = Path(_new)
                    if not ctx.writable(_path) or not ctx.writable(_dest):
                        print(f"  {ctx.write_denied(_dest)['hint']}")
                    elif _dest.exists():
                        print(f"  {_dest.name} already exists — not renaming.")
                    else:
                        _path.rename(_dest)
                        print(f"  renamed -> {_dest}")
                        _path = _dest
                _fixed += 1
            elif _ans.startswith("i"):
                # "this doesn't matter" — drop the rest of this kind for the
                # rest of the run. Nothing is written anywhere.
                _done_kinds.add(_f["symbol"])
                print(f"  ignoring {_f['symbol']} for this run"
                      f"{f' ({_same} findings)' if _same > 1 else ''}")
                _ignored += 1
            elif _ans.startswith("d"):
                ctx.defer(_target, _f)
                print(f"  deferred -> {ctx.debt_file}")
                _deferred += 1
            else:
                _skipped += 1
        # After the model edits, not before: autopep8 also repairs the
        # indentation of anything the model inserted.
        if _style and "format_file" in ctx.tools:
            _fr = ctx.tools["format_file"](filename=str(_path))
            if "error" in _fr:
                print(f"\n[Lint] autopep8: {_fr['error']}")
            else:
                print(f"\n[Lint] autopep8 fixed {len(_style)} style findings "
                      f"({_fr.get('changed', 0)} lines changed, no model used).")
                _fixed += len(_style)

        print(f"\n[Lint] {_fixed} fixed, {_skipped} skipped, "
              f"{_ignored} ignored, {_deferred} deferred.")
        if not ctx.finish_run(_path, _snapshot):
            print(f"{ctx.colour}[Lint]{ctx.reset} The result no longer "
                  f"compiles — all changes reverted. {_path.name} is as it was.")
        return

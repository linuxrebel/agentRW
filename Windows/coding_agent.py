#!/usr/bin/env python3
# Requires Python 3.7+  (3.8+ recommended)

import sys

if sys.version_info < (3, 7):
    _v = sys.version.split()[0]
    print(f"""
coding_agent.py requires Python 3.7 or newer.

  Your Python : {_v}
  Required    : 3.7+ (3.8+ recommended)

How to resolve:
  • Install a newer Python via your package manager:
      Fedora/RHEL : sudo dnf install python3
      Debian/Ubuntu: sudo apt install python3
  • Use pyenv to manage multiple Python versions: https://github.com/pyenv/pyenv
  • Use a conda/mamba environment with a newer Python

Your system Python at {sys.executable} will not be changed — only this script needs a newer interpreter.
""")
    sys.exit(1)

import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI, APIConnectionError, BadRequestError, InternalServerError, APIStatusError


DEFAULT_MODEL = "deepseek-coder-v2:16b"

SYSTEM_PROMPT = """
You are a coding assistant with access to file tools.

TOOLS
=====
{{tool_list_repr}}

CRITICAL RULES — FOLLOW EXACTLY:

RULE 1: NEVER ask the user to provide file contents. NEVER ask clarifying questions before reading a file. If the user mentions a file path, READ IT IMMEDIATELY using read_file.

RULE 2: Use the FULL absolute path EXACTLY as given in the [PATHS] tag. Do NOT alter dots, dashes, underscores, extensions, or any character. Copy it character-for-character.
  [PATHS] says: /srv/project/my.app.v2.py
  You use:      /srv/project/my.app.v2.py  (NOT my-app-v2.py, NOT my_app.py, NOT /srv/project/my.app.v2)

RULE 2b: NEVER guess, invent, or recall a path from memory. If no [PATHS] tag is present, ask the user for the exact path before doing anything. Do not substitute paths from previous messages.

RULE 3: A tool call is ONE bare line starting with "tool:" — no markdown, no backticks, no prefix:
  CORRECT:   tool: read_file({"filename":"/srv/project/my.app.v2.py","start_line":1,"max_lines":200})
  WRONG:     ```tool: read_file(...)```
  WRONG:     A: tool: read_file(...)
  WRONG:     Here is the tool call: tool: read_file(...)

RULE 4: When making a tool call, your ENTIRE response is just that one line. Nothing else.

RULE 5: After a tool_result arrives, analyze it and either make another tool call or give your answer.

RULE 6: To modify a file, use write_file with the COMPLETE new file content. Do not use pseudocode or placeholders — only real, runnable code.
  tool: write_file({"filename":"/full/path/file.py","content":"#!/usr/bin/python3\n..."})

RULE 7: If a tool_result contains an error, fix the arguments and try again. Do not give up after one error.

RULE 8: NEVER ask for approval, permission, or confirmation before making changes. If the user asks you to fix, edit, update, or write something — DO IT IMMEDIATELY with write_file or edit_file. Do not describe what you plan to do and wait. Do not produce bullet lists of "proposed changes". Act.
  WRONG: "I need approval to write these fixes: ..."
  WRONG: "Here is what I will do: ..."
  CORRECT: tool: write_file({"filename":"...","content":"..."})

RULE 9: run_command runs as the current user only. NEVER use sudo, su, doas, pkexec, or runuser. NEVER attempt privilege escalation. The tool will block and report an error if you try.

RULE 10: When the user asks you to run a shell command (ls, cat, grep, git, python3, make, etc.), use run_command to execute it directly. Do NOT map shell commands to file tools. Do NOT recreate shell output in JSON. run_command returns real shell stdout/stderr — use that.
  User: "run git status" → tool: run_command({"cmd": "git status"})
  WRONG: tool: list_files({"path": "..."})

RULE 11: Complete ONLY what the user asked for, then STOP and report results. Do NOT read, open, or explore additional files unless explicitly instructed. After list_files returns a listing, present it and wait. After read_file returns content, summarize it and wait. Do not chain extra tool calls out of curiosity.
  User: "ls this dir" → list_files → report listing → STOP.
  WRONG: list_files → then read_file on a random entry you noticed.

RULE 12: Every user message contains a [CURRENT DIR: /some/path] tag. That is the active working directory — a real absolute path. When the user refers to "the current directory", "this directory", or ".", use that exact path verbatim in your tool call. NEVER substitute a placeholder like %CURRENT_DIR%, $CWD, {path}, or any other variable syntax. Copy the path character-for-character from the [CURRENT DIR] tag.
  [CURRENT DIR: /mnt/data/git/AI]
  CORRECT: tool: list_files({"path": "/mnt/data/git/AI"})
  WRONG:   tool: list_files({"path": "%CURRENT_DIR%"})
  WRONG:   tool: list_files({"path": "."})
"""

def _init_ansi() -> bool:
    """Enable ANSI color support. Returns True if colors are available."""
    if sys.platform != "win32":
        return True
    # Try colorama first (pip install colorama)
    try:
        import colorama
        colorama.init()
        return True
    except ImportError:
        pass
    # Fall back to enabling VT processing via ctypes (Windows 10 v1511+)
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False

_ANSI = _init_ansi()
YOU_COLOR       = "\033[94m" if _ANSI else ""
ASSISTANT_COLOR = "\033[93m" if _ANSI else ""
RESET_COLOR     = "\033[0m"  if _ANSI else ""

_CHARS_PER_TOKEN = 4
TOKEN_BUDGET = 8000  # max non-system tokens before proactive trim

LOW_VRAM_PRESET = {
    "max_tokens": 512,
    "num_ctx":    2048,
    "token_budget": 2000,
}


# -----------------------------
# Ollama client (OpenAI compat)
# -----------------------------
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# -----------------------------
# Path utils
# -----------------------------
_agent_cwd = [Path.cwd()]  # mutable so tools always pick up current value

def resolve_abs_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (_agent_cwd[0] / p).resolve()


# -----------------------------
# Tools
# -----------------------------
def read_file_tool(
    filename: str,
    start_line: int = 1,
    max_lines: int = 200,
) -> Dict[str, Any]:
    """Read lines from a file. Use full absolute paths."""
    path = resolve_abs_path(filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {
            "error": "file_not_found",
            "file_path": str(path),
            "hint": "Use the exact full absolute path the user provided.",
        }
    except PermissionError:
        return {"error": "permission_denied", "file_path": str(path)}
    except Exception as e:
        return {"error": str(e), "file_path": str(path)}

    total = len(lines)
    start = max(start_line - 1, 0)
    end = min(start + max_lines, total)
    return {
        "file_path": str(path),
        "start_line": start_line,
        "end_line": end,
        "total_lines": total,
        "has_more": end < total,
        "content": "".join(lines[start:end]),
    }


def list_files_tool(path: str) -> Dict[str, Any]:
    """List files in a directory."""
    p = resolve_abs_path(path)
    try:
        entries = []
        for x in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            entry = {
                "name": x.name,
                "path": str(x),
                "type": "dir" if x.is_dir() else "file",
            }
            if x.is_symlink():
                entry["symlink_target"] = str(x.resolve())
            entries.append(entry)
        return {"path": str(p), "files": entries}
    except FileNotFoundError:
        return {"error": "directory_not_found", "path": str(p)}
    except NotADirectoryError:
        return {"error": "not_a_directory", "path": str(p)}
    except PermissionError:
        return {"error": "permission_denied", "path": str(p)}
    except Exception as e:
        return {"error": str(e), "path": str(p)}


# Patterns that indicate the model produced pseudocode instead of real code
import re as _re  # used throughout; imported here once at module level
_PSEUDOCODE_PATTERNS = [
    r'^\s*\.\.\.\s*$',                 # lone ellipsis line (placeholder stub)
    r'#\s*(TODO|FIXME)\s*:',           # TODO/FIXME markers with colon
    r'<[A-Z][A-Z_]{1,}>',             # <PLACEHOLDER> tokens (2+ uppercase chars)
    r'\bsome\s+(code|logic|variables)\b',  # "some code here"
]

def _looks_like_pseudocode(text: str) -> bool:
    for pat in _PSEUDOCODE_PATTERNS:
        if _re.search(pat, text, _re.IGNORECASE | _re.MULTILINE):
            return True
    return False


def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """Edit a file by replacing old_str with new_str. Use empty old_str to create.
    new_str must be complete, valid code — no pseudocode, ellipsis, or placeholders."""
    p = resolve_abs_path(path)

    # Guard: reject obviously broken new_str before touching the file
    if _looks_like_pseudocode(new_str):
        return {
            "error": "invalid_new_str",
            "hint": (
                "new_str contains pseudocode or placeholders (e.g. '...', "
                "'declare variables'). Provide complete, runnable code only."
            ),
        }

    try:
        if old_str == "":
            p.write_text(new_str, encoding="utf-8")
            return {"path": str(p), "action": "created"}

        text = p.read_text(encoding="utf-8")
        if old_str not in text:
            return {"path": str(p), "action": "not_found",
                    "hint": "old_str was not found verbatim in the file."}

        # Write .bak only if none exists yet — preserves original across multiple edits
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8")

        p.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
        return {"path": str(p), "action": "edited", "backup": str(bak)}
    except FileNotFoundError:
        return {"error": "file_not_found", "path": str(p)}
    except PermissionError:
        return {"error": "permission_denied", "path": str(p)}
    except Exception as e:
        return {"error": str(e), "path": str(p)}


def run_command_tool(cmd: str, timeout: int = 30) -> Dict[str, Any]:
    """Run a shell command as the current user. Never elevates privileges. Returns stdout, stderr, returncode."""
    import re
    import subprocess
    if re.search(r'\b(sudo|su|doas|pkexec|runuser)\b', cmd):
        return {
            "error": "privilege_escalation_blocked",
            "hint": "Commands that escalate privileges are not permitted (sudo, su, doas, pkexec, runuser).",
        }
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "hint": f"Command exceeded {timeout}s. Use a shorter operation or increase timeout."}
    except Exception as e:
        return {"error": str(e)}


def search_file_tool(filename: str, text: str) -> Dict[str, Any]:
    """Search for text inside a file (case-insensitive)."""
    p = resolve_abs_path(filename)
    try:
        matches = []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if text.lower() in line.lower():
                    matches.append({"line": i, "content": line.rstrip()})
        return {"file_path": str(p), "matches": matches[:100]}
    except FileNotFoundError:
        return {"error": "file_not_found", "file_path": str(p)}
    except PermissionError:
        return {"error": "permission_denied", "file_path": str(p)}
    except Exception as e:
        return {"error": str(e), "file_path": str(p)}


def write_file_tool(filename: str, content: str) -> Dict[str, Any]:
    """Overwrite an entire file with new content. Always backs up the original first.
    Use this when you need to rewrite most of a file. content must be complete, valid code."""
    p = resolve_abs_path(filename)
    try:
        bak = p.with_suffix(p.suffix + ".bak")
        existed = p.exists()

        if existed:
            original = p.read_text(encoding="utf-8")
            # Only write .bak if none exists yet — preserves original across multiple write attempts
            if not bak.exists():
                bak.write_text(original, encoding="utf-8")

            # Guard: refuse if new content looks truncated (< 60% of original size)
            if len(content.strip()) < len(original.strip()) * 0.6:
                return {
                    "error": "suspicious_truncation",
                    "hint": (
                        f"New content ({len(content)} chars) is less than 60% of original "
                        f"({len(original)} chars). Refusing to write — likely truncated output. "
                        "Provide the COMPLETE file content or use edit_file for partial changes."
                    ),
                    "backup": str(bak),
                }

        # Guard: syntax-check Python files before touching disk
        if p.suffix == ".py":
            try:
                compile(content, str(p), "exec")
            except SyntaxError as e:
                return {
                    "error": "syntax_error",
                    "hint": f"New content has a Python syntax error at line {e.lineno}: {e.msg}. File NOT written.",
                }

        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "action": "written", "backup": str(bak) if existed else None}
    except PermissionError:
        return {"error": "permission_denied", "path": str(p)}
    except Exception as e:
        return {"error": str(e), "path": str(p)}


TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
    "search_file": search_file_tool,
    "write_file": write_file_tool,
    "run_command": run_command_tool,
}


# -----------------------------
# Prompt builder
# -----------------------------
def tool_repr(name: str) -> str:
    fn = TOOL_REGISTRY[name]
    return f"""
{name}
{inspect.signature(fn)}
{fn.__doc__}
"""


def build_prompt() -> str:
    tools = ""
    for name in TOOL_REGISTRY:
        tools += tool_repr(name) + "\n----------------\n"

    # Use str.replace to avoid conflicts with JSON braces in the prompt body.
    return SYSTEM_PROMPT.replace("{{tool_list_repr}}", tools) \
                        .replace("{{", "{").replace("}}", "}")


# -----------------------------
# Tool parsing
# -----------------------------
def _parse_kwargs_syntax(args_str: str) -> Dict[str, Any]:
    """Parse Python keyword-argument style: key="value", key=123, key='value'"""
    import re
    result = {}
    # Match key=<value> where value is a quoted string, number, or bare word
    pattern = re.compile(
        r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')'
        r'|(\w+)\s*=\s*([\w.+-]+)'
    )
    for m in pattern.finditer(args_str):
        key = m.group(1) or m.group(4)
        if m.group(2) is not None:
            try:
                val = m.group(2).encode("raw_unicode_escape").decode("unicode_escape")
            except (UnicodeDecodeError, ValueError):
                val = m.group(2)
        elif m.group(3) is not None:
            try:
                val = m.group(3).encode("raw_unicode_escape").decode("unicode_escape")
            except (UnicodeDecodeError, ValueError):
                val = m.group(3)
        else:
            raw = m.group(5)
            try:
                val = json.loads(raw)
            except Exception:
                val = raw
        result[key] = val
    return result


def _fix_json_newlines(s: str) -> str:
    """Escape bare newlines/tabs inside JSON string literals."""
    result = []
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_str:
            if c == '\\':
                result.append(c)
                i += 1
                if i < len(s):
                    result.append(s[i])
                i += 1
                continue
            if c == '"':
                in_str = False
                result.append(c)
            elif c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            else:
                result.append(c)
        else:
            if c == '"':
                in_str = True
            result.append(c)
        i += 1
    return ''.join(result)


def _try_targeted_extract(tool_name: str, args_str: str) -> Dict[str, Any]:
    """Greedy extraction for known tool shapes when JSON and kwargs both fail.
    Uses last-quote anchoring so embedded \" in content still parses."""
    import re

    def _unescape(s: str) -> str:
        return (s.replace('\\n', '\n').replace('\\t', '\t')
                 .replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\'))

    if tool_name in ("write_file", "read_file", "search_file"):
        fn = re.search(r'"filename"\s*:\s*"([^"]+)"', args_str)
        if not fn:
            return {}
        result: Dict[str, Any] = {"filename": fn.group(1)}
        if tool_name == "write_file":
            ct = re.search(r'"content"\s*:\s*"(.*)"', args_str, re.DOTALL)
            if ct:
                result["content"] = _unescape(ct.group(1))
        elif tool_name == "search_file":
            tx = re.search(r'"text"\s*:\s*"([^"]*)"', args_str)
            if tx:
                result["text"] = tx.group(1)
        return result

    if tool_name in ("list_files",):
        p = re.search(r'"path"\s*:\s*"([^"]+)"', args_str)
        return {"path": p.group(1)} if p else {}

    if tool_name == "edit_file":
        p  = re.search(r'"path"\s*:\s*"([^"]+)"', args_str)
        os = re.search(r'"old_str"\s*:\s*"(.*?)"(?=\s*,\s*"new_str")', args_str, re.DOTALL)
        ns = re.search(r'"new_str"\s*:\s*"(.*)"', args_str, re.DOTALL)
        if p and ns:
            return {
                "path": p.group(1),
                "old_str": _unescape(os.group(1)) if os else "",
                "new_str": _unescape(ns.group(1)),
            }

    return {}


def extract_tools(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    import re
    out = []
    for m in re.finditer(r'tool:\s*(\w+)\s*\(', text):
        name = m.group(1).strip()
        # String-aware balanced-paren scan — skips ( ) inside string literals
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            c = text[i]
            if c == '\\':
                i += 2
                continue
            if c in ('"', "'"):
                q = c
                i += 1
                while i < len(text):
                    if text[i] == '\\':
                        i += 2
                        continue
                    if text[i] == q:
                        break
                    i += 1
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            i += 1
        args_str = text[m.end():i - 1]
        # 1. Raw JSON
        try:
            args = json.loads(args_str)
        except Exception:
            # 2. JSON after fixing bare newlines/tabs in string values
            try:
                args = json.loads(_fix_json_newlines(args_str))
            except Exception:
                # 3. Greedy targeted extraction (handles embedded quotes in content)
                args = _try_targeted_extract(name, args_str)
                if not args:
                    # 4. Keyword-argument fallback (last resort)
                    args = _parse_kwargs_syntax(args_str)
                    if not args:
                        continue
        out.append((name, args))
    return out


# -----------------------------
# Context compaction
# -----------------------------
def estimate_tokens(messages: list) -> int:
    return sum(len(m.get("content", "")) for m in messages) // _CHARS_PER_TOKEN


def _compress_tool_result(content: str) -> str:
    """Replace bulk data in a tool_result with a compact stub."""
    import re as _r
    m = _r.match(r'tool_result\((\{.*\})\)$', content, _r.DOTALL)
    if not m:
        return content
    try:
        data = json.loads(m.group(1))
    except Exception:
        return content

    if "error" in data or data.get("compressed"):
        return content  # keep errors and already-compressed stubs

    stub: Dict[str, Any] = {"compressed": True}

    if "content" in data and "file_path" in data:
        stub["action"] = "read_file"
        stub["path"] = data.get("file_path", "")
        stub["lines_read"] = data.get("end_line", 0) - data.get("start_line", 1) + 1
        stub["total_lines"] = data.get("total_lines", "?")
        stub["has_more"] = data.get("has_more", False)
    elif "files" in data:
        stub["action"] = "list_files"
        stub["path"] = data.get("path", "")
        stub["count"] = len(data.get("files", []))
    elif "matches" in data:
        stub["action"] = "search_file"
        stub["path"] = data.get("file_path", "")
        stub["match_count"] = len(data.get("matches", []))
    elif "action" in data:
        stub["action"] = data["action"]
        stub["path"] = data.get("path", "")
        if data.get("backup"):
            stub["backup"] = data["backup"]
    else:
        return content

    return f"tool_result({json.dumps(stub)})"


def compact_tool_results(messages: list, keep_recent: int = 2) -> int:
    """Compress old tool_result messages in-place. Returns chars freed."""
    indices = [
        i for i, m in enumerate(messages)
        if m["role"] == "user" and m.get("content", "").startswith("tool_result(")
    ]
    to_compress = indices[:-keep_recent] if keep_recent else indices
    saved = 0
    for i in to_compress:
        original = messages[i]["content"]
        compressed = _compress_tool_result(original)
        if compressed != original:
            saved += len(original) - len(compressed)
            messages[i] = {**messages[i], "content": compressed}
    return saved


def proactive_trim(messages: list, budget_tokens: int = TOKEN_BUDGET) -> int:
    """Drop oldest non-system message pairs until under budget. Returns count dropped."""
    budget_chars = budget_tokens * _CHARS_PER_TOKEN
    dropped = 0
    while True:
        non_sys_chars = sum(
            len(m.get("content", "")) for m in messages if m["role"] != "system"
        )
        if non_sys_chars <= budget_chars:
            break
        non_sys_idx = [i for i, m in enumerate(messages) if m["role"] != "system"]
        if len(non_sys_idx) <= 2:
            break
        for i in sorted(non_sys_idx[:2], reverse=True):
            del messages[i]
        dropped += 2
    return dropped


# -----------------------------
# LLM call
# -----------------------------
def call_llm(model: str, messages: list, gpu_layers: "list[int | None]" = None,
             max_tokens: int = 2000, num_ctx: int | None = None,
             token_budget: int = TOKEN_BUDGET) -> str:
    # Proactively compact and trim before sending
    compact_tool_results(messages)
    trimmed = proactive_trim(messages, budget_tokens=token_budget)
    if trimmed:
        print(f"\n[Context] Proactively dropped {trimmed} old message(s). "
              f"~{estimate_tokens(messages):,} tokens remaining.")

    # gpu_layers is a mutable 1-element list so callers can observe auto-halving
    layers = gpu_layers[0] if gpu_layers else None

    working = list(messages)
    while True:
        try:
            options: Dict[str, Any] = {}
            if layers is not None:
                options["num_gpu"] = layers
            if num_ctx is not None:
                options["num_ctx"] = num_ctx
            r = client.chat.completions.create(
                model=model,
                messages=working,
                max_tokens=max_tokens,
                extra_body={"options": options} if options else None,
            )
            # Sync caller's list in-place if we trimmed anything
            if len(working) < len(messages):
                trimmed = len(messages) - len(working)
                print(f"\n[Context] Trimmed {trimmed} old message(s) to fit context window.")
                messages[:] = working
            return r.choices[0].message.content or ""
        except BadRequestError as e:
            if "exceed_context_size" in str(e) or getattr(e, "status_code", None) == 400:
                # Find oldest non-system messages and drop them
                non_sys = [i for i, m in enumerate(working) if m["role"] != "system"]
                if len(non_sys) <= 2:
                    print(f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} Context full and nothing left to trim.")
                    return ""
                # Drop oldest two (typically a user+assistant or user+tool_result pair)
                del working[non_sys[0]:non_sys[1]+1]
            else:
                print(f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} Bad request: {e}")
                return ""
        except APIConnectionError:
            print(
                f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} Could not connect to Ollama at "
                f"http://localhost:11434 — is it running?\n"
                f"  Start it with:  ollama serve\n"
                f"  Then verify:    ollama list\n"
            )
            return ""
        except InternalServerError as e:
            msg = str(e)
            if "out of memory" in msg or "cudaMalloc" in msg or "CUDA" in msg:
                current = layers if layers is not None else 99
                if current == 0:
                    print(
                        f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} Out of memory at 0 GPU layers — "
                        f"model too large for system RAM.\n"
                        f"  Use a smaller or more quantized model (e.g. Q4_K_S instead of Q4_K_M).\n"
                    )
                    return ""
                reduced = max(current // 2, 0)
                if gpu_layers is not None:
                    gpu_layers[0] = reduced
                    layers = reduced
                print(
                    f"\n{ASSISTANT_COLOR}[OOM]{RESET_COLOR} GPU out of memory. "
                    f"Reduced gpu_layers {current} → {reduced} and retrying...\n"
                    f"  (Use /gpu-layers 0 for CPU-only, or try a smaller model)\n"
                )
                continue
            else:
                print(f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} Ollama server error (500): {e}")
            return ""
        except APIStatusError as e:
            print(f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} API error {e.status_code}: {e}")
            return ""


# -----------------------------
# Multi-line input
# -----------------------------
try:
    from prompt_toolkit import prompt as _pt_prompt
    from prompt_toolkit.formatted_text import ANSI as _ANSI
    from prompt_toolkit.key_binding import KeyBindings as _KB

    _kb = _KB()

    @_kb.add('escape', 'enter')  # Alt+Enter → insert newline
    def _newline(event):
        event.current_buffer.insert_text('\n')

    @_kb.add('enter')  # Enter → submit
    def _submit(event):
        event.current_buffer.validate_and_handle()

    def _read_input(prompt_str: str) -> str:
        return _pt_prompt(_ANSI(prompt_str), key_bindings=_kb, multiline=True)

    _INPUT_HINT = " [Alt+Enter=newline, ``` block=multiline]"

except ImportError:
    def _read_input(prompt_str: str) -> str:
        return input(prompt_str)

    _INPUT_HINT = " [``` on its own line = start/end code block]"


def _collect_backtick_block() -> str:
    """Collect lines until a lone ``` then return joined content."""
    print(f"  (code block — type ``` alone to finish)")
    lines = []
    while True:
        try:
            line = input("  ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "```":
            break
        lines.append(line)
    return "\n".join(lines)


def _summarise_result(tool_name: str, result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}  {result.get('hint', '')}"
    if tool_name == "list_files":
        n = len(result.get("files", []))
        return f"{n} entries in {result.get('path', '?')}"
    if tool_name == "read_file":
        return (f"{result.get('total_lines', '?')} lines"
                f"  [{result.get('file_path', '?')}]"
                + ("  (has more)" if result.get("has_more") else ""))
    if tool_name == "write_file":
        bak = result.get("backup")
        return f"written → {result.get('path', '?')}" + (f"  (backup: {bak})" if bak else "")
    if tool_name == "edit_file":
        return f"{result.get('action', 'done')} → {result.get('path', '?')}"
    if tool_name == "search_file":
        return f"{len(result.get('matches', []))} matches in {result.get('file_path', '?')}"
    if tool_name == "run_command":
        rc = result.get("returncode", "?")
        out = (result.get("stdout") or "").strip()
        preview = out[:120].replace("\n", "↵") if out else "(no output)"
        return f"exit={rc}  {preview}"
    return json.dumps(result)[:120]


# -----------------------------
# Main loop
# -----------------------------
def run(model: str, gpu_layers: int | None = None,
        max_tokens: int = 2000, num_ctx: int | None = None,
        token_budget: int = TOKEN_BUDGET):
    layers_ref = [gpu_layers]  # mutable so call_llm can update it on OOM
    cfg = {"max_tokens": max_tokens, "num_ctx": num_ctx, "token_budget": token_budget}
    hints = []
    if gpu_layers is not None: hints.append(f"gpu_layers={gpu_layers}")
    if num_ctx is not None:    hints.append(f"num_ctx={num_ctx}")
    if max_tokens != 2000:     hints.append(f"max_tokens={max_tokens}")
    hint_str = ("  [" + ", ".join(hints) + "]") if hints else ""
    print(f"Using model: {model}{hint_str}{_INPUT_HINT}")

    messages = [
        {"role": "system", "content": build_prompt()}
    ]

    while True:
        try:
            user = _read_input(f"{YOU_COLOR}You:{RESET_COLOR} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # Triple-backtick on its own → collect multi-line block
        if user == "```":
            user = _collect_backtick_block().strip()
            if not user:
                continue

        if user.lower() in {"exit", "quit", "/bye", "/exit", "/quit", "bye"}:
            print("Goodbye.")
            return

        if user.lower().startswith("/model"):
            parts = user.split(None, 1)
            if len(parts) == 2:
                model = parts[1].strip()
                print(f"[Model] Switched to: {model}")
            else:
                print(f"[Model] Current model: {model}")
            continue

        if user.lower() == "/compact":
            saved = compact_tool_results(messages, keep_recent=0)
            dropped = proactive_trim(messages, budget_tokens=cfg["token_budget"])
            print(f"[Compact] {saved:,} chars freed from tool results. "
                  f"{dropped} messages dropped. "
                  f"~{estimate_tokens(messages):,} tokens remaining.")
            continue

        if user.lower() == "/tokens":
            print(f"[Context] ~{estimate_tokens(messages):,} tokens in history.")
            continue

        if user.lower() == "/low-vram":
            cfg.update(LOW_VRAM_PRESET)
            print(f"[Low-VRAM] Applied preset: max_tokens={cfg['max_tokens']}, "
                  f"num_ctx={cfg['num_ctx']}, token_budget={cfg['token_budget']}")
            continue

        if user.lower().startswith("/gpu-layers"):
            parts = user.split(None, 1)
            if len(parts) == 2:
                try:
                    layers_ref[0] = int(parts[1])
                    print(f"[GPU] gpu_layers set to {layers_ref[0]}. Takes effect next call.")
                except ValueError:
                    print("[GPU] Usage: /gpu-layers <int>  (0 = CPU only)")
            else:
                cur = layers_ref[0]
                print(f"[GPU] Current gpu_layers: {'Ollama default' if cur is None else cur}")
            continue

        if user.lower() == "/reset":
            messages.clear()
            messages.append({"role": "system", "content": build_prompt()})
            print("[Reset] Conversation cleared. Model and system prompt retained.")
            continue

        if user.lower() == "/pwd":
            print(f"[CWD] {_agent_cwd[0]}")
            continue

        if user.lower() == "/ops":
            import subprocess as _sp
            _r = _sp.run(["ollama", "ps"], capture_output=True, text=True)
            if _r.stdout:
                print(_r.stdout, end="")
            if _r.stderr:
                print(_r.stderr, end="")
            continue

        if user.lower() == "/olist":
            import subprocess as _sp
            _r = _sp.run(["ollama", "list"], capture_output=True, text=True)
            if _r.stdout:
                print(_r.stdout, end="")
            if _r.stderr:
                print(_r.stderr, end="")
            continue

        if user.lower() == "/update":
            import subprocess as _sp, urllib.request as _ur, json as _json, re as _vre
            try:
                _vr = _sp.run(["ollama", "--version"], capture_output=True, text=True)
                _local = _vre.search(r"[\d.]+", _vr.stdout or "")
                _local = _local.group(0) if _local else "unknown"
                with _ur.urlopen("https://api.github.com/repos/ollama/ollama/releases/latest", timeout=8) as _resp:
                    _latest = _json.loads(_resp.read())["tag_name"].lstrip("v")
                if _local == _latest:
                    print(f"[Ollama] Version is current ({_local})")
                else:
                    print(f"[Ollama] Update needed — installed: {_local}  latest: {_latest}")
            except Exception as _e:
                print(f"[Ollama] Check failed: {_e}")
            continue

        # cd <path> — updates agent working directory; relative paths resolve against current cwd
        if user.lower().startswith("cd ") or user.lower().startswith("/cd "):
            parts = user.split()
            target = parts[1] if len(parts) > 1 else ""
            if not target or target == "~":
                _agent_cwd[0] = Path.home()
            else:
                candidate = Path(target).expanduser()
                if not candidate.is_absolute():
                    candidate = (_agent_cwd[0] / candidate).resolve()
                if candidate.is_dir():
                    _agent_cwd[0] = candidate
                else:
                    print(f"[CWD] Not a directory: {candidate}")
                    print(f"[CWD] Current dir is: {_agent_cwd[0]}  (use ~/... for home-relative paths)")
                    continue
            print(f"[CWD] {_agent_cwd[0]}")
            if len(parts) > 2:
                print(f"[CWD] Note: only the path was used. Send the rest as a separate message.")
            continue

        # Shell command passthrough — if first word is an executable in PATH, run it directly
        # cd is handled above (needs agent state). Everything else: universal passthrough.
        import shutil as _shutil, subprocess as _sp
        _first = user.split()[0] if user.split() else ""
        if _first and _shutil.which(_first):
            try:
                result = _sp.run(
                    user, shell=True, capture_output=True, text=True,
                    cwd=str(_agent_cwd[0]), timeout=60
                )
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="")
                if result.returncode != 0:
                    print(f"[exit {result.returncode}]")
            except KeyboardInterrupt:
                print()
            continue

        if not user:
            continue

        # Inject current working dir so model always knows where it is
        cwd_block = f"\n\n[CURRENT DIR: {_agent_cwd[0]}]"

        # If the message mentions a file or directory path, echo it back
        # explicitly so the model cannot misread or hallucinate the path.
        # Matches: /a/b, /a/b/c.py, ~/bin/foo — at least two path components.
        detected = _re.findall(r'(?:~|/[\w.~-]+)(?:/[\w.~-]+)+', user)
        if detected:
            path_block = "\n".join(f"  {p}" for p in detected)
            injected = (
                user +
                f"\n\n[PATHS — copy these character-for-character, do NOT change dots, dashes, or extensions]\n"
                f"{path_block}\n"
                "[Use read_file on each path immediately. No questions.]" +
                cwd_block
            )
            messages.append({"role": "user", "content": injected})
        else:
            messages.append({"role": "user", "content": user + cwd_block})

        consecutive_errors = 0
        tool_calls_this_turn = 0
        MAX_TOOL_CALLS = 4
        while True:
            print("\nThinking...")

            try:
                reply = call_llm(model, messages, gpu_layers=layers_ref,
                                 max_tokens=cfg["max_tokens"], num_ctx=cfg["num_ctx"],
                                 token_budget=cfg["token_budget"])
            except KeyboardInterrupt:
                print("\n[Cancelled]")
                messages.pop()
                break

            if not reply:
                # Connection error already printed; drop back to user prompt
                messages.pop()
                break

            tools = extract_tools(reply)

            if not tools:
                print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {reply}")
                messages.append({"role": "assistant", "content": reply})
                consecutive_errors = 0
                break

            # Hard cap: prevent runaway tool-call loops
            tool_calls_this_turn += len(tools)
            if tool_calls_this_turn > MAX_TOOL_CALLS:
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM] You have made {tool_calls_this_turn} tool calls this turn. "
                        f"Stop making tool calls immediately. Summarize what you found and give your final answer now."
                    )
                })
                final = call_llm(model, messages, gpu_layers=layers_ref,
                                 max_tokens=cfg["max_tokens"], num_ctx=cfg["num_ctx"],
                                 token_budget=cfg["token_budget"])
                if final:
                    print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {final}")
                    messages.append({"role": "assistant", "content": final})
                break

            # Record assistant's tool-call turn before injecting results
            messages.append({"role": "assistant", "content": reply})

            turn_had_error = False
            for name, args in tools:
                fn = TOOL_REGISTRY.get(name)
                if not fn:
                    result = {"error": "unknown_tool", "name": name}
                    turn_had_error = True
                else:
                    # Validate required args before calling to catch garbage parses
                    missing = [
                        p for p, param in inspect.signature(fn).parameters.items()
                        if param.default is inspect.Parameter.empty and p not in args
                    ]
                    if missing:
                        result = {"error": f"missing_required_args: {missing}",
                                  "hint": f"Required: {missing}. Got: {list(args.keys())}"}
                        turn_had_error = True
                    else:
                        print(f"[tool] {name} {args}")
                        try:
                            result = fn(**args)
                            if "error" in result:
                                turn_had_error = True
                        except TypeError as e:
                            result = {"error": f"bad_arguments: {e}"}
                            turn_had_error = True
                        except Exception as e:
                            result = {"error": str(e)}
                            turn_had_error = True

                print(f"[result] {_summarise_result(name, result)}")
                messages.append({
                    "role": "user",
                    "content": f"tool_result({json.dumps(result)})"
                })

            if turn_had_error:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    print(f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} 3 consecutive tool-error turns — "
                          "breaking loop to prevent infinite retry.")
                    break
            else:
                consecutive_errors = 0


def print_help():
    lv = LOW_VRAM_PRESET
    print(f"""
coding_agent.py — local LLM coding assistant (Ollama / OpenAI-compat API)

USAGE
  coding_agent.py [MODEL] [OPTIONS]
  cagent [MODEL] [OPTIONS]          (if symlinked)

─────────────────────────────────────────────────────
STARTUP OPTIONS
─────────────────────────────────────────────────────
  MODEL                Ollama model tag (default: {DEFAULT_MODEL})
                       Example: qwen2.5-coder:7b-instruct-q4_K_M

  --gpu-layers N       Number of model layers to load onto GPU.
                         0  = CPU only (no VRAM used)
                         N  = partial offload (splits across GPU+CPU)
                       Unset = Ollama decides (usually all layers).
                       On CUDA OOM, agent auto-halves N and retries.

  --num-ctx N          KV-cache context window in tokens.
                       Lower = less VRAM. Default: model's built-in value.
                       Example: --num-ctx 2048

  --max-tokens N       Max tokens the model may generate per reply.
                       Default: 2000. Lower this if replies get cut off
                       due to VRAM pressure.

  --low-vram           Preset for 4 GB GPU:
                         num_ctx={lv['num_ctx']}, max_tokens={lv['max_tokens']},
                         token_budget={lv['token_budget']}
                       Recommended with MoE models (deepseek-coder-v2:16b)
                       or small quants (qwen2.5-coder:7b-instruct-q4_K_M).

  --set-default MODEL  Save MODEL as the new default in config and exit.
                       Example: cagent --set-default qwen2.5-coder:7b-instruct-q4_K_M

  -h, --help           Show this help and exit.

─────────────────────────────────────────────────────
EXAMPLES
─────────────────────────────────────────────────────
  cagent                                        # default model, Ollama picks GPU split
  cagent qwen2.5-coder:7b-instruct-q4_K_M      # specific model
  cagent qwen2.5-coder:7b-instruct-q4_K_M --low-vram        # 4 GB GPU preset
  cagent deepseek-coder-v2:16b --low-vram                    # MoE, 4 GB GPU
  cagent mistral:7b --gpu-layers 20                          # partial offload
  cagent mistral:7b --num-ctx 1024 --max-tokens 256          # minimal VRAM
  cagent mistral:7b --gpu-layers 0                           # CPU-only, no VRAM

─────────────────────────────────────────────────────
INPUT MODES
─────────────────────────────────────────────────────
  Enter              Submit message
  Alt+Enter          Insert newline without submitting
                     (requires: pip install prompt_toolkit)
  ``` (alone)        Toggle multi-line block mode.
                     Type your text, then ``` alone on a line to submit.

─────────────────────────────────────────────────────
SLASH COMMANDS  (type during session)
─────────────────────────────────────────────────────
  /model             Show the active model name
  /model <name>      Switch model mid-session (history preserved)
                     Example: /model qwen2.5-coder:7b-instruct-q4_K_M

  /gpu-layers        Show current gpu_layers value
  /gpu-layers <N>    Change gpu_layers live without restarting
                     Example: /gpu-layers 0   (switch to CPU-only)

  /low-vram          Apply 4 GB preset live:
                     num_ctx={lv['num_ctx']}, max_tokens={lv['max_tokens']}, token_budget={lv['token_budget']}

  /compact           Manually trigger context compaction:
                     - Compresses old tool results to short stubs
                     - Trims oldest messages if over token budget
                     Run this if the model starts forgetting earlier context.

  /tokens            Show estimated token count of current conversation history

  /reset             Wipe conversation history entirely.
                     Keeps model, GPU settings, and system prompt.
                     Use when starting a completely new task.

  <shell cmd>        Any command found in $PATH runs directly — bypasses LLM entirely.
                     Runs in the agent's current working directory.
                     Examples: ls   cat file.py   git status   python3 script.py

  cd <path>          Change the agent's working directory.
  /cd <path>         Same as cd — either form works.
                     Relative paths resolve from current working dir.
                     Symlinks are followed. All subsequent file tool
                     calls resolve relative paths against this dir.
                     Example: cd /home/james/git/AI
                     Example: cd ..

  /pwd               Show the agent's current working directory.

  /ops               Show loaded Ollama models and GPU vs CPU memory usage.
                     Runs: ollama ps
                     Note: requires at least one prompt sent first or output will be empty.

  /olist             List all locally installed Ollama models.
                     Runs: ollama list

  /update             Check if Ollama is up to date.
                     Compares installed version against latest GitHub release.
                     Prints "version is current" or "update needed".

  /bye               Exit the agent  (also: exit, quit, /exit, /quit)

─────────────────────────────────────────────────────
MODEL TOOLS  (invoked automatically — you don't call these)
─────────────────────────────────────────────────────
  read_file    filename, [start_line=1], [max_lines=200]
               Read a file. Mention any file path in your message and
               the model will read it automatically before responding.
               Guards: path must exist; backup (.bak) never overwritten
               once created.

  write_file   filename, content
               Overwrite entire file with new content.
               Guards:
                 • .bak written on first call only (original preserved
                   across all retry attempts)
                 • Refuses if new content < 60% of original size
                   (prevents truncation-by-accident)
                 • Syntax-checks .py files before touching disk

  edit_file    path, old_str, new_str
               Replace first occurrence of old_str with new_str.
               Safer than write_file for small changes — only touches
               the matched region. Pass empty old_str to create a new file.
               Same .bak guard as write_file.

  search_file  filename, text
               Case-insensitive search inside a file.
               Returns up to 100 matching lines with line numbers.

  list_files   path
               List files and directories at a path.

  run_command  cmd, [timeout=30]
               Run a shell command as YOU (inherits your uid/gid and $PATH).
               Returns: stdout (capped 4000 chars), stderr (capped 2000 chars),
               returncode.
               Blocked: sudo, su, doas, pkexec, runuser — no privilege
               escalation ever permitted.
               Timeout default: 30s. Model can request longer for slow builds.

─────────────────────────────────────────────────────
CONTEXT MANAGEMENT
─────────────────────────────────────────────────────
  Token budget   : {TOKEN_BUDGET:,} tokens — history auto-trimmed before each LLM call
  Tool results   : compressed to stubs after first use to save space
  OOM recovery   : GPU layers auto-halved and request retried on CUDA OOM
  Manual compact : /compact to reclaim space mid-session

─────────────────────────────────────────────────────
RECOMMENDED MODELS FOR 4 GB VRAM
─────────────────────────────────────────────────────
  qwen2.5-coder:7b-instruct-q4_K_M   Best instruction-following, good code
  phi4-mini                           Fast, small, decent code quality
  starcoder2:3b                       Tiny, pure code completion
  deepseek-coder-v2:16b               MoE arch — low active params, fits 4 GB
                                      (avoid for instruction-following; wanders)
""")


if sys.platform == "win32":
    _CONFIG_PATH = Path(os.environ.get("APPDATA", Path.home())) / "coding_agent" / "config.json"
else:
    _CONFIG_PATH = Path.home() / ".config" / "coding_agent" / "config.json"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _build_default_string(model: str, flags: dict) -> str:
    parts = [model]
    if flags.get("low_vram"):                           parts.append("--low-vram")
    if flags.get("gpu_layers") is not None:             parts += ["--gpu-layers", str(flags["gpu_layers"])]
    if flags.get("num_ctx") is not None:                parts += ["--num-ctx", str(flags["num_ctx"])]
    if flags.get("max_tokens", 2000) != 2000:           parts += ["--max-tokens", str(flags["max_tokens"])]
    return " ".join(parts)


def _parse_default_string(s: str) -> tuple:
    import shlex
    parts = shlex.split(s)
    model = parts[0]
    flags: dict = {}
    i = 1
    while i < len(parts):
        t = parts[i]
        if t == "--low-vram":
            flags["low_vram"] = True
        elif t in ("--gpu-layers", "--num-ctx", "--max-tokens") and i + 1 < len(parts):
            key = t.lstrip("-").replace("-", "_")
            flags[key] = int(parts[i + 1])
            i += 1
        i += 1
    return model, flags


def _resolve_model(passed: str | None, cli_flags: dict) -> tuple:
    """Returns (model, effective_flags_dict)."""
    cfg = _load_config()
    this_str = _build_default_string(passed, cli_flags) if passed else None

    # No model passed on CLI
    if passed is None:
        saved = cfg.get("default")
        if saved:
            model, saved_flags = _parse_default_string(saved)
            # CLI flags override saved flags
            effective = {**saved_flags, **{k: v for k, v in cli_flags.items() if v not in (None, False, 2000)}}
            return model, effective
        print("First run: please provide a model name and any arguments. Run --help for options.")
        sys.exit(0)

    # Model passed — no config yet, save as default
    if not cfg.get("default"):
        cfg["default"] = this_str
        _save_config(cfg)
        print(f"[Config] Default set to: {this_str}")
        return passed, cli_flags

    # Compare full string (model + flags) — prompt if anything differs
    if this_str != cfg["default"] and cfg.get("ask_new_default", True):
        print(f"\nCurrent default : {cfg['default']}")
        print(f"This run        : {this_str}")
        answer = ""
        while answer not in ("y", "n", "d"):
            answer = input("Make this the new default? [y]es / [N]o (default) / [d]on't ask again: ").strip().lower()
            if not answer:
                answer = "n"
        if answer == "y":
            cfg["default"] = this_str
            _save_config(cfg)
            print(f"[Config] Default updated to: {this_str}")
        elif answer == "d":
            cfg["ask_new_default"] = False
            _save_config(cfg)
            print("[Config] Won't ask again. Use --set-default to change the default.")

    return passed, cli_flags


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--gpu-layers", type=int, default=None,
                        help="Max GPU layers (0=CPU only). Unset = Ollama default.")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="Context window size. Lower = less VRAM.")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Max output tokens per reply (default 2000).")
    parser.add_argument("--low-vram", action="store_true",
                        help=f"4GB VRAM preset: num_ctx={LOW_VRAM_PRESET['num_ctx']}, "
                             f"max_tokens={LOW_VRAM_PRESET['max_tokens']}, "
                             f"token_budget={LOW_VRAM_PRESET['token_budget']}")
    parser.add_argument("--set-default", metavar="MODEL",
                        help="Set a new default model in config and exit.")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    if args.set_default:
        cfg = _load_config()
        cfg["default"] = args.set_default
        _save_config(cfg)
        print(f"[Config] Default set to: {args.set_default}")
        sys.exit(0)

    cli_flags = dict(
        low_vram=args.low_vram,
        gpu_layers=args.gpu_layers,
        num_ctx=args.num_ctx,
        max_tokens=args.max_tokens,
    )
    model, effective_flags = _resolve_model(args.model, cli_flags)

    kwargs = dict(
        gpu_layers=effective_flags.get("gpu_layers"),
        num_ctx=effective_flags.get("num_ctx"),
        max_tokens=effective_flags.get("max_tokens", 2000),
        token_budget=TOKEN_BUDGET,
    )
    if effective_flags.get("low_vram"):
        kwargs.update(LOW_VRAM_PRESET)
        if effective_flags.get("num_ctx") is not None:   kwargs["num_ctx"]    = effective_flags["num_ctx"]
        if effective_flags.get("max_tokens", 2000) != 2000: kwargs["max_tokens"] = effective_flags["max_tokens"]
    run(model, **kwargs)

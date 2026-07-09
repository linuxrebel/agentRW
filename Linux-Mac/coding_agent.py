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
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI, APIConnectionError, BadRequestError, InternalServerError, APIStatusError


DEFAULT_MODEL = "deepseek-coder-v2:16b"

SYSTEM_PROMPT = """
You are a coding assistant with file tools. Act immediately — never ask permission.

TOOLS
=====
{{tool_list_repr}}

RULES
=====
1. NEVER output code as text. When asked to write/create a file, call write_file — do not display the code in chat.
   User: "create /home/james/foo.py" → tool: write_file({"filename":"/home/james/foo.py","content":"#!/usr/bin/env python3\n..."})
2. Tool call = one bare line, nothing else: tool: function_name({"key":"value"})
3. Paths: copy character-for-character from [PATHS] tags. Never guess or alter dots/dashes/extensions.
4. write_file: COMPLETE content only — no pseudocode, no placeholders, no ellipsis.
5. Newlines in write_file content must be \\n escapes, not literal newlines.
6. Never use sudo/su/doas/pkexec. run_command runs as current user only.
7. Shell commands (ls, git, grep, python3…): use run_command, not file tools.
8. Read files immediately when a path is mentioned — never ask the user to paste contents.
9. [CURRENT DIR: /path] in each message = working directory. Copy it verbatim.
10. On tool_result error: fix args and retry. Do not give up after one error.
11. Do only what was asked, then stop.
12. If write_file fails repeatedly, output the complete corrected code as a fenced code block (```python\\n...\\n```) — it will be saved to the requested path automatically.
"""

# Binaries that exist on most systems but are common English words —
# they misbehave (hang/block/no-op) when given natural language as arguments.
_PASSTHROUGH_SKIP = {"read", "write", "wait", "test", "true", "false"}


def _init_ansi() -> bool:
    """Enable ANSI color support. Returns True if colors are available."""
    if sys.platform != "win32":
        return True
    try:
        import colorama
        colorama.init()
        return True
    except ImportError:
        pass
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
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
    "max_tokens": 1800,
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


def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """Edit a file by replacing old_str with new_str. Use empty old_str to create.
    new_str must be complete, valid code — no pseudocode, ellipsis, or placeholders."""
    p = resolve_abs_path(path)
    try:
        if old_str == "":
            p.write_text(new_str, encoding="utf-8")
            return {"path": str(p), "action": "created"}

        text = p.read_text(encoding="utf-8")
        if old_str not in text:
            return {"path": str(p), "action": "not_found",
                    "hint": "old_str was not found verbatim in the file."}

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
            if not bak.exists():
                bak.write_text(original, encoding="utf-8")

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

        if p.suffix == ".py":
            try:
                compile(content, str(p), "exec")
            except SyntaxError as e:
                _hint = (
                    f"New content has a Python syntax error at line {e.lineno}: {e.msg}. File NOT written. "
                    "Common cause: a \\n escape inside a single-quoted string becomes a real newline, "
                    "breaking f-strings. Fix: use double-quoted strings (\"...\") or triple-quoted strings (\"\"\"...\"\"\") "
                    "instead of single-quoted strings when the content contains \\n escapes."
                )
                return {"error": "syntax_error", "hint": _hint}

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
    return SYSTEM_PROMPT.replace("{{tool_list_repr}}", tools) \
                        .replace("{{", "{").replace("}}", "}")


# -----------------------------
# Tool parsing
# -----------------------------
def _parse_kwargs_syntax(args_str: str) -> Dict[str, Any]:
    """Parse Python keyword-argument style: key="value", key=123, key='value'"""
    result = {}
    pattern = re.compile(
        r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\''
        r'|(\w+)\s*=\s*([\w.+-]+))'
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
    Uses last-quote anchoring so embedded quotes in content still parse."""
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
    out = []
    for m in re.finditer(r'tool\d*:\s*(\w+)\s*\(', text):
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
        if not isinstance(args, dict):
            continue
        out.append((name, args))
    return out


# -----------------------------
# Context compaction
# -----------------------------
def estimate_tokens(messages: list) -> int:
    return sum(len(m.get("content", "")) for m in messages) // _CHARS_PER_TOKEN


def compact_tool_results(messages: list, keep_recent: int = 2) -> int:
    """Truncate old tool_result messages in-place. Returns chars freed."""
    indices = [
        i for i, m in enumerate(messages)
        if m["role"] == "user" and m.get("content", "").startswith("tool_result(")
    ]
    to_compress = indices[:-keep_recent] if keep_recent else indices
    saved = 0
    for i in to_compress:
        orig = messages[i]["content"]
        if len(orig) > 300:
            compressed = orig[:300] + "…"
            saved += len(orig) - len(compressed)
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
    compact_tool_results(messages)
    trimmed = proactive_trim(messages, budget_tokens=token_budget)
    if trimmed:
        print(f"\n[Context] Proactively dropped {trimmed} old message(s). "
              f"~{estimate_tokens(messages):,} tokens remaining.")

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
            if len(working) < len(messages):
                trimmed = len(messages) - len(working)
                print(f"\n[Context] Trimmed {trimmed} old message(s) to fit context window.")
                messages[:] = working
            return r.choices[0].message.content or ""
        except BadRequestError as e:
            if "exceed_context_size" in str(e) or getattr(e, "status_code", None) == 400:
                non_sys = [i for i, m in enumerate(working) if m["role"] != "system"]
                if len(non_sys) <= 2:
                    print(f"\n{ASSISTANT_COLOR}[Error]{RESET_COLOR} Context full and nothing left to trim.")
                    return ""
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
                _new_model = parts[1].strip()
                _ol = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                _known = [ln.split()[0] for ln in _ol.stdout.splitlines()[1:] if ln.strip()]
                if _new_model in _known:
                    model = _new_model
                    print(f"[Model] Switched to: {model}. Switch will complete once you run the first command with this model.")
                else:
                    print(f"[Model] Not found locally: {_new_model}")
                    print(f"[Model] Run `ollama pull {_new_model}` to download it, or /olist to see available models.")
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

        if user.lower() in {"/help", "-h", "--help"}:
            print(
                "/help       /model      /gpu-layers  /low-vram\n"
                "/compact    /tokens     /reset       /pwd\n"
                "/ops        /olist      /update      /bye\n"
                "cd <path>"
            )
            continue

        if user.lower() == "/ops":
            _r = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
            _lines = [l for l in _r.stdout.splitlines() if l.strip()]
            if len(_lines) <= 1:
                print("[Ops] No models currently loaded — send a prompt first to load the model into memory.")
            else:
                print(_r.stdout, end="")
            if _r.stderr:
                print(_r.stderr, end="")
            continue

        if user.lower() == "/olist":
            _r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if _r.stdout:
                print(_r.stdout, end="")
            if _r.stderr:
                print(_r.stderr, end="")
            continue

        if user.lower() == "/update":
            try:
                _vr = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
                _local = re.search(r"[\d.]+", _vr.stdout or "")
                _local = _local.group(0) if _local else "unknown"
                _cr = subprocess.run(
                    ["curl", "-sf", "--max-time", "8",
                     "https://api.github.com/repos/ollama/ollama/releases/latest"],
                    capture_output=True, text=True, timeout=10
                )
                if _cr.returncode != 0 or not _cr.stdout:
                    print("[Ollama] Version check failed — no network or curl unavailable.")
                else:
                    _latest = json.loads(_cr.stdout)["tag_name"].lstrip("v")
                    if _local == _latest:
                        print(f"[Ollama] Version is current ({_local})")
                    else:
                        print(f"[Ollama] Update needed — installed: {_local}  latest: {_latest}")
            except Exception as _e:
                print(f"[Ollama] Check failed: {_e}")
            continue

        # cd <path> — updates agent working directory
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
        _first = user.split()[0] if user.split() else ""
        if _first and _first not in _PASSTHROUGH_SKIP and shutil.which(_first):
            try:
                result = subprocess.run(
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

        # If message mentions a file/directory path, echo it back so model can't misread it
        detected = [p.rstrip('.,;:!?)>') for p in re.findall(r'(?:~|/[\w.~-]+)(?:/[\w.~-]+)+', user)]
        if detected:
            path_block = "\n".join(f"  {p}" for p in detected)
            injected = (
                user +
                f"\n\n[PATHS — copy these character-for-character, do NOT change dots, dashes, or extensions]\n"
                f"{path_block}\n"
                "[Use these exact paths in your tool calls. Do not modify them.]" +
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
                messages.pop()
                break

            tools = extract_tools(reply)

            if not tools:
                print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {reply}")
                messages.append({"role": "assistant", "content": reply})
                consecutive_errors = 0

                # Auto-save fenced code blocks to paths mentioned in user message
                blocks = re.findall(r'```(?:\w+)?\n(.*?)```', reply, re.DOTALL)
                if blocks and detected:
                    for path_str in detected:
                        p = resolve_abs_path(path_str)
                        if p.suffix:  # has extension → it's a file target
                            result = write_file_tool(str(p), blocks[0].rstrip('\n'))
                            print(f"[auto-saved] {_summarise_result('write_file', result)}")
                            break

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
                    result = {"error": "unknown_tool", "name": name,
                              "hint": f"Available tools: {list(TOOL_REGISTRY.keys())}"}
                    turn_had_error = True
                else:
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

OPTIONS
  MODEL              Ollama model tag (default: {DEFAULT_MODEL})
  --gpu-layers N     GPU layers (0=CPU only; auto-halved on OOM)
  --num-ctx N        Context window size (lower = less VRAM)
  --max-tokens N     Max output tokens per reply (default: 2000)
  --low-vram         4 GB preset: num_ctx={lv['num_ctx']}, max_tokens={lv['max_tokens']}, token_budget={lv['token_budget']}
  --set-default M    Save M as default model and exit
  -h, --help         Show this help

SLASH COMMANDS
  /help  /model  /gpu-layers  /low-vram  /compact  /tokens
  /reset  /pwd  /ops  /olist  /update  /bye  cd <path>

MODEL TOOLS  (invoked automatically by the model)
  read_file    write_file    edit_file
  search_file  list_files    run_command

EXAMPLES
  coding_agent qwen2.5-coder:7b-instruct-q4_K_M --low-vram
  coding_agent deepseek-coder-v2:16b --low-vram
  coding_agent mistral:7b --gpu-layers 0
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


def _flags_to_str(flags: dict) -> str:
    parts = []
    if flags.get("low_vram"):                            parts.append("--low-vram")
    if flags.get("gpu_layers") is not None:              parts += ["--gpu-layers", str(flags["gpu_layers"])]
    if flags.get("num_ctx") is not None:                 parts += ["--num-ctx", str(flags["num_ctx"])]
    if flags.get("max_tokens", 2000) != 2000:            parts += ["--max-tokens", str(flags["max_tokens"])]
    return " ".join(parts)


def _resolve_model(passed: str | None, cli_flags: dict) -> tuple:
    """Returns (model, effective_flags_dict)."""
    cfg = _load_config()
    saved = cfg.get("default")  # {"model": "...", "low_vram": ..., ...}

    if passed is None:
        if saved and isinstance(saved, dict) and "model" in saved:
            model = saved["model"]
            saved_flags = {k: v for k, v in saved.items() if k != "model"}
            effective = {**saved_flags, **{k: v for k, v in cli_flags.items() if v not in (None, False, 2000)}}
            return model, effective
        if saved and isinstance(saved, str):
            print("Config format updated. Please re-run with a model name to set a new default.")
        else:
            print("First run: please provide a model name and any arguments. Run --help for options.")
        sys.exit(0)

    new_default = {"model": passed, **cli_flags}
    if not saved:
        cfg["default"] = new_default
        _save_config(cfg)
        print(f"[Config] Default set to: {passed}")
        return passed, cli_flags

    saved_model = saved.get("model", "?") if isinstance(saved, dict) else str(saved)
    saved_flags = {k: v for k, v in saved.items() if k != "model"} if isinstance(saved, dict) else {}
    saved_display = f"{saved_model} {_flags_to_str(saved_flags)}".strip()
    this_display = f"{passed} {_flags_to_str(cli_flags)}".strip()

    if new_default != saved and cfg.get("ask_new_default", True):
        print(f"\nCurrent default : {saved_display}")
        print(f"This run        : {this_display}")
        answer = ""
        while answer not in ("y", "n", "d"):
            answer = input("Make this the new default? [y]es / [N]o (default) / [d]on't ask again: ").strip().lower()
            if not answer:
                answer = "n"
        if answer == "y":
            cfg["default"] = new_default
            _save_config(cfg)
            print(f"[Config] Default updated to: {this_display}")
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
        cfg["default"] = {"model": args.set_default}
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
        if effective_flags.get("num_ctx") is not None:        kwargs["num_ctx"]    = effective_flags["num_ctx"]
        if effective_flags.get("max_tokens", 2000) != 2000:   kwargs["max_tokens"] = effective_flags["max_tokens"]
    run(model, **kwargs)

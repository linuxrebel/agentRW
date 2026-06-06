#!/usr/bin/env python3
# ollama-open-windows.py - Windows-compatible version of ollama-open (unsandboxed)
#
# Sibling: ollama-fsb-windows.py (sandboxed variant).
#
# Differences from ollama-open.py (Linux/Mac):
#   1. Uses colorama to enable ANSI escape codes in Windows terminals.
#   2. Agent Modelfiles are stored in %APPDATA%\ollama-fs\agents\.
#   3. Uses subprocess.run() instead of os.system() for `ollama create`.
#   4. Persistent shell is PowerShell (not bash); supports cd, ls, cat, git, etc.
#   5. Non-blocking shell read uses a thread + queue.Queue (Windows has no
#      select() on pipes - only sockets).
#
# Same unsandboxed semantics as the Linux/Mac version: no sandbox, full
# user-level filesystem access, shell-exec via run_shell_command, persistent
# cwd / env vars / PowerShell session state across calls.

import os
import sys
import time
import json
import difflib
import argparse
import subprocess
import threading
import itertools
import queue
import uuid
import ollama

# ---------------------------------------------------------------------------
# Windows ANSI support via colorama
# ---------------------------------------------------------------------------
try:
    import colorama
    colorama.init()
except ImportError:
    print("ERROR: 'colorama' is not installed.")
    print("Run:  pip install -r requirements-windows.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='ollama-open (Windows)',
    description=(
        'Terminal CLI agent that gives a local Ollama model full read/write '
        'and shell-exec access at the privilege of the invoking user. '
        'No sandbox - the agent can touch any path the user can.'
    ),
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        'Examples:\n'
        '  ollama-open-windows.py -m gemma4 C:\\Users\\me\\projects\\foo\n'
        '  ollama-open-windows.py -m gemma4 -a openclaw C:\\\n'
        '  ollama-open-windows.py -m gemma4 . --safe\n'
        '  ollama-open-windows.py -m gemma4 . --safe --debug\n'
        '  ollama-open-windows.py -a openclaw ~\n'
    )
)

parser.add_argument('-m', '--model',  metavar='MODEL', default=None,
                    help='Ollama model name to run  (e.g. gemma4, llama3)')
parser.add_argument('-a', '--agent',  metavar='AGENT', default=None,
                    help='Named Ollama agent / modelfile to launch (e.g. openclaw)')
parser.add_argument('start_dir',
                    help='Starting directory: where relative paths resolve from and where shell commands begin')
parser.add_argument('--safe',  action='store_true',
                    help='Safe mode: confirm every write, move, delete, and shell command before running')
parser.add_argument('--debug', action='store_true',
                    help='Append a JSONL trace to .ollama-fs-debug.jsonl in the starting directory')

args = parser.parse_args()

if args.agent and args.model:
    AGENT_NAME, MODEL_NAME, CHAT_TARGET = args.agent, args.model, args.agent
elif args.agent:
    AGENT_NAME, MODEL_NAME, CHAT_TARGET = args.agent, None, args.agent
elif args.model:
    AGENT_NAME, MODEL_NAME, CHAT_TARGET = None, args.model, args.model
else:
    parser.error('You must supply at least -m/--model or -a/--agent (or both).')

SAFE_MODE  = args.safe
DEBUG_MODE = args.debug
CWD = os.path.abspath(os.path.expanduser(args.start_dir))

if not os.path.isdir(CWD):
    parser.error(f"Starting directory does not exist or is not a directory: {CWD}")

os.chdir(CWD)

# Session cwd - mutable; tracks where the persistent shell currently is.
# File tools also resolve relative paths against this, so they follow `cd`.
SESSION_CWD = CWD

DEBUG_LOG_PATH = os.path.join(CWD, '.ollama-fs-debug.jsonl')

# %APPDATA%\ollama-open\agents\ (falls back to home if APPDATA unset)
_appdata = os.environ.get('APPDATA') or os.path.expanduser('~')
AGENTS_DIR = os.path.join(_appdata, 'ollama-open', 'agents')

# Shell defaults
SHELL_TIMEOUT_DEFAULT = 120
SHELL_OUTPUT_CAP = 8000

# ---------------------------------------------------------------------------
# Debug logger
# ---------------------------------------------------------------------------
def debug_log(event, payload):
    if not DEBUG_MODE:
        return
    record = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'event': event, **payload}
    try:
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    except Exception as exc:
        sys.stderr.write(f'[debug_log error] {exc}\n')

# ---------------------------------------------------------------------------
# Spinner (colorama makes \r and \033[K work on Windows)
# ---------------------------------------------------------------------------
class LoadingSpinner:
    def __init__(self, message="Thinking..."):
        self.message = message
        self.spinner = itertools.cycle(['-', '\\', '|', '/'])
        self.running = False
        self.thread  = None

    def _spin(self):
        while self.running:
            sys.stdout.write(f"\r{next(self.spinner)} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

# ---------------------------------------------------------------------------
# Starting-directory manifest (informational only; not a sandbox boundary)
# ---------------------------------------------------------------------------
def get_top_level_manifest():
    try:
        items = os.listdir(CWD)
        items = [i for i in items if i not in {'.ollama-fs-debug.jsonl'}]
        if not items:
            return f"Starting directory ({CWD}) is empty."
        lines = [f"Starting directory: {CWD}"]
        for item in sorted(items):
            path = os.path.join(CWD, item)
            if os.path.isdir(path):
                lines.append(f"  [Directory] {item}\\")
            else:
                lines.append(f"  [File]      {item}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error scanning starting directory: {e}"

# ---------------------------------------------------------------------------
# Tool definitions - paths may be absolute, relative to start_dir, or ~-expanded
# ---------------------------------------------------------------------------
_PATH_DOC = (
    "Path to operate on. May be absolute (e.g. C:\\Windows\\System32\\drivers\\etc\\hosts), "
    "relative to the starting directory, or use ~ for the user's home (e.g. ~/Documents/foo.txt). "
    "Forward slashes also work."
)

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'list_directory_contents',
            'description': 'List the files and subdirectories of any directory the invoking user can read.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string', 'description': _PATH_DOC}},
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_local_file',
            'description': 'Read the complete text content of any file the invoking user can read.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string', 'description': _PATH_DOC}},
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_local_file',
            'description': (
                'Write text content to a file. Use mode="overwrite" (default) to replace '
                'the file contents, or mode="append" to add to the end of an existing file '
                '(creates it if absent). Parent directories are created automatically. '
                'Use append mode when accumulating output across multiple calls, such as '
                'building up a report or log file incrementally.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'path':    {'type': 'string', 'description': _PATH_DOC},
                    'content': {'type': 'string', 'description': 'Exact text content to write or append.'},
                    'mode':    {
                        'type': 'string',
                        'enum': ['overwrite', 'append'],
                        'description': 'overwrite (default) replaces file contents; append adds to the end.',
                    },
                },
                'required': ['path', 'content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'move_local_file',
            'description': (
                'Move or rename a file or directory. Same directory + new name renames; '
                'different directory moves. Parent directories of the destination are created automatically.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'source_path':      {'type': 'string', 'description': _PATH_DOC},
                    'destination_path': {'type': 'string', 'description': _PATH_DOC},
                },
                'required': ['source_path', 'destination_path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'delete_local_file',
            'description': 'Permanently delete a single file (not a directory). Irreversible.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string', 'description': _PATH_DOC}},
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'run_shell_command',
            'description': (
                'Execute a PowerShell command with the same privileges as the invoking user. '
                'The PowerShell session is persistent: cd, $env: variables, defined functions, '
                'and sourced scripts carry across calls. Use this for git, ls/dir, cat/Get-Content, '
                'package managers, network tools - anything available in PowerShell.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The PowerShell command line to execute (e.g. "ls C:\\Windows", "git status", "python script.py").',
                    },
                    'timeout_seconds': {
                        'type': 'integer',
                        'description': f'Optional command timeout. Defaults to {SHELL_TIMEOUT_DEFAULT}s.',
                    },
                },
                'required': ['command'],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Path resolution - no sandbox, just expansion and normalisation
# ---------------------------------------------------------------------------
def _resolve(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(SESSION_CWD, expanded))

# ---------------------------------------------------------------------------
# File tool implementations
# ---------------------------------------------------------------------------
def list_directory_contents(path: str) -> str:
    abs_path = _resolve(path)
    if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
        return f"Error: '{abs_path}' does not exist or is not a directory."
    try:
        items = os.listdir(abs_path)
        if not items:
            return f"The directory '{abs_path}' is empty."
        lines = [f"Contents of '{abs_path}':"]
        for item in sorted(items):
            item_path = os.path.join(abs_path, item)
            kind = 'Directory' if os.path.isdir(item_path) else 'File'
            suffix = '\\' if os.path.isdir(item_path) else ''
            lines.append(f"  [{kind:9}] {item}{suffix}")
        result = "\n".join(lines)
        debug_log('tool_call', {'op': 'list', 'path': abs_path, 'result_lines': len(lines)})
        return result
    except PermissionError:
        return f"Error: Permission denied listing '{abs_path}'."
    except Exception as e:
        return f"Error reading directory: {e}"


def read_local_file(path: str) -> str:
    abs_path = _resolve(path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        debug_log('tool_call', {'op': 'read', 'path': abs_path, 'bytes': len(content)})
        return content
    except PermissionError:
        return f"Error: Permission denied reading '{abs_path}'."
    except FileNotFoundError:
        return f"Error: File not found: '{abs_path}'."
    except UnicodeDecodeError:
        return f"Error: '{abs_path}' is not a UTF-8 text file (binary?). Use run_shell_command for binary inspection."
    except Exception as e:
        return f"Error reading file: {e}"


def write_local_file(path: str, content: str, mode: str = 'overwrite') -> str:
    abs_path = _resolve(path)
    append_mode = (mode == 'append')

    if SAFE_MODE:
        if append_mode:
            existing_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0
            print(f"\n[Safe Mode] Append to '{abs_path}'  (existing size: {existing_size} bytes)")
            preview_lines = content.splitlines()[:20]
            print('--- content to append ---')
            print('\n'.join(preview_lines))
            if len(content.splitlines()) > 20:
                print(f"  ... ({len(content.splitlines()) - 20} more lines)")
        elif os.path.exists(abs_path):
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    existing_lines = f.readlines()
            except Exception:
                existing_lines = []
            new_lines = content.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                existing_lines, new_lines,
                fromfile=f'{abs_path} (current)',
                tofile=f'{abs_path} (proposed)',
            ))
            if diff:
                print(f"\n[Safe Mode] Proposed changes to '{abs_path}':")
                print(''.join(diff))
            else:
                print(f"\n[Safe Mode] No changes to '{abs_path}'. Skipping.")
                return f"Skipped: No changes detected in {abs_path}"
        else:
            print(f"\n[Safe Mode] New file: '{abs_path}'")
            preview_lines = content.splitlines()[:20]
            print('\n'.join(preview_lines))
            if len(content.splitlines()) > 20:
                print(f"  ... ({len(content.splitlines()) - 20} more lines)")

        action_word = 'append' if append_mode else 'write'
        while True:
            answer = input(f"\n  Confirm {action_word}? [y/n] > ").strip().lower()
            if answer in ('y', 'yes'):
                break
            elif answer in ('n', 'no'):
                debug_log('write_denied', {'path': abs_path, 'mode': mode})
                return f"{action_word.capitalize()} cancelled by user for '{abs_path}'."
            else:
                print("  Please enter y or n.")

    try:
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        open_mode = 'a' if append_mode else 'w'
        with open(abs_path, open_mode, encoding='utf-8') as f:
            f.write(content)
        debug_log('tool_call', {'op': 'write', 'path': abs_path, 'mode': mode, 'bytes': len(content)})
        if append_mode:
            return f"Success: Appended {len(content)} bytes to {abs_path}"
        return f"Success: Written to {abs_path}"
    except PermissionError:
        return f"Error: Permission denied writing '{abs_path}'."
    except Exception as e:
        return f"Error writing file: {e}"


def move_local_file(source_path: str, destination_path: str) -> str:
    src  = _resolve(source_path)
    dest = _resolve(destination_path)

    if not os.path.exists(src):
        return f"Error: Source '{src}' does not exist."
    if os.path.exists(dest):
        return f"Error: Destination '{dest}' already exists. Remove it first or choose a different name."

    action = "Rename" if os.path.dirname(src) == os.path.dirname(dest) else "Move"

    if SAFE_MODE:
        print(f"\n[{action}] About to {action.lower()} '{src}'  ->  '{dest}'")
        while True:
            answer = input(f"  {action}? [N/y] > ").strip().lower()
            if answer in ('y', 'yes'):
                break
            elif answer in ('n', 'no', ''):
                debug_log('move_denied', {'src': src, 'dest': dest})
                return f"{action} cancelled."
            else:
                print("  Please enter y or n (Enter = No).")

    try:
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.rename(src, dest)
        debug_log('tool_call', {'op': 'move', 'src': src, 'dest': dest})
        return f"Success: Moved '{src}' -> '{dest}'"
    except PermissionError:
        return f"Error: Permission denied moving '{src}'."
    except Exception as e:
        return f"Error moving file: {e}"


def delete_local_file(path: str) -> str:
    abs_path = _resolve(path)
    if not os.path.exists(abs_path):
        return f"Error: '{abs_path}' does not exist."
    if os.path.isdir(abs_path):
        return f"Error: '{abs_path}' is a directory. Only individual files can be deleted via this tool. Use run_shell_command + Remove-Item -Recurse if you really mean it."

    if SAFE_MODE:
        print(f"\n[Delete] About to permanently delete '{abs_path}'.")
        print( "  This operation cannot be undone.")
        while True:
            answer = input("  Delete? [N/y] > ").strip().lower()
            if answer in ('y', 'yes'):
                break
            elif answer in ('n', 'no', ''):
                debug_log('delete_denied', {'path': abs_path})
                return f"Delete cancelled for '{abs_path}'."
            else:
                print("  Please enter y or n (Enter = No).")

    try:
        os.remove(abs_path)
        debug_log('tool_call', {'op': 'delete', 'path': abs_path})
        return f"Success: Deleted '{abs_path}'"
    except PermissionError:
        return f"Error: Permission denied deleting '{abs_path}'."
    except Exception as e:
        return f"Error deleting file: {e}"


# ---------------------------------------------------------------------------
# Persistent PowerShell session
# ---------------------------------------------------------------------------
# Windows has no select() on pipes, so a background daemon thread drains
# stdout into a Queue. The run_shell_command read loop pulls from the queue
# with a timeout instead of blocking on select.
#
# PowerShell is launched with -Command - which makes it read script from
# stdin and execute as commands. After each user command we emit a sentinel
# line carrying exit code and $PWD.Path so we can detect completion and
# track the shell's current directory.
# ---------------------------------------------------------------------------
_SHELL = None
_SHELL_QUEUE = None
_SHELL_READER = None

def _shell_reader_loop(proc, q):
    """Daemon thread: pumps stdout lines from PowerShell into a Queue."""
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            q.put(line)
    except Exception:
        pass
    finally:
        q.put(None)  # EOF sentinel

def _ensure_shell():
    global _SHELL, _SHELL_QUEUE, _SHELL_READER
    if _SHELL is None or _SHELL.poll() is not None:
        # CREATE_NEW_PROCESS_GROUP lets us signal the shell with CTRL_BREAK
        # without affecting our own process.
        creationflags = 0
        if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        _SHELL = subprocess.Popen(
            ['powershell.exe', '-NoLogo', '-NoProfile', '-Command', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=SESSION_CWD,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        _SHELL_QUEUE = queue.Queue()
        _SHELL_READER = threading.Thread(
            target=_shell_reader_loop, args=(_SHELL, _SHELL_QUEUE), daemon=True
        )
        _SHELL_READER.start()
        # Quiet the shell.
        try:
            _SHELL.stdin.write('$ErrorActionPreference = "Continue"\n')
            _SHELL.stdin.flush()
        except Exception:
            pass
    return _SHELL

def _restart_shell():
    global _SHELL, _SHELL_QUEUE, _SHELL_READER
    if _SHELL is not None:
        try:
            _SHELL.kill()
        except Exception:
            pass
    _SHELL = None
    _SHELL_QUEUE = None
    _SHELL_READER = None


def run_shell_command(command: str, timeout_seconds=None) -> str:
    global SESSION_CWD
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        timeout_seconds = SHELL_TIMEOUT_DEFAULT

    if SAFE_MODE:
        print(f"\n[Shell] About to execute in {SESSION_CWD}:")
        print(f"  PS> {command}")
        while True:
            answer = input("  Execute? [N/y] > ").strip().lower()
            if answer in ('y', 'yes'):
                break
            elif answer in ('n', 'no', ''):
                debug_log('shell_denied', {'command': command})
                return f"Shell command cancelled: {command}"
            else:
                print("  Please enter y or n (Enter = No).")

    proc = _ensure_shell()
    marker = f"__OLLAMA_FS_DONE_{uuid.uuid4().hex}__"

    # Sentinel logic for PowerShell:
    #   $__success = $?              # last command success bool
    #   $__lec = $LASTEXITCODE       # native-command exit code (or stale)
    #   $__rc = if ($__lec) { $__lec } elseif ($__success) { 0 } else { 1 }
    sentinel_emit = (
        f'$__success = $?; '
        f'$__lec = $LASTEXITCODE; '
        f'$__rc = if ($__lec) {{ $__lec }} elseif ($__success) {{ 0 }} else {{ 1 }}; '
        f'Write-Output "{marker} $__rc $($PWD.Path)"'
    )

    try:
        proc.stdin.write(command + '\n')
        proc.stdin.write(sentinel_emit + '\n')
        proc.stdin.flush()
    except (BrokenPipeError, OSError) as e:
        _restart_shell()
        return f"Error: shell pipe broken ({e}). Shell restarted; please retry."

    output_lines = []
    exit_code = -1
    new_pwd = None
    deadline = time.time() + timeout_seconds
    timed_out = False

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            timed_out = True
            break
        try:
            line = _SHELL_QUEUE.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            continue
        if line is None:
            # shell died (EOF from reader thread)
            break
        if marker in line:
            # parse "MARKER <rc> <pwd>"
            idx = line.find(marker)
            tail = line[idx + len(marker):].strip()
            parts = tail.split(None, 1)
            try:
                exit_code = int(parts[0])
                if len(parts) >= 2:
                    new_pwd = parts[1].strip()
            except (ValueError, IndexError):
                pass
            break
        output_lines.append(line)

    if timed_out:
        _restart_shell()
        debug_log('shell_timeout', {'command': command, 'timeout': timeout_seconds, 'cwd': SESSION_CWD})
        return (
            f"Error: Command timed out after {timeout_seconds}s. "
            f"Shell session restarted; cwd reset to {SESSION_CWD}."
        )

    if new_pwd and os.path.isdir(new_pwd) and new_pwd != SESSION_CWD:
        SESSION_CWD = new_pwd

    output = ''.join(output_lines)
    truncated = False
    if len(output) > SHELL_OUTPUT_CAP:
        output = output[:SHELL_OUTPUT_CAP] + f"\n... [output truncated at {SHELL_OUTPUT_CAP} chars]"
        truncated = True

    debug_log('tool_call', {
        'op': 'shell', 'command': command, 'exit_code': exit_code,
        'cwd': SESSION_CWD, 'bytes': len(output), 'truncated': truncated,
    })

    header = f"PS> {command}    (cwd: {SESSION_CWD})"
    parts_out = [header, f"[exit code: {exit_code}]"]
    if output:
        parts_out.append(output.rstrip())
    else:
        parts_out.append("(no output)")
    return "\n".join(parts_out)


# ---------------------------------------------------------------------------
# Agent scaffolding
# ---------------------------------------------------------------------------
_MODELFILE_TEMPLATE = """\
FROM {base_model}

SYSTEM \"\"\"
You are {agent_name}, a CLI assistant with the same filesystem and shell
privileges as the invoking user. You can read, write, move, and delete files
anywhere that user has permission, and you can execute PowerShell commands
the same way they would in their terminal. Use absolute paths when possible.
Be precise, explain your actions, and confirm intent before destructive ops.
\"\"\"

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.1
PARAMETER top_p 0.9
PARAMETER top_k 40
"""

def scaffold_agent(agent_name: str, base_model: str) -> str:
    agent_dir      = os.path.join(AGENTS_DIR, agent_name)
    modelfile_path = os.path.join(agent_dir, 'Modelfile')
    if not os.path.exists(agent_dir):
        os.makedirs(agent_dir, exist_ok=True)
        print(f"[Agent] Created agent directory: {agent_dir}")
    if not os.path.exists(modelfile_path):
        content = _MODELFILE_TEMPLATE.format(base_model=base_model, agent_name=agent_name)
        with open(modelfile_path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print(f"[Agent] Scaffolded Modelfile: {modelfile_path}")
        print(f"[Agent] Edit that file to customise '{agent_name}', then re-run to rebuild.")
    else:
        print(f"[Agent] Modelfile found: {modelfile_path}")
    return modelfile_path


def ensure_agent_built(agent_name: str, modelfile_path: str):
    sentinel_path  = os.path.join(os.path.dirname(modelfile_path), '.built')
    needs_build    = True
    if os.path.exists(sentinel_path):
        if os.path.getmtime(modelfile_path) <= os.path.getmtime(sentinel_path):
            needs_build = False
    if needs_build:
        print(f"[Agent] Building '{agent_name}' from Modelfile...")
        result = subprocess.run(
            ['ollama', 'create', agent_name, '-f', modelfile_path],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[Agent] ERROR: 'ollama create' failed (exit code {result.returncode}).")
            sys.exit(1)
        with open(sentinel_path, 'w') as fh:
            fh.write(time.strftime('%Y-%m-%dT%H:%M:%S'))
        print(f"[Agent] '{agent_name}' built successfully.")
    else:
        print(f"[Agent] '{agent_name}' is up to date.")

# ---------------------------------------------------------------------------
# Startup ping
# ---------------------------------------------------------------------------
def startup_ping(chat_target: str):
    spinner = LoadingSpinner(f"Verifying '{chat_target}' is reachable (may take a minute)...")
    spinner.start()
    try:
        probe = ollama.chat(
            model=chat_target,
            messages=[{'role': 'user', 'content': 'Reply with only the single word READY and nothing else.'}],
            stream=False,
        )
        spinner.stop()
        reply = probe.get('message', {}).get('content', '').strip()
        print(f"OK '{chat_target}' is ready  (replied: \"{reply}\")")
        debug_log('startup_ping', {'target': chat_target, 'reply': reply, 'status': 'ok'})
    except Exception as exc:
        spinner.stop()
        print(f"FAILED '{chat_target}' did not respond.\n")
        print(f"[Startup Error] {exc}")
        print("  * Is the Ollama app running?     -> Open Ollama from the Start menu")
        print(f"  * Is the model pulled?           -> ollama pull {chat_target}")
        print(f"  * Is the agent registered?       -> ollama list")
        debug_log('startup_ping', {'target': chat_target, 'error': str(exc), 'status': 'failed'})
        sys.exit(1)


if AGENT_NAME:
    if not MODEL_NAME:
        print(f"[Agent] No -m flag supplied; assuming '{AGENT_NAME}' is already built in Ollama.")
    else:
        modelfile_path = scaffold_agent(AGENT_NAME, MODEL_NAME)
        ensure_agent_built(AGENT_NAME, modelfile_path)

startup_ping(CHAT_TARGET)

# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------
manifest = get_top_level_manifest()
USER = os.environ.get('USERNAME') or os.environ.get('USER') or 'the invoking user'

messages = [{
    'role': 'system',
    'content': (
        f"You are a CLI agent with the same filesystem and shell privileges as the "
        f"invoking user ({USER}). You can read, write, move, and delete files "
        f"anywhere that user can, including absolute paths like C:\\Windows, "
        f"C:\\Users\\{USER}, etc. You can also execute PowerShell commands as that "
        f"user via run_shell_command - the PowerShell session is persistent, so cd, "
        f"$env: vars, and defined functions carry across calls. Paths may be "
        f"absolute, relative to the starting directory ({CWD}), or use ~ for the "
        f"user's home. Prefer absolute paths when the user names a specific "
        f"location. Explain destructive actions before performing them.\n\n"
        f"IMPORTANT - REPORTING TOOL RESULTS:\n"
        f"When a tool returns data, report it accurately. Do NOT summarize, "
        f"paraphrase, or invent tool output. If the user asks you to ls, dir, "
        f"cat, Get-Content, list, show, or display something, reproduce the "
        f"actual tool output - never make up filenames or content based on what "
        f"you 'expect' a typical Windows system to contain. The tool already "
        f"truncates long output if needed, so if you got data back, that is the "
        f"real data.\n\n"
        f"Starting directory snapshot:\n{manifest}"
    )
}]

debug_log('session_start', {
    'chat_target': CHAT_TARGET,
    'model':       MODEL_NAME,
    'agent':       AGENT_NAME,
    'start_dir':   CWD,
    'safe_mode':   SAFE_MODE,
    'user':        USER,
})


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
print(f"Model:    {MODEL_NAME or '(via agent)'}")
if AGENT_NAME:
    print(f"Agent:    {AGENT_NAME}")
print(f"Target:   {CHAT_TARGET}")
print(f"User:     {USER}  (no sandbox - full user-level access)")
print(f"Start:    {CWD}")
print(f"Shell:    powershell.exe (persistent session)")
if SAFE_MODE:
    print("Mode:     [SAFE] - writes / moves / deletes / shell exec require confirmation")
if DEBUG_MODE:
    print(f"Debug:    logging to {DEBUG_LOG_PATH}")
print(f"\n{manifest}")
print("\n----------------------------------------------------------------")
print("Enter commands below. Type 'exit' or '/bye' to quit.")
print("----------------------------------------------------------------\n")

# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------
while True:
    try:
        user_prompt = input("User Input > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if user_prompt.lower() in ['exit', '/bye']:
        print("Goodbye!")
        break
    if not user_prompt:
        continue

    messages.append({'role': 'user', 'content': user_prompt})
    debug_log('user_input', {'content': user_prompt})
    spinner = LoadingSpinner("Working...")

    try:
        while True:
            spinner.start()
            response_stream = ollama.chat(
                model=CHAT_TARGET,
                messages=messages,
                tools=tools,
                stream=True,
            )

            full_response_message = None
            tool_calls    = []
            ai_reply_chunks = []

            for chunk in response_stream:
                msg_chunk = chunk.get('message', {})

                if msg_chunk.get('tool_calls'):
                    spinner.stop()
                    tool_calls.extend(msg_chunk['tool_calls'])
                    if not full_response_message:
                        full_response_message = msg_chunk
                    else:
                        full_response_message['tool_calls'].extend(msg_chunk['tool_calls'])

                if msg_chunk.get('content'):
                    if spinner.running:
                        spinner.stop()
                        sys.stdout.write("Assistant > ")
                        sys.stdout.flush()
                    sys.stdout.write(msg_chunk['content'])
                    sys.stdout.flush()
                    ai_reply_chunks.append(msg_chunk['content'])

            spinner.stop()

            if tool_calls:
                messages.append(full_response_message)
                for tool in tool_calls:
                    func_name = tool['function']['name']
                    args_     = tool['function']['arguments']

                    if func_name == 'move_local_file':
                        log_label = f"'{args_.get('source_path')}' -> '{args_.get('destination_path')}'"
                    elif func_name == 'run_shell_command':
                        log_label = f"PS> {str(args_.get('command', ''))[:80]}"
                    else:
                        log_label = f"'{args_.get('path', '')}'"
                    print(f"\n[System: {func_name} on {log_label}]")

                    if func_name == 'list_directory_contents':
                        result = list_directory_contents(args_.get('path', ''))
                    elif func_name == 'read_local_file':
                        result = read_local_file(args_.get('path', ''))
                    elif func_name == 'write_local_file':
                        result = write_local_file(
                            args_.get('path', ''),
                            args_.get('content', ''),
                            args_.get('mode', 'overwrite'),
                        )
                    elif func_name == 'move_local_file':
                        result = move_local_file(args_.get('source_path', ''), args_.get('destination_path', ''))
                    elif func_name == 'delete_local_file':
                        result = delete_local_file(args_.get('path', ''))
                    elif func_name == 'run_shell_command':
                        result = run_shell_command(
                            args_.get('command', ''),
                            args_.get('timeout_seconds'),
                        )
                    else:
                        result = f"Unknown tool: {func_name}"

                    debug_log('tool_result', {'tool': func_name, 'args': args_, 'result_preview': result[:200]})
                    messages.append({'role': 'tool', 'content': result, 'name': func_name})
                continue
            else:
                if ai_reply_chunks:
                    print("\n")
                    full_reply = "".join(ai_reply_chunks)
                    messages.append({'role': 'assistant', 'content': full_reply})
                    debug_log('assistant_reply', {'content': full_reply[:500]})
                break

    except KeyboardInterrupt:
        spinner.stop()
        _restart_shell()
        print("\n[Cancelled by Ctrl-C. Returning to prompt.]")
        debug_log('user_cancel', {})
        while messages and messages[-1].get('role') != 'system':
            popped = messages.pop()
            if popped.get('role') == 'user':
                break
        continue

    except Exception as e:
        spinner.stop()
        print(f"\nError: {e}")
        debug_log('error', {'message': str(e)})
        if messages and messages[-1]['role'] == 'user':
            messages.pop()

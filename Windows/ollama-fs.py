#!/usr/bin/env python3
# ollama-fs-windows.py — Windows-compatible version of ollama-fs
#
# Differences from ollama-fs.py:
#   1. Uses colorama to enable ANSI escape codes in Windows terminals
#      (cmd.exe, PowerShell, and Windows Terminal all work with colorama).
#   2. Agent Modelfiles are stored in %APPDATA%\ollama-fs\agents\ instead of
#      ~/.config/ollama-fs/agents/ — the standard Windows config location.
#   3. Uses subprocess.run() instead of os.system() for `ollama create` so
#      paths containing spaces are handled correctly on Windows.

import os
import sys
import time
import json
import difflib
import argparse
import threading
import itertools
import subprocess
import ollama

# ---------------------------------------------------------------------------
# Windows ANSI support via colorama
# ---------------------------------------------------------------------------
try:
    import colorama
    colorama.init()   # Wraps stdout/stderr so ANSI codes work in all terminals
except ImportError:
    print("ERROR: 'colorama' is not installed.")
    print("Run:  pip install -r requirements-windows.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='ollama-fs',
    description='A terminal CLI agent that gives a local Ollama model sandboxed file-system access.',
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        'Examples:\n'
        '  ollama-fs-windows.py -m gemma4 .\\my_project\n'
        '  ollama-fs-windows.py -m gemma4 -a openclaw .\\my_project\n'
        '  ollama-fs-windows.py -m gemma4 .\\my_project --safe\n'
        '  ollama-fs-windows.py -m gemma4 .\\my_project --safe --debug\n'
        '  ollama-fs-windows.py -a openclaw .\\my_project\n'
    )
)

parser.add_argument('-m', '--model',  metavar='MODEL', default=None,
                    help='Ollama model name to run  (e.g. gemma4, llama3)')
parser.add_argument('-a', '--agent',  metavar='AGENT', default=None,
                    help='Named Ollama agent / modelfile to launch (e.g. openclaw)')
parser.add_argument('workspace',
                    help='Path to the sandboxed workspace folder')
parser.add_argument('--safe',  action='store_true',
                    help='Enable safe mode: preview every write and ask for confirmation before committing')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug logging to .ollama-fs-debug.jsonl inside the workspace')

args = parser.parse_args()

if args.agent and args.model:
    AGENT_NAME  = args.agent
    MODEL_NAME  = args.model
    CHAT_TARGET = args.agent
elif args.agent:
    AGENT_NAME  = args.agent
    MODEL_NAME  = None
    CHAT_TARGET = args.agent
elif args.model:
    AGENT_NAME  = None
    MODEL_NAME  = args.model
    CHAT_TARGET = args.model
else:
    parser.error('You must supply at least -m/--model or -a/--agent (or both).')

SAFE_MODE  = args.safe
DEBUG_MODE = args.debug
TARGET_DIR = os.path.abspath(args.workspace)

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

DEBUG_LOG_PATH = os.path.join(TARGET_DIR, '.ollama-fs-debug.jsonl')

# ---------------------------------------------------------------------------
# Windows config dir  (%APPDATA%\ollama-fs\agents\)
# Falls back to ~/.config equivalent if APPDATA is not set (unusual but safe).
# ---------------------------------------------------------------------------
_appdata = os.environ.get('APPDATA') or os.path.expanduser('~')
AGENTS_DIR = os.path.join(_appdata, 'ollama-fs', 'agents')

# ---------------------------------------------------------------------------
# Debug logger  (only active when --debug is passed)
# ---------------------------------------------------------------------------
def debug_log(event: str, payload: dict):
    """Append a single JSON line to the debug log file. No-op without --debug."""
    if not DEBUG_MODE:
        return
    record = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'event': event, **payload}
    try:
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    except Exception as exc:
        sys.stderr.write(f'[debug_log error] {exc}\n')

# ---------------------------------------------------------------------------
# Loading spinner
# colorama.init() above makes \r and \033[K work correctly in Windows terminals.
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
# Workspace manifest
# ---------------------------------------------------------------------------
def get_top_level_manifest():
    """Scans and presents only the immediate top level of the workspace."""
    try:
        items = os.listdir(TARGET_DIR)
        MANIFEST_HIDDEN = {'.ollama-fs-debug.jsonl', '.git'}
        items = [i for i in items if i not in MANIFEST_HIDDEN]
        if not items:
            return "The workspace root directory is currently empty."
        manifest_lines = ["Workspace Root:"]
        for item in sorted(items):
            path = os.path.join(TARGET_DIR, item)
            if os.path.isdir(path):
                manifest_lines.append(f"  [Directory] {item}/")
            else:
                manifest_lines.append(f"  [File]      {item}")
        return "\n".join(manifest_lines)
    except Exception as e:
        return f"Error scanning root directory: {str(e)}"

# ---------------------------------------------------------------------------
# Ollama tool definitions
# ---------------------------------------------------------------------------
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'list_directory_contents',
            'description': 'List the files and folders inside a specific subfolder path relative to the workspace root.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'relative_path': {
                        'type': 'string',
                        'description': 'The path of the subfolder to inspect (e.g., "src" or "src/components")'
                    }
                },
                'required': ['relative_path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_local_file',
            'description': 'Read the complete text content of a file. Provide the relative path from the workspace root.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'relative_path': {
                        'type': 'string',
                        'description': 'Path to the file relative to the workspace root'
                    }
                },
                'required': ['relative_path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_local_file',
            'description': 'Write or overwrite text content to a file. Handles path and folder creation automatically.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'relative_path': {
                        'type': 'string',
                        'description': 'Target file path relative to the workspace root'
                    },
                    'content': {
                        'type': 'string',
                        'description': 'The exact text content to write'
                    }
                },
                'required': ['relative_path', 'content'],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------
def _safe_resolve(relative_path: str) -> str | None:
    safe_path = os.path.abspath(os.path.join(TARGET_DIR, relative_path))
    if safe_path == TARGET_DIR or safe_path.startswith(TARGET_DIR + os.sep):
        return safe_path
    return None

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def list_directory_contents(relative_path: str) -> str:
    safe_path = _safe_resolve(relative_path)
    if safe_path is None:
        debug_log('sandbox_violation', {'op': 'list', 'path': relative_path})
        return "Error: Access denied. Cannot inspect directories outside the designated workspace."
    if not os.path.exists(safe_path) or not os.path.isdir(safe_path):
        return f"Error: The directory path '{relative_path}' does not exist or is not a folder."
    try:
        items = os.listdir(safe_path)
        if not items:
            return f"The directory '{relative_path}' is empty."
        lines = [f"Contents of folder '{relative_path}':"]
        for item in sorted(items):
            item_path = os.path.join(safe_path, item)
            lines.append(f"  [{'Directory' if os.path.isdir(item_path) else 'File':9}] {item}{'/' if os.path.isdir(item_path) else ''}")
        result = "\n".join(lines)
        debug_log('tool_call', {'op': 'list', 'path': relative_path, 'result_lines': len(lines)})
        return result
    except Exception as e:
        return f"Error reading directory items: {str(e)}"


def read_local_file(relative_path: str) -> str:
    safe_path = _safe_resolve(relative_path)
    if safe_path is None:
        debug_log('sandbox_violation', {'op': 'read', 'path': relative_path})
        return "Error: Access denied. Cannot read files outside the designated workspace."
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        debug_log('tool_call', {'op': 'read', 'path': relative_path, 'bytes': len(content)})
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_local_file(relative_path: str, content: str) -> str:
    safe_path = _safe_resolve(relative_path)
    if safe_path is None:
        debug_log('sandbox_violation', {'op': 'write', 'path': relative_path})
        return "Error: Access denied. Cannot write files outside the designated workspace."

    if SAFE_MODE:
        if os.path.exists(safe_path):
            try:
                with open(safe_path, 'r', encoding='utf-8') as f:
                    existing_lines = f.readlines()
            except Exception:
                existing_lines = []
            new_lines = content.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                existing_lines, new_lines,
                fromfile=f'{relative_path} (current)',
                tofile=f'{relative_path} (proposed)',
            ))
            if diff:
                print(f"\n[Safe Mode] Proposed changes to '{relative_path}':")
                print(''.join(diff))
            else:
                print(f"\n[Safe Mode] No changes detected in '{relative_path}'. Skipping write.")
                return f"Skipped: No changes detected in {relative_path}"
        else:
            print(f"\n[Safe Mode] New file: '{relative_path}'")
            preview_lines = content.splitlines()[:20]
            print('\n'.join(preview_lines))
            if len(content.splitlines()) > 20:
                print(f"  ... ({len(content.splitlines()) - 20} more lines)")

        while True:
            answer = input("\n  Confirm write? [y/n] > ").strip().lower()
            if answer in ('y', 'yes'):
                break
            elif answer in ('n', 'no'):
                debug_log('write_denied', {'path': relative_path})
                return f"Write cancelled by user for '{relative_path}'."
            else:
                print("  Please enter y or n.")

    try:
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        debug_log('tool_call', {'op': 'write', 'path': relative_path, 'bytes': len(content)})
        return f"Success: Written to {relative_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

# ---------------------------------------------------------------------------
# Agent scaffolding
# ---------------------------------------------------------------------------
_MODELFILE_TEMPLATE = """\
FROM {base_model}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# Describe this agent's personality, role, and rules here.
# ---------------------------------------------------------------------------
SYSTEM \"\"\"
You are {agent_name}, a helpful and precise CLI assistant with sandboxed
file-system access. You follow instructions carefully, never modify files
outside the designated workspace, and always explain what you are doing.
\"\"\"

# ---------------------------------------------------------------------------
# PARAMETERS  (tune per agent as needed)
# ---------------------------------------------------------------------------
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
        content = _MODELFILE_TEMPLATE.format(
            base_model=base_model,
            agent_name=agent_name,
        )
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
        print(f"[Agent] Building '{agent_name}' from Modelfile (this may take a moment)...")
        # subprocess.run() handles paths with spaces correctly on Windows —
        # os.system() with shell=True can misparse them.
        result = subprocess.run(
            ['ollama', 'create', agent_name, '-f', modelfile_path],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[Agent] ERROR: 'ollama create' failed (exit code {result.returncode}).")
            print(f"        Check that Ollama is running and '{MODEL_NAME}' is pulled.")
            sys.exit(1)
        with open(sentinel_path, 'w') as fh:
            fh.write(time.strftime('%Y-%m-%dT%H:%M:%S'))
        print(f"[Agent] '{agent_name}' built successfully.")
    else:
        print(f"[Agent] '{agent_name}' is up to date (Modelfile unchanged).")


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


# Run scaffolding + ping before anything else touches the REPL
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

messages = [{
    'role': 'system',
    'content': (
        f'You are a CLI agent with direct read/write access to {TARGET_DIR}.\n'
        f'You can list subfolders on demand using tools if a user requests them.\n'
        f'Current top-level snapshot:\n{manifest}'
    )
}]

debug_log('session_start', {
    'chat_target': CHAT_TARGET,
    'model':       MODEL_NAME,
    'agent':       AGENT_NAME,
    'workspace':   TARGET_DIR,
    'safe_mode':   SAFE_MODE,
})

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
print(f"Model:  {MODEL_NAME or '(via agent)'}")
if AGENT_NAME:
    print(f"Agent:  {AGENT_NAME}")
print(f"Target: {CHAT_TARGET}")
print(f"Path:   {TARGET_DIR}")
if SAFE_MODE:
    print("Mode:   [SAFE] - all writes require confirmation")
if DEBUG_MODE:
    print(f"Debug:  logging to {DEBUG_LOG_PATH}")
print(f"\n{manifest}")
print("\n----------------------------------------------------------------")
print("Enter commands below. Type 'exit' or '/bye' to quit.")
print("----------------------------------------------------------------\n")

# ---------------------------------------------------------------------------
# Main REPL loop
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
            tool_calls      = []
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
                    rel_path  = args_.get('relative_path')

                    print(f"\n[System: {func_name} on '{rel_path}']")

                    if func_name == 'list_directory_contents':
                        result = list_directory_contents(rel_path)
                    elif func_name == 'read_local_file':
                        result = read_local_file(rel_path)
                    elif func_name == 'write_local_file':
                        result = write_local_file(rel_path, args_['content'])
                    else:
                        result = "Unknown tool"

                    debug_log('tool_result', {'tool': func_name, 'path': rel_path, 'result_preview': result[:200]})
                    messages.append({'role': 'tool', 'content': result, 'name': func_name})
                continue
            else:
                if ai_reply_chunks:
                    print("\n")
                    full_reply = "".join(ai_reply_chunks)
                    messages.append({'role': 'assistant', 'content': full_reply})
                    debug_log('assistant_reply', {'content': full_reply[:500]})
                break

    except Exception as e:
        spinner.stop()
        print(f"\nError: {e}")
        debug_log('error', {'message': str(e)})
        if messages and messages[-1]['role'] == 'user':
            messages.pop()

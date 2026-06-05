# ollama-fs (agentRW)

A terminal CLI utility that gives a local [Ollama](https://ollama.com) model **full read/write filesystem access and shell-execution capability at the same privilege level as the invoking user**. Run it, point it at a starting directory, and have a conversation with an AI that can actually read, create, edit, move, and delete files anywhere your account can — and run shell commands the same way you would in your terminal — without ever leaving your terminal or sending data to the cloud.

> ⚠️ **This is the unsandboxed variant.** The model has the same filesystem and shell privileges you do. If you want a sandboxed version that confines the agent to a single directory, use the original [`ollama-fs`](../agent-tool) instead. Read [Security](#security) before running.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- At least one tool-calling-capable model pulled (e.g. `ollama pull gemma4`)
- The `ollama` Python package (`pip install ollama`)
- Windows only: `colorama` (`pip install -r Windows/requirements-windows.txt`)

---

## Installation

No installation required beyond the dependencies above.

### macOS / Linux

```bash
chmod +x Linux-Mac/ollama-fs.py
ln -s "$(pwd)/Linux-Mac/ollama-fs.py" /usr/local/bin/ollama-fs
```

Then call it as `ollama-fs` from any directory.

> **macOS note:** If you installed Ollama via the `.dmg` app, it runs as a menu bar process rather than a background service. Make sure the Ollama app is open — the startup ping will tell you immediately if it isn't.

### Windows

```powershell
pip install -r Windows\requirements-windows.txt
python Windows\ollama-fs-windows.py -m gemma4 C:\Users\me\projects\foo
```

To call it without the `.py` extension from anywhere, add the folder to your `PATH` or create a `.bat` wrapper:

```bat
@echo off
python C:\path\to\ollama-fs-windows.py %*
```

Save as `ollama-fs.bat` on your `PATH` and call it as `ollama-fs` from any terminal.

---

## Usage

```
ollama-fs [-m MODEL] [-a AGENT] [--safe] [--debug] start_dir
```

At least one of `-m` or `-a` must be supplied.

| Flag | Description |
|------|-------------|
| `-m`, `--model MODEL` | Ollama base model to run (e.g. `gemma4`, `llama3`) |
| `-a`, `--agent AGENT` | Named agent to launch (e.g. `openclaw`, `hermes`) |
| `start_dir` | Starting directory: where relative paths resolve from and where shell commands begin. Not a sandbox boundary — the model can navigate elsewhere via `cd` or absolute paths. |
| `--safe` | Require confirmation before every write, move, delete, and shell command. Writes show a unified diff for existing files. |
| `--debug` | Append a JSONL trace of all events to `.ollama-fs-debug.jsonl` in the starting directory |

### Examples

```bash
# Run a model against the current directory; agent has full user-level access
ollama-fs -m gemma4 .

# Start in your projects folder; agent can still touch /etc, /var, ~/Documents, etc.
ollama-fs -m gemma4 ~/projects/foo

# Run a named agent (auto-scaffolded on first launch)
ollama-fs -m gemma4 -a openclaw ~

# Safe mode — confirm every write, move, delete, and shell command
ollama-fs -m gemma4 . --safe

# Safe mode + debug logging
ollama-fs -m gemma4 . --safe --debug

# Agent only (already registered in Ollama, no -m needed)
ollama-fs -a openclaw ~
```

---

## Tools available to the model

The agent has six tools wired into the Ollama function-calling API:

| Tool | What it does |
|------|--------------|
| `list_directory_contents` | Lists files and subdirectories of any path the invoking user can read |
| `read_local_file` | Returns the full UTF-8 text content of any readable file |
| `write_local_file` | Writes or appends text to any writable path. `mode="overwrite"` (default) replaces contents; `mode="append"` adds to the end (creates if absent). Parent directories are created automatically. |
| `move_local_file` | Move or rename a file or directory anywhere the user has permission |
| `delete_local_file` | Permanently delete a single file (refuses directories — use `run_shell_command` + `rm -rf` for those) |
| `run_shell_command` | Execute a shell command in a persistent shell session as the invoking user. Returns combined stdout/stderr, exit code, and the shell's cwd after the command. |

Paths accepted by the file tools may be **absolute** (`/etc/hostname`, `C:\Windows\System32\drivers\etc\hosts`), **relative to the current session directory**, or use **`~`** for the user's home.

---

## Persistent shell session

`run_shell_command` does **not** spawn a new shell per call. A single shell process (`/bin/bash --norc --noprofile` on Linux/macOS, `powershell.exe -NoLogo -NoProfile -Command -` on Windows) lives for the entire REPL session. This means:

- `cd /some/path` in one call carries over to the next call.
- `export FOO=bar` (or `$env:FOO = "bar"`) persists.
- Functions defined with `function foo() { ... }` or sourced from `.ps1` / shell scripts stay available.
- The file tools also follow the shell's current directory — `list_directory_contents(".")` after a `cd` lists the new location.

When a command times out (default 120 seconds, configurable per-call), the shell is killed and a fresh one is spawned at the current session cwd so state can't desync.

---

## Agents

When you supply both `-m` and `-a`, ollama-fs manages the full agent lifecycle:

1. **Scaffold** — Creates `~/.config/ollama-fs/agents/<agent_name>/Modelfile` (Linux/macOS) or `%APPDATA%\ollama-fs\agents\<agent_name>\Modelfile` (Windows) if it doesn't exist, pre-filled with your base model, an agentRW-appropriate system prompt, and tunable parameters.
2. **Build** — Runs `ollama create <agent_name>` to register the agent with Ollama.
3. **Sentinel** — Writes a `.built` timestamp next to the Modelfile. On subsequent launches, the agent is only rebuilt if the Modelfile has been edited since the last build.
4. **Ping** — Sends a probe to confirm the agent is responsive before opening the REPL.

Edit a Modelfile to customise the agent's personality, system prompt, or inference parameters — the next launch will detect the change and rebuild automatically.

### Modelfile parameters

The scaffolded template includes sensible defaults:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | `0.7` | Creativity (0 = deterministic, 1 = very creative) |
| `num_ctx` | `8192` | Context window size in tokens |
| `repeat_penalty` | `1.1` | Discourages repetitive output |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `top_k` | `40` | Top-k sampling limit |

---

## Safe Mode

With `--safe`, every destructive action is intercepted before touching disk or the shell:

- **Write (overwrite, existing file)** — Unified diff against the current contents.
- **Write (overwrite, new file)** — Preview of the first 20 lines.
- **Write (append)** — Preview of the content to be appended plus the current file size.
- **Move / rename** — Source and destination shown.
- **Delete** — Path shown with an explicit "this cannot be undone" notice.
- **Shell command** — Command line and the cwd it will run in.

You are prompted `[y/n]` (Enter defaults to *No* for move/delete/shell) to confirm or cancel. Cancellations are reported back to the model so it can respond gracefully.

---

## Security

**There is no sandbox.** The model can read, write, move, delete, and execute anywhere your user account can. If you can `rm -rf ~` in your shell, so can the model.

- The script inherits your normal user permissions — it does not escalate privileges, but it has all of yours.
- A confused or jailbroken model could damage files outside the starting directory. Use `--safe` for any session where you don't fully trust the model's reasoning or the prompts you're feeding it.
- Do not run this as root unless you specifically need to and accept the risk.
- The model's system prompt is hardened against fabricating tool output (it's told to report actual results verbatim), but it is not hardened against acting on prompt injection embedded in files it reads. Don't point this at attacker-controlled content while in non-safe mode.

If you want hard isolation, run inside a container, VM, or unprivileged user account whose home directory contains nothing irreplaceable.

---

## Cancelling the current turn

Press **Ctrl+C** during inference or a running shell command to cancel that turn cleanly:

- The in-flight shell command (if any) is killed.
- The persistent shell is respawned at the current session cwd, preserving your `cd` state.
- The user message that triggered the turn is rolled back so it doesn't pollute future context.
- You're returned to the `User Input >` prompt.

Press **Ctrl+C** (or Ctrl+D) at an empty `User Input >` prompt to exit the program. `exit` or `/bye` also quit.

---

## Debug Logging

With `--debug`, every event in the session is appended as a JSON line to `.ollama-fs-debug.jsonl` in the starting directory. Events include: `session_start`, `user_input`, `tool_call`, `tool_result`, `write_denied`, `move_denied`, `delete_denied`, `shell_denied`, `shell_timeout`, `user_cancel`, `startup_ping`, `assistant_reply`, and `error`.

Without `--debug`, no log file is created and there is zero overhead.

---

## Quitting

Type `exit` or `/bye` at the prompt, or press `Ctrl+C` / `Ctrl+D` at an empty prompt.

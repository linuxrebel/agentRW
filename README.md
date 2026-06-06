# ollama-fs

Two variants of a terminal CLI utility that gives a local [Ollama](https://ollama.com) model interactive read/write access to your machine — without sending data to the cloud. Pick the variant that matches your trust level.

| Variant | Script | Filesystem reach | Shell exec | Use when |
|---------|--------|-------------------|------------|----------|
| **Sandboxed** | `ollama-fsb.py` / `ollama-fsb-windows.py` | Confined to a single workspace folder; traversal blocked | None | You want hard isolation; you're running an unfamiliar model or feeding it untrusted prompts |
| **Open** | `ollama-open.py` / `ollama-open-windows.py` | Anywhere your user account can read/write | Yes (persistent bash/PowerShell session) | You want the agent to act as a real shell-using assistant across your whole machine |

Both variants share the same tool-calling architecture, agent scaffolding, safe-mode confirmations, and Ctrl-C cancellation; they differ only in what the model can touch.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- At least one tool-calling-capable model pulled (e.g. `ollama pull gemma4`)
- The `ollama` Python package (`pip install ollama`)
- Windows only: `colorama` (`pip install -r Windows/requirements-windows.txt`)

---

## Installation

### macOS / Linux

```bash
chmod +x Linux-Mac/ollama-fsb.py Linux-Mac/ollama-open.py
# Symlink whichever you use most onto your PATH:
ln -s "$(pwd)/Linux-Mac/ollama-fsb.py"  /usr/local/bin/ollama-fsb
ln -s "$(pwd)/Linux-Mac/ollama-open.py" /usr/local/bin/ollama-open
```

> **macOS note:** If you installed Ollama via the `.dmg` app, it runs as a menu bar process. Make sure the Ollama app is open — the startup ping will tell you immediately if it isn't.

### Windows

```powershell
pip install -r Windows\requirements-windows.txt
python Windows\ollama-fsb-windows.py  -m gemma4 .\my_project
python Windows\ollama-open-windows.py -m gemma4 C:\Users\me\projects\foo
```

Wrap either as a `.bat` on your `PATH` if you want to drop the `.py`:

```bat
@echo off
python C:\path\to\ollama-open-windows.py %*
```

---

## Usage

```
ollama-fsb  [-m MODEL] [-a AGENT] [--safe] [--debug] workspace
ollama-open [-m MODEL] [-a AGENT] [--safe] [--debug] start_dir
```

At least one of `-m` or `-a` must be supplied.

| Flag | Description |
|------|-------------|
| `-m`, `--model MODEL` | Ollama base model to run (e.g. `gemma4`, `llama3`) |
| `-a`, `--agent AGENT` | Named agent to launch (e.g. `openclaw`, `hermes`) |
| `workspace` (fsb) | Sandbox boundary. The model cannot touch anything outside this folder. |
| `start_dir` (open) | Starting directory for relative paths and shell commands. **Not** a boundary — the model can `cd` or use absolute paths to go anywhere. |
| `--safe` | Require confirmation before every write, move, delete, and (open only) shell command. Writes show a unified diff for existing files; appends show a content preview. |
| `--debug` | Append a JSONL trace of all events to `.ollama-fs-debug.jsonl` in the workspace / start_dir |

### Examples

```bash
# Sandboxed: agent can only touch ~/projects/foo
ollama-fsb -m gemma4 ~/projects/foo

# Open: agent has full user-level access; start in ~/projects/foo
ollama-open -m gemma4 ~/projects/foo

# Open + shell access; run a named agent
ollama-open -m gemma4 -a openclaw ~

# Safe mode — confirm every destructive action
ollama-fsb -m gemma4 ./my_project --safe
ollama-open -m gemma4 . --safe

# Safe mode + debug logging
ollama-open -m gemma4 . --safe --debug

# Agent only (already registered in Ollama, no -m needed)
ollama-open -a openclaw ~
```

---

## Tools available to the model

| Tool | fsb | open | What it does |
|------|:---:|:----:|--------------|
| `list_directory_contents` | ✅ | ✅ | List a directory the user can read |
| `read_local_file` | ✅ | ✅ | Read the full UTF-8 text content of a file |
| `write_local_file` | ✅ | ✅ | Write or append text. `mode="overwrite"` (default) replaces contents; `mode="append"` adds to the end (creates if absent). Parent directories auto-created. |
| `move_local_file` | ✅ | ✅ | Move or rename a file or directory |
| `delete_local_file` | ✅ | ✅ | Permanently delete a single file (not a directory) |
| `run_shell_command` | ❌ | ✅ | Execute a shell command in a persistent session. Returns combined stdout/stderr, exit code, cwd after execution. |

In **fsb**, paths are relative to the workspace root and any attempt to escape is blocked. In **open**, paths may be absolute, relative to the current session directory, or use `~`.

---

## Persistent shell session (open variant only)

`run_shell_command` does **not** spawn a new shell per call. A single shell process (`/bin/bash --norc --noprofile` on Linux/macOS, `powershell.exe -NoLogo -NoProfile -Command -` on Windows) lives for the entire REPL session. This means:

- `cd /some/path` in one call carries over to the next call.
- `export FOO=bar` (or `$env:FOO = "bar"`) persists.
- Functions defined with `function foo() { ... }` or sourced from `.ps1` / shell scripts stay available.
- The file tools also follow the shell's current directory — `list_directory_contents(".")` after a `cd` lists the new location.

When a command times out (default 120 seconds, configurable per-call), the shell is killed and a fresh one is spawned at the current session cwd so state can't desync.

---

## Agents

When you supply both `-m` and `-a`, both variants manage the full agent lifecycle:

1. **Scaffold** — Creates a per-variant agent directory:
   - `ollama-fsb`:  `~/.config/ollama-fsb/agents/<agent_name>/Modelfile`  (Linux/macOS) or `%APPDATA%\ollama-fsb\agents\<agent_name>\Modelfile`  (Windows)
   - `ollama-open`: `~/.config/ollama-open/agents/<agent_name>/Modelfile` (Linux/macOS) or `%APPDATA%\ollama-open\agents\<agent_name>\Modelfile` (Windows)

   Pre-filled with your base model, an appropriate system prompt for the chosen variant, and tunable parameters.
2. **Build** — Runs `ollama create <agent_name>` to register the agent with Ollama.
3. **Sentinel** — Writes a `.built` timestamp next to the Modelfile. On subsequent launches, the agent is only rebuilt if the Modelfile has been edited.
4. **Ping** — Sends a probe to confirm the agent is responsive before opening the REPL.

> The two variants keep their agent Modelfiles in separate directories, so you can have the same agent name (e.g. `openclaw`) scaffolded differently for each — `ollama-fsb` gets a sandbox-oriented SYSTEM prompt, `ollama-open` gets a full-access one. They still share the single Ollama model registry, so if you build `openclaw` under both, the second `ollama create` overwrites the first. Either use different names (`openclaw-fsb` / `openclaw-open`) or accept that whichever variant you used most recently wins.

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

With `--safe`, every destructive action is intercepted before touching disk or (in open) the shell:

- **Write (overwrite, existing file)** — Unified diff against the current contents.
- **Write (overwrite, new file)** — Preview of the first 20 lines.
- **Write (append)** — Preview of the content to be appended plus the current file size.
- **Move / rename** — Source and destination shown.
- **Delete** — Path shown with an explicit "this cannot be undone" notice.
- **Shell command** (open only) — Command line and the cwd it will run in.

You are prompted `[y/n]` (Enter defaults to *No* for move/delete/shell) to confirm or cancel. Cancellations are reported back to the model so it can respond gracefully.

---

## Security

### Sandboxed variant (`ollama-fsb`)

The model can only access files inside the workspace folder you specify. Every path is resolved to an absolute path and verified to stay within the boundary (`os.path.abspath` + `os.sep` prefix check). Directory traversal attempts (`../../etc/passwd`) are blocked and reported. The script inherits your normal user permissions — it does not escalate privileges.

Hidden from the model's manifest view (not blocked, just not advertised):
- `.git/`
- `.ollama-fs-debug.jsonl`

### Open variant (`ollama-open`)

**There is no sandbox.** The model can read, write, move, delete, and execute anywhere your user account can. If you can `rm -rf ~` in your shell, so can the model.

- A confused or jailbroken model could damage files outside the starting directory. Use `--safe` for any session where you don't fully trust the model's reasoning or the prompts you're feeding it.
- Do not run this as root unless you specifically need to and accept the risk.
- The model's system prompt is hardened against fabricating tool output (it's told to report actual results verbatim), but it is not hardened against acting on prompt injection embedded in files it reads. Don't point `ollama-open` at attacker-controlled content while in non-safe mode.
- For hard isolation, run inside a container, VM, or unprivileged user account whose home contains nothing irreplaceable — or use `ollama-fsb` instead.

---

## Cancelling the current turn (both variants)

Press **Ctrl+C** during inference or a running shell command to cancel that turn cleanly:

- The in-flight shell command (open only) is killed.
- In open, the persistent shell is respawned at the current session cwd, preserving your `cd` state.
- The user message that triggered the turn is rolled back so it doesn't pollute future context.
- You're returned to the `User Input >` prompt.

Press **Ctrl+C** (or Ctrl+D) at an empty `User Input >` prompt to exit the program. `exit` or `/bye` also quit.

---

## Debug Logging

With `--debug`, every event in the session is appended as a JSON line to `.ollama-fs-debug.jsonl` in the workspace / starting directory.

Events emitted by both variants: `session_start`, `user_input`, `tool_call`, `tool_result`, `write_denied`, `move_denied`, `delete_denied`, `user_cancel`, `startup_ping`, `assistant_reply`, `error`.

`ollama-fsb` additionally emits: `sandbox_violation` (when the model attempts a path outside the workspace).

`ollama-open` additionally emits: `shell_denied`, `shell_timeout`.

Without `--debug`, no log file is created and there is zero overhead.

---

## Quitting

Type `exit` or `/bye` at the prompt, or press `Ctrl+C` / `Ctrl+D` at an empty prompt.

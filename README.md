# ollama-fs

A terminal CLI utility that gives a local [Ollama](https://ollama.com) model sandboxed, interactive read/write access to a folder on your machine. Run it, point it at a directory, and have a conversation with an AI that can actually read, create, and edit your files — without ever leaving your terminal or sending data to the cloud.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- At least one model pulled (e.g. `ollama pull gemma4`)
- The `ollama` Python package (`pip install ollama`)

---

## Installation

No installation required. Just make the script executable:

```bash
chmod +x ollama-fs.py
```

Optionally symlink it onto your PATH:

```bash
ln -s /path/to/ollama-fs.py /usr/local/bin/ollama-fs
```

---

## Usage

```
ollama-fs.py [-m MODEL] [-a AGENT] [--safe] [--debug] workspace
```

At least one of `-m` or `-a` must be supplied.

| Flag | Description |
|------|-------------|
| `-m`, `--model MODEL` | Ollama base model to run (e.g. `gemma4`, `llama3`) |
| `-a`, `--agent AGENT` | Named agent to launch (e.g. `openclaw`, `hermes`) |
| `workspace` | Path to the sandboxed folder the model can access |
| `--safe` | Require confirmation before every file write; shows a diff for existing files |
| `--debug` | Log all tool calls and responses to `.ollama-fs-debug.jsonl` in the workspace |

### Examples

```bash
# Run a model directly against a folder
ollama-fs.py -m gemma4 ./my_project

# Run a named agent (auto-scaffolded on first launch)
ollama-fs.py -m gemma4 -a openclaw ./my_project

# Safe mode — confirm every write before it hits disk
ollama-fs.py -m gemma4 -a openclaw ./my_project --safe

# Safe mode + debug logging
ollama-fs.py -m gemma4 -a openclaw ./my_project --safe --debug

# Agent only (already registered in Ollama, no -m needed)
ollama-fs.py -a openclaw ./my_project
```

---

## Agents

When you supply both `-m` and `-a`, ollama-fs automatically manages the full agent lifecycle for you:

1. **Scaffold** — Creates `~/.config/ollama-fs/agents/<agent_name>/Modelfile` if it doesn't exist, pre-filled with your base model, a system prompt, and tunable parameters.
2. **Build** — Runs `ollama create <agent_name>` to register the agent with Ollama.
3. **Sentinel** — Writes a `.built` timestamp next to the Modelfile. On subsequent launches, the agent is only rebuilt if the Modelfile has been edited since the last build.
4. **Ping** — Sends a lightweight probe to confirm the agent is responsive before opening the REPL. A spinner runs during this step.

All agent Modelfiles live in `~/.config/ollama-fs/agents/` and are shared across workspaces. Edit a Modelfile to customise the agent's personality, system prompt, or inference parameters — the next launch will detect the change and rebuild automatically.

### Modelfile parameters

The scaffolded template includes sensible defaults for:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | `0.7` | Creativity (0 = deterministic, 1 = very creative) |
| `num_ctx` | `8192` | Context window size in tokens |
| `repeat_penalty` | `1.1` | Discourages repetitive output |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `top_k` | `40` | Top-k sampling limit |

---

## Safe Mode

With `--safe`, every file write is intercepted before touching disk:

- **Existing files** — A unified diff is shown so you can see exactly what will change.
- **New files** — A preview of the first 20 lines is shown.
- You are prompted `[y/n]` to confirm or cancel. Cancellations are reported back to the model so it can respond gracefully.

---

## Security

The model can only access files inside the workspace folder you specify. Every path provided by the model is resolved to an absolute path and verified to stay within the sandbox before any operation is performed. Directory traversal attempts (e.g. `../../etc/passwd`) are blocked and reported. The script inherits your normal user permissions — it does not escalate privileges.

Files hidden from the model's view (not accessible, just not shown in the workspace snapshot):
- `.git/`
- `.ollama-fs-debug.jsonl`

---

## Debug Logging

With `--debug`, every event in the session is appended as a JSON line to `.ollama-fs-debug.jsonl` inside the workspace. Events include: `session_start`, `user_input`, `tool_call`, `tool_result`, `sandbox_violation`, `write_denied`, `assistant_reply`, and `error`.

Without `--debug`, no log file is created and there is zero overhead.

---

## Quitting

Type `exit` or `/bye` at the prompt, or press `Ctrl+C` / `Ctrl+D`.

# coding_agent

A terminal CLI coding assistant that gives a local [Ollama](https://ollama.com) model interactive read/write/execute access to your filesystem — without ever leaving your terminal or sending data to the cloud.

One script, all platforms. Windows-specific behavior (config location, ANSI colors) is handled internally.

---

## Requirements

- Python 3.10+
- Ollama installed and running — see [ollama.com/download](https://ollama.com/download)
- At least one model pulled — browse [ollama.com/library](https://ollama.com/library?sort=newest)

```bash
pip install -r requirements.txt
```

`prompt_toolkit` is optional (Alt+Enter multi-line input); the agent falls back to plain `input()` without it. `colorama` installs on Windows only.

---

## Installation

### Linux / macOS

```bash
chmod +x coding_agent.py
ln -s "$PWD/coding_agent.py" ~/.local/bin/coding_agent
```

### Windows

```powershell
python coding_agent.py <model-name>
```

To call it without the `.py` extension, save a `coding_agent.bat` somewhere on your `PATH`:

```bat
@echo off
python C:\path\to\coding_agent.py %*
```

Config lives in `%APPDATA%\coding_agent\config.json` on Windows, `~/.config/coding_agent/config.json` elsewhere.

---

## First Run

Run without a model name and it will ask for one:

```
$ coding_agent
First run: please provide a model name and any arguments. Run --help for options.
```

Pass a model — it gets saved as your default:

```bash
coding_agent <model-name> --low-vram
```

After that, `coding_agent` alone uses the saved default. Passing a different model prompts you to update it.

---

## Usage

```
coding_agent.py [MODEL] [OPTIONS]
```

Run `coding_agent --help` for the full flag reference, and `/help` inside a session for the slash-command list. Both print from the source, so they never drift.

---

## Shell Commands

Any command found in your `$PATH` typed at the prompt runs directly — no LLM involved:

```
ls
git status
python3 script.py
grep -r "TODO" .
```

---

## Model Tools

The model calls these itself:

| Tool | Description |
|------|-------------|
| `read_file` | Read a file (with line range support) |
| `write_file` | Overwrite a file, backing up to `.bak` first |
| `edit_file` | Patch a file by replacing a string match |
| `search_file` | Case-insensitive grep |
| `list_files` | Directory listing |
| `run_command` | Run a shell command as the current user (no sudo) |

---

## File Safety

- **Backup**: `.bak` written once on first modification — the original survives across retries.
- **No privilege escalation**: `run_command` blocks `sudo`, `su`, `doas`, `pkexec`, `runuser`.

The agent is a harness, not a critic: `write_file` does not vet the content it is handed. It writes what the model produced and leaves judging the result to you. A file on disk with a bug is more useful than a refusal and no file.

---

## Quitting

Type `/bye`, `exit`, or `quit`, or press `Ctrl+C` / `Ctrl+D`.

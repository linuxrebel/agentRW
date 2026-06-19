# coding_agent

A terminal CLI coding assistant that gives a local [Ollama](https://ollama.com) model interactive read/write/execute access to your filesystem — without ever leaving your terminal or sending data to the cloud.

---

## Requirements

- Python 3.7+ (3.8+ recommended)
- Ollama should be installed and running (See [https://ollama.com/download](https://ollama.com/download) for instructions on how to do this)
- At least one model pulled (go to [https://ollama.com/library?sort=newest](https://ollama.com/library?sort=newest) to find models)
- The `openai` Python package (`pip install openai`)
- Optional: `pip install prompt_toolkit` for Alt+Enter multi-line input

---

## Installation

### Linux / macOS

```bash
pip install -r Linux-Mac/requirements.txt
chmod +x Linux-Mac/coding_agent.py
ln -s /path/to/Linux-Mac/coding_agent.py ~/.local/bin/cagent
```

### Windows

```powershell
pip install -r Windows\requirements.txt
python Windows\coding_agent.py qwen2.5-coder:7b-instruct-q4_K_M
```

To call it without the `.py` extension from anywhere, create a `.bat` wrapper:

```bat
@echo off
python C:\path\to\coding_agent.py %*
```

Save as `cagent.bat` somewhere on your `PATH`.

Config is stored in `%APPDATA%\coding_agent\config.json`.

ANSI colors work automatically in Windows Terminal and PowerShell 7.
For older consoles, `colorama` (included in `requirements.txt`) handles it.

---

## First Run

The first time you run without a model name, it will prompt you:

```
cagent
First run: please provide a model name and any arguments. Run --help for options.
```

Pass a model name — it gets saved as your default:

```bash
cagent qwen2.5-coder:7b-instruct-q4_K_M --low-vram
```

After that, `cagent` alone uses your saved default.

---

## Usage

```
coding_agent.py [MODEL] [OPTIONS]
cagent [MODEL] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `MODEL` | Ollama model tag (saved as default on first use) |
| `--gpu-layers N` | GPU layers (0 = CPU only; auto-halved on OOM) |
| `--num-ctx N` | KV cache context window size |
| `--max-tokens N` | Max output tokens per reply (default: 2000) |
| `--low-vram` | 4 GB VRAM preset: num_ctx=2048, max_tokens=512 |
| `--set-default MODEL` | Save a new default model to config and exit |
| `-h`, `--help` | Full usage reference |

---

## Shell Commands

Any command found in your `$PATH` typed at the prompt runs directly — no LLM involved:

```
ls
cat file.py
git status
python3 script.py
grep -r "TODO" .
```

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/model [name]` | Show or switch the active model |
| `/gpu-layers [N]` | Show or set GPU layers live |
| `/low-vram` | Apply 4 GB preset mid-session |
| `/compact` | Compress context to free token budget |
| `/tokens` | Show estimated token count |
| `/reset` | Wipe conversation history |
| `cd <path>` | Change working directory |
| `/pwd` | Show current working directory |
| `/bye` | Exit (also: Ctrl+C, Ctrl+D) |

---

## Model Tools

The model has access to these tools automatically:

| Tool | Description |
|------|-------------|
| `read_file` | Read a file (with line range support) |
| `write_file` | Overwrite a file (truncation guard + .bak + syntax check) |
| `edit_file` | Patch a file by replacing a string match |
| `search_file` | Case-insensitive grep |
| `list_files` | Directory listing |
| `run_command` | Run a shell command as the current user (no sudo) |

---

## File Safety

- **Backup**: `.bak` written once on first modification — original preserved across all retries
- **Truncation guard**: refuses to write if new content is < 60% of original size
- **Syntax check**: `.py` files are compiled before touching disk — syntax errors are rejected
- **No privilege escalation**: `run_command` blocks `sudo`, `su`, `doas`, `pkexec`, `runuser`

---

## Recommended Models (4 GB VRAM)

| Model | Notes |
|-------|-------|
| `qwen2.5-coder:7b-instruct-q4_K_M` | Best instruction-following, good code quality |
| `phi4-mini` | Fast, small, decent code quality |
| `starcoder2:3b` | Tiny, pure code completion |
| `deepseek-coder-v2:16b` | MoE architecture — low active params, fits 4 GB |

Run all with `--low-vram` on 4 GB VRAM.

---

## Quitting

Type `/bye`, `exit`, or `quit` at the prompt, or press `Ctrl+C` / `Ctrl+D`.

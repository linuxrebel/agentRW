# coding_agent

A terminal coding agent for hardware that can't run the big ones.

Same shape as [OpenCode](https://opencode.ai), Claude Code, or Aider: you type at a
prompt, a model reads and writes your files and runs commands, you stay in the
terminal. The difference is what it's built around.

Those tools assume a capable frontier model and spend context freely — large system
prompts, whole files in the window, rich scaffolding. That's the right call when
you're driving Claude or GPT through an API.

**This one assumes the opposite.** It targets a 7B model on a 4 GB GPU, where the
scarce resource is not model capability but the context window. Every design choice
follows from that:

- The system prompt is **593 tokens**, or **539 under `--low-vram`** — 28% and
  26% of a 2048-token window. Measured and trimmed rather than guessed at.

- Shell commands run *directly*, never through the model. Typing `ls` or `git status`
  costs zero tokens.
- Tools return summaries, not dumps. Raw pylint output on a 1000-line file is
  ~2400 tokens; the same findings grouped are ~150.
- **Deterministic work goes to real tools.** `/lint` sends 5 findings to a model
  and hands the other 11 to autopep8 — one subprocess, no tokens, and it cannot
  alter the code either side of the whitespace.
- A tool costs nothing to *have*, only to *advertise*. Anything the model does
  not need to call itself stays out of the prompt and remains callable.

It is a **harness, not a critic**: it writes what the model produces and leaves
judging the result to you. An earlier version validated content before writing and
that turned out to be the single biggest source of apparent "bad model output" —
see `CLAUDE.md`.

Measured on a 35-line file with six models, from a pylint baseline of 2.61:
gemma4:31b-cloud 9.57, qwen2.5-coder:7b 9.13, ornith:latest 8.26, and 7.39 for
qwen3.5, qwen3.6 and ornith:35b. **All six improve the file; none break it.**
The 7.39 floor is autopep8 alone — a model that contributes nothing costs you
time, not correctness.

Runs fully local against [Ollama](https://ollama.com), so nothing leaves your
machine unless you point it at a cloud model on purpose.

**Not for you if** you have API access to a frontier model and no hardware
constraint — use OpenCode or Claude Code, which do far more with that budget.

---

## Requirements

- Python 3.10+
- Ollama installed and running — see [ollama.com/download](https://ollama.com/download)
- At least one model pulled — browse [ollama.com/library](https://ollama.com/library?sort=newest)

```bash
pip install -r requirements.txt
```

`prompt_toolkit` is optional (Alt+Enter multi-line input); the agent falls back to plain `input()` without it. `colorama` installs on Windows only.

`pylint` and `autopep8` power the bundled plugins. They are also distro packages
(Fedora `python3-pylint` / `python3-autopep8`, Debian `pylint` /
`python3-autopep8`) and the plugins invoke them as executables on `PATH`, so
either source works. Prefer the distro package or a venv — pip on top of a
distro copy is a known way to break a system Python. Without them the plugins
stay dormant and say so; nothing else is affected.

---

## Installation

One script runs everywhere — `coding_agent.py`. Platform differences (config
location, ANSI colors) are handled inside it. All that differs below is how you put
it on your `PATH`.

**Linux / macOS**

```bash
chmod +x coding_agent.py
ln -s "$PWD/coding_agent.py" ~/.local/bin/coding_agent
```

**Windows** — run it directly:

```powershell
python coding_agent.py <model-name>
```

Or save a `coding_agent.bat` on your `PATH` to drop the extension:

```bat
@echo off
python C:\path\to\coding_agent.py %*
```

Config lives in `~/.config/coding_agent/config.json`, or
`%APPDATA%\coding_agent\config.json` on Windows.

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

## Cloud Models

Ollama proxies cloud-tagged models through the same local endpoint the agent
already talks to, so **no configuration change is needed** — the agent cannot tell
the difference.

Sign in once, in a normal terminal:

```bash
ollama signin
```

Do this *outside* the agent. Slash commands and the shell passthrough capture
output rather than attaching a terminal, so an interactive auth flow would hang
until it times out. It is one-time machine setup, not a session command.

Then just use one. **No pull needed** — cloud models resolve server-side:

```bash
coding_agent gpt-oss:20b-cloud
```

`/model gpt-oss:20b-cloud` also works mid-session.

To see what your plan actually covers:

```
/cloud-models          # models you can actually use, nothing else
/cloud-models all      # the full catalog
```

The default checks against your account, because nothing in the API advertises
entitlement — availability is only discoverable by asking. On a free account
roughly a third of the catalog comes back usable. `all` is a plain catalog
listing and makes no API calls.

**`ollama pull` proves nothing.** It fetches a manifest, and the subscription is
only checked at inference time, so a pull of a model you cannot use still
succeeds. Use `/cloud-models`, not a successful pull.

**Your code leaves your machine.** Everything else in this README describes a
fully local setup; a `:cloud` tag sends file contents and commands to Ollama's
servers. That is the trade, and the one reason to think before typing the tag.

**Local-only flags are ignored, not errors.** `--low-vram` and `--gpu-layers`
become `num_gpu`/`num_ctx` inference options for hardware you are no longer
using; cloud endpoints accept and discard them (verified — identical token
counts with `num_gpu` at 0, 99, or unset). The context-trimming budget is tuned
for a small window and will trim history a large cloud model could hold, so
raise it if you work this way often.

**Non-Ollama APIs** (Moonshot direct, OpenRouter, anything else OpenAI-compatible)
are a different case — `base_url` and `api_key` are currently hardcoded near the
top of `coding_agent.py` and would need editing.

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
| `write_file` | Overwrite a file (scoped to allowed dirs), backing up to `.bak` first |
| `edit_file` | Patch a file by replacing a string match |
| `search_file` | Case-insensitive grep |
| `list_files` | Directory listing |
| `run_command` | Run a shell command as the current user (no sudo) |

Every tool is also callable directly — `/read_file foo.py start_line=50` — and
`/tools` shows which are advertised to the model and what each costs.

---

## Plugins

Drop a `.py` file into `tools/`. It is discovered — no registration, no manifest,
no code changes. Two naming conventions are the whole API:

| suffix | becomes |
|---|---|
| `*_tool` | a tool the model can call, and `/name` |
| `*_command` | a slash command `/name` |

A plugin gates itself on what it needs, in ordinary Python:

```python
if shutil.which("pylint"):
    def lint_command(ctx, args): ...
```

Miss the requirement and the command is never registered — **absent, not
broken**. It does not appear in `/help` and `/lint` falls through to the model
like any unknown word. `/plugins` shows what is registered and what is dormant.

**Commands are free; tools are not.** A command costs nothing until you invoke
it. A tool's docstring rides in the system prompt, which is re-sent with *every*
request — so an advertised tool costs its tokens every turn, used or not.
Loading is irrelevant: a plugin loads in 0.2 ms.

Dispatch and advertisement are separate, so a tool can be callable without being
advertised:

```
/tools                    what is advertised, and the cost of each
/tools off lint_file      drop from the prompt, still callable as /lint_file
/tools core               core six only
```

`--low-vram` does the last one automatically. `/lint` reaches the same 9.13
whether or not `lint_file` is advertised, because the plugin calls it directly.

Two plugins ship with the agent:

| plugin | provides | needs |
|---|---|---|
| `tools/lint.py` | `lint_file`, `/lint` | pylint |
| `tools/format.py` | `format_file` | autopep8 |

**`/lint <file>`** walks pylint findings one at a time — fix, skip, ignore the
whole kind, defer to `DEBT.md`, or see the raw message. It explains what each
finding means in plain English, hands every style finding to autopep8 without
asking, and reverts the entire run if the result no longer compiles.

Writing one: see [PLUGINS.md](PLUGINS.md) — conventions, the `ctx` API, the four
contract obligations, and the trust model.

---

## Safety

The agent runs a local model with your privileges. Anything it reads — a file, a
command's output — becomes text the model acts on, so a file containing
instructions can try to steer it. Two controls exist for that:

- **Writes are scoped.** `write_file` and `edit_file` only write under the working
  directory. `~/.bashrc`, `~/.ssh/`, and the script itself are out of reach. Add
  directories with `--allow-write DIR` (repeatable); `cd` moves the scope.
- **Commands are confirmed.** Every `run_command` the model proposes is shown to
  you first. `a` approves the rest of the session, `--yes` skips the prompt
  entirely. Declining is the only thing standing between a poisoned file and
  execution, so read what you approve.

Lesser guarantees:

- **Backup**: `.bak` written once on first modification, owner-only (`0600`), so
  the copy does not widen access to a file's contents.
- **No accidental sudo**: `run_command` rejects `sudo`, `su`, `doas`, `pkexec`,
  `runuser`. This catches the model reaching for them by habit. It is **not** a
  security boundary — the command runs under a shell, which expands the string
  after that check, so anything deliberate gets through. The confirmation prompt
  is the real control.

`write_file` does not vet the *content* it is handed — it is a harness, not a
critic. It writes what the model produced and leaves judging the result to you.
A file on disk with a bug is more useful than a refusal and no file.

---

## Quitting

Type `/bye`, `exit`, or `quit`, or press `Ctrl+C` / `Ctrl+D`.

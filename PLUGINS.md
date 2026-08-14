# Writing a plugin

A plugin is a directory containing ordinary Python and an `install.md`. It adds
tools the model can call and slash commands you can type. There is no framework
to learn: two naming conventions and one object handed to your command.

Nothing imports your files explicitly. Nothing registers them. They are
discovered.

```
tools/linuxrebel/lint/
  install.md      what you are, what you ship, what you need
  plugin.py       the code
  README.md       optional, for humans
```

Plugins live in their own git repos and are namespaced by owner, so two authors
can both publish a `lint` and both can be installed.

---

## install.md

Markdown, so it renders on GitHub and GitLab where people will read it.

```markdown
# linuxrebel/lint 1.0.0

Interactive pylint session — walks findings one at a time.

## Files
- plugin.py
- symbols.json

## Requires
- pylint
- autopep8

## API
1
```

| section | meaning |
|---|---|
| heading | `owner/name version` — your namespaced identity |
| `## Files` | **every file you ship.** Only declared `.py` files are imported |
| `## Requires` | executables you need, reported by `/plugins` when missing |
| `## API` | lowest `ctx.api` you work against. A host older than this refuses to load you, and says why |

Declaring your files is not bookkeeping. An undeclared `.py` sitting in your
directory is **never imported**, so nothing can be smuggled in alongside a
legitimate plugin. It is also what lets an installer extract only what you
listed.

If `install.md` is malformed the plugin is refused. That is the author's
problem — the harness reports what it could not read and moves on.

Your directory is otherwise yours. Config, caches, data, tests: keep what you
like beside your code. `/plugins` reports registered plugins only and ignores
the rest. `PLUGIN_DIR` is injected into your module so you can find your own
files.

---

## The two conventions

```python
def wordcount_tool(filename: str) -> dict:
    """Count lines and words in a file."""      # shown to the model
    ...

def wordcount_command(ctx, args: str) -> None:
    """One line, shown in /help."""
    ...
```

| suffix | becomes |
|---|---|
| `*_tool` | a tool the model can call, and `/name` on the prompt |
| `*_command` | a slash command `/name` |

`build_prompt()` reads your tool's `inspect.signature()` and `__doc__`, so a
discovered tool is indistinguishable from a built-in one. Keep the docstring
short — it is paid for on every single turn.

---

## The contract

Five obligations. A plugin that does not meet them is a bug, not a variation.

### 1. Declare what you need

```python
REQUIRES = {
    "pylint": {"pip": "pylint", "fedora": "python3-pylint", "debian": "pylint"},
}
```

`/plugins` reports this so the owner can see why something is dormant and how
to resolve it.

### 2. Gate your own registration

If a requirement is missing, do not register. Ordinary Python:

```python
if shutil.which("pylint"):

    def lint_command(ctx, args): ...
```

The `def` never runs, so the command is never registered. It is **absent**, not
broken — `/lint` falls through to the model like any unknown word, and does not
clutter `/help`. At a hundred plugins this is what keeps the command list
honest.

### 3. Say what is missing, never install it

When you cannot run, name the missing thing and stop. Do not invoke a package
manager, and do not pip into the system interpreter — that is a known way to
break a distro Python. The owner decides how to meet the need.

```python
if not shutil.which("autopep8"):
    return {"error": "autopep8_not_installed", "hint": "pip install autopep8"}
```

### 4. Answer `help`

Every command must handle `help` as its argument:

```python
def lint_command(ctx, args: str) -> None:
    if args.strip().lower() in ("help", "-h", "--help"):
        print("""/lint <file> [symbol] [max=N]
        ...what it does, every option, what it requires...""")
        return
```

`/lint help` is how someone discovers a plugin they did not write.

### 5. Say what you touch, in your README

Installing your plugin means running your code. The convention elsewhere is to
tell users to read the source first — which nobody does, and which quietly
moves the blame onto them for not doing it.

Write a **What this will do** section instead. Two lists, plain language,
checkable against your code by anyone who cares to:

```markdown
## What this will do

**It will:**
- Read and rewrite the .py file you point /lint at, one line at a time, and
  only after you approve each change
- Write a .bak beside that file before the first change
- Append to DEBT.md in your working directory when you answer `defer`
- Run pylint and autopep8 as subprocesses
- Send short snippets to whichever model agentRW is pointed at. If that is a
  cloud model, those lines leave the machine

**It will not:**
- Touch any file except the one you named
- Apply anything the user has not seen and agreed to
- Install packages, change configuration, or reach the network itself
```

Be specific about three things in particular, because they are what someone
would actually want to know and cannot infer:

- **what you write**, and whether the original is recoverable
- **what leaves the machine**, especially anything sent to a model
- **what you run**, if you shell out

A user who cannot read Python can still decide from that. A user who can read
Python now knows what to check.

Your command receives `ctx`. **Use only what is listed here.** The harness
renames and restructures its internals freely; `ctx` is the contract that does
not move. Check `ctx.api` if you care.

```
ctx.api             int, currently 1. Bumped on a breaking change.

session
ctx.model           model name in use
ctx.cfg             {"num_ctx":…, "token_budget":…}
ctx.layers          mutable GPU-layer ref for call_llm
ctx.cwd             the agent's working directory
ctx.tools           the tool registry, e.g. ctx.tools["lint_file"](filename=…)

paths and writing
ctx.resolve_path(s) -> Path, honouring the agent's `cd`
ctx.writable(path)  -> bool, is this inside the allowed write scope
ctx.write_denied(p) -> the standard refusal dict

findings pipeline
ctx.gather_findings(target, only="")     flatten a detector's output
ctx.propose_fix(model, cfg, layers, lines, finding)
                                          computed when determined, else asks
ctx.apply_fix(path, lines, finding, new) replace / insert / write, checked
ctx.finish_run(path, snapshot) -> bool   revert the run if it broke the file
ctx.defer(target, finding)               append to DEBT.md
ctx.debt_file                            the ledger's filename

output
ctx.summarise(tool_name, result)         one-line console echo
ctx.render(result)                       full human-readable rendering
ctx.colour, ctx.reset                    ANSI, empty strings if unsupported
```

`resolve_abs_path` is also injected into your module's globals, so a plugin
stays importable standalone for testing:

```python
def _abs(filename):
    injected = globals().get("resolve_abs_path")
    return injected(filename) if injected else Path(filename).expanduser().resolve()
```

Reaching past `ctx` into harness internals is how plugins break on upgrade.
Two references to a private `_writable` survived a refactor here and crashed at
runtime; they only surfaced because `ctx` made them visible as unresolved names.

---

## Rules that are not negotiable

**Tools aggregate, never dump.** Raw pylint output on a 1000-line file is
~2400 tokens — more than a `--low-vram` window holds. The same findings grouped
and counted are ~150. Return counts and the top few, with an argument for
drilling into one. Every byte you return is paid for out of the model's
capacity to think.

**Prefer a tool to the model.** If a real program can produce the answer, call
it. autopep8 fixes eleven style findings deterministically, in one subprocess,
with no tokens, and *cannot* alter the code either side of the whitespace.
Asking a model to fix indentation produced a line that kept the wrong indent
and silently dropped a `*` from `join(*lines)` — a style request that caused a
runtime error. Every finding moved off the model is a win, not a shortfall.

**Say what the finding means.** Detectors speak their own dialect. pylint's
"Module name doesn't conform to snake_case" never mentions that it means the
filename. Translate: what it means, what to do, and what happens if you ignore
it.

**Do not exceed the stated scope.** If the finding is about whitespace, change
whitespace. A fix that also "improves" the code is a defect even when the
improvement is defensible.

---

## What a plugin may add, and what it costs

| addition | cost |
|---|---|
| a command | **nothing** until you invoke it |
| a tool | **nothing** until you advertise it |
| a tool with `model_facing = True` | ~40 tokens in the system prompt, every turn, forever |
| prompt text | not available — see below |

Commands are free because nothing about them reaches the model until you type
one. That is why `/lint` can afford a 90-line loop and long per-finding
explanations: none of it is paid for unless used.

Plugin tools are free for the same reason, by default. A registered tool is
always **callable** — `/your_tool arg` works, and the model can call it if it
knows the name — but it is not **advertised** unless it says so. Advertised
means its name, signature, and docstring sit in the system prompt and in the
tool schema, re-sent on every single turn.

```python
def your_tool(path: str) -> dict:
    """One short line. This is what the model reads."""
    ...

your_tool.model_facing = True   # advertise it — costs ~40 tokens/turn
```

Set it only when the model must reach for the tool unprompted to do its job.
Twenty plugins each advertising "just one tool" is 800 tokens off a 2048-token
window before you have typed anything. If the user is the one deciding when to
run it — lint this file, run the tests — leave it unadvertised, or write a
command instead.

Either way the user has the final say per session: `/tools on <name>` and
`/tools off <name>`, and `/tools` shows the current bill.

**Plugins cannot add system prompt text.** There is no hook for it, by design.
A plugin adding standing instructions would alter every future turn,
invisibly and permanently — a much worse failure than any one-off action.

Your tool's docstring is the one exception, since an advertised tool has to
describe itself to be used. It is capped at 240 characters, flattened to a
single line, and stripped of control characters. Write a short, plain
description.

---

## The trust model

**Installing a plugin means running someone's code.** `tools/*.py` is imported
at startup, so top-level code executes before any convention applies. A plugin
can read your files, reach the network, or modify another plugin. Nothing here
sandboxes that.

What the harness does limit:

- **No prompt-text hook**, so a plugin cannot silently rewrite the model's
  standing instructions.
- **Docstrings are capped and sanitised**, bounding the one channel where
  plugin text reaches the model. Note this bounds *size and formatting only* —
  a docstring that reads "ignore all previous instructions" still says that.
  Sanitising cannot stop persuasion.
- **The harness module is not injected.** `ctx` is the whole surface. Reaching
  into internals requires an explicit import, which is visible in review.
- **Writes stay scoped.** `ctx.writable()` reflects the same working-directory
  limit the model has, so a plugin using it cannot wander.

Taken together these prevent accidents and make hostile behaviour conspicuous.
They do not stop a determined author.

The usual advice here is "read the source before you install it". Nobody reads
four hundred lines of someone else's Python, and pretending otherwise just
moves the blame onto the user. See obligation 5.

---

## Uninstalling

Remove the directory. The tools and commands disappear with it.

---

## Worked example

No plugins ship with the agent, so the examples are repos you can read:

[**arwPyFormat**](https://github.com/linuxrebel/arwPyFormat) is the smallest
complete plugin: one tool, a `REQUIRES` block, a clear error when autopep8 is
absent, and no command. Around 50 lines.

[**arwRunTests**](https://github.com/linuxrebel/arwRunTests) is the middle
shape: a tool, a gated command with `help`, its own state file in `PLUGIN_DIR`,
and a self-check under `__main__`.

[**arwLint**](https://github.com/linuxrebel/arwLint) is the full shape: plain
English translation of a detector's messages, per-finding classification of how
a fix can be applied, and a run it reverts if the result stops compiling.

# Writing a plugin

A plugin is one ordinary Python file in `tools/`. It adds tools the model can
call and slash commands you can type. There is no framework to learn: two
naming conventions and one object handed to your command.

Nothing imports your file explicitly. Nothing registers it. It is discovered.

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

Four obligations. A plugin that does not meet them is a bug, not a variation.

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

---

## ctx — the plugin API

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
| a tool | ~40 tokens in the system prompt, every turn, forever |
| prompt text | not available — see below |

Commands are free because nothing about them reaches the model until you type
one. That is why `/lint` can afford a 90-line loop and long per-finding
explanations: none of it is paid for unless used.

Tools are not free. Twenty plugins each adding "just one tool" is 800 tokens
off a 2048-token window before you have typed anything. Add a tool only when
the model genuinely needs to call it; otherwise write a command.

**Plugins cannot add system prompt text.** There is no hook for it, by design.
A plugin adding standing instructions would alter every future turn,
invisibly and permanently — a much worse failure than any one-off action.

Your tool's docstring is the one exception, since it must reach the model to be
callable. It is capped at 240 characters, flattened to a single line, and
stripped of control characters. Write a short, plain description.

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
They do not stop a determined author. Read a plugin before you install it — the
whole point of the design is that a plugin is one readable file.

---

## Uninstalling

Move the file out of `tools/`. The tools and commands disappear with it.

Anything else you keep in `tools/` — config, cache, data — is your business.
`/plugins` reports registered plugins only and ignores everything else.

---

## Worked example

`tools/format.py` is the smallest complete plugin: one tool, a `REQUIRES`
block, a clear error when autopep8 is absent, and no command.

`tools/lint.py` is the full shape: a tool, a gated command, `help`, plain
English translation of a detector's messages, and per-finding classification of
how a fix can be applied.

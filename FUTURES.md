# Futures

Decided but not built. Each entry records what was chosen and why, so it can be
picked up without re-litigating. Not a wishlist — things here have been argued
through and deferred on purpose.

`Future_Router.md` is a separate, deeper design for one idea. This is the
backlog.

---

## User-defined command names

**Decided 2026-08-07. Deferred until install/uninstall works.**

A plugin *suggests* a command name; `tools.json` *decides* it. Same separation
as everywhere else in the plugin design — the plugin declares, the host
resolves.

```json
{
  "enabled": ["linuxrebel/lint", "someone/lint"],
  "commands": {
    "lint":  "linuxrebel/lint",
    "plint": "someone/lint",
    "l":     "linuxrebel/lint"
  }
}
```

**Why it matters more than convenience.** With namespaced identity, two authors
can both ship a plugin wanting `/lint`. Without user-owned binding the winner is
decided by load order, which is decided by directory name — arbitrary, and the
person affected has no recourse. This turns a collision into a choice.

Three constraints, none optional:

- **Aliases resolve before `_canonical_command`**, so a user alias beats
  prefix-matching. Otherwise `/l` lands somewhere surprising.
- **Aliases cannot shadow builtins.** `/help` or `/model` rebound to a plugin is
  a hijack vector, and the trust model already treats plugins as untrusted.
- **`/plugins` shows the binding.** A plugin's own `help` will say `/lint`;
  yours might be `/plint`. Without showing it, renames confuse rather than help.

Resolution order becomes: user alias → builtin → plugin's requested name → tool
name → prefix/plural guess.

No `/alias` command at first. Editing `tools.json` is honest and it is already
the file you would look at. Add one if it chafes.

---

## ctx.ask, and namespacing the findings helpers

**Blocking for any non-lint plugin.**

`ctx` has no way to call a model. The only route is `ctx.propose_fix`, which is
hard-shaped to "rewrite ONE line for this finding" with `FIX_PROMPT` baked in.
A plugin like `GitHub-Actions-Builder`, whose whole job is *describe a workflow,
get YAML*, has no route to a model at all — it would have to reach past `ctx`
into harness internals, which is exactly what `ctx` exists to prevent.

```
ctx.ask(messages, max_tokens=…, send_tools=False) -> str
```

Raw model access with the harness's context-trimming and error handling, no
lint-shaped prompt.

Six of the nineteen `ctx` names are the lint pipeline wearing a general API's
clothes: `gather_findings`, `propose_fix`, `apply_fix`, `finish_run`, `defer`,
`debt_file`. Group them so the split is visible:

```
ctx.ask, ctx.tools, ctx.resolve_path …     every plugin
ctx.findings.gather / propose / apply …    detector-shaped plugins only
```

That is a `ctx.api` bump to 2. **Cheap now, expensive once third-party plugins
exist** — do it before the API has outside users.

---

## Plugin-command tests

Four were designed; none built. The suite currently exercises no plugin command
at all, which is why a `NameError` on `_writable` reached runtime.

Worth keeping permanently — all cheap, no model, milliseconds:

1. **`ctx` completeness against `PLUGINS.md`.** Parse documented `ctx.*` names,
   assert each exists on a real `plugin_context()`. This is the
   "an update must not break someone's plugin" guarantee expressed as code, and
   it also catches the doc drifting from reality.
2. **Plugins use only `ctx`.** Static AST check for unresolved names in
   `tools/**/*.py`. No execution, no dependencies. Highest value per
   millisecond in the suite — it is what caught `_writable`.
3. **Discovery and gating**, against a temp directory: `*_tool` and `*_command`
   register; a conditionally-defined command with the condition false registers
   nothing; a syntax-error plugin is skipped without taking others down;
   `_`-prefixed files ignored.
4. **Contract compliance.** For each registered command, `help` produces output
   and `REQUIRES` is a dict.

Explicitly *not* worth keeping: a full `/lint` run against a live model. It
needs ollama, pylint and autopep8, takes 40–500s, and is non-deterministic —
qwen applied 13, 14, 15 and 16 findings across identical runs. Right as a
development check, wrong as a standing test. Keep the driver in scratchpad.

---

## run_tests — the detector that catches what syntax checking cannot

`/lint` reverts a run whose result does not compile. That is necessary and
demonstrably not sufficient.

A gemma4 run scored **9.57 with clean syntax and still died at runtime**:

```
TypeError: sequence item 0: expected str instance, tuple found
```

A line rewrite had silently changed `join(*lines)` to `join(lines)`. pylint
rated the broken file 9.57. `compile()` passed it. Only *executing* the code
distinguished the two.

`run_tests` as a category-1 detector — pytest is already installed — plus a
verify step in `/lint`: apply, run, revert from `.bak` if previously-passing
tests now fail. That is a fact, not a judgement, so it does not violate
harness-not-critic.

Caveat learned the hard way: a program killed by a timeout loses buffered
stdout, which reads as a false failure. Use `-u`.

---

## More deterministic detectors

Every finding moved off a model is a win. Anything with a real tool should get
one:

| finding | tool |
|---|---|
| `unused-import` | `autoflake` |
| import order | `isort` (already installed here) |
| `invalid-name` on a module | a plain rename — no model needed |

What a docstring should actually *say* is the only part that genuinely needs a
model. Shrink the residue and a 7B suffices — that is the whole SLM thesis, and
it argues for adding tools rather than a bigger model.

---

## Remote install sources

Local path and git URL were chosen. A registry was considered and rejected —
no infrastructure, no review process, and every ecosystem (VS Code, pip, npm,
Obsidian) supports local install anyway and none started with a registry.

The manifest format is what makes a registry possible later **without changing
the plugin**. That is the only reason to care about it now.

Deferred with it:

- **Provenance in `tools.json`** — source URL, version, tarball sha256. Makes
  "where did this come from" and "has it changed" answerable.
- **Update** — re-download, diff against installed, require an explicit enable.
- **Signature verification.** Only meaningful with a key distribution story,
  which means a registry. Not now.

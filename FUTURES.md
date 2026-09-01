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

---

## Local memory: routing index + write arbitration (OzBrain pattern)

**Proposed 2026-08-22. Not started.**

Evaluated [ozbrain.com](https://ozbrain.com/) — a hosted, multi-tenant MCP
knowledge base implementing Karpathy's agent-maintained-wiki pattern
(https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). The
hosting model (Postgres + RLS, per-account encryption, connector auth,
$0–$99/mo tiers) is irrelevant to a single-user home lab. The *retrieval and
write-arbitration mechanism* is not — it is a direct answer to the problem a
small local model (gemma4 on Bairn) actually has: a context window too small
to hold everything it needs to know, and no discipline for what goes in when
it writes something back.

`repo-browser` (SQLite FTS5 + `nomic-embed-text` embeddings) already does
semantic retrieval across 400+ repos. What it does not do is the four things
OzBrain adds on top of retrieval:

1. **Routing index over raw dump.** A small, always-loaded index — article
   name, one-line description, freshness tag — read first, before pulling any
   full article. This is the actual fix for small-context-window pain, more
   than the storage/embedding backend. Right now `repo-browser` relies purely
   on embedding search per query; there is no persistent top-level index the
   model reads before deciding what to pull.
2. **Size discipline at write time.** Cap article/note size; when a note grows
   past a threshold, split and re-link automatically. Never make gemma4 read
   one giant markdown file — same principle as "shrink the residue" in the
   detector table above, applied to memory instead of lint findings.
3. **Freshness tags.** Cheap metadata (`fresh` / `aging`) forcing a recheck
   instead of silently trusting stale content. Trivial to add: a
   last-verified timestamp per note, surfaced in the index.
4. **Conflict-on-write.** New info that contradicts existing canon pauses for
   review instead of silently overwriting. This is the one piece with real
   cost — it requires a model call to compare new content against canon
   before committing the write, which is exactly the kind of overhead a 7B
   model on Bairn feels. Worth prototyping as a category-1 detector rather
   than a full model judgement call: cheap heuristic diff/contradiction check
   first, model escalation only when the heuristic is ambiguous — same
   division of labor as the deterministic-detectors table above (offload to
   a tool wherever possible, model only for what genuinely needs judgement).

**Provenance per write** (which agent, when) is the cheap fifth item — useful
once more than one local agent (browser-use, Ollama-fs, StudySkills,
job-application-package) writes to the same store. Low cost, add alongside
freshness tags.

**Explicitly not adopting:** the hosting/tenancy layer, the SaaS billing
model, or OzBrain itself as a dependency. The site's own "copy this prompt"
onboarding block is itself worth noting as a pattern to avoid replicating
uncritically — it's written to be pasted into an agent as instructions,
which is a reasonable UX choice for their product but not something to
import into `agentRW`'s design without scrutiny.

**Where this plugs in:** most naturally as a `memory`-shaped plugin on top of
`repo-browser`'s existing SQLite/embedding layer, or as a `ctx`-level
primitive (`ctx.memory.query` / `ctx.memory.write`) once `ctx.ask` and the
namespacing split above land — same "declare, don't reach past ctx" model as
everything else in this file.

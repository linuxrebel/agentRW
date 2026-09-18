# Lint Pipeline → Plugin (ctx.ask) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the 178 lines of lint-shaped fix pipeline out of the agentRW core and into the arwLint plugin where it belongs, replacing the six lint-specific `ctx` helpers with one general `ctx.ask`.

**Architecture:** The harness core keeps only a general model-access primitive (`ctx.ask`) plus the already-general `ctx` names (tools, paths, output). All lint-specific logic — gather findings, propose a fix, apply it, revert on breakage, defer to DEBT.md — becomes private to `arwLint/plugin.py`, which already owns `lint_file`. This restores the project's own thesis: the core knows nothing about any one plugin's job. We choose **delete-and-move** over the namespacing alternative in the spec, because only one plugin (arwLint) uses these helpers — namespacing would keep them in the core, deletion removes them.

**Tech Stack:** Python 3.9+, Ollama (OpenAI-compat), pytest (dev-only), the agentRW plugin API.

**Spec:** `FUTURES.md` → section "ctx.ask, and namespacing the findings helpers" (the design argument and the `ctx.ask` signature travel with this plan).

## Global Constraints

- **Python 3.9+.** macOS system Python (3.9.6) must run it. No 3.10+ syntax.
- **Harness writes, it does not judge.** No content validation added to any write path (agentRW `CLAUDE.md`).
- **Parse tolerantly.** Do not touch `extract_tools` shapes.
- **agentRW git workflow:** work on `farkitall`; commit; push `farkitall` to BOTH `gh-origin` and `gl-origin`; `git checkout main && git merge --ff-only farkitall`; push `main` to both; `git checkout farkitall` before finishing.
- **Plugin repos (arwLint/arwPyFormat/arwRunTests):** single remote `origin` (GitHub), branch `main`. Commit + push `origin main`.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Before claiming agentRW works:** `python3 test_extract_tools.py` (exit 0).
- **/opt deploy:** after core changes land, copy changed files to `/opt/agentRW/` (files are `james`-owned, writable without sudo). Updated plugins deploy to `/opt/agentRW/tools/linuxrebel/<name>/`.
- **No live-model test in a standing suite.** A real `/lint` run is non-deterministic and needs ollama+pylint+autopep8 — keep it as a manual smoke step, never a pytest.

---

## File Structure

**agentRW (`/mnt/data/git/AI/agentRW`):**
- `coding_agent.py` — MODIFY. Add `ctx.ask` to `plugin_context()`, bump `PLUGIN_API` to 2, delete the lint pipeline (functions + prompts + `DEBT_FILE` + six `ctx` names).
- `test_session_store.py` — MODIFY. Add tests asserting the new/removed `ctx` surface.

**arwLint (`/mnt/data/git/AI/arwLint`):**
- `plugin.py` — MODIFY. Add the moved pipeline as module-private functions using `ctx.ask`; rewrite `/lint` call sites to call them; add `ctx.api` guard.
- `install.md` — MODIFY. `## API` → `2`.
- `test_pipeline.py` — CREATE. Unit tests for the pure movers (`_clean_proposal`, `_apply_insert`) with a stub ctx.

**arwPyFormat / arwRunTests:** VERIFY ONLY — they use only stable `ctx` names (`tools`, `cwd`, `colour`, `reset`), so they must keep loading unchanged under host API 2.

---

## Task 1: Add `ctx.ask`, bump PLUGIN_API to 2 (agentRW core)

**Files:**
- Modify: `coding_agent.py` — `plugin_context()` (594-616), `PLUGIN_API = 1` (591)
- Test: `test_session_store.py`

**Interfaces:**
- Produces: `ctx.ask(messages, max_tokens=300, send_tools=False, no_think=False) -> str` — raw model access through the harness's trimming/error handling, bound to the live session model/layers/cfg. `ctx.api == 2`.

- [ ] **Step 1: Write the failing test**

```python
# test_session_store.py
import coding_agent as ca

def test_ctx_has_ask_and_api_2():
    ctx = ca.plugin_context("m", {"num_ctx": 2048, "token_budget": 2000}, [None])
    assert ctx.api == 2
    assert callable(ctx.ask)

def test_ctx_ask_calls_call_llm(monkeypatch):
    seen = {}
    def fake_call_llm(model, messages, **kw):
        seen["model"] = model; seen["kw"] = kw
        return "ok"
    monkeypatch.setattr(ca, "call_llm", fake_call_llm)
    ctx = ca.plugin_context("mymodel", {"num_ctx": 2048, "token_budget": 2000}, [7])
    out = ctx.ask([{"role": "user", "content": "hi"}], max_tokens=50)
    assert out == "ok"
    assert seen["model"] == "mymodel"
    assert seen["kw"]["send_tools"] is False
    assert seen["kw"]["max_tokens"] == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/git/AI/agentRW && python3 -m pytest test_session_store.py -k ctx_ -v` (or `python3 test_session_store.py` if the suite is a `__main__` runner)
Expected: FAIL — `ctx.api` is 1 / `ctx` has no `ask`.

- [ ] **Step 3: Bump the API constant**

`coding_agent.py:591`
```python
PLUGIN_API = 2
```

- [ ] **Step 4: Add `ask` to plugin_context**

In `plugin_context(model, cfg, layers_ref)` (594-616), add a closure and expose it. Insert before `return`:
```python
    def _ask(messages, max_tokens=300, send_tools=False, no_think=False):
        # Raw model access for plugins: the harness's trimming + error handling,
        # no lint-shaped prompt. Bound to the live session model/layers/cfg.
        return call_llm(model, messages, gpu_layers=layers_ref,
                        max_tokens=max_tokens, num_ctx=cfg.get("num_ctx"),
                        token_budget=cfg.get("token_budget", TOKEN_BUDGET),
                        send_tools=send_tools, no_think=no_think)
```
and add `ask=_ask,` to the `types.SimpleNamespace(...)` argument list (next to `api=PLUGIN_API,`).

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest test_session_store.py -k ctx_ -v`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
cd /mnt/data/git/AI/agentRW && git add coding_agent.py test_session_store.py
git commit -m "feat(plugin-api): add ctx.ask, bump PLUGIN_API to 2

Raw model access for plugins — the harness's trimming and error handling
with no lint-shaped prompt. Prerequisite for moving the lint pipeline out
of the core (FUTURES.md: ctx.ask). API bump to 2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Add the pipeline to arwLint as private functions (using ctx.ask)

Move the pipeline into `arwLint/plugin.py` as module-private helpers. This task adds them **alongside** the still-working `ctx.*` calls (call sites are rewritten in Task 3, core is deleted in Task 4), so `/lint` keeps working throughout.

**Files:**
- Modify: `/mnt/data/git/AI/arwLint/plugin.py`
- Test: `/mnt/data/git/AI/arwLint/test_pipeline.py` (create)

**Interfaces:**
- Consumes: `ctx.ask` (Task 1), `ctx.tools`, `ctx.resolve_path`, `ctx.writable`, `ctx.write_denied`, `ctx.summarise`.
- Produces (module-private in plugin.py):
  - `_gather_findings(ctx, path, only="") -> list[dict]`
  - `_propose_or_compute(ctx, lines, finding) -> str`
  - `_apply_fix(ctx, path, lines, finding, new) -> dict`
  - `_finish_run(path, snapshot) -> bool`
  - `_defer(ctx, path, finding, note="") -> None`
  - `DEBT_FILE = "DEBT.md"` (module constant, plugin-owned)

- [ ] **Step 1: Write the failing test (pure movers only)**

```python
# /mnt/data/git/AI/arwLint/test_pipeline.py
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "arwlint_plugin", pathlib.Path(__file__).with_name("plugin.py"))
mod = importlib.util.module_from_spec(spec)
mod.__dict__["resolve_abs_path"] = lambda p: pathlib.Path(p)
mod.__dict__["PLUGIN_DIR"] = pathlib.Path(__file__).parent
spec.loader.exec_module(mod)

def test_clean_proposal_strips_fence_and_backticks():
    assert mod._clean_proposal("```\n    x = 1\n```") == "    x = 1"
    assert mod._clean_proposal("`\"\"\"doc\"\"\"`") == '"""doc"""'
    assert mod._clean_proposal("nope\nnope") == ""      # >1 line => ""
    assert mod._clean_proposal("UNFIXABLE") == ""

def test_apply_insert_shebang_goes_below_line1():
    lines = ["#!/usr/bin/env python3\n", "x=1\n"]
    f = {"action_kind": "insert_top", "line": 1}
    out = mod._apply_insert(lines, f, '"""mod."""')
    assert out[0] == "#!/usr/bin/env python3\n"
    assert out[1].strip() == '"""mod."""'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/git/AI/arwLint && python3 -m pytest test_pipeline.py -v`
Expected: FAIL — `_clean_proposal` / `_apply_insert` not defined in plugin.py.

- [ ] **Step 3: Copy the pure helpers verbatim into plugin.py**

Copy these **unchanged** from `coding_agent.py` into `arwLint/plugin.py` (module level, above the `lint_command`):
- `FIX_PROMPT` (1781-1784), `INSERT_PROMPT` (1786-1788)
- `DEBT_FILE = "DEBT.md"` (1790)
- `_parses` (1854-1859)
- `_clean_proposal` (2046-2067)
- `_apply_insert` (2014-2043)
- `_finish_run` (1875-1885)

`_apply_insert` and `_clean_proposal` use `re` — ensure `import re` is present in plugin.py (add if missing). `_finish_run` uses `Path` — plugin.py already imports `Path`.

- [ ] **Step 4: Add the ctx-bound helpers (the parts that called harness internals)**

Add these adapted versions. Changes from core: `call_llm(...)` → `ctx.ask(...)`; `TOOL_REGISTRY[...]`/`write_file_tool` → `ctx.tools[...]`; `resolve_abs_path` → `ctx.resolve_path`. Paste into plugin.py:

```python
def _gather_findings(ctx, path, only=""):
    lint = ctx.tools.get("lint_file")
    if not lint:
        return []
    res = lint(filename=path, symbol=only or "*")
    if "error" in res:
        return [{"error": res["error"]}]
    out = [{**o, "symbol": o.get("symbol", only)} for o in res.get("occurrences", [])]
    return sorted(out, key=lambda f: -f["line"])   # bottom-up: line nums stay valid

def _propose_fix(ctx, lines, finding):
    n = finding["line"]
    lo, hi = max(n - 3, 0), min(n + 2, len(lines))
    context = "".join(f"{i+1}: {lines[i]}" for i in range(lo, hi))
    target = lines[n - 1].rstrip("\n")
    kind = finding.get("action_kind", "line")
    if kind in ("insert_after", "insert_top"):
        where = "at the very top of the file" if kind == "insert_top" \
            else f"immediately after line {n}"
        msgs = [{"role": "system", "content": INSERT_PROMPT},
                {"role": "user", "content": (
                    f"Issue: {finding['symbol']} — {finding['message']}\n"
                    f"Goal: {finding.get('action', '')}\n\n"
                    f"Context:\n{context}\n"
                    f"Write the ONE line to insert {where}.")}]
    else:
        msgs = [{"role": "system", "content": FIX_PROMPT},
                {"role": "user", "content": (
                    f"Issue: {finding['symbol']} — {finding['message']}\n"
                    f"Goal: {finding.get('action', 'fix the issue on this line')}\n\n"
                    f"Context:\n{context}\n"
                    f"Rewrite ONLY line {n}:\n{target}")}]
    raw = ctx.ask(msgs, max_tokens=300, send_tools=False) or ""
    return _clean_proposal(raw)

def _propose_or_compute(ctx, lines, finding):
    kind = finding.get("action_kind", "line")
    if kind.startswith("reindent"):
        want = int(kind.split(":")[1]) if ":" in kind else 4
        return " " * want + lines[finding["line"] - 1].lstrip().rstrip("\n")
    return _propose_fix(ctx, lines, finding)

def _apply_fix(ctx, path, lines, finding, new):
    kind = finding.get("action_kind", "line")
    if kind == "line" or kind.startswith("reindent"):
        out = list(lines)
        out[finding["line"] - 1] = new + "\n"
    elif kind.startswith("insert"):
        out = _apply_insert(lines, finding, new)
    else:
        return {"error": "no_automatic_fix", "action_kind": kind}
    return ctx.tools["write_file"](str(path), "".join(out))

def _defer(ctx, path, finding, note=""):
    ledger = ctx.resolve_path(DEBT_FILE)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(f"- [ ] {path}:{finding['line']} {finding['symbol']} — "
                f"{finding['message']}{(' (' + note + ')') if note else ''}\n")
```

Note: core `_apply_fix` routed through `_apply_checked` → `write_file_tool`; `_apply_checked` was a pass-through kept only for signature symmetry (validation lives in `_finish_run`). Dropped here — call `ctx.tools["write_file"]` directly.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest test_pipeline.py -v`
Expected: PASS. (`plugin.py` still imports and the pure movers behave.)

- [ ] **Step 6: Commit (arwLint)**

```bash
cd /mnt/data/git/AI/arwLint && git add plugin.py test_pipeline.py
git commit -m "feat(pipeline): bring the fix pipeline in-plugin via ctx.ask

Adds gather/propose/apply/finish/defer as module-private helpers using
ctx.ask instead of the harness's lint-shaped ctx names. Call sites still
use ctx.* until the next commit; core still owns them until agentRW drops
them. Pure movers unit-tested.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Rewrite arwLint call sites to the local helpers; guard on ctx.api

**Files:**
- Modify: `/mnt/data/git/AI/arwLint/plugin.py` (call sites 260, 314, 348, 369-370, 387), `install.md`

**Interfaces:**
- Consumes: the Task 2 module-private helpers, `ctx.api`.

- [ ] **Step 1: Add the API guard at the top of `lint_command`**

Immediately after the `def lint_command(ctx, args):` opening / its docstring, before first use of `ctx`:
```python
        if getattr(ctx, "api", 1) < 2:
            print("[Lint] needs agentRW plugin API 2+ (host has "
                  f"{getattr(ctx, 'api', 1)}). Update agentRW.")
            return
```

- [ ] **Step 2: Rewrite the five call sites**

| Line | From | To |
|---|---|---|
| 260 | `ctx.gather_findings(_target, _only)` | `_gather_findings(ctx, _target, _only)` |
| 314 | `ctx.propose_fix(ctx.model, ctx.cfg, ctx.layers, _lines, _f)` | `_propose_or_compute(ctx, _lines, _f)` |
| 348 | `ctx.apply_fix(_path, _lines, _f, _new)` | `_apply_fix(ctx, _path, _lines, _f, _new)` |
| 369 | `ctx.defer(_target, _f)` | `_defer(ctx, _target, _f)` |
| 370 | `print(f"  deferred -> {ctx.debt_file}")` | `print(f"  deferred -> {DEBT_FILE}")` |
| 387 | `ctx.finish_run(_path, _snapshot)` | `_finish_run(_path, _snapshot)` |

Note line 314 changes from `ctx.propose_fix` (which was `_propose_fix`) to `_propose_or_compute` — the reindent/compute branch that core did inside `_propose_or_compute`. Core's `ctx.propose_fix` pointed at `_propose_fix` and the reindent-compute was a *separate* core function never exposed on ctx; the plugin's `reindent` display branch (317) relied on `_propose_fix` returning the computed line. Verify: core `plugin_context` bound `propose_fix=_propose_or_compute` (NOT `_propose_fix`). Confirm which by reading `coding_agent.py:606`; bind the local call to whichever the core `ctx.propose_fix` actually pointed at. (As of audit: `propose_fix=_propose_or_compute`.)

- [ ] **Step 3: Bump install.md API**

`/mnt/data/git/AI/arwLint/install.md` → under `## API`, change `1` to `2`.

- [ ] **Step 4: Verify the plugin still imports and loads**

Run:
```bash
cd /mnt/data/git/AI/arwLint && python3 -c "import importlib.util,pathlib; \
s=importlib.util.spec_from_file_location('p','plugin.py'); m=importlib.util.module_from_spec(s); \
m.__dict__['resolve_abs_path']=lambda x:pathlib.Path(x); m.__dict__['PLUGIN_DIR']=pathlib.Path('.'); \
s.loader.exec_module(m); print('import OK; grep for stale ctx refs:')" \
&& ! grep -nE 'ctx\.(gather_findings|propose_fix|apply_fix|finish_run|defer|debt_file)' plugin.py && echo "no stale ctx pipeline refs"
```
Expected: `import OK` then `no stale ctx pipeline refs`.

- [ ] **Step 5: Commit (arwLint)**

```bash
cd /mnt/data/git/AI/arwLint && git add plugin.py install.md
git commit -m "refactor(lint): call in-plugin pipeline, require plugin API 2

/lint no longer depends on the harness's lint-shaped ctx names; it uses
its own gather/propose/apply/finish/defer via ctx.ask. Guards on ctx.api.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Delete the lint pipeline from the agentRW core

Now that arwLint is self-contained, remove the dead pipeline + the six lint-specific `ctx` names from the core.

**Files:**
- Modify: `coding_agent.py` — `plugin_context()` (remove 6 names), delete pipeline functions + prompts + `DEBT_FILE`
- Test: `test_session_store.py`

**Interfaces:**
- Produces: a `ctx` whose surface is `{api, model, cfg, layers, cwd, tools, resolve_path, writable, write_denied, summarise, render, colour, reset, ask}` — no `gather_findings/propose_fix/apply_fix/finish_run/defer/debt_file`.

- [ ] **Step 1: Write the failing test**

```python
# test_session_store.py
def test_ctx_no_lint_pipeline_names():
    ctx = ca.plugin_context("m", {"num_ctx": 2048, "token_budget": 2000}, [None])
    for gone in ("gather_findings", "propose_fix", "apply_fix",
                 "finish_run", "defer", "debt_file"):
        assert not hasattr(ctx, gone), f"{gone} should be gone from ctx"

def test_core_pipeline_functions_removed():
    for gone in ("_gather_findings", "_propose_fix", "_propose_or_compute",
                 "_apply_fix", "_apply_insert", "_apply_checked", "_finish_run",
                 "_clean_proposal", "_defer", "FIX_PROMPT", "INSERT_PROMPT",
                 "DEBT_FILE"):
        assert not hasattr(ca, gone), f"{gone} should be removed from core"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/git/AI/agentRW && python3 -m pytest test_session_store.py -k "no_lint or pipeline_functions" -v`
Expected: FAIL — names still present.

- [ ] **Step 3: Remove the six names from plugin_context**

In `coding_agent.py` `plugin_context()` delete these argument lines from the `SimpleNamespace`:
```python
        gather_findings=_gather_findings,
        propose_fix=_propose_or_compute,
        apply_fix=_apply_fix,
        finish_run=_finish_run,
        defer=_defer,
        debt_file=DEBT_FILE,
```
(and drop the now-stale `# the finding pipeline` comment above them).

- [ ] **Step 4: Delete the pipeline definitions**

Delete these functions/constants from `coding_agent.py` entirely:
`FIX_PROMPT`, `INSERT_PROMPT`, `DEBT_FILE`, `_gather_findings`, `_propose_fix`, `_parses`, `_apply_checked`, `_finish_run`, `_apply_fix`, `_propose_or_compute`, `_apply_insert`, `_clean_proposal`, `_defer`.

Keep `_file_hash`, the `INGEST_*` prompts, and `ingest_file` — those are the memory subsystem, out of scope. `_parses` is used ONLY by the lint pipeline (`_finish_run`); confirm with `grep -n '_parses\|_finish_run' coding_agent.py` before deleting — remove only if no non-lint caller remains.

- [ ] **Step 5: Run tests to verify pass + nothing else broke**

Run:
```bash
python3 -c "import ast;ast.parse(open('coding_agent.py').read());print('parse OK')"
python3 -m pytest test_session_store.py -v
python3 test_extract_tools.py < /dev/null; echo "extract exit=$?"
```
Expected: parse OK; session_store tests PASS; `extract exit=0`.

- [ ] **Step 6: Commit (agentRW)**

```bash
cd /mnt/data/git/AI/agentRW && git add coding_agent.py test_session_store.py
git commit -m "refactor(core): remove lint pipeline — arwLint owns it now

Deletes ~178 lines of lint-shaped fix pipeline and the six lint-specific
ctx names. The core no longer knows any single plugin's job; arwLint runs
its own pipeline via ctx.ask (FUTURES.md: ctx.ask). Net: core shrinks.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Verify arwPyFormat + arwRunTests still load under API 2

They use only stable `ctx` names. The host API-check refuses a plugin whose declared API is **greater** than the host's, so declaring 1 against host 2 still loads. No code change expected — this task proves it.

**Files:**
- Verify: `/mnt/data/git/AI/arwPyFormat/plugin.py`, `/mnt/data/git/AI/arwRunTests/plugin.py`

- [ ] **Step 1: Confirm they use only surviving ctx names**

Run:
```bash
grep -noE 'ctx\.[a-z_]+' /mnt/data/git/AI/arwPyFormat/plugin.py /mnt/data/git/AI/arwRunTests/plugin.py \
  | grep -E 'gather_findings|propose_fix|apply_fix|finish_run|defer|debt_file' \
  && echo "FAIL: uses a removed name" || echo "OK: only stable ctx names"
```
Expected: `OK: only stable ctx names`.

- [ ] **Step 2: Load all three plugins through the real loader against a temp tools dir**

Run:
```bash
cd /mnt/data/git/AI/agentRW && python3 -c "
import coding_agent as ca, pathlib
found = ca.load_plugins(pathlib.Path('/opt/agentRW/tools'))
print('loaded tools:', sorted(found))
print('plugin status:', [(p['file'], p.get('error','')) for p in ca.PLUGIN_STATUS])
"
```
Expected: format/lint/runtests present, no `error` entries. (Note: `/opt` still has the OLD arwLint until Task 6; this only proves the loader + other two plugins under API 2. Full check is Task 6.)

- [ ] **Step 3: No commit** (verification only; if a plugin *did* break, that is a new task, not a silent fix here).

---

## Task 6: Deploy, live smoke, push everything

**Files:** none new — deployment + integration.

- [ ] **Step 1: Push agentRW per the dual-remote workflow**

```bash
cd /mnt/data/git/AI/agentRW
git push gh-origin farkitall && git push gl-origin farkitall
git checkout main && git merge --ff-only farkitall
git push gh-origin main && git push gl-origin main
git checkout farkitall
```

- [ ] **Step 2: Push arwLint**

```bash
cd /mnt/data/git/AI/arwLint && git push origin main
```

- [ ] **Step 3: Deploy core + plugin to /opt**

```bash
cp /mnt/data/git/AI/agentRW/coding_agent.py /opt/agentRW/coding_agent.py
cp /mnt/data/git/AI/arwLint/plugin.py   /opt/agentRW/tools/linuxrebel/lint/plugin.py
cp /mnt/data/git/AI/arwLint/install.md  /opt/agentRW/tools/linuxrebel/lint/install.md
python3 -c "import ast;ast.parse(open('/opt/agentRW/coding_agent.py').read());print('core deployed OK')"
```

- [ ] **Step 4: Live smoke — /lint against a real file (manual, non-standing)**

Make a scratch file with a known lint finding, run one fix, confirm it still works end-to-end and reverts on breakage:
```bash
printf 'import os\nx=1\n' > /tmp/lint_smoke.py
cd /opt/agentRW && printf '/lint /tmp/lint_smoke.py max=1\ns\nq\n/bye\n' | python3 coding_agent.py 2>&1 | grep -iE 'findings|fixed|reverted|autopep8|Lint'
```
Expected: the `[Lint] N findings …` header prints and the run completes without a traceback. (A model is invoked; output varies — you are checking the pipeline runs in-plugin, not a specific score.)

- [ ] **Step 5: Confirm the weight actually dropped**

```bash
cd /mnt/data/git/AI/agentRW && python3 -c "print(len(open('coding_agent.py').read().splitlines()),'lines')"
```
Expected: ~2730 lines or fewer (was 2911; ~178 removed, minus a few added for `ctx.ask`).

- [ ] **Step 6: Final state check**

```bash
git -C /mnt/data/git/AI/agentRW branch --show-current   # must be: farkitall
git -C /mnt/data/git/AI/agentRW status --short           # clean
git -C /mnt/data/git/AI/arwLint status --short           # clean
```

---

## Self-Review

**Spec coverage (FUTURES.md "ctx.ask …"):**
- "ctx has no way to call a model … `ctx.ask(messages, max_tokens=…, send_tools=False)`" → Task 1. ✔
- "Six of the nineteen ctx names are the lint pipeline …" (gather_findings, propose_fix, apply_fix, finish_run, defer, debt_file) → removed in Task 4, moved in Task 2. ✔
- "That is a `ctx.api` bump to 2 … do it before the API has outside users" → Task 1 (bump) + Task 3 (plugin guards on api). ✔
- Namespacing alternative (`ctx.findings.gather`) → explicitly rejected in Architecture (delete-and-move instead, since one consumer). ✔

**Placeholder scan:** No TBD/TODO; every code step has real code; the 178-line move cites exact source line ranges rather than reprinting (a verbatim move, not a rewrite) — the changed lines (call_llm→ctx.ask, TOOL_REGISTRY→ctx.tools, resolve_abs_path→ctx.resolve_path) are shown in full in Task 2 Step 4.

**Type consistency:** Local helpers take `ctx` as first arg consistently (`_gather_findings(ctx, …)`, `_propose_or_compute(ctx, …)`, `_apply_fix(ctx, …)`, `_defer(ctx, …)`); `_finish_run(path, snapshot)` and `_clean_proposal`/`_apply_insert` take no ctx (pure). Call sites in Task 3 match these signatures. `DEBT_FILE` referenced as module global in `_defer` and the print at line 370. ✔

**One open verification carried into execution (Task 3 Step 2 note):** confirm whether core `ctx.propose_fix` bound `_propose_fix` or `_propose_or_compute`, and point the local call at the same one. Audit shows `_propose_or_compute`; the executor re-checks `coding_agent.py` before rewriting line 314.

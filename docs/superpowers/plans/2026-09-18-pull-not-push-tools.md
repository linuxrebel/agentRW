# Pull-not-Push Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop pushing the tool schema every turn; let the model pull tools on demand, so weak models stop tool-hunting on chat and every turn spends less of the 2048-token window.

**Architecture:** Two-pass turn. Pass 1 sends the question with **no** OpenAI schema and a system prompt that lists tools as names+one-liners plus a `USE:` protocol. If the model replies `USE: <tool>...`, the harness activates those tools and re-runs the same turn with the schema (pass 2); otherwise the pass-1 reply is the answer. The harness parses only its own `USE:` sentinel — never the user's natural language. Reuses the existing advertise/dispatch split.

**Tech Stack:** Python 3.9+, Ollama (OpenAI-compat), pytest-style `__main__` test runners (match `test_session_store.py`).

**Spec:** `docs/superpowers/specs/2026-09-18-pull-not-push-tools-design.md`

## Global Constraints

- **Python 3.9+.** No 3.10+ syntax.
- **Harness writes, does not judge; parse tolerantly** (agentRW `CLAUDE.md`) — unchanged.
- **No new deps.** No registry/embeddings/search-tool. ~8 tools; selection is moot.
- **Harness parses zero user natural language for tool gating.** Only its own `USE:` sentinel.
- **git:** work on `farkitall`; commit; push `farkitall` to `gh-origin` + `gl-origin`; `checkout main && merge --ff-only farkitall`; push `main` both; `checkout farkitall` before finishing.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Before claiming it works:** `python3 test_extract_tools.py` (exit 0) and `python3 test_session_store.py` (all pass).
- **/opt deploy:** `coding_agent.py` is `james`-owned (copy directly, no sudo).
- **Models under test:** `ornith-1.5:9b` (default), `MotherMAE/ornith-latest:latest`, `dolphin3:latest`, `qwen3.5:latest`.
- **No guesses:** Phase 1 does not start until Phase 0 clears its bar.

---

## PHASE 0 — Viability experiment (GATES everything)

### Task 0: Measure whether the models can pull

**Files:**
- Create: `scratchpad/exp/rig5.py` (throwaway; not committed)

**Interfaces:**
- Produces: a decision — PROCEED or REJECT — plus a per-model grid of chat-USE-rate and task-USE-then-call success.

- [ ] **Step 1: Write the rig**

A standalone rig (pattern of the session's rig2–4): load `/opt/agentRW/coding_agent.py` as `ca`, build a **pull prompt** string, and for each model run pass 1 (no schema) then, if `USE:` appears, pass 2 (schema on, only named tools advertised). Classify.

```python
import importlib.util, json, re
from pathlib import Path
spec = importlib.util.spec_from_file_location("ca", "/opt/agentRW/coding_agent.py")
ca = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(ca)
except SystemExit: pass

TOOLS_ONELINE = "\n".join(f"- {n}: {(f.__doc__ or '').strip().splitlines()[0]}"
                         for n, f in ca.TOOL_REGISTRY.items())
PULL_SYS = (
    "You are a coding assistant. You have NO tools loaded right now.\n"
    "Available tools (names only):\n" + TOOLS_ONELINE + "\n\n"
    "If the request needs to read, write, search, or run files or commands, "
    "reply with EXACTLY one line: `USE: <tool> [<tool> ...]` and nothing else. "
    "Otherwise just answer normally.")

USE_RE = re.compile(r'^\s*USE:\s*(.+)$', re.MULTILINE)
def parse_use(reply):
    m = USE_RE.search(reply or "")
    if not m: return []
    return [w for w in re.split(r'[\s,]+', m.group(1).strip()) if w in ca.TOOL_REGISTRY]

QS = [("CHAT","what languages can you code in"),
      ("CHAT","explain what recursion is in one sentence"),
      ("TASK","read the file /opt/agentRW/requirements.txt"),
      ("TASK","create a file /tmp/exp_hi.py that prints hello"),
      ("TASK","list the files in /tmp")]
MODELS = ["ornith-1.5:9b","MotherMAE/ornith-latest:latest","dolphin3:latest","qwen3.5:latest"]
OUT = Path(__file__).with_name("rig5.jsonl")

with OUT.open("w") as fh:
    for model in MODELS:
        for kind,q in QS:
            for r in range(4):
                # PASS 1: pull prompt, NO schema
                p1 = ca.call_llm(model, [{"role":"system","content":PULL_SYS},
                                         {"role":"user","content":q}],
                                 max_tokens=200, num_ctx=None,
                                 token_budget=ca.TOKEN_BUDGET, send_tools=False)
                use = parse_use(p1)
                called = []
                if use:
                    # PASS 2: real schema, only requested tools advertised
                    ca._active_tools.clear(); ca._active_tools.update(use)
                    p2 = ca.call_llm(model, [{"role":"system","content":ca.build_prompt()},
                                             {"role":"user","content":q}],
                                     max_tokens=200, num_ctx=None,
                                     token_budget=ca.TOKEN_BUDGET, send_tools=True)
                    called = [t[0] for t in ca.extract_tools(p2)]
                rec = {"model":model,"kind":kind,"q":q,"rep":r,
                       "use":use,"called":called,"p1_head":(p1 or "")[:80].replace("\n"," ")}
                fh.write(json.dumps(rec)+"\n"); fh.flush()
                print(f"{model:30} {kind} r{r} use={use} called={called}")
print("ALL DONE")
```

- [ ] **Step 2: Run it (background; slow — 4 model loads)**

Run: `cd scratchpad/exp && python3 rig5.py` (background). ~80 calls.

- [ ] **Step 3: Classify against the bar**

```python
import json
from collections import defaultdict
rows=[json.loads(l) for l in open("rig5.jsonl")]
agg=defaultdict(lambda:[0,0,0,0])  # model -> [chat_use, chat_n, task_ok, task_n]
for r in rows:
    a=agg[r["model"]]
    if r["kind"]=="CHAT": a[1]+=1; a[0]+= (len(r["use"])>0)
    else: a[3]+=1; a[2]+= (len(r["called"])>0)
for m,(cu,cn,tk,tn) in agg.items():
    print(f"{m:30} chat_USE={cu}/{cn} (want 0)   task_pull_ok={tk}/{tn} (want high)")
```

- [ ] **Step 4: DECISION GATE**

Proceed to Phase 1 **only if**, on `ornith-1.5:9b` **and** `qwen3.5`:
`chat_USE ≈ 0/8` **and** `task_pull_ok ≥ 6/12` (≈ push baseline).
Otherwise **STOP** — report to the user that pull is not viable on the good
models and the harness stays push. Record the grid either way in memory.

*(No commit — rig5 is scratchpad. The deliverable is the decision + grid.)*

---

## PHASE 1 — Implementation (only after Task 0 clears the gate)

> If Phase 0 refined the `PULL_SYS` wording, use the wording that won. The code
> structure below is unchanged by wording.

### Task 1: `parse_use_request` — the sentinel parser

**Files:**
- Modify: `coding_agent.py` (new module-level function near `extract_tools`)
- Test: `test_session_store.py`

**Interfaces:**
- Produces: `parse_use_request(reply: str) -> List[str]` — tool names from a leading
  `USE:` line, filtered to `TOOL_REGISTRY`; `[]` if none.

- [ ] **Step 1: Write the failing test**

```python
def test_parse_use_request():
    assert ca.parse_use_request("USE: read_file") == ["read_file"]
    assert ca.parse_use_request("USE: read_file write_file") == ["read_file","write_file"]
    assert ca.parse_use_request("sure, here is an answer") == []
    assert ca.parse_use_request("USE: not_a_tool") == []   # filtered to registry
    print("  parse_use_request extracts + filters USE line   ok")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -c "import importlib.util,pathlib as P; s=importlib.util.spec_from_file_location('t','test_session_store.py'); t=importlib.util.module_from_spec(s); s.loader.exec_module(t); t.test_parse_use_request()"`
Expected: AttributeError / FAIL.

- [ ] **Step 3: Implement**

```python
_USE_RE = re.compile(r'^\s*USE:\s*(.+)$', re.MULTILINE)

def parse_use_request(reply: str) -> List[str]:
    """Tool names from a model's `USE: ...` line, filtered to the registry.

    The harness's own sentinel — the model's request to pull tools. Not the
    user's words: the harness never parses natural language for tool gating.
    """
    m = _USE_RE.search(reply or "")
    if not m:
        return []
    return [w for w in re.split(r'[\s,]+', m.group(1).strip()) if w in TOOL_REGISTRY]
```

- [ ] **Step 4: Run to verify pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(pull): parse_use_request — the USE: sentinel parser

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2: Pull system prompt (names + one-liners + USE protocol)

**Files:**
- Modify: `coding_agent.py` — `SYSTEM_PROMPT`, `build_prompt()` (and a names+one-liner renderer)
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: `TOOL_REGISTRY`, `_active_tools`.
- Produces: `build_prompt(pull: bool = True) -> str`. In pull mode the prompt lists
  tools as `- name: one-liner`, includes the `USE:` instruction, and contains no
  full signatures.

- [ ] **Step 1: Write the failing test**

```python
def test_pull_prompt_lists_names_and_use_protocol():
    p = ca.build_prompt(pull=True)
    assert "USE:" in p
    assert "read_file" in p and "write_file" in p
    # names+one-liners only: no full signature parens block for tools
    assert "(filename" not in p
    # smaller than the old push prompt
    assert len(p) < len(ca.build_prompt(pull=False))
    print("  pull prompt: names+USE, no schemas, smaller       ok")
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL (`build_prompt` takes no `pull`).

- [ ] **Step 3: Implement**

Add a names+one-liner renderer and a `pull` branch:
```python
def _tool_names_block() -> str:
    return "\n".join(f"- {n}: {_safe_doc(f).split('. ')[0]}"
                     for n, f in TOOL_REGISTRY.items() if n in _active_tools)

PULL_INSTRUCTION = (
    "You have NO tools loaded right now. If the request needs to read, write, "
    "search, or run files or commands, reply with EXACTLY one line: "
    "`USE: <tool> [<tool> ...]` and nothing else. Otherwise answer normally.")

def build_prompt(pull: bool = True) -> str:
    dirs = "\n".join(f"  {d}" for d in [_agent_cwd[0], *_extra_write_dirs])
    if pull:
        tools = "\n" + PULL_INSTRUCTION + "\n\nTools:\n" + _tool_names_block() + "\n"
    else:
        tools = "".join(_tool_block(n, f) for n, f in TOOL_REGISTRY.items()
                        if n in _active_tools)
    return SYSTEM_PROMPT.replace("{{tool_list_repr}}", tools) \
                        .replace("{{writable_dirs}}", dirs) \
                        .replace("{{", "{").replace("}}", "}")
```

- [ ] **Step 4: Run to verify pass.** Expected: PASS. Also `python3 -c "import ast;ast.parse(open('coding_agent.py').read())"`.

- [ ] **Step 5: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(pull): names-only pull prompt with USE protocol

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 3: Two-pass turn in run()

**Files:**
- Modify: `coding_agent.py` — the turn logic in `run()` (the `call_llm` at the top of the per-turn `while True:` loop)
- Test: `test_session_store.py` (via a stubbed `call_llm`, matching the existing monkeypatch pattern)

**Interfaces:**
- Consumes: `parse_use_request` (Task 1), `build_prompt(pull=...)` (Task 2).
- Behavior: pass 1 uses `send_tools=False` and the pull prompt. If
  `parse_use_request(reply)` is non-empty, activate those tools for this turn,
  rebuild the system message with `build_prompt(pull=False)`, and re-run once
  with `send_tools=True`. Else the pass-1 reply is the turn's answer.

- [ ] **Step 1: Write the failing test (stubbed call_llm)**

```python
def test_two_pass_pull_then_call():
    # pass1 returns USE, pass2 returns a real call; assert both happen in order.
    calls = []
    orig = ca.call_llm
    try:
        def fake(model, messages, **kw):
            calls.append(kw.get("send_tools"))
            return "USE: read_file" if len(calls)==1 else 'read_file({"filename":"/x"})'
        ca.call_llm = fake
        reply, tools = ca._turn_reply("read /x", "m", [None],
                                      {"num_ctx":None,"token_budget":ca.TOKEN_BUDGET,
                                       "max_tokens":200})
    finally:
        ca.call_llm = orig
    assert calls == [False, True]           # pass1 no schema, pass2 schema
    assert tools and tools[0][0] == "read_file"
    print("  two-pass: USE on pass1, tool call on pass2       ok")
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL (`_turn_reply` missing).

- [ ] **Step 3: Implement — extract the turn reply into a testable helper**

Add `_turn_reply(user, model, layers_ref, cfg) -> Tuple[str, list]` that does the
two-pass and returns `(final_reply, extract_tools(final_reply))`; call it from
`run()` in place of the current single `call_llm`. Minimal extraction — moves the
existing call, adds the pull branch:
```python
def _turn_reply(user, model, layers_ref, cfg):
    sys1 = build_prompt(pull=True)
    r1 = call_llm(model, [{"role":"system","content":sys1},
                          {"role":"user","content":user}],
                  gpu_layers=layers_ref, max_tokens=cfg["max_tokens"],
                  num_ctx=cfg["num_ctx"], token_budget=cfg["token_budget"],
                  send_tools=False)
    want = parse_use_request(r1)
    if not want:
        return r1, []
    _active_tools.clear(); _active_tools.update(want)
    sys2 = build_prompt(pull=False)
    r2 = call_llm(model, [{"role":"system","content":sys2},
                          {"role":"user","content":user}],
                  gpu_layers=layers_ref, max_tokens=cfg["max_tokens"],
                  num_ctx=cfg["num_ctx"], token_budget=cfg["token_budget"],
                  send_tools=True)
    return r2, extract_tools(r2)
```
*(Note: this simplifies the turn to a fresh system+user each pass. Reconciling
with the running `messages` history / folding is part of this task — keep the
existing history append behavior; only the model call becomes two-pass. The
executor wires `_turn_reply` into the loop preserving `remember(...)` calls.)*

- [ ] **Step 4: Run to verify pass** + full suites:
`python3 test_session_store.py` (all pass) and `python3 test_extract_tools.py </dev/null` (exit 0).

- [ ] **Step 5: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(pull): two-pass turn — pull tools on demand

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 4: Retire `_is_smalltalk` if pull makes it redundant

**Files:**
- Modify: `coding_agent.py` (remove `_SMALLTALK`, `_is_smalltalk`, and the smalltalk branch in `run()`) — **only if** a live check confirms chat no longer hunts under pull.
- Test: `test_session_store.py`

- [ ] **Step 1: Live check** — with pull wired, run the canonical chat questions through the four models (reuse rig5-style pass-1-only). If chat USE/tool ≈ 0 without the smalltalk guard, it is redundant.
- [ ] **Step 2:** If redundant, delete `_SMALLTALK`, `_is_smalltalk`, the `turn_is_smalltalk` branch; add a test asserting `not hasattr(ca, "_is_smalltalk")`. If NOT redundant, keep it and note why in a comment. (Data decides; no guess.)
- [ ] **Step 3:** Run suites; commit.

### Task 5: Deploy, live smoke, push

- [ ] **Step 1:** `python3 test_session_store.py` + `python3 test_extract_tools.py </dev/null` both green.
- [ ] **Step 2:** Live smoke against `cagent` default: a chat question answers directly (1 call, no `[tool]`), a task ("create /tmp/pull_smoke.py that prints hi") produces `USE:` then a `write_file` and the file exists.
- [ ] **Step 3:** Push `farkitall` both remotes; ff-merge `main`; push both; back to `farkitall`.
- [ ] **Step 4:** `cp coding_agent.py /opt/agentRW/coding_agent.py`; parse-check.
- [ ] **Step 5:** Update memory ([[agentrw-cross-session-recall]] neighborhood) with the pull design + Phase 0 grid.

---

## Self-Review

**Spec coverage:** pull protocol → Tasks 1–3; names-only prompt → Task 2; USE sentinel, zero user-NL parsing → Task 1; viability gate (no guesses) → Phase 0 Task 0; smalltalk retirement → Task 4; non-goals (no registry/embeddings) → honored (nothing of the sort in any task). ✔

**Placeholder scan:** Phase 0 rig and Tasks 1–3 carry real code. Task 3 flags one genuine integration judgement (wiring `_turn_reply` into the loop while preserving `remember`/folding) — called out explicitly, not hidden as "handle history." Task 4 is data-gated by design, not a placeholder. Phase-0-refined prompt wording is the only deferred literal, and its structure is fixed.

**Type consistency:** `parse_use_request(str)->List[str]` used identically in rig5, Task 1, Task 3. `build_prompt(pull: bool=True)` used with `pull=True`/`pull=False` consistently in Tasks 2–3. `_turn_reply(user, model, layers_ref, cfg)->Tuple[str,list]` defined and called with matching args in Task 3's test and impl. ✔

**Open risk carried to execution:** Task 3's two-pass currently rebuilds a fresh `[system,user]` per pass; the executor must preserve the session's `messages`/folding/`remember` semantics when wiring it into `run()`. Flagged, not hidden.

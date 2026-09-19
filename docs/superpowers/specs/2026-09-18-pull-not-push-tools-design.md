# Pull-not-Push Tool Advertisement — Design

Status: **decided (architecture), experiment-gated (viability)**. 2026-09-18.

## The problem, measured this session

The harness sends the OpenAI tool schema on **every** turn (`send_tools=True` in
`call_llm`, always). Three issues step on each other:

1. **Tiny context / low-or-no VRAM.** On a 2048-token window, cramming every
   tool schema into every request is wasteful *regardless of model*.
2. **The all-present assumption.** The harness assumes every tool that might be
   wanted must already be advertised, every turn.
3. **Strong models fight the harness and win; weak ones give up.** Both classes
   answer a plain chat question fine under bare `ollama run`. In-harness, the
   pushed schema drags weak models into tool-calling on chat.

### Isolation data (rigs 2–4, this session; models: ornith-1.5:9b, MotherMAE/ornith-latest, dolphin3, qwen3.5)

- **The tool schema is the cause of chat tool-hunting, not the system prompt.**
  MotherMAE chat tool-hunts: `bare 0/12, sys(prompt only) 0/12, tools(schema) 8/12, both 10/12`.
  Strong control ornith-1.5:9b: `0/12` in every condition — overcomes the schema.
- **The schema is load-bearing for tasks.** Task tool-use with vs without schema:
  ornith `7/9 → 1/9`, MotherMAE `9/9 → 5/9`, qwen `8/9 → 7/9`, dolphin `4/9 → 3/9`.
  Dropping the schema globally would gut real tasks on the default model.
- Therefore push + a global "always on" is wrong for the constraint, and a
  harness-side natural-language gate (path/verb detection) is the same trap in
  new clothes — the harness is bad at language, the model is good at it.

## Decision: pull, not push

The model — which understands the request — **requests** tools; the harness
serves them and **parses no user natural language**. This is minimal tool
deferral, sized for agentRW's ~8 tools (not the registry/embeddings/search-tool
apparatus that a hundreds-of-tools system needs).

Reuses what already exists: advertisement is already separate from dispatch
(`_active_tools` vs `TOOL_REGISTRY`; unadvertised tools stay callable —
`coding_agent.py`). We make advertisement **pull-driven per turn** instead of
static-and-always.

### Protocol v1 (exact wording finalized by the Phase 0 experiment)

- The system prompt lists tools as **names + one-line descriptions only** — plain
  text, NOT the OpenAI schema — plus this instruction: *"You have no tools loaded
  right now. If the request needs to read, write, search, or run files/commands,
  reply with exactly `USE: <tool> [<tool> ...]` on its own line and nothing else.
  Otherwise answer normally."*
- **Pass 1:** `send_tools=False`. Harness inspects the reply for a leading
  `USE:` line (its own sentinel — not the user's words):
  - No `USE:` → the reply is the answer. Serve it. (Chat → no schema present →
    no hunt, per the `sys` data = 0/12.)
  - `USE: read_file` → harness activates those tools, `send_tools=True`, re-runs
    the **same** user turn. Model emits the real call; existing dispatch handles it.
- The harness parses only its own `USE:` sentinel from the model. Zero user-NL
  parsing. No path regex, no verb lists, no ambiguous-word tables for tool gating.

### Non-goals

- No registry / BM25 / embeddings / `tool_search` tool (N≈8; selection is moot).
- No per-turn NL heuristics to guess "tool-shaped" turns.
- No fine-tuning.
- Not touching the shell-passthrough ambiguous-word logic (separate concern).

## Open question → Phase 0 gate (no guesses)

Can the target models actually **do** the pull?
- (a) Chat turn → **no** `USE:` / no tool call (want ~0).
- (b) Task turn → correct `USE:` on pass 1, correct tool call on pass 2 (want ≥ push baseline).

**Success bar:** on the models actually used (ornith-1.5:9b, qwen3.5), chat
USE-rate ≈ 0 **and** task success ≥ the push baseline (ornith/qwen ≈ 6–8/9).
Weak/flaky models (MotherMAE, dolphin3) reported but not gating.

**If the good models cannot pull reliably → pull is rejected; keep push.** The
implementation phase does not begin until Phase 0 clears this bar.

## Success criteria (measured)

Per model, ≥4 reps: chat `USE`/tool rate, task USE-then-call success rate.
Compare against the push baseline already captured in rig4.

# File Ingest — Design

**Date:** 2026-08-31
**Status:** Approved for planning
**Scope:** agentRW (`coding_agent.py`)
**Related:** `2026-08-31-cross-session-recall-design.md` (ingest is the distilled
"write side"; recall is the "read side")

## Problem

agentRW reads files 200 lines at a time (`read_file_tool`, page by page). To
"learn what a large file does," the model pages through it; older pages get
folded to one-line pointers by `proactive_trim`. By the end the model has *seen*
the whole file but retains it only as a lossy folded trail — and the next
session starts blind again. There is no durable, distilled understanding of a
file.

Cross-session recall (separate spec) fixes retrieval of past *dialogue*, and
explicitly does **not** index raw file-page dumps. Ingest fills the other half:
produce one compact, durable **digest** per file, small enough to live in the
~2048-token window and reusable across sessions.

## Goal

`/ingest <path>` → a compact file digest that:
- fits the window (raw pages never have to);
- persists across sessions, keyed to file content so it self-invalidates on
  change;
- reuses cached work when the file is unchanged;
- adds **no new dependency or service** — native SQLite + the existing
  `call_llm` + `read_file` paging;
- degrades cleanly, never crashing the agent.

## Decisions (settled during brainstorming)

1. **Trigger: `/ingest <path>` human command only.** No model-facing tool in
   v1 — ingest spends many model calls, too costly to let the model fire
   freely. A tool can be added later behind the same gating recall uses.
2. **Chunking: fixed 200-line windows with ~20-line overlap.** Reuses the
   existing pager; language-agnostic (agentRW is polyglot). Overlap keeps a
   function split across a boundary from being lost. Structural/AST chunking is
   rejected now (Python-only, large); a seam is left.
3. **Summarization: map-reduce.** Each chunk → one `call_llm` summarize call
   (map). Then one final `call_llm` over all chunk summaries → a compact
   file-level digest (reduce). Per-chunk summaries are kept for drill-down.
   Rejected: one-level concat (digest too big for the window) and rolling
   refine (sequential + re-sends the running summary each call).
4. **Storage + staleness: two tables, sha256 content hash.** `file_digests`
   (canonical per-file digest) + `file_chunks` (map summaries). Keyed on
   `(path, content_hash)`; reuse on hash match, recompute on mismatch. Hash
   over mtime — robust both ways.
5. **Output: foldable digest into the window + console, atomic on failure.**
   Digest injected as one foldable message (not pinned) so recall/FTS picks it
   up and the budget is respected; per-chunk summaries stay in the DB. Console
   shows progress. Digest hard-capped (~200 words). Failure aborts atomically,
   storing nothing partial.

## Key facts about the existing code

- `call_llm(model, messages, gpu_layers=..., max_tokens=..., ...)`
  (`coding_agent.py:1262`) is the single model entry point; precedent for a
  small one-off call exists (`max_tokens=300`, line 1517).
- The per-message `summary` field is **deterministic formatting**
  (`_summarise_result`), *not* a model summary. Ingest's chunk/file summaries
  are real `call_llm` calls.
- `read_file_tool(path, start_line, max_lines=200)` (`coding_agent.py:234`) is
  the pager to reuse.
- `SessionStore` (`coding_agent.py:976`) owns `sessions.db`; ingest adds tables
  there. Digests reach recall by riding the `messages_fts` index the recall
  spec adds — no separate FTS.

## Components

All in `coding_agent.py`, near `SessionStore` and the command handling.

### 1. Schema (add to `SCHEMA`)

```sql
CREATE TABLE IF NOT EXISTS file_digests (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL,
    cwd          TEXT,
    content_hash TEXT NOT NULL,
    lines        INTEGER,
    n_chunks     INTEGER,
    digest       TEXT NOT NULL,
    model        TEXT,
    created      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS file_digests_path ON file_digests(path, content_hash);
CREATE TABLE IF NOT EXISTS file_chunks (
    digest_id  INTEGER NOT NULL,
    chunk_no   INTEGER NOT NULL,
    start_line INTEGER,
    end_line   INTEGER,
    summary    TEXT,
    PRIMARY KEY (digest_id, chunk_no)
);
```

### 2. `SessionStore` methods

- `find_digest(path, content_hash) -> Optional[row]` — cache lookup for reuse.
- `save_digest(path, cwd, content_hash, lines, n_chunks, digest, model, chunk_summaries) -> int`
  — insert `file_digests` + `file_chunks` in one transaction (atomic).
- `chunks_for(path) -> list[(chunk_no, start_line, end_line, summary)]` — for
  `/recall` drill-down into a file.
- All wrapped in the store's existing defensive try/except; failure disables
  the feature, never the agent.

### 3. Ingest driver (module-level function, e.g. `ingest_file`)

Signature: `ingest_file(path, model, store, layers_ref) -> dict`.

Flow:
1. Resolve + read the file (`resolve_abs_path`, reuse `read_file` validation).
   Missing/binary/unreadable → return `{"error": ..., "hint": "use read_file"}`.
2. Compute `content_hash = sha256(file_bytes)`.
3. `store.find_digest(path, hash)` → if hit, return the cached digest
   (`cached=True`), skip all model calls.
4. Page the file into 200-line chunks with 20-line overlap.
5. **Map:** for each chunk, `call_llm` with a tight summarize prompt
   (`max_tokens` ~150). Print `summarizing chunk i/N…` to console.
   - One chunk total (file ≤ 200 lines) → that single summary *is* the digest;
     skip the reduce step.
6. **Reduce:** `call_llm` over the concatenated chunk summaries → the file-level
   digest, capped to ~200 words (+ `_cap_tool_result` backstop).
7. `store.save_digest(...)` atomically.
8. Return `{"path", "lines", "n_chunks", "digest", "cached": False}`.

Any `call_llm` failure mid-run → abort, save nothing, return an error dict.

### 4. `/ingest <path>` command handler

- Call `ingest_file`.
- On success: print the digest to console; **inject** it into the window as one
  foldable message via the normal `remember`/`store.add` path
  (`no_index=False`, so it enters `messages_fts` and becomes recallable now and
  across sessions). Header line: `path · N lines · ingested digest`.
- On `cached`: same, but note "(cached)".
- On error: print the hint, inject nothing.

### 5. Recall drill-down (small integration)

When `/recall` surfaces an ingested-digest message, the digest header names the
path; `store.chunks_for(path)` lets a follow-up show which chunk(s) cover a
topic. (Thin; can be a follow-on if it complicates v1.)

## Data flow

```
/ingest <path>
  ├─ read file ─▶ sha256
  ├─ cache hit? ──▶ reuse digest ─────────────┐
  └─ miss:                                     │
       page (200-line, 20 overlap)             │
        └─ map: call_llm per chunk  (console progress)
             └─ reduce: call_llm ▶ digest (~200w, capped)
                  └─ save_digest (file_digests + file_chunks, atomic)
                                                │
  digest ──▶ console + foldable window message ─┘
                 └─(store.add, no_index=0)─▶ messages_fts ─▶ recallable
```

## Error handling

- **Missing/binary/unreadable file:** graceful refuse + point at `read_file`.
- **ollama down / `call_llm` fails mid-run:** abort atomically, store nothing,
  clear message; agent keeps running.
- **Store unavailable:** ingest still summarizes and prints to console; it just
  cannot cache or make the digest recallable (matches "a worse session, not a
  dead one").
- **Digest oversized:** capped at reduce prompt; `_cap_tool_result` backstop.

## Testing

Extend `test_session_store.py` (plain asserts, offline tmp DB; stub `call_llm`
so no model/network is needed):

- `save_digest` + `find_digest` round-trip; `(path, hash)` cache hit returns the
  stored digest.
- Changed content (different hash) → cache miss.
- Chunking: an N-line file yields the expected chunk count with 20-line overlap;
  a ≤200-line file yields one chunk and skips reduce.
- `ingest_file` with a stubbed `call_llm` produces a digest and persists chunks;
  a stub that raises aborts atomically (no `file_digests` row written).
- Digest injected via ingest is findable by `search()` (ties to recall spec);
  raw file pages are not.

## Out of scope / rejected alternatives

- **Model-facing `ingest_tool`** — deferred (many model calls; too costly to let
  the model trigger freely). Add later behind recall-style gating.
- **Structural/AST chunking** — Python-only, large; seam left at the chunker.
- **One-level / rolling-refine summarization** — rejected (window fit /
  sequential cost).
- **Auto-ingest on large `read_file`** — rejected (hidden, expensive, surprising).
- **New dependencies of any kind** — none introduced.

# Cross-Session Recall — Design

**Date:** 2026-08-31
**Status:** Approved for planning
**Scope:** agentRW (`coding_agent.py`)

## Problem

agentRW persists every message to a single global SQLite store
(`SESSION_DB = ~/.config/coding_agent/sessions.db`, `coding_agent.py:2282`).
But the store is only ever read back for the *current* session — the window is
rebuilt per session and past sessions are invisible. All the history is on disk;
none of it is retrievable into a new conversation.

This gap was surfaced while comparing agentRW to the TencentDB-Agent-Memory
stack. That stack's one idea worth borrowing for a token-starved local agent is
**retrieval, not stuffing**: keep everything on disk, pull only the few most
relevant snippets into the window on demand. Its actual infrastructure (Docker
services, ClickHouse, Redis, extra LLM calls, an OpenAI-compatible injecting
proxy) is the opposite of agentRW's ethos and is explicitly **not** adopted —
see "Rejected alternatives".

## Goal

Give agentRW cross-session recall that:
- adds **no new dependency and no new service** — native SQLite only;
- costs **zero tokens** unless recall is actually used;
- never blows the ~2048-token window it is designed around;
- degrades cleanly, never crashing the agent (matches existing `SessionStore`).

## Decisions (settled during brainstorming)

1. **Relevance = FTS5 keyword search (BM25).** Native to SQLite, zero deps,
   zero window cost. For a coding agent, literal terms (file names, error
   strings, function names, commands) give a high hit rate. A clean seam is
   left for semantic embeddings later, but embeddings are rejected now: they
   fight the 4 GB-GPU / 7B-model hardware constraint that is agentRW's whole
   reason to exist.
2. **Triggers: manual primary + model-driven auto.**
   - `/recall [--all] <query>` — human slash command, zero prompt cost.
   - `recall_tool(query, all=False)` — model-facing tool the model calls when
     it decides it is short on context.
   - Session-start hint for discoverability.
   - **Not** blanket auto-injection every turn (pollutes the tiny window).
3. **Scope: same-cwd default, global on demand.** Default ranks/limits to the
   current directory's past sessions; `--all` (command) / `all=True` (tool)
   searches everything.
4. **Result shape: summary + snippet, K=4, capped.** Each hit is the stored
   `summary` plus a ~200-char excerpt around the match plus a pointer; the whole
   block is run through the existing `_cap_tool_result`. Full message content is
   never auto-injected — it stays one deliberate step away.
5. **Index: external-content FTS5.** `messages_fts` shadows `messages` (uses
   `messages.id` as rowid, stores no duplicate text) kept in sync by triggers.
   Index both `content` and `summary`. Auto-backfill once from pre-existing
   rows. Fall back cleanly if the sqlite build lacks FTS5.

## Components

All changes live in `coding_agent.py`, near `SessionStore` (~line 933) and the
tool/command plumbing.

### 1. Schema addition

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    summary,
    content='messages',
    content_rowid='id'
);
-- keep the shadow index in sync with messages
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, summary)
    VALUES (new.id, new.content, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, summary)
    VALUES ('delete', old.id, old.content, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, summary)
    VALUES ('delete', old.id, old.content, old.summary);
    INSERT INTO messages_fts(rowid, content, summary)
    VALUES (new.id, new.content, new.summary);
END;
```

Because sync is via triggers, `SessionStore.add()` needs **no** change — the
`folded` UPDATE and any future edits stay indexed automatically.

### 2. `SessionStore._ensure_fts()` (called from `__init__`)

- Detect FTS5 support (attempt to create the virtual table; on
  `sqlite3.OperationalError` mentioning fts5, set `self.fts = False`).
- Create the table + triggers when supported.
- Backfill once: if `messages_fts` is empty but `messages` is not,
  `INSERT INTO messages_fts(rowid, content, summary) SELECT id, content, summary FROM messages`.
- Wrapped in the same defensive try/except as the rest of the store; failure
  disables recall, never the agent.

### 3. `SessionStore.search(query, cwd=None, k=4)`

- FTS5 path:
  ```sql
  SELECT m.session_id, m.seq, s.cwd, m.summary,
         snippet(messages_fts, 0, '', '', '…', 12) AS snip
  FROM messages_fts
  JOIN messages  m ON m.id = messages_fts.rowid
  JOIN sessions  s ON s.id = m.session_id
  WHERE messages_fts MATCH ?
    AND (? IS NULL OR s.cwd = ?)
  ORDER BY bm25(messages_fts)
  LIMIT ?;
  ```
- No-FTS5 path: `LIKE '%query%'` scan over `messages.content`, same return
  tuple shape `(session_id, seq, cwd, summary, snippet)`.
- Excludes the current session's own rows (recall is about *past* context).
- Any error → returns `[]`, logged once, never raises.

### 4. `format_recall(hits) -> str`

- Empty → `"[recall] no matches."`.
- Else per hit: `[sess {session_id} #{seq}] {summary} — "…{snippet}…"`.
- Whole block passed through `_cap_tool_result` so recall can never exceed its
  budget slice.

### 5. Entry points — both feed the window

- **`/recall [--all] <query>`**: parse `--all` → `cwd=None`, else
  `cwd=str(_agent_cwd[0])`. Run `search`, `format_recall`, print to console,
  **and** append the block to the window as a foldable message so the model can
  use it.
- **`recall_tool(query, all=False)`**: same core; result returns as a normal
  tool result (already foldable/trimmable). Tight docstring to keep the
  advertised cost minimal.

### 6. Advertisement gating

`recall_tool` is advertised **only when the DB holds history from an earlier
session** (a prior `sessions` row / messages outside the current session
exist). Fresh DB or first-ever session → tool hidden, 0 tokens/turn. Once
history exists → tool advertised (~40 tokens/turn, the honest price of
self-serve auto). Uses the existing tool-advertisement mechanism.

### 7. Session-start hint

On startup, if prior sessions exist **for this cwd**, print one line:

```
[Recall] N past session(s) here. /recall <topic> to pull, or I'll check when unsure.
```

No token cost — console only, not injected.

## Data flow

```
store.add(msg) ──(trigger)──▶ messages_fts stays current   (no code in add())

/recall / recall_tool(query)
    └▶ SessionStore.search(query, cwd, k=4)
         └▶ rows → format_recall() → _cap_tool_result()
              └▶ foldable message in window  (pointer per hit → re-fetch full
                                              original via existing lookup)
```

## Error handling

- **No FTS5 in the sqlite build:** `LIKE` fallback, one-line notice, agent runs
  normally.
- **Any search/index error:** caught, returns empty, logged once, never raised
  — same contract as `SessionStore` today ("losing history is a worse session,
  not a dead one").
- **Stale cwd** (user ran `/cwd` mid-session): scope compares against live
  `_agent_cwd[0]`; `sessions` rows keep their start-cwd. Accepted minor — coding
  sessions mostly stay in one directory; `--all` covers the rest.

## Testing

Extend `test_session_store.py` (plain asserts, offline tmp DB, no framework —
matching existing style):

- `add()` makes a row searchable via `search()`.
- Keyword hit ranks the matching message.
- cwd scope filters to the current directory by default.
- `--all` / `all=True` returns matches across directories.
- Backfill populates `messages_fts` from rows inserted before the index existed.
- FTS5-absent path (simulated) returns the same tuple shape via `LIKE`.
- `format_recall` output is capped (never exceeds the tool-result slice).

## Out of scope / rejected alternatives

- **TencentDB-Agent-Memory stack** (Docker, ClickHouse, Redis, injecting proxy,
  extra LLM calls) — heavyweight, team-scale, and directly opposed to agentRW's
  offline single-file token-frugal design. Not adopted.
- **Semantic embeddings (B/C from Decision 1)** — deferred; a clean seam is left
  at `SessionStore.search` so a reranker or vector path can slot in without
  touching callers.
- **Auto-inject every turn** — rejected; pollutes the ~2048-token window.
- **New dependencies of any kind** — none introduced.

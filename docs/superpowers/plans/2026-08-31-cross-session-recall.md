# Cross-Session Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give agentRW native SQLite FTS5 keyword recall over its existing global session store, so a session can pull relevant snippets from past sessions via `/recall` and a gated model-facing `recall` tool.

**Architecture:** Add an external-content FTS5 index (`messages_fts`) that shadows the existing `messages` table via triggers, plus a `no_index` flag so raw file-page dumps stay out. A `SessionStore.search()` returns ranked hits; `format_recall()` renders them capped. Two entry points feed the window: an inline `/recall [--all] <query>` command and a `recall` tool that is registered for dispatch but advertised only when prior history exists.

**Tech Stack:** Python 3.9+, stdlib `sqlite3` (FTS5), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-31-cross-session-recall-design.md`

## Global Constraints

- **No new dependencies or services.** stdlib `sqlite3` only.
- **Never crash the agent.** Every new store method is wrapped in defensive
  try/except; failure disables recall, not the agent (matches existing
  `SessionStore`).
- **Respect the window.** Recall output passes through `_cap_tool_result`.
- **Match existing style.** Tests are plain functions with `assert` in
  `test_session_store.py`, run from `__main__`; the module is loaded as `ca`
  via `importlib`. No test framework.
- **Single file.** All production code lands in `coding_agent.py`.
- Key existing anchors: `SCHEMA` (`coding_agent.py:943`), `SessionStore`
  (`:976`), `SessionStore.add` (`:1006`), `_cap_tool_result` (`:1109`),
  `TOOL_REGISTRY` (`:697`), `_active_tools` (`:731`), `build_prompt` (`:756`),
  `_agent_cwd` (module-global list), `run()` (`:1715`), `remember()` closure
  (`:1734`), the inline command branches (e.g. `/tokens` near `:1808`),
  `SESSION_DB` (`:2282`).

---

### Task 1: Store — schema, FTS index, `no_index`, and `search()`

**Files:**
- Modify: `coding_agent.py` — `SCHEMA` (`:943`), `SessionStore.__init__` (`:979`), `SessionStore.add` (`:1006`); add methods to `SessionStore`.
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: existing `SessionStore(db_path, model, cwd)`, `self.db`, `self.session_id`.
- Produces:
  - `SessionStore.add(seq, role, content, summary="", no_index=False)` (new kwarg, backward compatible).
  - `SessionStore.search(query, cwd=None, k=4) -> list[tuple]` returning
    `(session_id, seq, cwd, summary, snippet)`, excluding the current session,
    cwd-filtered when `cwd` is not None, ranked by bm25 (or LIKE fallback).
  - `SessionStore.has_prior_history() -> bool` and
    `SessionStore.prior_sessions_for_cwd(cwd) -> int`.
  - `self.fts: bool` attribute (True when FTS5 is available).

- [ ] **Step 1: Write failing tests**

Add to `test_session_store.py`:

```python
def _seed(store, rows):
    """rows: list of (session_id, seq, role, content, summary, no_index)."""
    for sid, seq, role, content, summary, no_index in rows:
        store.session_id = sid
        store.add(seq, role, content, summary, no_index=no_index)

def test_search_finds_past_sessions_by_keyword():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping keyword test)      ok")
            return
        cur_session = store.session_id
        _seed(store, [
            (10, 1, "user", "the auth token expiry bug in login.py", "auth bug", False),
            (10, 2, "assistant", "fixed by using <= not <", "fix", False),
            (11, 1, "user", "unrelated chatter about lunch", "lunch", False),
        ])
        store.session_id = cur_session          # search excludes current session
        hits = store.search("auth token", cwd=None, k=4)
        assert hits, "should find the auth message"
        assert any("auth" in (h[3] or "") or "auth" in (h[4] or "") for h in hits)
        assert all(h[0] != cur_session for h in hits), "must exclude current session"
        print("  recall finds past-session keyword hits         ok")

def test_search_scope_is_cwd_by_default_and_global_on_demand():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping scope test)        ok")
            return
        cur = store.session_id
        # session 20 lived in /proj/a, session 21 in /proj/b
        store.db.execute("INSERT INTO sessions (id, started, model, cwd) "
                         "VALUES (20,'t','m','/proj/a'),(21,'t','m','/proj/b')")
        _seed(store, [
            (20, 1, "user", "widget rendering glitch", "widget", False),
            (21, 1, "user", "widget rendering glitch", "widget", False),
        ])
        store.session_id = cur
        local = store.search("widget", cwd="/proj/a", k=4)
        assert local and all(h[2] == "/proj/a" for h in local), "cwd scope failed"
        allscope = store.search("widget", cwd=None, k=4)
        assert {h[2] for h in allscope} >= {"/proj/a", "/proj/b"}, "global scope failed"
        print("  recall scopes to cwd, global on demand          ok")

def test_no_index_rows_are_not_recalled():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping no_index test)     ok")
            return
        cur = store.session_id
        _seed(store, [
            (30, 1, "user", "def frobnicate(): pass  # jquery blob", "read_file page", True),
            (30, 2, "user", "we discussed frobnicate in the meeting", "discussion", False),
        ])
        store.session_id = cur
        hits = store.search("frobnicate", cwd=None, k=4)
        assert hits, "should find the discussion"
        assert all(h[1] != 1 for h in hits), "raw file-page dump must not be recalled"
        print("  raw file-page dumps excluded from recall        ok")

def test_backfill_indexes_preexisting_rows():
    with tempfile.TemporaryDirectory() as d:
        dbp = Path(d) / "s.db"
        # First store writes rows, then we drop the FTS table to simulate a
        # pre-index database, and a fresh store must backfill on open.
        s1 = ca.SessionStore(dbp, model="test", cwd="/proj/a")
        if not s1.fts:
            print("  (fts5 unavailable — skipping backfill test)     ok")
            return
        s1.session_id = 40
        s1.add(1, "user", "backfill me: xyzzy marker", "seed", no_index=False)
        s1.db.execute("DROP TABLE messages_fts")
        s1.db.commit()
        s1.db.close()
        s2 = ca.SessionStore(dbp, model="test", cwd="/proj/a")   # should backfill
        s2.session_id = 999
        hits = s2.search("xyzzy", cwd=None, k=4)
        assert hits, "backfill should make pre-existing rows searchable"
        print("  backfill indexes pre-existing rows              ok")

def test_history_flags():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        assert store.has_prior_history() is False, "fresh db has no prior history"
        assert store.prior_sessions_for_cwd("/proj/a") == 0
        store.db.execute("INSERT INTO sessions (id, started, model, cwd) "
                         "VALUES (50,'t','m','/proj/a')")
        store.db.execute("INSERT INTO messages (session_id, seq, role, content) "
                         "VALUES (50,1,'user','hi')")
        store.db.commit()
        assert store.has_prior_history() is True
        assert store.prior_sessions_for_cwd("/proj/a") == 1
        assert store.prior_sessions_for_cwd("/proj/b") == 0
        print("  prior-history flags report correctly            ok")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python test_session_store.py`
Expected: `AttributeError`/failure — `search`, `has_prior_history`, `fts`, or the `no_index` kwarg do not exist yet.

- [ ] **Step 3: Add the `no_index` column to `messages` in `SCHEMA`**

In `SCHEMA` (`coding_agent.py:951`), add the column to the `messages` table:

```sql
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    summary    TEXT,
    folded     INTEGER NOT NULL DEFAULT 0,
    no_index   INTEGER NOT NULL DEFAULT 0
);
```

Because `CREATE TABLE IF NOT EXISTS` will not alter an existing table, also add
a defensive migration inside `_ensure_fts` (Step 5) using `ALTER TABLE ... ADD
COLUMN` guarded by a check, so older databases gain the column.

- [ ] **Step 4: Thread `no_index` through `add()`**

Modify `SessionStore.add` (`coding_agent.py:1006`):

```python
def add(self, seq: int, role: str, content: str, summary: str = "",
        no_index: bool = False) -> None:
    if not self.live:
        return
    try:
        self.db.execute(
            "INSERT INTO messages (session_id, seq, role, content, summary, no_index) "
            "VALUES (?,?,?,?,?,?)",
            (self.session_id, seq, role, content, summary, int(no_index)))
        self.db.commit()
    except sqlite3.Error:
        pass
```

- [ ] **Step 5: Add `_ensure_fts()`, call it from `__init__`, and add the query methods**

In `SessionStore.__init__`, after `self.db.executescript(SCHEMA)` (`:986`) and
before the `INSERT INTO sessions`, add `self.fts = False` then
`self._ensure_fts()`. (Keep it inside the existing try/except so a failure
disables recall, not the store.)

Add these methods to `SessionStore`:

```python
def _ensure_fts(self) -> None:
    """Create the FTS5 shadow index + triggers, migrate old DBs, backfill once.

    FTS5 is not in every sqlite build. If it is missing, self.fts stays False
    and search() falls back to LIKE. Never raises past this method.
    """
    try:
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(messages)")]
        if "no_index" not in cols:
            self.db.execute("ALTER TABLE messages ADD COLUMN "
                            "no_index INTEGER NOT NULL DEFAULT 0")
        self.db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content, summary, content='messages', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages
            WHEN new.no_index = 0 BEGIN
                INSERT INTO messages_fts(rowid, content, summary)
                VALUES (new.id, new.content, new.summary);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages
            WHEN old.no_index = 0 BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, summary)
                VALUES ('delete', old.id, old.content, old.summary);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages
            WHEN old.no_index = 0 BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, summary)
                VALUES ('delete', old.id, old.content, old.summary);
                INSERT INTO messages_fts(rowid, content, summary)
                VALUES (new.id, new.content, new.summary);
            END;
        """)
        # Backfill once: index pre-existing rows the triggers never saw.
        n = self.db.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        if n == 0:
            self.db.execute(
                "INSERT INTO messages_fts(rowid, content, summary) "
                "SELECT id, content, summary FROM messages WHERE no_index = 0")
        self.db.commit()
        self.fts = True
    except sqlite3.Error:
        self.fts = False

def search(self, query: str, cwd: "Optional[str]" = None, k: int = 4) -> list:
    """Top-k past-session matches: (session_id, seq, cwd, summary, snippet).

    Excludes the current session. cwd filters to that directory; None = global.
    Uses FTS5/bm25 when available, else a LIKE scan. Never raises.
    """
    if not self.live or not query.strip():
        return []
    try:
        if self.fts:
            sql = ("SELECT m.session_id, m.seq, s.cwd, m.summary, "
                   "snippet(messages_fts, 0, '', '', '…', 12) "
                   "FROM messages_fts "
                   "JOIN messages m ON m.id = messages_fts.rowid "
                   "JOIN sessions s ON s.id = m.session_id "
                   "WHERE messages_fts MATCH ? AND m.session_id != ? "
                   "AND (? IS NULL OR s.cwd = ?) "
                   "ORDER BY bm25(messages_fts) LIMIT ?")
            rows = self.db.execute(
                sql, (query, self.session_id, cwd, cwd, k)).fetchall()
        else:
            like = f"%{query}%"
            sql = ("SELECT m.session_id, m.seq, s.cwd, m.summary, "
                   "substr(m.content, 1, 160) "
                   "FROM messages m JOIN sessions s ON s.id = m.session_id "
                   "WHERE m.no_index = 0 AND m.content LIKE ? "
                   "AND m.session_id != ? AND (? IS NULL OR s.cwd = ?) "
                   "LIMIT ?")
            rows = self.db.execute(
                sql, (like, self.session_id, cwd, cwd, k)).fetchall()
        return [tuple(r) for r in rows]
    except sqlite3.Error:
        return []

def has_prior_history(self) -> bool:
    if not self.live:
        return False
    try:
        row = self.db.execute(
            "SELECT 1 FROM messages WHERE session_id != ? LIMIT 1",
            (self.session_id,)).fetchone()
        return row is not None
    except sqlite3.Error:
        return False

def prior_sessions_for_cwd(self, cwd: str) -> int:
    if not self.live:
        return 0
    try:
        row = self.db.execute(
            "SELECT COUNT(*) FROM sessions WHERE cwd = ? AND id != ?",
            (cwd, self.session_id)).fetchone()
        return row[0] if row else 0
    except sqlite3.Error:
        return 0
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python test_session_store.py`
Expected: the six new tests print `ok` (or the fts-unavailable skip line); all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(recall): FTS5 message index, no_index flag, and SessionStore.search"
```

---

### Task 2: `format_recall()` — render hits, capped

**Files:**
- Modify: `coding_agent.py` — add module-level `format_recall` near `_cap_tool_result` (`:1109`).
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: hit tuples `(session_id, seq, cwd, summary, snippet)` from `search()`; `_cap_tool_result`.
- Produces: `format_recall(hits, budget_tokens=TOKEN_BUDGET) -> str`.

- [ ] **Step 1: Write failing test**

```python
def test_format_recall_renders_and_caps():
    assert "no matches" in ca.format_recall([]).lower()
    hits = [(7, 3, "/proj/a", "auth bug", "token expiry …")]
    out = ca.format_recall(hits)
    assert "sess 7" in out and "#3" in out and "auth bug" in out
    big = [(i, i, "/p", "s" * 500, "x" * 5000) for i in range(50)]
    assert len(ca.format_recall(big, budget_tokens=2000)) < 2000 * ca._CHARS_PER_TOKEN
    print("  recall output renders and stays capped          ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_session_store.py`
Expected: FAIL — `format_recall` not defined.

- [ ] **Step 3: Implement `format_recall`**

```python
def format_recall(hits: list, budget_tokens: int = TOKEN_BUDGET) -> str:
    """Render recall hits as a compact, capped block for the window."""
    if not hits:
        return "[recall] no matches."
    lines = ["[recall] past-session matches:"]
    for session_id, seq, cwd, summary, snippet in hits:
        label = (summary or "").strip() or (snippet or "").strip()[:80]
        lines.append(f"  [sess {session_id} #{seq}] {label} — \"…{snippet}…\"")
    return _cap_tool_result("\n".join(lines), budget_tokens)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_session_store.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(recall): format_recall renders capped hit blocks"
```

---

### Task 3: `/recall [--all] <query>` inline command

**Files:**
- Modify: `coding_agent.py` — add `do_recall` helper (module level, near `format_recall`); add an inline command branch in `run()` next to the other `/` branches (e.g. after the `/tokens` branch, ~`:1815`); add `/recall` to the command list at `:120-121`.
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: `SessionStore.search`, `format_recall`.
- Produces: `do_recall(store, cwd, query, all_scope=False, budget_tokens=TOKEN_BUDGET) -> str` — the testable core the command branch calls.

- [ ] **Step 1: Write failing test**

```python
def test_do_recall_respects_all_flag():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping do_recall test)    ok")
            return
        cur = store.session_id
        store.db.execute("INSERT INTO sessions (id, started, model, cwd) "
                         "VALUES (60,'t','m','/proj/a'),(61,'t','m','/proj/b')")
        _seed(store, [
            (60, 1, "user", "sqlite migration plan", "migration", False),
            (61, 1, "user", "sqlite migration plan", "migration", False),
        ])
        store.session_id = cur
        local = ca.do_recall(store, "/proj/a", "migration", all_scope=False)
        assert "sess 60" in local and "sess 61" not in local, "cwd scope"
        glob = ca.do_recall(store, "/proj/a", "migration", all_scope=True)
        assert "sess 61" in glob, "--all should reach other cwds"
        print("  do_recall honors cwd default and --all          ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_session_store.py`
Expected: FAIL — `do_recall` not defined.

- [ ] **Step 3: Implement `do_recall`**

```python
def do_recall(store, cwd: str, query: str, all_scope: bool = False,
              budget_tokens: int = TOKEN_BUDGET) -> str:
    scope = None if all_scope else cwd
    return format_recall(store.search(query, cwd=scope, k=4), budget_tokens)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_session_store.py`
Expected: PASS.

- [ ] **Step 5: Wire the inline command branch**

Add `/recall` to the command list near `coding_agent.py:120-121` (so tab/help
know it). Then add this branch in `run()` alongside the other `/` branches
(place it before the generic `/<tool>` dispatch at `:2045`):

```python
        if user.lower().startswith("/recall"):
            _ra = user.split(None, 1)
            _rest = _ra[1] if len(_ra) > 1 else ""
            _all = False
            if _rest.startswith("--all"):
                _all = True
                _rest = _rest[len("--all"):].strip()
            if not _rest:
                print("[Recall] usage: /recall [--all] <query>")
                continue
            _block = do_recall(store, str(_agent_cwd[0]), _rest,
                               all_scope=_all, budget_tokens=cfg["token_budget"])
            print(_block)
            remember("user", _block, summary=f"recall: {_rest[:60]}")
            continue
```

(`remember` injects it into the window so the model can use it; it is foldable
like any message. The recall block is small and safe to index.)

- [ ] **Step 6: Smoke-test the command path manually**

Run: `python -c "import coding_agent"` to confirm the module still imports
(no syntax error in `run()`).
Expected: no output, exit 0.

- [ ] **Step 7: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(recall): /recall [--all] command injects hits into the window"
```

---

### Task 4: `recall` model tool + advertisement gating

**Files:**
- Modify: `coding_agent.py` — add module-global `_active_store = [None]` near `_agent_cwd`; add `recall_tool`; register it in `TOOL_REGISTRY` (`:697`) under the name `recall`; ensure it is excluded from default `_active_tools` (`:731`); in `run()` set `_active_store[0] = store` and gate advertising after the store is created (before/at the first `build_prompt`).
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: `_active_store[0]` (a `SessionStore`), `_agent_cwd[0]`, `SessionStore.search`.
- Produces: `recall_tool(query, all=False) -> dict` with keys `matches` (list of
  `{"session": int, "seq": int, "cwd": str, "summary": str, "snippet": str}`)
  or `{"matches": []}`. Registered for dispatch as `recall`.

- [ ] **Step 1: Write failing test**

```python
def test_recall_tool_returns_matches_dict():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping recall_tool test)  ok")
            return
        cur = store.session_id
        store.db.execute("INSERT INTO sessions (id, started, model, cwd) "
                         "VALUES (70,'t','m','/proj/a')")
        _seed(store, [(70, 1, "user", "kafka consumer lag spike", "lag", False)])
        store.session_id = cur
        ca._active_store[0] = store
        ca._agent_cwd[0] = ca.Path("/proj/a")
        res = ca.recall_tool("kafka lag")
        assert isinstance(res, dict) and res.get("matches"), res
        assert res["matches"][0]["session"] == 70
        assert "recall" in ca.TOOL_REGISTRY, "recall must be dispatchable"
        assert "recall" not in ca._active_tools, "recall must be off by default"
        print("  recall_tool returns matches; off by default     ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_session_store.py`
Expected: FAIL — `recall_tool` / `_active_store` not defined.

- [ ] **Step 3: Add `_active_store` and `recall_tool`; register + de-advertise**

Near the `_agent_cwd` module global, add:

```python
_active_store = [None]   # set by run(); lets recall_tool reach the live store
```

Add the tool (module level, near the other `*_tool` functions). Keep the
docstring tight — it is advertised at ~40 tokens/turn once gated on:

```python
def recall_tool(query: str, all: bool = False) -> Dict[str, Any]:
    """Search past sessions for prior context. all=True searches every directory."""
    store = _active_store[0]
    if store is None:
        return {"matches": []}
    scope = None if all else str(_agent_cwd[0])
    hits = store.search(query, cwd=scope, k=4)
    return {"matches": [
        {"session": s, "seq": q, "cwd": c, "summary": summ, "snippet": snip}
        for (s, q, c, summ, snip) in hits]}
```

Register it for dispatch under the name `recall`. At `TOOL_REGISTRY` (`:697`),
add `"recall": recall_tool` to the merged dict. Then ensure the default
`_active_tools` (`:731`) does **not** include it — if the existing comprehension
would include it, add `- {"recall"}` / `discard("recall")` right after the set
is built:

```python
_active_tools.discard("recall")   # advertised only when prior history exists
```

- [ ] **Step 4: Gate advertising in `run()`**

In `run()`, after `store = SessionStore(...)` (`:1731`), add:

```python
    _active_store[0] = store
    if store.has_prior_history():
        _active_tools.add("recall")
    else:
        _active_tools.discard("recall")
```

The system prompt is built at `:1729` (`messages = [{"role": "system", ...}]`)
**before** the store exists. Move that `build_prompt()` call to **after** this
gating block, or rebuild it: set `messages[0] = {"role": "system", "content":
build_prompt()}` immediately after the gating block so the advertisement
reflects the gate on turn one.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python test_session_store.py`
Expected: PASS (or fts-skip line). All existing tests still pass.

- [ ] **Step 6: Import smoke test**

Run: `python -c "import coding_agent"`
Expected: exit 0, no error.

- [ ] **Step 7: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(recall): gated model-facing recall tool"
```

---

### Task 5: Session-start hint

**Files:**
- Modify: `coding_agent.py` — in `run()`, after the gating block from Task 4, print a one-line hint when prior sessions exist for this cwd.
- Test: `test_session_store.py` (covered by `test_history_flags` from Task 1; add one direct assertion on the count used by the hint).

**Interfaces:**
- Consumes: `SessionStore.prior_sessions_for_cwd`.
- Produces: console output only (no return value, no tokens).

- [ ] **Step 1: Add the hint**

After the Task 4 gating block in `run()`:

```python
    _n_here = store.prior_sessions_for_cwd(str(_agent_cwd[0]))
    if _n_here:
        print(f"[Recall] {_n_here} past session(s) here. "
              f"/recall <topic> to pull, or I'll check when unsure.")
```

- [ ] **Step 2: Verify the count method (already tested)**

`test_history_flags` (Task 1) already asserts `prior_sessions_for_cwd` returns
the right count. Re-run to confirm still green:

Run: `python test_session_store.py`
Expected: PASS.

- [ ] **Step 3: Import smoke test**

Run: `python -c "import coding_agent"`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add coding_agent.py
git commit -m "feat(recall): session-start hint when prior sessions exist for the cwd"
```

---

### Task 6: Set `no_index` at the `read_file` result call site

**Files:**
- Modify: `coding_agent.py` — `remember()` closure (`:1734`) gains a `no_index`
  param threaded to `store.add`; the tool-result `remember(...)` call in the
  dispatch loop (~`:2231`) passes `no_index=(name == "read_file")`.
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: `SessionStore.add(..., no_index=)` (Task 1), the dispatch loop's
  in-scope `name` variable.
- Produces: `read_file` tool-result messages stored with `no_index=1`, so
  `search()` never returns them.

- [ ] **Step 1: Write failing test**

```python
def test_read_file_results_are_not_indexed():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping read_file test)    ok")
            return
        cur = store.session_id
        store.session_id = 80
        # Simulate the dispatch loop storing a read_file page (no_index=True)
        store.add(1, "user", "tool_result({\"content\": \"def zzqmarker(): pass\"})",
                  summary="read_file: 200 lines", no_index=True)
        # ...and a normal discussion message that mentions the same token.
        store.add(2, "user", "we should refactor zzqmarker next", "note", no_index=False)
        store.session_id = cur
        hits = store.search("zzqmarker", cwd=None, k=4)
        assert hits, "the discussion message should be found"
        assert all(h[1] != 1 for h in hits), "read_file page must be excluded"
        print("  read_file results are not recalled              ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_session_store.py`
Expected: this test would already pass at the store layer (Task 1 honors
`no_index`), but it documents the contract the call site must satisfy. If it
passes here, proceed — the call-site change in Step 3 is what makes the running
agent actually pass `no_index=True`.

- [ ] **Step 3: Thread `no_index` through `remember` and set it at the call site**

In the `remember` closure (`coding_agent.py:1734`):

```python
    def remember(role: str, content: str, summary: str = "",
                 pinned: bool = False, no_index: bool = False) -> dict:
        seq[0] += 1
        m = {"role": role, "content": content, "seq": seq[0]}
        if summary:
            m["summary"] = summary
        if pinned:
            m["pinned"] = True
        messages.append(m)
        store.add(seq[0], role, content, summary, no_index=no_index)
        return m
```

In the dispatch loop where a tool result is stored (~`:2231`), pass the flag
(`name` is already in scope there):

```python
                remember("user",
                         _cap_tool_result(f"tool_result({json.dumps(result)})",
                                          cfg["token_budget"]),
                         summary=f"{name}: {_summarise_result(name, result)}",
                         no_index=(name == "read_file"))
```

- [ ] **Step 4: Run tests + import smoke test**

Run: `python test_session_store.py && python -c "import coding_agent"`
Expected: all green, import exits 0.

- [ ] **Step 5: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(recall): exclude read_file results from the recall index"
```

---

## Self-Review

**Spec coverage:**
- FTS5 keyword/bm25 + LIKE fallback → Task 1 (`search`, `self.fts`). ✓
- External-content FTS5 + triggers + backfill → Task 1 (`_ensure_fts`). ✓
- `no_index` column + trigger `WHEN` + honored in `search` → Task 1; set at the
  `read_file` call site → Task 6. ✓
- Same-cwd default, `--all`/`all=` global → Tasks 1, 3, 4. ✓
- Hits = summary+snippet, K=4, capped → Tasks 1, 2. ✓
- `/recall` command feeds window → Task 3. ✓
- Gated `recall` model tool → Task 4. ✓
- Session-start hint → Task 5. ✓
- Never-crash / degrade cleanly → try/except in every store method; `self.fts` fallback. ✓

**Placeholder scan:** none — every step has runnable code or an exact command.

**Type consistency:** `search()` returns `(session_id, seq, cwd, summary,
snippet)` everywhere (Tasks 1–4); `format_recall`/`do_recall` consume that
shape; `recall_tool` maps it to the documented dict. `no_index` is `int` in SQL,
`bool` at the `add()` boundary. Names (`_active_store`, `has_prior_history`,
`prior_sessions_for_cwd`, `do_recall`, `format_recall`, `recall_tool`) are used
identically across tasks.

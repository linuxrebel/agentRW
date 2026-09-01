# File Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `/ingest <path>` — a map-reduce file digest, cached by content hash, injected foldable so it rides the existing recall FTS index.

**Architecture:** Two new SQLite tables in the existing `sessions.db` (`file_digests`, `file_chunks`). A pure chunker splits a file into overlapping 200-line windows; a driver `ingest_file` maps each chunk through `call_llm`, reduces the summaries to one capped digest, and persists atomically keyed on `(path, sha256)`. A `/ingest` command handler in `run()` calls the driver and injects the digest via the existing `remember(no_index=False)` path, so it enters `messages_fts` and becomes recallable. No new dependency; native sqlite3 + the existing `call_llm` and `resolve_abs_path`.

**Tech Stack:** Python 3, sqlite3 (FTS5 already wired), Ollama via the existing `call_llm`. Tests: plain-assert `test_session_store.py`, offline tmp DB, stubbed `call_llm`.

**Spec:** `docs/superpowers/specs/2026-08-31-file-ingest-design.md`

## Global Constraints

- No new dependency or service. sqlite3 + existing `call_llm` + `resolve_abs_path` only.
- The store must never crash the agent: every `SessionStore` method wraps its DB work in `try/except sqlite3.Error` and returns a safe default, matching the existing methods.
- Store unavailable (`not self.live`) → methods no-op / return the empty default; ingest still summarizes and prints, it just cannot cache.
- `/ingest` is a human command only. No model-facing tool in v1 — do not touch `TOOL_REGISTRY`, `_active_tools`, or `CORE_TOOLS`.
- Digest hard-capped ~200 words at the reduce prompt, with `_cap_tool_result` as backstop.
- Chunk summaries and the file digest are real `call_llm` calls with `send_tools=False` (precedent: `coding_agent.py:1658`, `max_tokens=300`). The per-message `summary` field is deterministic `_summarise_result` — unrelated, do not reuse it.
- Failure mid-run aborts atomically: store nothing partial, return an error dict.

---

### Task 1: Schema + digest storage methods

**Files:**
- Modify: `coding_agent.py` — `SCHEMA` string (starts `coding_agent.py:957`); add methods to `class SessionStore` (near `counts`, `coding_agent.py:1206`)
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: existing `SessionStore` (`self.db`, `self.live`, `self.session_id`), `import hashlib` (add if absent — check top of file).
- Produces:
  - `SessionStore.find_digest(path: str, content_hash: str) -> Optional[dict]` — returns `{"id","path","cwd","content_hash","lines","n_chunks","digest","model","created"}` or `None`.
  - `SessionStore.save_digest(path: str, cwd: str, content_hash: str, lines: int, n_chunks: int, digest: str, model: str, chunk_summaries: list) -> Optional[int]` — inserts one `file_digests` row + one `file_chunks` row per `chunk_summaries` item in a single transaction; returns the new digest id, or `None` if the store is dead. `chunk_summaries` is a list of `(chunk_no, start_line, end_line, summary)`.
  - `SessionStore.chunks_for(path: str) -> list` — latest digest's chunks as `[(chunk_no, start_line, end_line, summary), ...]`, newest digest for that path; `[]` if none.

- [ ] **Step 1: Write the failing test**

Add to `test_session_store.py`, after the recall tests:

```python
def test_digest_roundtrip_and_cache():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="m", cwd="/proj/a")
        did = store.save_digest(
            "/proj/a/big.py", "/proj/a", "hash1", lines=350, n_chunks=2,
            digest="does X and Y", model="m",
            chunk_summaries=[(0, 1, 200, "part one"), (1, 181, 350, "part two")])
        assert did, "save_digest should return an id"
        hit = store.find_digest("/proj/a/big.py", "hash1")
        assert hit and hit["digest"] == "does X and Y"
        assert hit["n_chunks"] == 2
        assert store.find_digest("/proj/a/big.py", "hash2") is None, "changed hash = miss"
        chunks = store.chunks_for("/proj/a/big.py")
        assert [c[0] for c in chunks] == [0, 1]
        assert chunks[1][3] == "part two"
        print("  digest round-trip + hash cache                 ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -c "import test_session_store as t; t.test_digest_roundtrip_and_cache()"`
Expected: FAIL — `AttributeError: 'SessionStore' object has no attribute 'save_digest'`.

- [ ] **Step 3: Add the schema tables**

In the `SCHEMA` string (`coding_agent.py:957`), before the closing `"""`, append:

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

- [ ] **Step 4: Add the three methods**

Confirm `import hashlib` is present at the top of `coding_agent.py`; if not, add it (needed in Task 3, harmless here). Add to `class SessionStore`, right after `counts` (`coding_agent.py:1206-1216`):

```python
    def find_digest(self, path: str, content_hash: str) -> "Optional[dict]":
        """Cached digest for this exact file content, or None."""
        if not self.live:
            return None
        try:
            row = self.db.execute(
                "SELECT id, path, cwd, content_hash, lines, n_chunks, digest, "
                "model, created FROM file_digests "
                "WHERE path=? AND content_hash=? ORDER BY id DESC LIMIT 1",
                (path, content_hash)).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        keys = ("id", "path", "cwd", "content_hash", "lines", "n_chunks",
                "digest", "model", "created")
        return dict(zip(keys, row))

    def save_digest(self, path: str, cwd: str, content_hash: str, lines: int,
                    n_chunks: int, digest: str, model: str,
                    chunk_summaries: list) -> "Optional[int]":
        """Insert one file_digests row + its file_chunks in one transaction.

        Atomic: a failure part-way rolls back, so a cache lookup never finds a
        digest whose chunks are missing.
        """
        if not self.live:
            return None
        try:
            cur = self.db.execute(
                "INSERT INTO file_digests "
                "(path, cwd, content_hash, lines, n_chunks, digest, model, created) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (path, cwd, content_hash, lines, n_chunks, digest, model,
                 datetime.now().isoformat(timespec="seconds")))
            digest_id = cur.lastrowid
            self.db.executemany(
                "INSERT INTO file_chunks "
                "(digest_id, chunk_no, start_line, end_line, summary) "
                "VALUES (?,?,?,?,?)",
                [(digest_id, cn, sl, el, s) for (cn, sl, el, s) in chunk_summaries])
            self.db.commit()
            return digest_id
        except sqlite3.Error:
            self.db.rollback()
            return None

    def chunks_for(self, path: str) -> list:
        """Chunks of the newest digest for path: (chunk_no, start, end, summary)."""
        if not self.live:
            return []
        try:
            row = self.db.execute(
                "SELECT id FROM file_digests WHERE path=? ORDER BY id DESC LIMIT 1",
                (path,)).fetchone()
            if not row:
                return []
            return list(self.db.execute(
                "SELECT chunk_no, start_line, end_line, summary FROM file_chunks "
                "WHERE digest_id=? ORDER BY chunk_no", (row[0],)))
        except sqlite3.Error:
            return []
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -c "import test_session_store as t; t.test_digest_roundtrip_and_cache()"`
Expected: PASS — prints `digest round-trip + hash cache                 ok`.

- [ ] **Step 6: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(ingest): file_digests/file_chunks tables + store methods"
```

---

### Task 2: The chunker

**Files:**
- Modify: `coding_agent.py` — add a module-level function near the other file helpers (after `read_file_tool`, `coding_agent.py:266`)
- Test: `test_session_store.py`

**Interfaces:**
- Produces: `_chunk_lines(lines: list, size: int = 200, overlap: int = 20) -> list` — returns `[(start_line, end_line, text), ...]` where `start_line`/`end_line` are 1-based inclusive line numbers and `text` is the joined slice. A file of `size` lines or fewer yields exactly one chunk. Windows advance by `size - overlap`.

- [ ] **Step 1: Write the failing test**

```python
def test_chunker_windows_and_overlap():
    lines = [f"line {i}\n" for i in range(1, 351)]   # 350 lines
    chunks = ca._chunk_lines(lines, size=200, overlap=20)
    # windows advance by 180: [1..200], [181..350]
    assert len(chunks) == 2, f"expected 2 chunks, got {len(chunks)}"
    assert chunks[0][0] == 1 and chunks[0][1] == 200
    assert chunks[1][0] == 181 and chunks[1][1] == 350
    assert chunks[0][2].startswith("line 1\n")
    # small file -> exactly one chunk, no reduce needed downstream
    small = ca._chunk_lines([f"l{i}\n" for i in range(50)], size=200, overlap=20)
    assert len(small) == 1 and small[0][0] == 1 and small[0][1] == 50
    print("  chunker windows + overlap + single-chunk       ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -c "import test_session_store as t; t.test_chunker_windows_and_overlap()"`
Expected: FAIL — `AttributeError: module 'ca' has no attribute '_chunk_lines'`.

- [ ] **Step 3: Write the implementation**

Add after `read_file_tool` (`coding_agent.py:266`):

```python
def _chunk_lines(lines: list, size: int = 200, overlap: int = 20) -> list:
    """Split lines into overlapping windows: [(start_line, end_line, text), ...].

    Line numbers are 1-based inclusive. Windows advance by size - overlap so a
    function split across a boundary survives in the next chunk. A file of
    `size` lines or fewer is one chunk (the caller then skips the reduce step).
    """
    total = len(lines)
    if total <= size:
        return [(1, total, "".join(lines))] if total else [(1, 0, "")]
    step = max(size - overlap, 1)
    out = []
    start = 0
    while start < total:
        end = min(start + size, total)
        out.append((start + 1, end, "".join(lines[start:end])))
        if end == total:
            break
        start += step
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -c "import test_session_store as t; t.test_chunker_windows_and_overlap()"`
Expected: PASS — prints `chunker windows + overlap + single-chunk       ok`.

- [ ] **Step 5: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(ingest): overlapping line chunker"
```

---

### Task 3: The ingest driver

**Files:**
- Modify: `coding_agent.py` — add prompt constants + `ingest_file` near the other `call_llm` users (after `_propose_or_compute`, `coding_agent.py:1727`)
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: `_chunk_lines` (Task 2), `SessionStore.find_digest`/`save_digest` (Task 1), module-level `call_llm` (`coding_agent.py:1403`), `resolve_abs_path` (`coding_agent.py:156`), `_cap_tool_result` (`coding_agent.py:1232`), `hashlib`.
- Produces: `ingest_file(path: str, model: str, store, layers_ref: list, cfg: dict) -> dict`.
  - Success: `{"path", "lines", "n_chunks", "digest", "cached": bool}`.
  - Error: `{"error": <slug>, "path": path, "hint": <str>}` — same shape as `_fs_error`.
  - Calls `call_llm` by module global name so tests can stub `ca.call_llm`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ingest_file_maps_reduces_and_caches():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "big.py"
        f.write_text("".join(f"line {i}\n" for i in range(1, 351)))  # 2 chunks
        store = ca.SessionStore(Path(d) / "s.db", model="m", cwd=d)
        calls = {"n": 0}
        def fake_llm(model, messages, **kw):
            calls["n"] += 1
            return f"summary {calls['n']}"
        orig = ca.call_llm
        ca.call_llm = fake_llm
        try:
            res = ca.ingest_file(str(f), "m", store, [None], {"num_ctx": 2048, "token_budget": 2048})
            assert res.get("digest"), res
            assert res["n_chunks"] == 2 and res["cached"] is False
            assert calls["n"] == 3, "2 map calls + 1 reduce"
            row = store.find_digest(str(f), ca._file_hash(str(f)))
            assert row and row["digest"] == res["digest"]
            assert len(store.chunks_for(str(f))) == 2
            # second run hits cache: no new call_llm
            before = calls["n"]
            res2 = ca.ingest_file(str(f), "m", store, [None], {"num_ctx": 2048, "token_budget": 2048})
            assert res2["cached"] is True and calls["n"] == before
        finally:
            ca.call_llm = orig
        print("  ingest maps, reduces, caches                   ok")


def test_ingest_aborts_atomically_on_llm_failure():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "big.py"
        f.write_text("".join(f"line {i}\n" for i in range(1, 351)))
        store = ca.SessionStore(Path(d) / "s.db", model="m", cwd=d)
        def boom(*a, **k):
            raise RuntimeError("ollama down")
        orig = ca.call_llm
        ca.call_llm = boom
        try:
            res = ca.ingest_file(str(f), "m", store, [None], {"num_ctx": 2048, "token_budget": 2048})
            assert "error" in res, "should return an error dict"
        finally:
            ca.call_llm = orig
        assert store.find_digest(str(f), ca._file_hash(str(f))) is None, "nothing persisted"
        print("  ingest aborts atomically on llm failure        ok")


def test_ingest_refuses_missing_file():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="m", cwd=d)
        res = ca.ingest_file(str(Path(d) / "nope.py"), "m", store, [None],
                             {"num_ctx": 2048, "token_budget": 2048})
        assert res.get("error"), res
        assert "hint" in res
        print("  ingest refuses a missing file                  ok")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -c "import test_session_store as t; t.test_ingest_refuses_missing_file()"`
Expected: FAIL — `AttributeError: module 'ca' has no attribute 'ingest_file'`.

- [ ] **Step 3: Write the implementation**

Add after `_propose_or_compute` (`coding_agent.py:1727`):

```python
INGEST_MAP_PROMPT = (
    "You summarize a slice of a source file for a durable index. In 2-3 "
    "sentences, say what this slice defines and does. Name key functions, "
    "classes, and side effects. No preamble, no code fences.")

INGEST_REDUCE_PROMPT = (
    "You are given per-slice summaries of one file, in order. Write a single "
    "digest of the whole file in at most 200 words: its purpose, its main "
    "components, and how they fit. No preamble, no code fences.")


def _file_hash(path: str) -> str:
    """sha256 of the file's bytes. Content-addressed staleness."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(path: str, model: str, store, layers_ref: list, cfg: dict) -> dict:
    """Map-reduce a file into one cached, capped digest. Never raises."""
    abspath = resolve_abs_path(path)
    try:
        with open(abspath, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as e:
        return _fs_error(e, abspath)
    except UnicodeDecodeError as e:
        return _fs_error(e, abspath)

    key = str(abspath)
    content_hash = _file_hash(key)
    cached = store.find_digest(key, content_hash)
    if cached:
        return {"path": key, "lines": cached["lines"],
                "n_chunks": cached["n_chunks"], "digest": cached["digest"],
                "cached": True}

    chunks = _chunk_lines(lines, size=200, overlap=20)
    num_ctx = cfg.get("num_ctx")
    token_budget = cfg.get("token_budget", TOKEN_BUDGET)
    chunk_summaries = []
    try:
        for i, (sl, el, text) in enumerate(chunks):
            print(f"[Ingest] summarizing chunk {i + 1}/{len(chunks)}…")
            msgs = [{"role": "system", "content": INGEST_MAP_PROMPT},
                    {"role": "user", "content": f"Lines {sl}-{el}:\n{text}"}]
            summary = call_llm(model, msgs, gpu_layers=layers_ref, max_tokens=150,
                               num_ctx=num_ctx, token_budget=token_budget,
                               send_tools=False) or ""
            chunk_summaries.append((i, sl, el, summary.strip()))

        if len(chunk_summaries) == 1:
            digest = chunk_summaries[0][3]
        else:
            joined = "\n".join(f"[{sl}-{el}] {s}" for (_, sl, el, s) in chunk_summaries)
            print("[Ingest] reducing to file digest…")
            msgs = [{"role": "system", "content": INGEST_REDUCE_PROMPT},
                    {"role": "user", "content": joined}]
            digest = (call_llm(model, msgs, gpu_layers=layers_ref, max_tokens=300,
                               num_ctx=num_ctx, token_budget=token_budget,
                               send_tools=False) or "").strip()
    except Exception as e:  # any call_llm failure aborts the whole run
        return {"error": "ingest_failed", "path": key, "hint": str(e)}

    digest = _cap_tool_result(digest, token_budget)
    store.save_digest(key, str(_agent_cwd[0]), content_hash, len(lines),
                      len(chunks), digest, model, chunk_summaries)
    return {"path": key, "lines": len(lines), "n_chunks": len(chunks),
            "digest": digest, "cached": False}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -c "import test_session_store as t; t.test_ingest_refuses_missing_file(); t.test_ingest_file_maps_reduces_and_caches(); t.test_ingest_aborts_atomically_on_llm_failure()"`
Expected: PASS — three `ok` lines.

- [ ] **Step 5: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(ingest): map-reduce driver with sha256 cache + atomic abort"
```

---

### Task 4: `/ingest` command + recall tie-in

**Files:**
- Modify: `coding_agent.py` — add `/ingest` to the command-name list (`coding_agent.py:120`); add the handler in `run()` next to the `/recall` handler (`coding_agent.py:1969-1983`)
- Test: `test_session_store.py`

**Interfaces:**
- Consumes: `ingest_file` (Task 3), `remember` (`coding_agent.py:1888`), `store`, `model`, `layers_ref`, `cfg`, `_agent_cwd` — all in `run()` scope.
- Produces: a `/ingest <path>` command. On success it prints the digest and injects one message via `remember(..., no_index=False)` so it enters `messages_fts` and is recallable now and next session. On error it prints the hint and injects nothing.

- [ ] **Step 1: Write the failing test**

This asserts the recall tie-in at the store level (the property `/ingest` relies on): a digest added as a normal message is found by `search()` from a later session, and a raw file page is not. It does not spin up the `run()` loop.

```python
def test_ingested_digest_is_recallable():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="m", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping recall tie test)  ok")
            return
        cur = store.session_id
        store.session_id = 90
        # what /ingest injects (foldable, indexed):
        store.add(1, "user", "[ingest] parser.py — tokenizes and builds the AST",
                  summary="ingest: parser.py", no_index=False)
        # a raw page dump must never be recalled:
        store.add(2, "user", 'tool_result({"content":"astzzq internals"})',
                  summary="read_file", no_index=True)
        store.session_id = cur
        hits = store.search("AST", cwd=None, k=4)
        assert hits, "digest should be recallable"
        assert all(h[1] != 2 for h in hits), "raw page must not be recalled"
        print("  ingested digest is recallable                  ok")
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python -c "import test_session_store as t; t.test_ingested_digest_is_recallable()"`
Expected: PASS already (it exercises existing `search`/`no_index`). This test locks the invariant `/ingest` depends on; keep it. Proceed to wire the command.

- [ ] **Step 3: Register the command name**

In the command-name list at `coding_agent.py:120`, add `"/ingest"` (so it is recognized as a slash command, consistent with `/recall`):

```python
    "/help", "/model", "/gpu-layers", "/low-vram", "/compact", "/tokens", "/recall", "/ingest",
```

- [ ] **Step 4: Add the handler in `run()`**

Immediately after the `/recall` handler block (ends `coding_agent.py:1983` with `continue`), add:

```python
        if user.lower().startswith("/ingest"):
            _ia = user.split(None, 1)
            _ipath = _ia[1].strip() if len(_ia) > 1 else ""
            if not _ipath:
                print("[Ingest] usage: /ingest <path>")
                continue
            _res = ingest_file(_ipath, model, store, layers_ref, cfg)
            if _res.get("error"):
                print(f"[Ingest] {_res.get('hint', _res['error'])} "
                      f"(use read_file to inspect it directly)")
                continue
            _tag = " (cached)" if _res.get("cached") else ""
            _hdr = (f"[ingest] {_res['path']} · {_res['lines']} lines · "
                    f"digest{_tag}")
            print(_hdr)
            print(_res["digest"])
            remember("user", f"{_hdr}\n{_res['digest']}",
                     summary=f"ingest: {Path(_res['path']).name}", no_index=False)
            continue
```

- [ ] **Step 5: Run the full suite**

Run: `python test_session_store.py`
Expected: ends with `all session-store tests passed` (add the four new tests to the `__main__` block first — Step 6).

- [ ] **Step 6: Register the new tests in `__main__`**

In the `if __name__ == "__main__":` block at the end of `test_session_store.py`, add before the final `print`:

```python
    test_digest_roundtrip_and_cache()
    test_chunker_windows_and_overlap()
    test_ingest_file_maps_reduces_and_caches()
    test_ingest_aborts_atomically_on_llm_failure()
    test_ingest_refuses_missing_file()
    test_ingested_digest_is_recallable()
```

- [ ] **Step 7: Run the full suite again**

Run: `python test_session_store.py`
Expected: `all session-store tests passed`.

- [ ] **Step 8: Smoke-check the command help + import**

Run: `python -c "import coding_agent"`
Expected: no exception (module imports; new function/constants parse).

- [ ] **Step 9: Commit**

```bash
git add coding_agent.py test_session_store.py
git commit -m "feat(ingest): /ingest command injects recallable digest"
```

---

## Self-Review

**Spec coverage:**
- Trigger `/ingest <path>` human command only → Task 4 (no TOOL_REGISTRY change). ✓
- 200-line / 20-overlap chunking reusing the pager concept → Task 2 `_chunk_lines`. ✓
- Map-reduce, per-chunk kept, single-chunk skips reduce → Task 3. ✓
- Two tables + sha256 staleness, reuse on match → Task 1 + Task 3 cache path. ✓
- Foldable digest into window + console, atomic on failure → Task 4 handler (`remember`, no_index=False) + Task 3 try/except abort. ✓
- Recall drill-down via `chunks_for` → Task 1 method (thin; UI wiring left as follow-on per spec §5). ✓
- Error handling: missing/binary/unreadable → `_fs_error`; llm fails → abort atomic; store dead → methods no-op; oversized → `_cap_tool_result`. ✓
- Testing bullets → Tasks 1-4 tests cover round-trip, cache miss, chunk count/overlap/single, map-reduce persist, atomic abort, recallable-vs-raw. ✓

**Placeholder scan:** none — every step has runnable code or an exact command.

**Type consistency:** `save_digest` `chunk_summaries` is `(chunk_no, start_line, end_line, summary)` in Task 1 and produced in that shape in Task 3; `_chunk_lines` returns `(start_line, end_line, text)` consumed positionally in Task 3; `_file_hash`/`ingest_file`/`find_digest` signatures match across tests and impl.

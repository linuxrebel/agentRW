#!/usr/bin/env python3
"""Folding must never lose anything, and must never lose the goal.

The failure this guards against: the window filled, the harness deleted the
oldest messages to make room, and the oldest message was the user's task. The
agent then kept working with a directory listing and no idea what it was for.
"""
import importlib.util
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ca", Path(__file__).parent / "coding_agent.py")
ca = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ca)
except SystemExit:
    pass


def test_fold_keeps_goal_and_loses_nothing():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd=d)
        assert store.live, "store should open"

        goal = "refactor the parser in foo.py"
        listing = "X" * 8000
        rows = [
            ("user", goal, "goal: " + goal, True),
            ("assistant", "A" * 400, "tool call", False),
            ("user", listing, "list_files: 118 entries in /home/james", False),
            ("assistant", "B" * 400, "", False),
            ("user", "Y" * 4000, "read_file: 400 lines", False),
            ("assistant", "C" * 400, "", False),
            ("user", "now run the tests", "", False),
            ("assistant", "D" * 400, "", False),
        ]
        messages = [{"role": "system", "content": "S" * 2000}]
        for i, (role, content, summary, pinned) in enumerate(rows, start=1):
            m = {"role": role, "content": content, "seq": i}
            if summary:
                m["summary"] = summary
            if pinned:
                m["pinned"] = True
            messages.append(m)
            store.add(i, role, content, summary)

        before = sum(len(m["content"]) for m in messages if m["role"] != "system")
        folded = ca.proactive_trim(messages, budget_tokens=2000, store=store)
        after = sum(len(m["content"]) for m in messages if m["role"] != "system")

        assert folded > 0, "should have folded something"
        assert after < before, "folding should free space"
        assert after <= 2000 * ca._CHARS_PER_TOKEN, f"still over budget: {after}"

        # The goal survives verbatim. This is the whole point.
        goals = [m for m in messages if m.get("pinned")]
        assert len(goals) == 1 and goals[0]["content"] == goal, "goal was not preserved"

        # The oversized listing is gone from the window...
        window = "".join(m["content"] for m in messages)
        assert listing not in window, "big listing should be folded out"
        # ...and its summary is what replaced it, so the model still knows it happened.
        assert "118 entries" in window, "folded message should leave its summary"

        # ...but it is still on disk, in full. Folded != lost.
        dump = store.export_markdown()
        assert listing in dump, "folded content must still be recoverable"
        assert goal in dump
        assert "folded out of context" in dump, "export should mark what was folded"

        stored, folded_n = store.counts()
        assert stored == len(rows), f"expected {len(rows)} stored, got {stored}"
        assert folded_n == folded, f"db folded={folded_n}, memory folded={folded}"
    print("  fold keeps the goal, loses nothing            ok")


def test_a_broken_store_does_not_stop_the_agent():
    # Losing history is a worse session. It is not a dead one.
    store = ca.SessionStore(Path("/proc/nonexistent/nope.db"))
    assert not store.live
    store.add(1, "user", "hi")          # must not raise
    store.set_state("goal", "hi")
    store.mark_folded(1)
    assert store.counts() == (0, 0)
    print("  unusable store degrades quietly               ok")


def test_oversized_single_result_is_capped():
    capped = ca._cap_tool_result("tool_result(" + "x" * 40000 + ")", 2000)
    assert len(capped) < 5000
    assert "truncated" in capped, "must say it was cut, not silently lie"
    print("  oversized tool result capped, and says so     ok")


def test_arguments_are_parsed_as_tolerantly_as_calls():
    """The harness salvages malformed tool calls. It has to do the same for
    the values inside them, or the tolerance stops at the envelope."""
    here = Path(__file__).parent
    plain = ca.read_file_tool(str(here / "coding_agent.py"), max_lines=1)
    assert "error" not in plain, plain

    for wrapped in ('"{}"', "'{}'", " {} "):
        arg = wrapped.format(here / "coding_agent.py")
        got = ca.read_file_tool(arg, max_lines=1)
        assert "error" not in got, f"{arg!r} should resolve, got {got}"
        assert got["total_lines"] == plain["total_lines"]
    print("  quoted and padded paths resolve               ok")


def test_a_filter_matching_nothing_offers_a_way_out():
    here = str(Path(__file__).parent)
    # Junk that is syntactically a pattern but matches nothing.
    got = ca.list_files_tool(here, "(<!DOCTYPE html>)|(html)")
    assert got["names"] == []
    assert got["entries_without_pattern"] > 0, "must say what is actually there"
    assert "hint" in got, "a dead end is a harness failure, not an answer"
    # And a pattern Path.match rejects outright must not end the turn.
    assert "error" not in ca.list_files_tool(here, "[")
    print("  empty filter result stays actionable          ok")


def test_errors_are_slugs_not_prose():
    """Every error is a fixed identifier a caller can branch on. str(e) puts
    prose in that slot — error="[Errno 21] Is a directory" is not a code."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        binary = Path(d) / "b.bin"
        binary.write_bytes(bytes(range(256)))
        me = str(Path(__file__).parent / "coding_agent.py")

        checks = [
            (ca.read_file_tool(d), "not_a_file"),
            (ca.read_file_tool(str(binary)), "not_text"),
            (ca.read_file_tool(me, start_line=10 ** 7), "start_line_past_end"),
            (ca.search_file_tool(d, "x"), "not_a_file"),
            (ca.list_files_tool(me), "not_a_directory"),
            (ca.list_files_tool(str(Path(d) / "nope")), "directory_not_found"),
        ]
        for got, want in checks:
            assert got.get("error") == want, f"expected {want}, got {got}"
            assert got.get("hint"), f"{want} gives the caller nowhere to go"
    print("  filesystem errors are stable slugs           ok")


def test_search_reports_what_it_cut():
    me = str(Path(__file__).parent / "coding_agent.py")
    got = ca.search_file_tool(me, "e")
    assert got["found"] > ca.MAX_SEARCH_MATCHES, "need a file with many hits"
    assert len(got["matches"]) == ca.MAX_SEARCH_MATCHES
    assert got["not_shown"] == got["found"] - ca.MAX_SEARCH_MATCHES
    empty = ca.search_file_tool(me, "zzz-not-present-zzz")
    assert empty["found"] == 0 and empty.get("hint")
    print("  truncated search says how much it cut        ok")


# ---------------------------------------------------------------------------
# Cross-session recall (FTS5)
# ---------------------------------------------------------------------------
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


def test_format_recall_renders_and_caps():
    assert "no matches" in ca.format_recall([]).lower()
    hits = [(7, 3, "/proj/a", "auth bug", "token expiry …")]
    out = ca.format_recall(hits)
    assert "sess 7" in out and "#3" in out and "auth bug" in out
    big = [(i, i, "/p", "s" * 500, "x" * 5000) for i in range(50)]
    assert len(ca.format_recall(big, budget_tokens=2000)) < 2000 * ca._CHARS_PER_TOKEN
    print("  recall output renders and stays capped          ok")


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
        ca._agent_cwd[0] = Path("/proj/a")
        res = ca.recall_tool("kafka lag")
        assert isinstance(res, dict) and res.get("matches"), res
        assert res["matches"][0]["session"] == 70
        assert "recall" in ca.TOOL_REGISTRY, "recall must be dispatchable"
        assert "recall" not in ca._active_tools, "recall must be off by default"
        ca._active_store[0] = None
        print("  recall_tool returns matches; off by default     ok")


def test_read_file_results_are_not_indexed():
    with tempfile.TemporaryDirectory() as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd="/proj/a")
        if not store.fts:
            print("  (fts5 unavailable — skipping read_file test)    ok")
            return
        cur = store.session_id
        store.session_id = 80
        store.add(1, "user", 'tool_result({"content": "def zzqmarker(): pass"})',
                  summary="read_file: 200 lines", no_index=True)
        store.add(2, "user", "we should refactor zzqmarker next", "note", no_index=False)
        store.session_id = cur
        hits = store.search("zzqmarker", cwd=None, k=4)
        assert hits, "the discussion message should be found"
        assert all(h[1] != 1 for h in hits), "read_file page must be excluded"
        print("  read_file results are not recalled              ok")


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


def test_ingest_tool_is_advertised_and_bridged():
    assert getattr(ca.ingest_tool, "model_facing", False) is True, "should be advertised"
    assert ca.TOOL_REGISTRY.get("ingest") is ca.ingest_tool, "must be registered"
    # no active session -> graceful refuse, not a crash
    ca._active_ingest[0] = None
    r = ca.ingest_tool("whatever.py")
    assert r.get("error") == "ingest_unavailable", r
    # bridged: the tool returns whatever the live closure produces
    ca._active_ingest[0] = lambda p: {"path": p, "digest": "D", "cached": False}
    try:
        assert ca.ingest_tool("x.py")["digest"] == "D"
    finally:
        ca._active_ingest[0] = None
    print("  ingest_tool advertised + bridged               ok")


def test_ingest_max_chunks_guard_trips_before_model():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "huge.py"
        f.write_text("".join(f"line {i}\n" for i in range(3200)))  # >15 chunks
        store = ca.SessionStore(Path(d) / "s.db", model="m", cwd=d)
        def boom(*a, **k):
            raise AssertionError("model must not be called when over the cap")
        orig = ca.call_llm
        ca.call_llm = boom
        try:
            res = ca.ingest_file(str(f), "m", store, [None],
                                 {"num_ctx": 2048, "token_budget": 2048}, max_chunks=15)
            assert res.get("error") == "too_large", res
        finally:
            ca.call_llm = orig
        assert store.find_digest(str(f), ca._file_hash(str(f))) is None, "nothing persisted"
        print("  ingest max_chunks guard trips before model     ok")


def test_no_think_sets_reasoning_effort():
    import types
    captured = {}
    def fake_create(**kw):
        captured.update(kw)
        msg = types.SimpleNamespace(tool_calls=None, content="ok", reasoning=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    orig = ca.client.chat.completions.create
    ca.client.chat.completions.create = fake_create
    try:
        ca.call_llm("m", [{"role": "user", "content": "hi"}], send_tools=False, no_think=True)
        assert captured.get("reasoning_effort") == "none", "no_think must disable reasoning"
        captured.clear()
        ca.call_llm("m", [{"role": "user", "content": "hi"}], send_tools=False)
        assert "reasoning_effort" not in captured, "default must not set reasoning_effort"
    finally:
        ca.client.chat.completions.create = orig
    print("  no_think passes reasoning_effort=none           ok")


def test_reply_text_falls_back_to_reasoning():
    import types
    # content empty (reasoning model hit the length limit before answering)
    msg = types.SimpleNamespace(tool_calls=None, content="", reasoning="I think X")
    assert ca._reply_text(msg) == "I think X", "empty content should fall back to reasoning"
    # normal content wins over reasoning
    msg2 = types.SimpleNamespace(tool_calls=None, content="real answer", reasoning="scratch")
    assert ca._reply_text(msg2) == "real answer"
    # nothing anywhere -> empty string, not a crash
    msg3 = types.SimpleNamespace(tool_calls=None, content="", reasoning=None)
    assert ca._reply_text(msg3) == ""
    print("  reply_text falls back to reasoning on blank     ok")


def test_ingest_guards_empty_digest():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "big.py"
        f.write_text("".join(f"line {i}\n" for i in range(1, 351)))
        store = ca.SessionStore(Path(d) / "s.db", model="m", cwd=d)
        orig = ca.call_llm
        ca.call_llm = lambda *a, **k: ""      # model returns nothing usable
        try:
            res = ca.ingest_file(str(f), "m", store, [None],
                                 {"num_ctx": 2048, "token_budget": 2048})
            assert res.get("error") == "empty_digest", res
        finally:
            ca.call_llm = orig
        assert store.find_digest(str(f), ca._file_hash(str(f))) is None, "blank must not persist"
        print("  ingest refuses to persist an empty digest      ok")


def test_ctx_has_ask_and_api_2():
    ctx = ca.plugin_context("m", {"num_ctx": 2048, "token_budget": 2000}, [None])
    assert ctx.api == 2
    assert callable(ctx.ask)
    print("  ctx exposes ask, api is 2                       ok")


def test_ctx_ask_calls_call_llm():
    seen = {}
    orig = ca.call_llm
    try:
        def fake(model, messages, **kw):
            seen["model"] = model
            seen["kw"] = kw
            return "ok"
        ca.call_llm = fake
        ctx = ca.plugin_context("mymodel",
                                {"num_ctx": 2048, "token_budget": 2000}, [7])
        out = ctx.ask([{"role": "user", "content": "hi"}], max_tokens=50)
    finally:
        ca.call_llm = orig
    assert out == "ok"
    assert seen["model"] == "mymodel"
    assert seen["kw"]["send_tools"] is False
    assert seen["kw"]["max_tokens"] == 50
    print("  ctx.ask routes through call_llm                 ok")


def test_ctx_no_lint_pipeline_names():
    ctx = ca.plugin_context("m", {"num_ctx": 2048, "token_budget": 2000}, [None])
    for gone in ("gather_findings", "propose_fix", "apply_fix",
                 "finish_run", "defer", "debt_file"):
        assert not hasattr(ctx, gone), f"{gone} should be gone from ctx"
    print("  ctx no longer carries the lint pipeline names   ok")


def test_core_pipeline_functions_removed():
    for gone in ("_gather_findings", "_propose_fix", "_propose_or_compute",
                 "_apply_fix", "_apply_insert", "_apply_checked", "_finish_run",
                 "_clean_proposal", "_defer", "_parses",
                 "FIX_PROMPT", "INSERT_PROMPT", "DEBT_FILE"):
        assert not hasattr(ca, gone), f"{gone} should be removed from core"
    print("  core no longer defines the lint pipeline        ok")


def test_parse_use_request():
    assert ca.parse_use_request("USE: read_file") == ["read_file"]
    assert ca.parse_use_request("USE: read_file write_file") == ["read_file", "write_file"]
    assert ca.parse_use_request("sure, here is an answer") == []
    assert ca.parse_use_request("USE: not_a_tool") == []   # filtered to registry
    # weak models abbreviate the tool name and tack the path on the USE line
    assert ca.parse_use_request("USE: read /home/x.py") == ["read_file"]
    assert ca.parse_use_request("USE: write") == ["write_file"]
    assert ca.parse_use_request("USE: re") == []           # read_file+recall -> ambiguous, dropped
    # and they echo the instruction's own markdown back at it
    assert ca.parse_use_request("USE: `write_file`") == ["write_file"]
    # ...including around the sentinel itself, which the instruction backticks
    assert ca.parse_use_request('`USE: read_file "/x"`') == ["read_file"]
    assert ca.parse_use_request("**USE:** read_file") == ["read_file"]
    assert ca.parse_use_request("USE: **read_file**, `write_file`") == ["read_file", "write_file"]
    print("  parse_use_request extracts + filters USE line   ok")


def test_pull_prompt_lists_names_and_use_protocol():
    p = ca.build_prompt(pull=True)
    assert "USE:" in p
    assert "read_file" in p and "write_file" in p
    # names+one-liners only: no full signature parens block for tools
    assert "(filename" not in p
    # smaller than the old push prompt
    assert len(p) < len(ca.build_prompt(pull=False))
    print("  pull prompt: names+USE, no schemas, smaller       ok")


def test_two_pass_pull_then_call():
    # pass1 returns USE, pass2 returns a real call; assert both happen in order.
    calls = []
    orig = ca.call_llm
    active = set(ca._active_tools)
    try:
        def fake(model, messages, **kw):
            calls.append(kw.get("send_tools"))
            return "USE: read_file" if len(calls) == 1 else 'read_file({"filename":"/x"})'
        ca.call_llm = fake
        messages = [{"role": "system", "content": "seed"},
                    {"role": "user", "content": "read /x"}]
        reply, tools = ca._turn_reply(messages, "m", [None],
                                      {"num_ctx": None, "token_budget": ca.TOKEN_BUDGET,
                                       "max_tokens": 200}, None, active)
    finally:
        ca.call_llm = orig
        ca._active_tools.clear()
        ca._active_tools.update(active)
    assert calls == [True, True]            # pass1 meta-schema only, pass2 full
    assert tools and tools[0][0] == "read_file"
    print("  two-pass: USE on pass1, tool call on pass2       ok")


def test_pass1_tool_call_is_honoured():
    # A model that skips USE: and just emits the call on pass1 must still
    # dispatch — discarding it printed a real write_file call to the user.
    calls = []
    active = set(ca._active_tools)
    orig = ca.call_llm
    try:
        def fake(model, messages, **kw):
            calls.append(kw.get("send_tools"))
            return '{"name":"write_file","arguments":{"filename":"/x","content":"hi"}}'
        ca.call_llm = fake
        messages = [{"role": "system", "content": "seed"},
                    {"role": "user", "content": "write /x"}]
        reply, tools = ca._turn_reply(messages, "m", [None],
                                      {"num_ctx": None, "token_budget": ca.TOKEN_BUDGET,
                                       "max_tokens": 200}, None, active)
    finally:
        ca.call_llm = orig
        ca._active_tools.clear()
        ca._active_tools.update(active)
    assert calls == [True]               # no pass2 needed, the call is already here
    assert tools and tools[0][0] == "write_file"
    print("  pass1 tool call honoured, not printed            ok")


def test_edit_waits_for_the_read_in_the_same_reply():
    # Live: one reply held read_file(hiMe.py) + edit_file(hiMe.py). Both ran,
    # but the edit's old_str was written before the read returned, so it could
    # only guess. Arguments cannot use a result that does not exist yet.
    was = ca._agent_cwd[0]
    ca._agent_cwd[0] = Path("/home/james")
    try:
        keep, held = ca._defer_blind_edits(
            [("read_file", {"filename": "hiMe.py"}),
             ("edit_file", {"path": "/home/james/hiMe.py", "old_str": "t", "new_str": "x"})])
        assert [n for n, _ in keep] == ["read_file"]
        assert [n for n, _ in held] == ["edit_file"], "blind edit must be deferred"
        # a different file is untouched
        _, none_held = ca._defer_blind_edits(
            [("read_file", {"filename": "/home/james/a.py"}),
             ("edit_file", {"path": "/home/james/b.py", "old_str": "t", "new_str": "x"})])
        assert none_held == [], "only the file being read is deferred"
        # an edit on its own always runs
        solo, held2 = ca._defer_blind_edits(
            [("edit_file", {"path": "/home/james/hiMe.py", "old_str": "t", "new_str": "x"})])
        assert len(solo) == 1 and held2 == []
    finally:
        ca._agent_cwd[0] = was
    print("  blind edit waits for its read                    ok")


def test_bad_old_str_is_an_error_that_points_at_read_file():
    # As {"action": "not_found"} it carried no "error" key, so the turn's error
    # counter never incremented and a guessing model looped to the call cap.
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.py"
        f.write_text("temp_f = temp_c * 9 / 5 + 32\n")
        ca._extra_write_dirs.append(Path(d))
        try:
            got = ca.edit_file_tool(str(f), "temp = round(temp, 2)", "temp = round(temp)")
        finally:
            ca._extra_write_dirs.remove(Path(d))
    assert got.get("error") == "old_str_not_found", got
    # the tool just read the file to compare — it must hand back what it has
    assert "temp_f" in got.get("file_contents", ""), "must return the text it holds"
    assert "file_contents" in got.get("hint", ""), "must name the recovery"
    assert "ERROR" in ca._summarise_result("edit_file", got), "user must see it failed"
    print("  bad old_str errors and returns the file          ok")


def test_edit_survives_lost_indentation_when_unambiguous():
    # Live: the model copied a block out of the file and dropped the 4-space
    # base indent. Every line correct, exactly one match, refused on whitespace.
    src = ('if ok:\n'
           '    temp_c = data["t"]\n'
           '    temp_f = temp_c * 9 / 5 + 32\n'
           '    print(temp_f)\n')
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.py"
        f.write_text(src)
        ca._extra_write_dirs.append(Path(d))
        try:
            got = ca.edit_file_tool(
                str(f),
                'temp_c = data["t"]\ntemp_f = temp_c * 9 / 5 + 32',
                'temp_c = data["t"]\ntemp_f = round(temp_c * 9 / 5 + 32)')
        finally:
            ca._extra_write_dirs.remove(Path(d))
        assert got.get("action") == "edited", got
        assert "indentation" in got.get("note", ""), "must say it adjusted indent"
        after = f.read_text()
        assert "    temp_f = round(temp_c * 9 / 5 + 32)" in after, after
        assert after.startswith("if ok:\n"), "surrounding lines untouched"
    # ambiguity is never resolved by guessing
    twice = "a = 1\nb = 2\n\na = 1\nb = 2\n"
    assert ca._reindent_match(twice, "a = 1\nb = 2", "a = 9\nb = 9") is None
    # and text genuinely absent stays absent
    assert ca._reindent_match(src, "nothing like this", "x") is None
    print("  edit survives lost indentation, once only        ok")


def test_backup_is_never_a_write_target():
    # write_file reports the .bak path to the model; one then edited the backup
    # of the file it had just changed, destroying the recoverable copy.
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.py"
        f.write_text("a = 1\n")
        bak = Path(d) / "x.py.bak"
        bak.write_text("original\n")
        ca._extra_write_dirs.append(Path(d))
        try:
            got = ca.edit_file_tool(str(bak), "original", "clobbered")
            wrote = ca.write_file_tool(str(bak), "clobbered")
        finally:
            ca._extra_write_dirs.remove(Path(d))
        assert got.get("error") == "backup_is_not_a_target", got
        assert wrote.get("error") == "backup_is_not_a_target", wrote
        assert bak.read_text() == "original\n", "backup must be untouched"
    print("  backup file is never writable                    ok")


def test_truncated_tool_call_is_closed_and_parsed():
    # Live: a real edit_file call arrived one `}` short and was printed at the
    # user as prose instead of running.
    blob = '```json\n{"name": "edit_file", "arguments": {"old_str": "a", "new_str": "b"}\n```'
    got = ca.extract_tools(blob)
    assert got and got[0][0] == "edit_file" and got[0][1]["new_str"] == "b", got
    # but genuine garbage still fails rather than being invented into a call
    assert ca.extract_tools('{"name": "edit_file", "arguments": {{{{{{') == []
    print("  truncated tool call closed and dispatched        ok")


def test_missing_path_filled_only_when_unambiguous():
    ef = ca.TOOL_REGISTRY["edit_file"]
    a = {"old_str": "a", "new_str": "b"}
    # one file in play this turn -> the harness already knows the target
    assert ca._fill_missing_path(ef, a, ["/home/james/hiMe.py"])["path"] == "/home/james/hiMe.py"
    # two, or none -> never guess; the error stands
    assert "path" not in ca._fill_missing_path(ef, a, ["/a.py", "/b.py"])
    assert "path" not in ca._fill_missing_path(ef, a, [])
    # an explicit path is never overridden
    assert ca._fill_missing_path(ef, {**a, "path": "/x.py"}, ["/other.py"])["path"] == "/x.py"
    print("  missing path filled only when unambiguous        ok")


def test_native_tool_calls_are_kept_as_data():
    """The spec returns tool calls as structure with an id. Flattening them to
    `name({...})` and re-parsing the string is what every text parser below —
    and five invented USE: dialects — existed to undo."""
    class _Fn:
        def __init__(self, n, a): self.name, self.arguments = n, a

    class _Call:
        def __init__(self, i, n, a): self.id, self.function = i, _Fn(n, a)

    class _Msg:
        def __init__(self, calls): self.tool_calls, self.content = calls, ""

    got = ca._native_calls(_Msg([_Call("call_abc", "read_file", '{"filename":"/x.py"}')]))
    assert got == [("call_abc", "read_file", {"filename": "/x.py"})], got
    # some servers hand back an already-decoded dict
    assert ca._native_calls(_Msg([_Call("c2", "edit_file", {"path": "/x"})])) \
        == [("c2", "edit_file", {"path": "/x"})]
    # unparseable arguments fall through to the text parser instead of raising
    assert ca._native_calls(_Msg([_Call("c3", "read_file", "not json")])) == []
    assert ca._native_calls(_Msg(None)) == []
    print("  native tool calls kept as data, with ids         ok")


def test_names_mentioned_is_format_independent():
    # Every real dialect this cost us, live. All named the right tools; only
    # the wrapper differed, so the fallback reads names, not wrappers.
    assert ca.names_mentioned("EXACTLY one line: USE: read_file write_file") \
        == ["read_file", "write_file"]
    assert ca.names_mentioned("use_tools read_file write_file") \
        == ["read_file", "write_file"]
    assert ca.names_mentioned('```EXTRACTION: USE edit_file "hiMe.py"\n```') == ["edit_file"]
    # ...and must not turn chat into a tool turn
    assert ca.names_mentioned("Hello! How can I assist you today?") == []
    assert ca.names_mentioned("USE: none") == []
    assert ca.names_mentioned("x read_file " * 80) == [], "long prose is an answer"
    assert ca.names_mentioned("") == []
    print("  tool names found whatever the wrapper            ok")


def test_toolbox_is_offered_as_a_tool():
    # A native tool-caller handed no schema invents a text format, and a
    # different one each time (EDIT:, `USE:`, EXTRACTION: USE). Give it one
    # real tool to call so it has no reason to improvise.
    sch = ca._meta_schema()
    assert len(sch) == 1 and sch[0]["function"]["name"] == "use_tools"
    assert "names" in sch[0]["function"]["parameters"]["properties"]
    assert "use_tools" in ca.TOOL_REGISTRY          # so _TOOL_CALL_RE matches it
    assert "use_tools" not in ca._active_tools      # but never in the pull menu
    print("  toolbox offered as use_tools, not in menu        ok")


def test_use_tools_call_activates_and_reruns():
    baseline = {"read_file", "edit_file", "write_file"}
    active = set(ca._active_tools)
    orig, calls = ca.call_llm, []
    try:
        def fake(model, messages, **kw):
            calls.append(kw.get("schema") is not None)
            if len(calls) == 1:
                return '{"name":"use_tools","arguments":{"names":"read_file"}}'
            return 'read_file({"filename":"/x"})'
        ca.call_llm = fake
        msgs = [{"role": "system", "content": "seed"},
                {"role": "user", "content": "read /x"}]
        reply, tools = ca._turn_reply(msgs, "m", [None],
                                      {"num_ctx": None, "token_budget": ca.TOKEN_BUDGET,
                                       "max_tokens": 200}, None, baseline)
    finally:
        ca.call_llm = orig
        ca._active_tools.clear()
        ca._active_tools.update(active)
    assert calls == [True, False]        # pass1 meta-schema, pass2 the real set
    assert tools and tools[0][0] == "read_file"
    # use_tools itself must never reach the dispatch loop as a tool to run
    assert all(n != "use_tools" for n, _ in tools)
    print("  use_tools activates then re-runs with schema     ok")


def test_path_and_filename_are_interchangeable():
    # read_file takes `filename`, edit_file takes `path`. A model that called one
    # carries the name to the next; live that cost two wasted steps in a row.
    rf, ef = ca.TOOL_REGISTRY["read_file"], ca.TOOL_REGISTRY["edit_file"]
    assert ca._normalise_args(rf, {"path": "/x"}) == {"filename": "/x"}
    assert ca._normalise_args(ef, {"filename": "/x", "old_str": "a", "new_str": "b"}) \
        == {"path": "/x", "old_str": "a", "new_str": "b"}
    assert ca._normalise_args(rf, {"file_path": "/x"}) == {"filename": "/x"}
    # a correct call is left exactly as it is
    assert ca._normalise_args(rf, {"filename": "/x"}) == {"filename": "/x"}
    # nothing to rename onto, nothing invented
    assert ca._normalise_args(rf, {"text": "hi"}) == {"text": "hi"}
    print("  path/filename accepted interchangeably           ok")


def test_edit_file_states_its_contract():
    # The menu shows only the first sentence, so "read first" has to live there
    # or the model has no reason to read before editing.
    first = ca._safe_doc(ca.TOOL_REGISTRY["edit_file"]).split(". ")[0]
    assert "read the file first" in first, first
    print("  edit_file contract visible in pull menu          ok")


def test_tools_returned_to_the_box_each_step():
    # Step 1 pulls edit_file. Step 2 must start from the full baseline again,
    # or the model is locked out of read_file for the rest of the turn and
    # invents the old_str edit_file needs. This is the hiMe.py failure.
    baseline = {"read_file", "edit_file", "write_file", "list_files"}
    active = set(ca._active_tools)
    orig = ca.call_llm
    seen = []
    try:
        wanted = iter(["USE: edit_file", "USE: read_file"])

        def fake(model, messages, **kw):
            if kw.get("schema") is None:          # pass 2: the real tool set
                return "done"
            # pass 1: what the menu offered this step, before anything was taken
            seen.append(set(ca._active_tools))
            return next(wanted)
        ca.call_llm = fake
        cfg = {"num_ctx": None, "token_budget": ca.TOKEN_BUDGET, "max_tokens": 200}
        msgs = [{"role": "system", "content": "seed"},
                {"role": "user", "content": "edit /x"}]
        ca._turn_reply(msgs, "m", [None], cfg, None, baseline)
        assert ca._active_tools == {"edit_file"}       # tool in hand
        ca._turn_reply(msgs, "m", [None], cfg, None, baseline)
        assert ca._active_tools == {"read_file"}       # swapped, not stuck
    finally:
        ca.call_llm = orig
        ca._active_tools.clear()
        ca._active_tools.update(active)
    assert seen == [baseline, baseline]                # whole box offered both steps
    print("  tools returned to the box each step              ok")


def test_smalltalk_retired():
    # pull removes the need for the greeting whitelist; it must be gone.
    assert not hasattr(ca, "_is_smalltalk")
    assert not hasattr(ca, "_SMALLTALK")
    print("  smalltalk guard retired under pull               ok")


if __name__ == "__main__":
    test_ctx_has_ask_and_api_2()
    test_ctx_ask_calls_call_llm()
    test_ctx_no_lint_pipeline_names()
    test_core_pipeline_functions_removed()
    test_errors_are_slugs_not_prose()
    test_search_reports_what_it_cut()
    test_arguments_are_parsed_as_tolerantly_as_calls()
    test_a_filter_matching_nothing_offers_a_way_out()
    test_fold_keeps_goal_and_loses_nothing()
    test_a_broken_store_does_not_stop_the_agent()
    test_oversized_single_result_is_capped()
    test_search_finds_past_sessions_by_keyword()
    test_search_scope_is_cwd_by_default_and_global_on_demand()
    test_no_index_rows_are_not_recalled()
    test_backfill_indexes_preexisting_rows()
    test_history_flags()
    test_format_recall_renders_and_caps()
    test_do_recall_respects_all_flag()
    test_recall_tool_returns_matches_dict()
    test_read_file_results_are_not_indexed()
    test_digest_roundtrip_and_cache()
    test_chunker_windows_and_overlap()
    test_ingest_file_maps_reduces_and_caches()
    test_ingest_aborts_atomically_on_llm_failure()
    test_ingest_refuses_missing_file()
    test_ingested_digest_is_recallable()
    test_reply_text_falls_back_to_reasoning()
    test_ingest_tool_is_advertised_and_bridged()
    test_ingest_max_chunks_guard_trips_before_model()
    test_no_think_sets_reasoning_effort()
    test_ingest_guards_empty_digest()
    test_parse_use_request()
    test_pull_prompt_lists_names_and_use_protocol()
    test_two_pass_pull_then_call()
    test_pass1_tool_call_is_honoured()
    test_edit_waits_for_the_read_in_the_same_reply()
    test_bad_old_str_is_an_error_that_points_at_read_file()
    test_edit_survives_lost_indentation_when_unambiguous()
    test_backup_is_never_a_write_target()
    test_truncated_tool_call_is_closed_and_parsed()
    test_missing_path_filled_only_when_unambiguous()
    test_native_tool_calls_are_kept_as_data()
    test_names_mentioned_is_format_independent()
    test_toolbox_is_offered_as_a_tool()
    test_use_tools_call_activates_and_reruns()
    test_path_and_filename_are_interchangeable()
    test_edit_file_states_its_contract()
    test_tools_returned_to_the_box_each_step()
    test_smalltalk_retired()
    print("all session-store tests passed")

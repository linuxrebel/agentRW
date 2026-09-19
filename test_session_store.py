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
                                       "max_tokens": 200}, None)
    finally:
        ca.call_llm = orig
        ca._active_tools.clear()
        ca._active_tools.update(active)
    assert calls == [False, True]           # pass1 no schema, pass2 schema
    assert tools and tools[0][0] == "read_file"
    print("  two-pass: USE on pass1, tool call on pass2       ok")


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
    test_smalltalk_retired()
    print("all session-store tests passed")

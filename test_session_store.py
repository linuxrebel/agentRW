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


if __name__ == "__main__":
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
    print("all session-store tests passed")

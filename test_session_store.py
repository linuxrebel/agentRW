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


if __name__ == "__main__":
    test_arguments_are_parsed_as_tolerantly_as_calls()
    test_a_filter_matching_nothing_offers_a_way_out()
    test_fold_keeps_goal_and_loses_nothing()
    test_a_broken_store_does_not_stop_the_agent()
    test_oversized_single_result_is_capped()
    print("all session-store tests passed")

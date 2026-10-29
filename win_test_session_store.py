#!/usr/bin/env python3
"""Windows runner for test_session_store.py.

Every test is imported from test_session_store.py and run unchanged, except
where noted. Two Windows differences are handled here, not in the tests:

1. Windows will not delete a file that is still open. The tests leave their
   SQLite store open when their TemporaryDirectory exits, so cleanup raised
   after the test body had already passed. The temp directory below skips that
   failed cleanup and retries it once the run ends and the stores are closed.
2. test_recall_tool_returns_matches_dict seeds a literal "/proj/a"; see its
   win_ version for why that misses on Windows.
3. test_a_broken_store_does_not_stop_the_agent uses /proc/nonexistent as a
   path that cannot be created. On Windows that is an ordinary path on the
   current drive, so the store creates it. The win_ version uses a path whose
   parent is a regular file, which no platform can create.

Run: python win_test_session_store.py
"""
import gc
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

import test_session_store as base
from win_test_extract_tools import run

ca = base.ca
_leftover_dirs = []


class _WinTemporaryDirectory(tempfile.TemporaryDirectory):
    """TemporaryDirectory that defers cleanup when a file is still open."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ignore_cleanup_errors", True)
        super().__init__(*args, **kwargs)
        _leftover_dirs.append(self.name)


# Only the test module's view of tempfile changes; the real module is untouched.
base.tempfile = types.SimpleNamespace(
    **{n: getattr(tempfile, n) for n in dir(tempfile) if not n.startswith("__")})
base.tempfile.TemporaryDirectory = _WinTemporaryDirectory


def _clean_leftovers() -> None:
    gc.collect()              # closes the sqlite connections the tests left open
    for d in _leftover_dirs:
        shutil.rmtree(d, ignore_errors=True)


def win_test_a_broken_store_does_not_stop_the_agent():
    # Losing history is a worse session. It is not a dead one.
    with tempfile.TemporaryDirectory() as d:
        blocker = Path(d) / "a_file"
        blocker.write_text("not a directory")
        store = ca.SessionStore(blocker / "nested" / "nope.db")
        assert not store.live
        store.add(1, "user", "hi")          # must not raise
        store.set_state("goal", "hi")
        store.mark_folded(1)
        assert store.counts() == (0, 0)
    print("  unusable store degrades quietly               ok")


def win_test_recall_tool_returns_matches_dict():
    """Same as test_recall_tool_returns_matches_dict, but the seeded session's
    cwd is written the way the agent writes it, str(Path(...)). The original
    seeds the literal "/proj/a", which Windows renders as "\\proj\\a", so the
    lookup missed. The agent itself is consistent: run() stores and recall_tool
    compares the same str(_agent_cwd[0])."""
    proj = str(Path("/proj/a"))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        store = ca.SessionStore(Path(d) / "s.db", model="test", cwd=proj)
        if not store.fts:
            print("  (fts5 unavailable — skipping recall_tool test)  ok")
            return
        cur = store.session_id
        store.db.execute("INSERT INTO sessions (id, started, model, cwd) "
                         "VALUES (70,'t','m',?)", (proj,))
        base._seed(store, [(70, 1, "user", "kafka consumer lag spike", "lag", False)])
        store.session_id = cur
        ca._active_store[0] = store
        ca._agent_cwd[0] = Path(proj)
        try:
            res = ca.recall_tool("kafka lag")
            assert isinstance(res, dict) and res.get("matches"), res
            assert res["matches"][0]["session"] == 70
            assert "recall" in ca.TOOL_REGISTRY, "recall must be dispatchable"
            assert "recall" not in ca._active_tools, "recall must be off by default"
        finally:
            ca._active_store[0] = None
            store.db.close()
    print("  recall_tool returns matches; off by default     ok")


# Same order as test_session_store.py's __main__, with the two swaps.
TESTS = [
    base.test_saved_default_accepts_string_form,
    base.test_ctx_has_ask_and_api_2,
    base.test_ctx_ask_calls_call_llm,
    base.test_ctx_no_lint_pipeline_names,
    base.test_core_pipeline_functions_removed,
    base.test_errors_are_slugs_not_prose,
    base.test_search_reports_what_it_cut,
    base.test_arguments_are_parsed_as_tolerantly_as_calls,
    base.test_a_filter_matching_nothing_offers_a_way_out,
    base.test_fold_keeps_goal_and_loses_nothing,
    win_test_a_broken_store_does_not_stop_the_agent,
    base.test_oversized_single_result_is_capped,
    base.test_search_finds_past_sessions_by_keyword,
    base.test_search_scope_is_cwd_by_default_and_global_on_demand,
    base.test_no_index_rows_are_not_recalled,
    base.test_backfill_indexes_preexisting_rows,
    base.test_history_flags,
    base.test_format_recall_renders_and_caps,
    base.test_do_recall_respects_all_flag,
    win_test_recall_tool_returns_matches_dict,
    base.test_read_file_results_are_not_indexed,
    base.test_digest_roundtrip_and_cache,
    base.test_chunker_windows_and_overlap,
    base.test_ingest_file_maps_reduces_and_caches,
    base.test_ingest_aborts_atomically_on_llm_failure,
    base.test_ingest_refuses_missing_file,
    base.test_ingested_digest_is_recallable,
    base.test_reply_text_falls_back_to_reasoning,
    base.test_ingest_tool_is_advertised_and_bridged,
    base.test_ingest_max_chunks_guard_trips_before_model,
    base.test_no_think_sets_reasoning_effort,
    base.test_ingest_guards_empty_digest,
    base.test_parse_use_request,
    base.test_pull_prompt_lists_names_and_use_protocol,
    base.test_two_pass_pull_then_call,
    base.test_smalltalk_retired,
]


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("win_test_session_store.py is the Windows runner; "
                 "use test_session_store.py on this platform.")
    try:
        code = run(TESTS, echo_ok=False)   # these tests print their own ok line
    finally:
        _clean_leftovers()
    sys.exit(code)

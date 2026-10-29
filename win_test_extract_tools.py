#!/usr/bin/env python3
"""Windows runner for test_extract_tools.py.

Every test is imported from test_extract_tools.py and run unchanged, except the
two that depend on Unix: they are replaced below by Windows equivalents that
check the same guarantee. Keep test_extract_tools.py the source of truth; add
a win_ replacement here only when a test cannot run on Windows as written.

Run: python win_test_extract_tools.py
"""
import ctypes
import ctypes.wintypes
import io
import os
import pathlib
import sys
import tempfile
import traceback

import test_extract_tools as base


def _peak_working_set_mb() -> int:
    """Peak RSS of this process in MB. Windows stand-in for resource.getrusage."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE,
                                           ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                    ctypes.byref(pmc), pmc.cb)
    assert ok, "GetProcessMemoryInfo failed"
    return pmc.PeakWorkingSetSize // (1024 * 1024)


def win_test_writes_are_scoped():
    """Same scope checks as test_writes_are_scoped.

    The .bak 0600 check is replaced: Windows has no POSIX mode bits, chmod only
    toggles read-only, and os.stat always reports 0o666. Access there is set by
    the directory's ACLs. What still holds is that the .bak keeps the original.
    """
    import coding_agent as ca
    work, outside = tempfile.mkdtemp(), tempfile.mkdtemp()
    ca._agent_cwd[0] = pathlib.Path(work)
    ca._extra_write_dirs.clear()

    assert ca.write_file_tool("ok.py", "x=1")["action"] == "written"
    for denied in (os.path.join(outside, "evil.py"),
                   os.path.expanduser("~/.bashrc"),
                   "../../escape.py"):
        assert ca.write_file_tool(denied, "x=1").get("error") == "write_outside_allowed_dirs", denied
    assert ca.edit_file_tool(os.path.join(outside, "e.py"), "", "x=1") \
             .get("error") == "write_outside_allowed_dirs"

    ca._extra_write_dirs.append(pathlib.Path(outside))
    assert ca.write_file_tool(os.path.join(outside, "now.py"), "x=1")["action"] == "written"
    ca._extra_write_dirs.clear()

    ca.write_file_tool("ok.py", "x=2")
    bak = pathlib.Path(work, "ok.py.bak")
    assert bak.read_text(encoding="utf-8") == "x=1", "the .bak must hold the original"


def win_test_output_is_capped():
    """Same guarantee as test_output_is_capped: unbounded output kills the
    command instead of filling RAM.

    `yes` and `/dev/urandom` do not exist under cmd.exe, so Python stands in
    for both: one endless text stream, one endless binary stream.
    """
    import coding_agent as ca
    py = f'"{sys.executable}"'
    endless = {
        "text":   f'{py} -c "while 1: print(\'yes read dr-strange.py\')"',
        "binary": f'{py} -c "import os,sys; [sys.stdout.buffer.write(os.urandom(65536)) for _ in iter(int, 1)]"',
    }

    before = _peak_working_set_mb()
    for kind, cmd in endless.items():
        out, _err, _rc, capped = ca._run_capped(cmd, timeout=30)
        assert capped is True, kind
        assert len(out) <= ca.MAX_CAPTURE_BYTES + 65536, (kind, len(out))
    after = _peak_working_set_mb()
    assert after - before < 500, f"peak working set grew {after - before} MB"

    # ordinary commands unaffected
    out, _e, rc, capped = ca._run_capped("echo hello", timeout=10)
    assert out.strip() == "hello" and rc == 0 and capped is False
    assert ca._run_capped("exit 3", timeout=10)[2] == 3

    # `yes` must not reach the passthrough silently (platform-independent)
    assert "yes" in ca._AMBIGUOUS_WORDS
    real_stdin = sys.stdin
    try:
        ca._word_is_command.clear()
        sys.stdin = io.StringIO("\n")
        assert ca._resolve_ambiguous("yes", "yes, do that") is False
        sys.stdin = io.StringIO("")
        ca._word_is_command.clear()
        assert ca._resolve_ambiguous("yes", "yes, do that") is False
        ca._word_is_command.clear()
        sys.stdin = io.StringIO("1\n")
        assert ca._resolve_ambiguous("sort", "sort file.txt") is True
    finally:
        sys.stdin = real_stdin
        ca._word_is_command.clear()


# Same order as test_extract_tools.py's __main__, with the two swaps.
TESTS = [
    base.test_extract_tools,
    base.test_advertising_defaults,
    base.test_native_tool_calls_are_read,
    base.test_command_aliases,
    base.test_json_tool_call_format,
    win_test_output_is_capped,
    base.test_run_command_reports_timeout,
    base.test_schema_description_is_capped,
    base.test_path_echo_matches_windows_paths,
    base.test_ui_arg_mapping,
    base.test_lint_file_summarises,
    base.test_plugin_loading,
    base.test_install_md_parsing,
    win_test_writes_are_scoped,
    base.test_command_confirmation_defaults_to_no,
]


def run(tests, echo_ok: bool = True) -> int:
    """Run every test, even after a failure, so one run shows all of them."""
    failed = []
    for t in tests:
        try:
            t()
            if echo_ok:
                print(f"  {t.__name__:<48} ok")
        except Exception:
            failed.append(t.__name__)
            print(f"  {t.__name__:<48} FAIL")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("win_test_extract_tools.py is the Windows runner; "
                 "use test_extract_tools.py on this platform.")
    sys.exit(run(TESTS))

#!/usr/bin/env python3
"""Guards tool-call parsing: bare calls and positional args must not be
treated as chat text (that bug silently swallowed every write_file)."""
import io
import os
import pathlib
import stat
import sys
import tempfile

from coding_agent import extract_tools

CASES = [
    ('tool: write_file({"filename":"/tmp/x.py","content":"print(1)"})',
     ('write_file', {'filename': '/tmp/x.py', 'content': 'print(1)'})),
    ('write_file({"filename":"/tmp/x.py","content":"print(1)"})',
     ('write_file', {'filename': '/tmp/x.py', 'content': 'print(1)'})),
    ("run_command('echo hi')", ('run_command', {'cmd': 'echo hi'})),
    ('read_file("/tmp/x.py")', ('read_file', {'filename': '/tmp/x.py'})),
    ('list_files(".")', ('list_files', {'path': '.'})),
    ('write_file(filename="/tmp/y.py", content="x=1")',
     ('write_file', {'filename': '/tmp/y.py', 'content': 'x=1'})),
    ('tool: edit_file({"path":"/tmp/x.py","old_str":"a","new_str":"b"})',
     ('edit_file', {'path': '/tmp/x.py', 'old_str': 'a', 'new_str': 'b'})),
    # Python-literal dict, single quotes (phi3 emits this)
    ("write_file({'filename': '/tmp/x.py', 'content': 'print(1)'})",
     ('write_file', {'filename': '/tmp/x.py', 'content': 'print(1)'})),
    # Triple-quoted body, unbalanced close — must still recover
    ("write_file({'filename': 't.py', 'content': '''a = \"x\"\\nb = 2''})",
     ('write_file', {'filename': 't.py', 'content': 'a = "x"\nb = 2'})),
    ('no tools here, just prose about files', None),
]


def test_extract_tools():
    for text, want in CASES:
        got = extract_tools(text)
        assert got == ([want] if want else []), f"{text!r} -> {got!r}"


def test_writes_are_scoped():
    """write_file/edit_file must refuse anything outside the allowed dirs."""
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

    # .bak must not widen access to the original's contents
    ca.write_file_tool("ok.py", "x=2")
    mode = stat.S_IMODE(os.stat(os.path.join(work, "ok.py.bak")).st_mode)
    assert mode == 0o600, oct(mode)


def test_command_confirmation_defaults_to_no():
    """Non-interactive or empty answer must decline, never run."""
    import coding_agent as ca
    ca._auto_approve[0] = False
    real_stdin = sys.stdin
    try:
        for answer, expected in [("", False), ("n\n", False), ("y\n", True), ("a\n", True)]:
            ca._auto_approve[0] = False
            sys.stdin = io.StringIO(answer)
            assert ca._confirm_command("rm -rf ~") is expected, repr(answer)
        assert ca._auto_approve[0] is True  # 'a' sticks for the session
    finally:
        sys.stdin = real_stdin
        ca._auto_approve[0] = False


if __name__ == "__main__":
    test_extract_tools()
    test_writes_are_scoped()
    test_command_confirmation_defaults_to_no()
    print("ok")

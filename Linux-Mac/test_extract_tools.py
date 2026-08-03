#!/usr/bin/env python3
"""Guards tool-call parsing: bare calls and positional args must not be
treated as chat text (that bug silently swallowed every write_file)."""
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


if __name__ == "__main__":
    test_extract_tools()
    print("ok")

#!/usr/bin/env python3
"""Guards tool-call parsing: bare calls and positional args must not be
treated as chat text (that bug silently swallowed every write_file)."""
import inspect
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


def test_lint_file_summarises():
    """lint_file must aggregate. A dump would defeat the point of the tool."""
    import json
    import shutil as _sh
    import coding_agent as ca
    if not _sh.which("pylint") or "lint_file" not in ca.TOOL_REGISTRY:
        return  # optional dependency / plugin not installed
    lint = ca.TOOL_REGISTRY["lint_file"]
    work = tempfile.mkdtemp()
    src = os.path.join(work, "sample.py")
    with open(src, "w") as f:
        f.write("import os\nimport sys\n" + "".join(
            f"def f{i}(): return {i}\n" for i in range(40)))

    r = lint(src)
    assert "score" in r and r["total"] > 0, r
    assert len(r["top_issues"]) <= 6, r["top_issues"]
    # The whole point: the result stays small no matter how many messages.
    assert len(json.dumps(r)) < 1200, len(json.dumps(r))

    sym = r["top_issues"][0]["symbol"]
    d = lint(src, symbol=sym)
    assert d["symbol"] == sym and d["count"] >= 1, d
    assert len(d["occurrences"]) <= 20

    assert lint(os.path.join(work, "gone.py"))["error"] == "file_not_found"


def _make_plugin(root, owner, name, body, files=("plugin.py",), api=1,
                 requires=("thing",)):
    """Write a plugin in the tools/<owner>/<name>/ layout."""
    pkg = root / owner / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "plugin.py").write_text(body)
    (pkg / "install.md").write_text(
        f"# {owner}/{name} 1.0.0\n\ndesc\n\n## Files\n"
        + "".join(f"- {f}\n" for f in files)
        + "\n## Requires\n" + "".join(f"- {r}\n" for r in requires)
        + f"\n## API\n{api}\n")
    return pkg


def test_plugin_loading():
    """A plugin is tools/<owner>/<name>/ with an install.md declaring its files."""
    import coding_agent as ca
    d = pathlib.Path(tempfile.mkdtemp())

    assert ca.load_plugins(d) == {}                          # empty
    assert ca.load_plugins(d / "nope") == {}                 # missing

    _make_plugin(d, "me", "demo",
                 "def demo_tool(x: str) -> dict:\n"
                 '    """Demo."""\n'
                 "    return {'x': x}\n")
    # declared but absent, and present but undeclared
    _make_plugin(d, "me", "partial", "def ghost_tool(): return {}\n",
                 files=("plugin.py", "missing.py"))
    (d / "me" / "partial" / "sneaky.py").write_text("def sneaky_tool(): return {}\n")
    _make_plugin(d, "me", "broken", "def oops_tool(\n")      # syntax error
    _make_plugin(d, "me", "future", "def soon_tool(): return {}\n", api=99)
    (d / "me" / "notaplugin").mkdir()                        # no install.md

    ca.PLUGIN_STATUS.clear()
    found = ca.load_plugins(d)

    assert "demo" in found, found                            # discovered
    assert found["demo"](x="hi") == {"x": "hi"}              # and it runs
    assert "ghost" in found, found                           # missing file skipped,
    assert "sneaky" not in found, found                      # undeclared never runs
    assert "oops" not in found, found                        # broken isolated
    assert "soon" not in found, found                        # api too new

    # A discovered tool must be indistinguishable from a core one to the prompt.
    assert found["demo"].__doc__ and str(inspect.signature(found["demo"]))

    import shutil as _sh
    _sh.rmtree(d / "me" / "demo")
    assert "demo" not in ca.load_plugins(d)                  # uninstalled


def test_install_md_parsing():
    """install.md is markdown so it renders on a forge. Malformed is the
    author's problem — we return what was readable and refuse."""
    import coding_agent as ca
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "install.md"

    p.write_text("# owner/thing 2.1.0\n\ndesc\n\n## Files\n- a.py\n- data.json\n"
                 "\n## Requires\n- pylint\n\n## API\n1\n")
    m = ca.read_install_md(p)
    assert m["name"] == "owner/thing" and m["version"] == "2.1.0", m
    assert m["files"] == ["a.py", "data.json"], m
    assert m["requires"] == ["pylint"] and m["api"] == 1, m

    p.write_text("nothing useful here\n")                    # malformed
    m = ca.read_install_md(p)
    assert m["files"] == [] and m.get("name") is None, m     # readable, empty


def test_ui_arg_mapping():
    """/tool <args> maps bare tokens positionally, k=v by name, ints coerced."""
    import coding_agent as ca

    def parse(cmd):
        parts = cmd.lstrip("/").split()
        sig = inspect.signature(ca.TOOL_REGISTRY[parts[0]])
        params = list(sig.parameters)

        def coerce(p, raw):
            ann = sig.parameters[p].annotation
            return int(raw) if ann is int else raw

        args, pos = {}, 0
        for tok in parts[1:]:
            k = tok.split("=", 1)[0]
            if "=" in tok and k in params:
                args[k] = coerce(k, tok.split("=", 1)[1])
            elif pos < len(params):
                args[params[pos]] = coerce(params[pos], tok)
                pos += 1
        return args

    assert parse("/read_file a.py") == {"filename": "a.py"}
    # ints must be coerced or the tool raises TypeError
    assert parse("/read_file a.py start_line=5") == {"filename": "a.py", "start_line": 5}
    assert parse("/search_file a.py needle") == {"filename": "a.py", "text": "needle"}


def test_fix_loop_pieces():
    """/fix must gather every finding in ONE detector run, apply, and defer."""
    import shutil as _sh
    import coding_agent as ca
    if not _sh.which("pylint") or "lint_file" not in ca.TOOL_REGISTRY:
        return

    work = pathlib.Path(tempfile.mkdtemp())
    ca._agent_cwd[0] = work
    ca._extra_write_dirs.clear()
    src = work / "sample.py"
    src.write_text("import os\nimport sys\nx=1\n")

    calls = []
    real = ca.TOOL_REGISTRY["lint_file"]
    ca.TOOL_REGISTRY["lint_file"] = lambda **kw: (calls.append(kw), real(**kw))[1]
    try:
        found = ca._gather_findings(str(src))
    finally:
        ca.TOOL_REGISTRY["lint_file"] = real

    # The whole point of symbol="*": one run, not one per kind.
    assert len(calls) == 1, calls
    assert len(found) >= 3, found
    assert all({"line", "symbol", "message"} <= set(f) for f in found), found
    # Descending: an edit shifts every line below it, so working bottom-up keeps
    # unvisited line numbers valid. Ascending left 13 of 17 findings stale.
    assert found == sorted(found, key=lambda f: -f["line"])

    # apply
    assert ca.edit_file_tool(str(src), "x=1", "x = 1")["action"] == "edited"
    assert "x = 1" in src.read_text()

    # defer writes a ledger line instead of losing the finding
    ca._defer("sample.py", found[0])
    assert (work / ca.DEBT_FILE).read_text().startswith("- [ ] sample.py:")


def test_output_is_capped():
    """Unbounded output must kill the command, not fill RAM.

    `yes read dr-strange.py ...` went through the passthrough, and
    capture_output=True buffered it to 29 GB before the OOM killer took the
    whole agent down. Truncating after the fact is too late.
    """
    import resource
    import coding_agent as ca

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    for cmd in ("yes read dr-strange.py and tell me if you see any code errors",
                "yes",
                "cat /dev/urandom"):          # binary: must not crash the reader
        out, _err, _rc, capped = ca._run_capped(cmd, timeout=30)
        assert capped is True, cmd
        assert len(out) <= ca.MAX_CAPTURE_BYTES + 65536, (cmd, len(out))

    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    assert after - before < 500, f"RSS grew {after - before} MB"

    # ordinary commands unaffected
    out, _e, rc, capped = ca._run_capped("echo hello", timeout=10)
    assert out.strip() == "hello" and rc == 0 and capped is False
    assert ca._run_capped("exit 3", timeout=10)[2] == 3

    # `yes` must not reach the passthrough silently — it is prompted for, and
    # the default answer treats the line as a message.
    assert "yes" in ca._AMBIGUOUS_WORDS
    real_stdin = sys.stdin
    try:
        ca._word_is_command.clear()
        sys.stdin = io.StringIO("\n")          # bare Enter = default
        assert ca._resolve_ambiguous("yes", "yes, do that") is False
        sys.stdin = io.StringIO("")            # EOF must also decline
        ca._word_is_command.clear()
        assert ca._resolve_ambiguous("yes", "yes, do that") is False
        ca._word_is_command.clear()
        sys.stdin = io.StringIO("1\n")
        assert ca._resolve_ambiguous("sort", "sort file.txt") is True
    finally:
        sys.stdin = real_stdin
        ca._word_is_command.clear()


def test_native_tool_calls_are_read():
    """Native tool_calls must not be lost.

    gemma4:31b-cloud answers with finish_reason="tool_calls" and content "".
    Reading only .content dropped the whole reply and the turn ended silently.
    """
    import types
    import coding_agent as ca

    def call(name, args):
        return types.SimpleNamespace(
            function=types.SimpleNamespace(name=name, arguments=args))

    msg = types.SimpleNamespace(content="", tool_calls=[
        call("list_files", '{"path":"/home/james/bin"}')])
    assert ca.extract_tools(ca._reply_text(msg)) == [
        ("list_files", {"path": "/home/james/bin"})]

    # Core tools only — plugin tools are optional and may not be installed.
    multi = types.SimpleNamespace(content=None, tool_calls=[
        call("read_file", '{"filename":"a.py"}'),
        call("search_file", '{"filename":"a.py","text":"x"}')])
    assert len(ca.extract_tools(ca._reply_text(multi))) == 2

    # plain text replies unaffected
    assert ca._reply_text(types.SimpleNamespace(content="hi", tool_calls=None)) == "hi"
    assert ca._reply_text(types.SimpleNamespace(content=None, tool_calls=None)) == ""

    # schema is derived from the registry, so plugins are included
    names = {t["function"]["name"] for t in ca.TOOLS_SCHEMA}
    assert names == set(ca.TOOL_REGISTRY), names
    rf = next(t for t in ca.TOOLS_SCHEMA if t["function"]["name"] == "read_file")
    params = rf["function"]["parameters"]
    assert params["required"] == ["filename"], params
    assert params["properties"]["start_line"]["type"] == "integer", params


def test_command_aliases():
    """Common slash typos resolve; ambiguous ones are left alone."""
    import coding_agent as ca
    c = ca._canonical_command
    assert c("/lint") == "lint"                 # the interactive session
    assert c("/models") == "model"              # plural toggle
    assert c("/cloud-model") == "cloud-models"  # singular toggle
    assert c("/token") == "tokens"
    assert c("/read") == "read_file"
    assert c("/model") == "model"               # exact wins over toggle
    assert c("/cloud-models") == "cloud-models"
    assert c("/help") == "help"
    assert c("/c") == "c"                       # ambiguous: compact/cloud-models
    assert c("/nosuchthing") == "nosuchthing"   # unknown falls through


def test_json_tool_call_format():
    """Sending a tools schema makes some models answer with the JSON call
    object as plain text. qwen2.5-coder does this on ~2 turns in 3, and it
    was being dropped entirely."""
    import coding_agent as ca
    cases = [
        ('{"name": "read_file", "arguments": {"filename": "/x/y.py"}}',
         [("read_file", {"filename": "/x/y.py"})]),
        ('{"name":"run_command","arguments":{"cmd":"ls -la"}}',
         [("run_command", {"cmd": "ls -la"})]),
        ('I will do this:\n{"name": "list_files", "arguments": {"path": "."}}\nok',
         [("list_files", {"path": "."})]),
        ('{"name":"read_file","arguments":"{\\"filename\\":\\"a.py\\"}"}',
         [("read_file", {"filename": "a.py"})]),
        ('```json\n{"name":"read_file","arguments":{"filename":"a.py"}}\n```',
         [("read_file", {"filename": "a.py"})]),
        ('{"name":"not_a_tool","arguments":{}}', []),
        ('just prose, no calls at all', []),
        # the paren form must still win, unchanged
        ('read_file({"filename":"/x/y.py"})', [("read_file", {"filename": "/x/y.py"})]),
    ]
    for text, want in cases:
        assert ca.extract_tools(text) == want, (text, ca.extract_tools(text))


if __name__ == "__main__":
    test_extract_tools()
    test_native_tool_calls_are_read()
    test_command_aliases()
    test_json_tool_call_format()
    test_output_is_capped()
    test_fix_loop_pieces()
    test_ui_arg_mapping()
    test_lint_file_summarises()
    test_plugin_loading()
    test_install_md_parsing()
    test_writes_are_scoped()
    test_command_confirmation_defaults_to_no()
    print("ok")

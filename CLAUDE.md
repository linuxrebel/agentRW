# CLAUDE.md — agentRW

Repo-specific instructions. Merge with the global `~/.claude/CLAUDE.md`.

## Git workflow

Remotes are `gh-origin` (GitHub) and `gl-origin` (GitLab). **There is no remote
named `origin`** — "push to origin" means push to both.

1. Do all work on `farkitall`.
2. Commit, push `farkitall` to **both** remotes.
3. `git checkout main && git merge --ff-only farkitall`
4. Push `main` to **both** remotes.
5. **`git checkout farkitall` before finishing. Always.**

Step 5 is not optional. Ending a session on `main` caused a later round of edits
to land directly on `main`, which inverted the merge direction and had to be
untangled by hand.

## Design invariants

Break these only on an explicit decision, never as a side effect of cleanup.

**The harness writes, it does not judge.** `write_file_tool` must not validate
the content it is handed — no syntax checks, no truncation heuristics, no
"suspicious content" gates. Those existed once and were the actual cause of what
looked like bad model output: they returned `{"error": ...}`, which tripped
`turn_had_error` and burned the retry budget, so files were never written and the
model appeared to flail. A file on disk with a bug beats a refusal and no file.
Judging the result is the user's job.

**Parse tolerantly.** Models emit tool calls in whatever syntax they feel like —
bare `write_file(...)` with no `tool:` prefix, single-quoted Python dicts, triple
quotes, unbalanced quotes. `extract_tools` accepts all of it by design. Tightening
the parser to reject malformed input recreates the "file never gets written" bug.
Add shapes, do not remove them.

**The agent cannot modify its own source.** `_writable()` calls `.resolve()`
before checking a path against the allowed write directories, so symlinks are
followed first. Running with cwd `~/bin`, where the `cagent` symlink lives, still
cannot write through that link to the repo. This looks incidental. It is not — do
not "simplify" `_writable` to skip `.resolve()` or compare unresolved paths.

**Writes are scoped, commands are confirmed.** Writes are confined to the working
directory (`--allow-write DIR` widens). Every model-proposed `run_command` is
confirmed by the user (`--yes` skips). These are the controls against a file's
contents steering the model into execution — the sudo regex is *not* one, since
`shell=True` expands the string after that check.

## Before claiming something works

Run `python3 test_extract_tools.py`. It covers tool-call parsing across every
syntax, the write scope (including traversal and `~/.bashrc`), `.bak`
permissions, and that command confirmation defaults to no.

## Context budget

This targets a 4 GB GPU. The system prompt is ~641 tokens; `--low-vram` gives a
2048-token window total. Anything added to the prompt is paid for on every turn —
see `Future_Router.md` for the measured numbers and the plan to move mechanical
work into system space instead.

# /lint — what two full runs turned up

Written 2026-08-06 against `coding_agent.py` at `3590e60` plus uncommitted
`/lint` work. Both runs walked all 17 findings in `/home/james/bin/dr-strange.py`
(35 lines) auto-accepting every proposed fix.

---

## 1. The context-window hypothesis holds

This was the thing the design was built around, and it is confirmed.

| | qwen2.5-coder:7b (local) | gemma4:31b-cloud |
|---|---|---|
| model calls | 9 | 3 |
| prompt tokens, peak | 141 | 142 |
| completion, peak | 37 | 285 |
| **total, peak** | **156** | **414** |
| avg per call | 122 | 132 |
| time | 45s (5.0s avg) | 4s (1.5s avg) |

Against a 2048-token `--low-vram` window. Nothing accumulates between findings:
each one is a fresh ~130-token prompt. A 300-finding file costs the same *per
decision* as a 3-finding one, which was the whole point.

gemma4's 414 peak is entirely completion — it wrote a 285-token essay
deliberating about a blank line instead of answering. That is a prompt-obedience
problem, not a context problem.

---

## 2. Both runs destroyed the file, at the same finding, for the same reason

**This is a harness bug, not a model bug.** `edit_file_tool`:

```python
if old_str == "":
    p.write_text(new_str)      # replaces the ENTIRE file
    return {"path": str(p), "action": "created"}
```

Finding 4 in both runs was `bad-indentation` on a **blank line**. The loop reads
the target line, gets `""`, and calls `edit_file_tool(path, "", new)`. The empty
`old_str` is interpreted as "create this file", so the whole file is replaced by
whatever the model returned for that one line.

- local 7B left the file as: `` 16:``` ``
- gemma4 left the file as its 285-token deliberation, ending in `UNFIXABLE`

Every finding after that failed, because the file no longer had the lines they
referred to.

Severity: this is reachable any time `edit_file` is called with an empty
`old_str`, including directly by a model. It predates the `/lint` work.

---

## 3. Line numbers go stale after an insert

Findings are gathered once, up front. Every `insert_top` / `insert_after` shifts
every later line by one, and nothing re-maps them.

Run 2 reached finding 4 of 17 before the file was destroyed; findings 5–17 all
pointed at lines that no longer existed. Even without bug 2 this would have
mis-targeted every subsequent edit — which is worse than failing, because a
shifted line number still names a *real* line, just the wrong one.

Fixes, either works:
- process findings in descending line order, so an edit never moves a line
  that has not been visited yet
- re-run the detector after each applied insert (correct, but 21s per pass on a
  large file)

---

## 4. Model output is inserted verbatim, unsanitised

The local 7B returned, and the harness wrote:

````
+ ```python
def my_function():
    """Function docstring"""
````

and elsewhere the literal word `UNFIXABLE` wrapped in a code fence — which the
`UNFIXABLE` check missed, because it tests equality against the whole reply
rather than looking inside it.

Needed before any proposal is applied:
- strip markdown fences
- take the first line only when one line was asked for
- reject anything containing `UNFIXABLE`
- reject multi-line output for a single-line rewrite

gemma4 did not need this — its answers were clean, single-line
(`"""Module docstring"""`, `"""Handle the dialogue sequence."""`). So this is a
robustness gap the harness must cover for weaker models, not a universal defect.

---

## 4b. The prompt contradicts itself

`FIX_PROMPT` ends with:

> Output ONLY the corrected line(s) ... **Preserve indentation exactly.** If the
> issue cannot be fixed by rewriting these lines, output UNFIXABLE.

For `bad-indentation` the fix *is* the indentation. The model is told to change
it and preserve it in the same breath. Probed directly against line 28
(`'  bargain()\n'`, 2 spaces, inside a `for` body), gemma4 returned:

```
run1 -> 'bargain()'
run2 -> 'bargain()'
run3 -> 'bargain()'
```

Zero indentation, deterministically. Applied across a whole file this produces
`IndentationError: expected an indented block after 'for' statement`.

Related: `_propose_fix` sends the raw pylint message but **not** the `action`
field the plugin already computes ("rewrite the line with 4-space indentation").
The model is given the complaint and withheld the instruction.

---

## 4c. A sanitiser must never strip leading whitespace

Worth recording because it is easy to get wrong and fails exactly like a model
error. The first isolation run used:

```python
t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())   # WRONG
```

`text.strip()` removes leading whitespace, so every indentation fix was
de-indented before it was written. Strip fences and *trailing* space only.

---

## 5. Isolation run: harness bugs neutralised

Same loop, same model, with blank-line findings skipped, findings processed in
descending line order, and proposals sanitised correctly:

```
applied=16 rejected=0 blank-skipped=0 no-fix=1
calls=16  peak_prompt=185  peak_total=202  time=18s
```

Sensible docstrings ("""Bargain with Dormammu.""", """Prints a dialogue between
a speaker and their lines."""), no fences, no refusals, nothing overwritten.
The only remaining breakage was the indentation contradiction in 4b — the model
did exactly what it was asked.

---

## 5b. What is model-dependent and what is not

| symptom | harness | model |
|---|---|---|
| file replaced from a blank-line finding | ✅ | |
| stale line numbers after insert | ✅ | |
| fences / multi-line output inserted raw | ✅ (no sanitising) | ✅ (7B only) |
| indentation stripped on bad-indentation | ✅ (prompt contradicts itself) | |
| `action` computed but never sent to the model | ✅ | |
| 285-token essay instead of one line | | ✅ (gemma4, once) |
| context exhaustion | — none observed — | |

**The answer is: 100% harness.** Both models failed at identical points for
identical reasons. With the harness bugs neutralised the cloud model applied
16 of 17 findings cleanly. The 7B is messier and still needs sanitising, but it
caused no failure the cloud model avoided.

---

## 5c. Some findings need no model at all

The single line that broke qwen2.5-coder's whole run:

```
line 9 >> '  print("\n".join(lines) + "\n")'     # 2 spaces, expected 4
```

It came back with the indentation *unfixed* and the `*` silently dropped from
`join(*lines)` — a style request that produced a runtime `TypeError`. gemma4
made the same substitution on an earlier run.

But look at what the detector said: **"Found 2 spaces, expected 4"**. The
correct output is fully determined. There is nothing to decide.

`bad-indentation` is now `action_kind: "reindent:N"` and the harness computes
`" " * N + line.lstrip()` directly. On this file that is 11 of 17 findings which
now cost zero tokens, zero seconds, are exactly right every time, and — the
part that matters — **cannot alter the code**, only leading whitespace. A
reindent operation is structurally incapable of dropping a `*`.

This is the Future_Router taxonomy applied one level deeper than intended. It
is not only whole skills that split into detection and judgment: individual
*findings* do too. Before sending a finding to a model, ask whether the
detector already stated the answer.

---

## 5d. The dr-strange.py `join` trap

Worth recording, because three separate models failed on it identically.

```python
def dialogue(speaker, *lines):     # says "any number of lines"
    print("\n".join(*lines))       # but unpacks, because...
def strange(*lines):
    dialogue("Dr Strange", lines)  # ...callers pass ONE tuple
```

The signature is variadic; every caller passes a single sequence. So `lines` is
a tuple containing a tuple and `join(*lines)` is the workaround.

**This does not excuse the models.** The finding was `bad-indentation` on that
line. The requested change was the leading whitespace, nothing else. Rewriting
`join(*lines)` to `join(lines)` is outside the scope of the instruction whether
or not the suspicion was defensible — and the result compiled cleanly while
failing at runtime, so nothing downstream caught it. Three models, same
overreach, same silent breakage.

It is the same failure seen earlier in the session, when a model asked to fix
indentation also reindented the whole file, added an `if __name__` guard, and
changed the iteration pattern, while reporting only the indentation change.
The defect is scope discipline, not code judgement.

The lesson for the harness: do not ask a model to respect a narrow scope and
hope. Make the operation *incapable* of exceeding it. `reindent` rebuilds the
line as `" " * N + line.lstrip()`, so the code after the whitespace is not
passed through a model at all and cannot come back altered.

The signature was still worth fixing — `def dialogue(speaker, lines)`, two
characters, callers unchanged — but as a readability improvement, not as an
excuse.

---

## 5e. The metric is the result, not the model's share

autopep8 now handles every style finding — 11 of 17 on this file — in one
subprocess, deterministically, with no model involved. It is tempting to read
that as the model being "carried". That reading is wrong.

autopep8 exists to solve whitespace. Using a probabilistic system for a solved
problem is the mistake; reaching for the deterministic tool is the correct
engineering, not a fallback. Every finding moved off the model is a win.

So the design goal is to **minimise** model involvement, not to demonstrate it:

| | findings | tokens |
|---|---|---|
| autopep8 | 11 | 0 |
| model (docstring content) | 5 | 187 peak |
| explained, no automatic fix | 1 | 0 |

Two consequences follow.

**It changes what a small local model has to be.** It does not need broad
capability — it needs to handle the residue that has no deterministic
solution, and to be terse enough to parse when it does. Shrink the residue and
a 7B suffices. That is the whole SLM thesis, and it is an argument for adding
tools rather than for a bigger model.

**A weak model becomes slow rather than dangerous.** The floor is whatever the
tools achieve; the model can only add above it. ornith burns 80s per finding
producing prose that gets rejected, and the file still improves — because
nothing it says is required for the deterministic half.

The corollary for adding tools: anything with a real tool should get one.
`autoflake` for unused imports, `isort` for import order, a plain rename for
`invalid-name`. What is left — what a docstring should actually *say* — is the
only part that genuinely needs a model.

---

## 6. Order to fix

1. `edit_file_tool` must not treat an empty `old_str` as "replace whole file".
   A blank-line finding is either a no-op or an explicit insert.
2. `FIX_PROMPT` must not say "preserve indentation exactly", and `_propose_fix`
   must send the `action` text the plugin already computed.
3. Sanitise proposals before applying: strip fences, first line only, reject
   anything containing UNFIXABLE — and never touch leading whitespace.
4. Fix line drift — descending order is the cheaper of the two options.

None of these are about the streaming architecture, which measured exactly as
intended: ~130 tokens per finding, constant, on both models.

# Future Router — offloading detection to the system, reserving judgment for the model

Status: **design speculation**, nothing implemented. Written 2026-08-03 against
`coding_agent.py` at commit `1d5f3ba`.

The premise: this harness is meant to stay lightweight and hand work to the
system wherever the system is better at it — the shell passthrough already does
this for bash. The question is how far that idea generalizes, and whether things
like `pylint` or `ponytail` can become programs that run in system space and
*report findings* to the model, rather than instructions the model has to carry
in its context window.

---

## 1. The measurement that started this

Run on `coding_agent.py` (1034 lines), on the machine this targets:

| Path | Tokens | Notes |
|---|---|---|
| Raw `pylint` text output | ~2,441 | exceeds the entire `--low-vram` window (2048) |
| Raw `pylint` JSON output | ~8,929 | |
| Same findings, grouped and counted system-side | **~135** | 99% reduction |
| The source file itself, if the model had to read it | ~10,000 | |

The model never opens the file. `pylint` opens the file. The model receives 135
tokens describing what was found.

That ratio is the whole argument. Everything below is an attempt to generalize it.

---

## 2. Three categories of skill

Skills sort into three buckets, and the bucket determines the implementation:

| # | Category | Shape | Implementation | Cost to context |
|---|---|---|---|---|
| 1 | **Easy offload** | fully mechanical, returns facts | a tool, one call | ~40 tokens of signature |
| 2 | **Streamable** | mechanical detection, per-item judgment | the router (§3) | ~920 tokens, constant |
| 3 | **Cannot offload** | no observable output, pure prompt shaping | stays in the prompt | full skill text, always resident |

Examples: `pylint`, `pytest`, `git log` are category 1 — there is nothing to
decide, only to report. `ponytail-review`, `ponytail-audit`, and security triage
are category 2 — a program finds the candidates, a model rules on each.
`caveman` is category 3: it shapes how output reads and has nothing to detect,
so there is no system-side work to hand off.

Category 3 is the honest ceiling. Those skills either cost their full token
price or do not run — no architecture recovers them. The useful consequence is
that it is worth *knowing* which of your skills are category 3, because those
are the ones to write a condensed local variant of rather than port verbatim.

### The underlying split

Categories 1 and 2 differ only in how much judgment survives detection. Both
divide into two parts that currently live in the same prompt:

- **Detection** — mechanical, decidable by a program. Finding the candidates.
- **Judgment** — needs a model. Deciding what a candidate means and what to do.

Today both cost context. The leverage is moving detection into system space and
spending the model's limited window only on judgment.

| Skill | Detection → system | Judgment → model |
|---|---|---|
| pylint | all of it | which findings matter here |
| ponytail-review | unused imports, dead code, single-implementation classes, hand-rolled stdlib, wrappers that only delegate | "is this abstraction speculative, or load-bearing?" |
| tests | run them, collect failures | why it failed, what to change |
| security | grep for `shell=True`, `eval`, hardcoded secrets, unscoped writes | is it reachable, does it matter here |
| ponytail-debt | harvest every `ponytail:` comment | is this still worth doing |

`pylint` is 100% detection, which is why it is pure win — it should be a tool
(`lint_file`), not a skill. `ponytail` is roughly 60/40: a program can find the
candidates, but only a model can decide *does this need to exist at all*, because
that requires knowing intent.

---

## 3. The core idea: one finding at a time

The obvious implementation is batch — run the detector, hand the model a summary,
let it work. That saves the *reading*, but not the *acting*: to fix something the
model still needs enough context to edit correctly, and a 40-finding batch drags
the whole file back into the window.

The better shape is a **router that streams findings one at a time**:

```
detector  ──yields──▶  finding  ──▶  model decides  ──▶  router applies  ──▶  next
                                     fix / ignore / defer
```

Per iteration the model sees:

- the finding (kind, file, line)
- the relevant snippet, not the file
- whatever extra context the *detector* decided is needed (callers, definition site)
- three choices

Then the window resets and the next finding arrives.

**Why this is the important property:** context cost becomes constant per finding
instead of linear in the number of findings. A codebase with 500 findings works on
a 2048-token model exactly as well as one with 5. Nothing accumulates.

Rough budget per iteration, against the real numbers:

| Component | Tokens |
|---|---|
| System prompt | ~641 |
| One finding + snippet | ~150 |
| Verdict instruction | ~80 |
| Model reply | ~50 |
| **Total** | **~920** |

Fits inside `--low-vram` (2048) with room for a fix to be written, and it does not
grow.

---

## 4. The three verdicts

The model returns one of:

- **fix** — model writes the patch, router applies it via `edit_file`, re-runs the
  detector on that file to confirm the finding cleared.
- **ignore** — dropped, recorded so the same finding is not re-raised next run.
- **defer** — appended to a ledger with the model's one-line reason.

`defer` is the interesting one. It is exactly what `ponytail:` comments and
`ponytail-debt` already do by hand — a deliberate shortcut with a named ceiling.
Making it a first-class verdict means deferred work accumulates in a file rather
than evaporating.

---

## 5. Finding schema

Everything hinges on detectors emitting a common shape, so the router does not
care what produced a finding:

```python
{
  "id":       "pylint:broad-exception-caught:coding_agent.py:168",  # stable, for dedup
  "kind":     "broad-exception-caught",
  "severity": "low",
  "file":     "coding_agent.py",
  "line":     168,
  "snippet":  "    except Exception as e:\n        return {\"error\": str(e)}",
  "context":  "",          # detector-supplied: callers, definition site, if needed
  "message":  "Catching too general exception Exception",
  "suggest":  "narrow to OSError",   # optional, detectors may propose
}
```

A detector is then anything that yields these. `pylint` via `--output-format=json`
is an adapter of maybe 20 lines. A ponytail detector is a set of AST queries. A
grep-based security detector is a dict of patterns.

---

## 6. What it would take

Roughly, smallest thing that works:

| Piece | Effort | Notes |
|---|---|---|
| Finding schema + dedup by `id` | small | just a dict and a set |
| `pylint` adapter | ~20 lines | JSON output already structured |
| Router loop (yield → ask → apply → next) | ~60 lines | the real work |
| Verdict parsing | small | reuse `extract_tools`; verdicts are just tool calls |
| Ledger file for `defer` | ~15 lines | append-only markdown |
| State file for resume | ~20 lines | which findings are settled |
| Ponytail AST detectors | largest | one query per rung, incremental |

The pylint path is a weekend. The ponytail path is open-ended — each heuristic is
independent, so it can grow one detector at a time and be useful from the first.

Two things fall out of existing decisions in this repo:

- **Verdicts reuse the tool-call parser.** `fix(...)`, `ignore(...)`, `defer(...)`
  are the same shape as `write_file(...)`, so `extract_tools` already handles them.
  No new parsing.
- **State must survive a crash.** A router that loses its place on a tool crash
  and re-asks 200 settled findings is worse than useless. The state file is not
  optional.

---

## 7. The design rule that makes it work

**Tools aggregate, never dump.**

That is the entire difference between 135 tokens and 8,929 — same information, one
groups and counts before returning.

`run_command` currently does `result.stdout[:4000]`, which is dumb truncation: it
keeps the first 4000 characters and silently discards whatever mattered. The
upgrade is structured summarization — counts, top-N, and a `detail` argument so the
model can request specifics on *one* item once it knows which it cares about.

Cheap overview, detail on demand.

---

## 8. What does not port — category 3 in detail

Behavioral guidance with no observable output. `caveman` shapes how output reads
and has nothing to detect. Ponytail's first rung, *does this need to exist at
all*, is the same: it requires knowing what the user actually wanted, which no
static analysis recovers.

Note that a single skill can straddle categories. Ponytail is mostly 2 — the
ladder's middle rungs (already in the codebase? stdlib does it? one line?) are
all detectable — but its first rung is 3. The router captures the mechanical
part and leaves the rest in the prompt. That is a feature, not a compromise:
the 60% that ports is the 60% that is tedious to do by hand.

For genuine category 3 skills the only lever is *length*. A local variant of
ponytail is the ladder plus "shortest diff wins" — roughly 120 tokens against
the 595 the full `ponytail-review` skill costs. Write condensed variants rather
than porting frontier-model skill files verbatim.

---

## 9. Risks worth naming up front

- **A snippet may not be enough to fix correctly.** The most common failure mode:
  the model patches the line it was shown and breaks a caller it never saw. This is
  the detector's job to mitigate — a finding about a shared function must carry its
  callers in `context`. If detectors are lazy here, the router produces confident
  wrong fixes, which is worse than no router.
- **A 7B's verdict quality is unproven.** Three-way classification on a single
  finding is a much easier task than holistic review, which is the point — but it
  still needs measuring before trusting `fix` to auto-apply. Start with `defer` as
  the default and require explicit `fix`.
- **Detector precision matters more than recall.** 200 findings of which 150 are
  noise trains the operator to rubber-stamp. Better to ship four detectors that are
  almost always right than twenty that are usually wrong.
- **Auto-apply needs the same gate as `run_command`.** A router that edits files
  unattended is a bigger blast radius than a single confirmed command. The existing
  write-scoping applies, but `fix` should be confirmable.

---

## 10. Why this matters for the actual goal

The point of a small local model is not that it is as capable as a large one — it
is that capability spent on the right thing is enough. Context spent holding a
linter's instruction manual is context not spent reasoning.

Offloading detection means the model gets to be *precise about one finding* instead
of *overwhelmed by a thousand lines*. That is the same instinct as the bash
passthrough, applied one level up: use the system for what it is good at —
exhaustive, mechanical, cheap — and the model for what it is good at — deciding
what a thing means and whether it matters.

---

## Next concrete step

`lint_file` as a tool. `pylint` is already installed, the adapter is ~25 lines, and
it returns the 135-token summary measured in §1. It is useful standalone and it is
the first detector the router would consume.

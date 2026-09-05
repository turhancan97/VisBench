# The leaderboard

Every probe against every backbone, in one place:
[`LEADERBOARD.md`](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md).

It is **generated** from
[`results/corpus/visbench.jsonl`](https://github.com/turhancan97/VisBench/blob/main/results/corpus/visbench.jsonl)
by
[`scripts/render_tables.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/render_tables.py),
and a test in the fast suite fails if it drifts from the records — so a number
there and the run behind it cannot disagree. The same generator fills the board
on each probe page from the same corpus.

## Read this before quoting any of it

The full rules are in {doc}`how to read a board </guides/reading-a-board>`.
The three that most often go wrong:

**"Which backbone is best" is not a well-formed question against this corpus.**
`mae_vitb16` is first on six of the sixteen boards and last on four. A summary
that picks a winner is discarding the result.

**A count over a corpus is a fact about that corpus, not about a backbone.**
Three of MAE's counts have moved without its features changing — twice because
a column was added, once because a *board* was. Re-read a count off
`LEADERBOARD.md` rather than out of prose, including the prose above.

**A ceiling travels beside a score and must never be ranked on.** `ceiling_*`
says what the feature grid made *available*, not what the backbone recovered,
and because it falls with the grid, ranking on it would rank feature resolution
directly.

## What may sit in one table at all

`comparability_key` decides, and it is stricter than "same probe": the task, the
level, the `protocol` string, the dataset and its fingerprint, the split, the
requested pooling, the feature mode and the resolved layers all have to match.

That strictness is the point of the *Bench* half of the name. Two consequences
worth knowing:

- **A second dataset under an existing probe name does not merge into its
  board — it makes that board unrenderable.** `board_for` refuses a task with
  more than one comparability group, which is why scene and fine-grained
  classification are distinct probe *names* rather than flags.
- **`results/controls/` is rankable and still excluded.** Those records pass
  `comparability_key` against the boards they were run to explain, and answer a
  different question: the corpus says what a backbone scores, a control says
  what changes when one thing about one backbone moves. Nothing there feeds a
  generated table.

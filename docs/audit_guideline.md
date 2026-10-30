# Filter-audit annotation guideline

**Goal.** You are labelling GitHub issues so we can measure how well the
OpenRE-Bench mining filter separates *requirements* from *non-requirements*.
Fill the `label_rater1` (or `label_rater2`) column in
`docs/precision_audit_sample.csv`. For the eight disagreements, fill
`adjudicated` with a final `requirement` / `not` call. Then run
`uv run python scripts/audit_stats.py` to refresh the paper's numbers
(primary rates prefer `adjudicated`; consensus-only macros are kept too).

The sheet has 80 rows in two strata (do **not** treat them differently while
labelling — judge each issue on its own text):

- `retained` (50 rows): issues that survived all seven filter stages.
- `signal_dropped` (30 rows): issues the requirement-signal stage removed.

## The one question to answer

> Does this issue describe a **requirement** — a request to add or change a
> capability of the software (a new feature, enhancement, or behavioural
> change the author wants the system to support)?

Label each row exactly one of:

- `requirement` — yes, it asks for a new/changed capability.
- `not` — no (see below). (`no`, `non`, `0` are also accepted.)
- leave blank or `?` if you genuinely cannot tell (counts as unlabelled).

Judge from the **title and body text shown in the sheet**, not from whether the
issue was later accepted, labelled, or merged. Ignore how it is phrased: an
issue is still a requirement even if it never uses the words
"should/must/need/support".

### Count as `requirement`

- A concrete request to add functionality, an option, an API, a format, a
  model, a command, etc.
- A request to change existing behaviour to better serve a use case.
- A "it would be nice if the system could X" framed around a capability.

### Count as `not`

- A **bug/defect/regression** report (something is broken and should be fixed,
  with no new capability requested).
- A **support/usage question** ("how do I …?") or troubleshooting.
- **Pure design discussion / meta** with no concrete capability ask
  (roadmap chatter, "thoughts?", governance, CI/release housekeeping).
- **Duplicate or empty** content, or a tracking/umbrella issue with no
  substance of its own.

Edge cases: a "bug" whose real content is "the system should additionally
support X" is a `requirement`. A feature-tagged issue that is actually a
question is `not`. When an issue mixes a bug fix and a feature ask, label by
its **primary** intent.

## Worked examples (real rows from the sheet)

1. `huggingface/transformers#23036` — "[New model] Bark for realistic
   text-to-speech". Asks to add support for a new model. → **requirement**.
2. `elastic/elasticsearch#156292` — "Add a meta to indicate that the results
   are really approximated". Asks for a new response indicator. →
   **requirement**.
3. `scikit-learn/scikit-learn#12076` — "StratifiedGroupShuffleSplit … It would
   be nice to have a manner to perform GroupShuffleSplit with some level of
   stratification." A concrete new-capability request — even though it uses no
   signal keyword, so the filter dropped it. → **requirement** (a
   false-negative of the heuristic; this is exactly what the rejected-set audit
   is meant to catch).
4. A reply that only says "I'm hitting the same error, any workaround?" with no
   capability ask → **not** (support question).

## What the numbers mean

- **Retained-set precision** = fraction of `retained` rows you labelled
  `requirement`. High precision means the filter keeps mostly real
  requirements.
- **Rejected-set false-negative rate** = fraction of `signal_dropped` rows you
  labelled `requirement`. This estimates how many genuine requirements the
  requirement-signal stage threw away.
- **Cohen's kappa** and the **disagreement count** summarise how consistently
  the two raters applied this guideline. Point estimates use the rows where the
  two raters agree.

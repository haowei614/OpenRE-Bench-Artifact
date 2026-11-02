# Filtered issue pool — schema

`filtered_issue_pool.jsonl` is the issue-level dataset behind OpenRE-Bench's
mined cases. It contains the **1396 GitHub issues that survived the active
filter stages** of the mining pipeline (`ingestion/`), across five projects
(v1.0 default: requirement-signal gate **off**). Of these, **18** were
aggregated into the five `CaseInput` cases in `data/case_studies/`.

One JSON object per line (JSONL, UTF-8). Regenerate with
`uv run python scripts/build_filtered_pool.py`.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `repo` | string | `owner/name` of the source project. |
| `slug` | string | Short project id used in case names (`sklearn`, `transformers`, `vscode`, `elasticsearch`, `pandas`). |
| `domain` | string | Domain label from `ingestion/config.yaml`. |
| `number` | int | GitHub issue number. |
| `url` | string | Canonical issue URL (`html_url`); the authoritative source of the original text. |
| `title` | string | Issue title, de-identified (see below). |
| `labels` | list[string] | Issue labels at fetch time. |
| `state` | string | `open` / `closed`. |
| `created_at` | string | ISO-8601 creation timestamp. |
| `closed_at` | string \| null | ISO-8601 close timestamp, or null if open. |
| `comments` | int | Comment count at fetch time. |
| `reactions_plus_one` | int | `+1` reaction count. |
| `reactions_heart` | int | `heart` reaction count. |
| `reactions_total` | int | Total reaction count. |
| `body` | string | Issue body, de-identified (see below). |
| `passed_stages` | list[string] | Filter stages this issue passed, in order. With the v1.0 default (`requirement_signal: false`) the list is `label_allow_list`, `state_reactions_gate`, `min_body_length`, `bug_exclusion`, `mega_thread_cap`, `tfidf_dedup`. When the gate is enabled for ablation, `requirement_signal` is inserted after `bug_exclusion`. |
| `selected_into_case` | string \| null | Case name (e.g. `sklearn_2021q4`) if this issue was aggregated into a `CaseInput`; null otherwise. 18 issues are non-null. |

## Provenance and counts

- Snapshot: the five raw API captures under `ingestion/snapshots/`
  (`issues_<slug>_<timestamp>.json`), fetched 2026-09-29.
- Per-project retained counts (v1.0): sklearn 261, transformers 358, vscode 258,
  elasticsearch 150, pandas 369 (= 1396). These match the `dedup` column of the
  mining funnel (paper Table II; `ingestion/filtered/*_filtered.json`).
- The full per-stage funnel (including the issues each stage dropped) lives in
  `ingestion/filtered/<slug>_filtered.json` under `funnel` and `rejections`.

## De-identification

`title` and `body` have GitHub author @-mentions replaced with `@USER`
(`ingestion/_textproc.py:anonymize_mentions`), including inside code spans. A
small allowlist of code/markup tokens (e.g. `@timestamp`, `@param`,
`@property`) is left verbatim so technical text is not corrupted. The body is
otherwise the issue text as fetched from the GitHub API.

## Licensing

The original issue text is the intellectual property of its authors and GitHub;
it is redistributed here for **research use**, with the source `url` retained
for every record so readers can consult the canonical original. The curation,
field selection, stage labels, and de-identification are released under
CC BY 4.0. See `DATA_LICENSE` for details.

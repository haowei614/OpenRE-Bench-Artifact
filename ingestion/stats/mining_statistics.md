# OpenRE-Bench Mining Statistics

Per-project issue funnel from raw issues to the final aggregated case. Each column after `raw` shows survivors after that filter stage.

> `Raw` is the fetched issue sample (most-recent label-matching issues, capped at `github.max_issues_per_project`), not the project's all-time issue total. The `Label` drop is typically 0 because the API query already restricts to the configured labels.

| Project | Domain | Raw | Labels | State | Length | Bug | Signal | Discussion | Dedup | Final Issues | Words | Window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scikit-learn/scikit-learn | ML Library | 600 | 600 | 425 | 344 | 267 | 267 | 261 | 261 | 4 | 400 | 2021q4 |
| huggingface/transformers | ML / NLP | 600 | 600 | 423 | 380 | 367 | 367 | 363 | 358 | 4 | 377 | 2024q3 |
| microsoft/vscode | Editor / DevTool | 600 | 600 | 399 | 304 | 258 | 258 | 258 | 258 | 3 | 323 | 2026q3 |
| elastic/elasticsearch | Database / Search | 600 | 600 | 203 | 164 | 150 | 150 | 150 | 150 | 4 | 384 | 2026q1 |
| pandas-dev/pandas | Data Processing | 600 | 600 | 446 | 442 | 370 | 370 | 370 | 369 | 3 | 328 | 2023q3 |

## Issues Dropped by Each Filter

| Project | Label | State | Length | Bug | Signal | Discussion | Dedup |
|---|---|---|---|---|---|---|---|
| scikit-learn/scikit-learn | 0 | 175 | 81 | 77 | 0 | 6 | 0 |
| huggingface/transformers | 0 | 177 | 43 | 13 | 0 | 4 | 5 |
| microsoft/vscode | 0 | 201 | 95 | 46 | 0 | 0 | 0 |
| elastic/elasticsearch | 0 | 397 | 39 | 14 | 0 | 0 | 0 |
| pandas-dev/pandas | 0 | 154 | 4 | 72 | 0 | 0 | 1 |

**Totals:** 3000 raw issues across 5 projects -> 18 issues in 5 final case(s).

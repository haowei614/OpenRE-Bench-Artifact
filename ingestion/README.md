# OpenRE-Bench Case Ingestion

Mines real requirements from GitHub feature-request issues and converts them
into OpenRE-Bench cases that the existing harness can run unchanged. The full
methodology (selection criteria, filters, aggregation, thresholds and their
rationale) is documented in [`../docs/ingestion_design.md`](../docs/ingestion_design.md).

## What it produces

- 3–5 mined cases in `data/case_studies/<case_name>_input.json` (same 3-field
  `CaseInput` schema as the existing cases).
- A provenance sidecar `data/mined_cases_metadata.json` linking each case to its
  source GitHub issues.
- A mining statistics table (`stats/mining_statistics.csv` and `.md`) showing
  the per-project funnel from raw issues to the final case.
- Frozen raw snapshots in `snapshots/` (with `fetched_at`) for reproducibility.

## Prerequisites

- The repo environment (`uv sync --all-groups`). No extra dependencies: the
  pipeline uses only the standard library plus `PyYAML` and `scikit-learn`,
  which are already available in the locked environment.
- A **GitHub token** for the fetch steps (steps 1–2). All later steps run
  offline on cached snapshots.

### Getting and providing a token

Create a token with read-only access to public repositories:

- Fine-grained: <https://github.com/settings/personal-access-tokens/new> →
  Repository access: *Public Repositories (read-only)*.
- or Classic: <https://github.com/settings/tokens/new> → scope `public_repo`.

Provide it **without committing it** using either:

```bash
# Option A: inline on the command (token never persisted)
GITHUB_TOKEN=<your_token> uv run python ingestion/fetch_issues.py

# Option B: gitignored local file (auto-read by the scripts)
echo "GITHUB_TOKEN=<your_token>" > ingestion/.env.local
```

`ingestion/.env.local` is gitignored. The token is never written to snapshots,
logs, or committed files.

## Running the pipeline

End to end (steps 1–7):

```bash
GITHUB_TOKEN=<your_token> uv run python ingestion/run_all.py
```

Offline replay from committed snapshots (no token, no network):

```bash
uv run python ingestion/run_all.py --skip-fetch
```

Or run steps individually:

| Step | Script | Needs token | Output |
|------|--------|-------------|--------|
| 1 | `fetch_projects.py` | yes | `snapshots/projects_*.json` |
| 2 | `fetch_issues.py` | yes | `snapshots/issues_<slug>_*.json` |
| 3 | `filter_issues.py` | no | `filtered/<slug>_filtered.json` |
| 4 | `aggregate_cases.py` | no | `aggregated/<slug>_aggregated.json` |
| 5 | `convert_to_schema.py` | no | `data/case_studies/<case>_input.json`, metadata |
| 6 | `validate_cases.py` | no | validates against `CaseInput` |
| 7 | `build_stats.py` | no | `stats/mining_statistics.{csv,md}` |

Example single step:

```bash
uv run python ingestion/filter_issues.py --config ingestion/config.yaml
```

## Configuration

All thresholds, projects, filters, aggregation knobs, and the random seed live
in [`config.yaml`](config.yaml). Change behaviour there, not in code. To swap a
project, edit the `projects:` list (owner/repo, slug, domain, query labels).

## Verification (smoke test)

Run one mined case through all three adapters under the cheapest setting and
record runtime / token estimates / cost:

```bash
# no-key deterministic smoke path (single_agent needs no LLM credentials)
uv run python ingestion/smoke_test.py --case <case_name> --setting single_agent
```

Writes `artifacts/smoke/cost_report.md` and per-adapter artifacts. For
LLM-backed settings, configure `OPENAI_API_KEY` / a local endpoint as described
in the top-level `README.md`.

## Reproducibility notes

- Every source of randomness (dedup vectorizer, any sampling) uses
  `reproducibility.random_seed` from `config.yaml` (default 42).
- Aggregation (`time_window`) is deterministic: best quarter by count (ties →
  earliest), then top issues by reactions (ties → issue number ascending).
- Snapshots record `fetched_at`; re-running fetch overwrites with a new
  timestamped file, while `--skip-fetch` reuses the committed snapshots.

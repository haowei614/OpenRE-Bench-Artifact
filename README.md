# OpenRE-Bench

> A reproducible, extensible benchmarking infrastructure for comparing
> multi-agent Requirements Engineering (RE) frameworks under identical
> conditions, with a GitHub issue-mining pipeline that grows the case corpus
> from real feature requests.

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License: AGPL v3](https://img.shields.io/badge/license-AGPLv3-green)
![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)

OpenRE-Bench runs heterogeneous LLM multi-agent RE frameworks behind one shared
contract (`CaseInput` → Phase 1–4 artifacts → run record), scores them with a
common metric suite, and lets a new framework join by implementing a single
adapter. It originated as the evaluation back-end of the RE 2026 paper **QUARE**
and is extended here with a reproducible **issue-mining pipeline**, five **mined
cases**, a **filter audit**, and **reference runs on the mined cases**.

## Artifact Location

- Source repository: <https://github.com/haowei614/OpenRE-Bench-Artifact>
- Software citation: [`CITATION.cff`](CITATION.cff)

### Version map

| Git ref | Paper / use | Archival DOI |
|---------|-------------|--------------|
| tag `quare-2026` (= `v1.0.5-artifact`) | QUARE (IEEE RE 2026) artifact freeze | [10.5281/zenodo.20790776](https://doi.org/10.5281/zenodo.20790776) |
| tag `msr2027-v1.0` | MSR extension: issue mining, filter audit, mined-case reference runs | [10.5281/zenodo.23875694](https://doi.org/10.5281/zenodo.23875694) |
| tag `msr2027-v1.1` | MSR 2027 submission: v1.0 plus the 16-project scale demonstration | [10.5281/zenodo.24111289](https://doi.org/10.5281/zenodo.24111289) |
| branch `msr2027` | Development tip of the MSR extension | — |

The `quare-2026` tag is frozen and must remain untouched. Reviewers of the MSR
submission should use tag `msr2027-v1.1`, which the paper cites.

## Quick Offline Reproduction (no API key, no cost)

This path needs **no API key** and incurs **no cost**. After a one-time
environment sync it runs in seconds (well under 10 minutes end to end). Each
command is also runnable under Docker (see below).

```bash
uv sync --all-groups                       # pinned Python 3.13 environment

# (i) Re-derive the five mined cases offline from the committed API snapshots
#     (deterministic, seed 42; no GitHub token needed):
uv run python ingestion/run_all.py --skip-fetch

# (ii) Deterministic, key-free smoke test of the harness on one mined case:
uv run openre_bench --run-case \
  --case-input data/case_studies/vscode_2026q3_input.json \
  --artifacts-dir artifacts/smoke-quare \
  --run-record artifacts/smoke-quare/run_record.json \
  --system quare --setting single_agent

# (iii) Recompute the paper's reference-runs table from the committed run
#       outputs (no LLM calls):
uv run python scripts/recompute_reference_table.py
```

- (i) prints the seven-stage funnel and writes the five `*_input.json` cases;
  the only file that changes on replay is the timestamp in
  `data/mined_cases_metadata.json`.
- (ii) prints `Validation: passed` and writes Phase 1–4 artifacts under
  `artifacts/smoke-quare/`.
- (iii) prints MARE/iReDev/QUARE means ± SD for the
  `negotiation_integration_verification` setting (5 cases × 3 seeds, n = 15),
  matching the paper's reference-runs table.

### Under Docker

```bash
docker build -t openre-bench .
docker run --rm openre-bench uv run python ingestion/run_all.py --skip-fetch
docker run --rm openre-bench uv run python scripts/recompute_reference_table.py
docker run --rm -e HF_HOME=/tmp/openre-bench-hf-cache openre-bench uv run pytest
```

## Full Reviewer Checks

The locked environment targets Python 3.13. Python 3.14/free-threaded builds are
not recommended because transitive dependencies such as `tokenizers` may not
provide compatible wheels.

```bash
uv sync --all-groups
uv run openre_bench --version                 # -> 0.1.0
uv run ruff check .                           # passes
HF_HOME=/tmp/openre-bench-hf-cache uv run pytest   # 651 tests pass
```

The same checks run under Docker with the commands in the previous section
(the image copies `ingestion/` so the offline mining replay works in-container).

## Models and Cost

Two models appear in this artifact, used in two different experiments:

| Experiment | Model | Outputs |
|-----------|-------|---------|
| This paper's reference runs (mined + hand-built cases) | `claude-haiku-4-5-20251001` | `experiment_outputs/mined-claude/`, `experiment_outputs/mined-claude-s202303/`, `experiment_outputs/mined-claude-v1-sklearn-pandas/`, `experiment_outputs/handbuilt-claude/` |
| Originating QUARE (RE 2026) experiment (hand-built cases) | `gpt-4o-mini-2024-07-18` | `experiment_outputs/mare-iredev-quare/` |

`scripts/recompute_reference_table.py` merges the reference-runs table as:
unchanged cases from `mined-claude` + `mined-claude-s202303` (elasticsearch /
transformers / vscode), plus refreshed `sklearn_2021q4` and `pandas_2023q3`
rows from `mined-claude-v1-sklearn-pandas` (signal-off v1.0 case texts). The
older sklearn/pandas runs remain in the first two directories so the packaged
`usage_report.json` totals (180 runs / USD 10.96) stay accountable.

The harness is model-agnostic: any OpenAI- or Anthropic-compatible endpoint
works. Reference-run cost is recorded in each experiment's `usage_report.json`
(actual, not an estimate, from the packaged runs):

- Single-seed mined matrix: **60 runs, 752 LLM calls, USD 3.59**
  (`experiment_outputs/mined-claude/usage_report.json`).
- Full three-seed mined matrix (seeds 101/202/303): **180 runs, 2,266 calls,
  USD 10.96** (that file plus `mined-claude-s202303/usage_report.json`).

The key-free `single_agent` setting reproduces the harness and the mining
pipeline offline at **zero cost**.

## Running LLM-backed Settings

LLM-backed settings require model credentials. The harness uses LiteLLM, so the
hosted model is selected with `OPENAI_MODEL` and the key with `OPENAI_API_KEY`
(a git-ignored `.api_key` file takes priority); set `OPENAI_MODEL` to the
provider model name you want. The paper's reference runs used
`OPENAI_MODEL=claude-haiku-4-5-20251001`.

```bash
cp .env.example .env     # set OPENAI_API_KEY and OPENAI_MODEL, or use .api_key
uv run openre_bench --check-openai   # validate credentials
uv run openre_bench --run-case \
  --case-input data/case_studies/vscode_2026q3_input.json \
  --artifacts-dir artifacts/vscode-quare \
  --run-record artifacts/vscode-quare/run_record.json \
  --system quare --setting negotiation_integration_verification
```

Reproduce the full matrix (5 cases × 3 frameworks × 4 settings × 3 seeds = 180
runs):

```bash
uv run openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir experiment_outputs/mined-claude-fresh \
  --matrix-seeds 101,202,303 \
  --systems mare,iredev,quare \
  --matrix-workers 3
```

Any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio) also works for smoke
testing; `uv run openre_bench --llm-ping` returns `pong`. Local-model outputs
will not match the paper's hosted-model numbers.

## Review Paths (pre-generated evidence, no key)

- Reference-runs CSVs: `experiment_outputs/mined-claude*/comparison_metrics_*.csv`
- Per-run phase artifacts: `experiment_outputs/*/runs/`
- Mining funnel and per-stage decisions: `ingestion/filtered/`, `ingestion/stats/`
- Published issue pool (1396 filtered issues): [`data/filtered_issue_pool.jsonl`](data/filtered_issue_pool.jsonl) ([schema](data/filtered_issue_pool.schema.md))
- Filter-audit sample + guideline: [`docs/precision_audit_sample.csv`](docs/precision_audit_sample.csv), [`docs/audit_guideline.md`](docs/audit_guideline.md)
- Human-evaluation outputs (originating QUARE study): [`human_eval/analysis/`](human_eval/analysis/)

The pre-generated artifacts are paper evidence, not an alternative mechanism for
generating new LLM outputs.

## Scale Demonstration (16 further projects)

The unchanged pipeline (same filter settings as v1.0, requirement-signal gate off)
was run on 16 further projects, adding Go and Rust: 9,350 fetched issues, 4,600
after filtering, and 16 schema-valid cases in 437 s end to end (bounded by GitHub
API rate limits). This extension is **not** part of the audited v1.0 pool or the
v1.0 case suite, and it is kept in separate files:

- Config: [`ingestion/config_scale.yaml`](ingestion/config_scale.yaml)
- Funnel statistics: [`ingestion/stats/mining_statistics_scale.md`](ingestion/stats/mining_statistics_scale.md)
- Cases: [`data/case_studies_scale/`](data/case_studies_scale/) (metadata: `data/mined_cases_metadata_scale.json`)

Replay offline from the committed snapshots (no token needed):

```bash
uv run python ingestion/run_all.py --config ingestion/config_scale.yaml --skip-fetch
```

## Repository Map

| Path | Purpose |
|------|---------|
| [`src/openre_bench/`](src/openre_bench/) | Shared CLI, pipeline, schemas, LLM client, validators, comparison harness, and metric computation. |
| [`openre_bench_quare/`](openre_bench_quare/) | QUARE adapter (first-party). |
| [`openre_bench_mare/`](openre_bench_mare/) | MARE adapter — paper-faithful reimplementation (no public code exists). |
| [`openre_bench_iredev/`](openre_bench_iredev/) | iReDev adapter — paper-faithful reimplementation (no public code exists). |
| [`ingestion/`](ingestion/) | GitHub issue-mining pipeline: 7 steps, `config.yaml`, frozen API snapshots, per-stage funnel, precision-audit sampler. |
| [`data/case_studies/`](data/case_studies/) | Case inputs: 5 hand-built + 5 mined. |
| [`data/filtered_issue_pool.jsonl`](data/filtered_issue_pool.jsonl) | The 1396 issues surviving the v1.0 filter stages (issue-level dataset; requirement-signal gate off by default). |
| [`experiment_outputs/`](experiment_outputs/) | Reference runs (claude-haiku-4-5 on mined/hand-built; gpt-4o-mini QUARE experiment). |
| [`human_eval/`](human_eval/) | Human-evaluation traceability and agreement analyses (QUARE study). |
| [`scripts/`](scripts/) | Reproduction helpers (`recompute_reference_table.py`, `audit_stats.py`, `build_filtered_pool.py`, human-eval tooling). |
| [`docs/`](docs/) | CLI, artifact-format, methodology, ingestion design, and audit guideline. |
| [`tests/`](tests/) | 651-test regression and smoke suite. |

## Further Documentation

- CLI and runtime configuration: [`docs/reference/cli.md`](docs/reference/cli.md)
- Artifact formats and run records: [`docs/reference/artifacts.md`](docs/reference/artifacts.md)
- Mining pipeline design: [`ingestion/README.md`](ingestion/README.md), [`docs/ingestion_design.md`](docs/ingestion_design.md)
- Filter-audit annotation guideline: [`docs/audit_guideline.md`](docs/audit_guideline.md)
- Human-evaluation navigation: [`human_eval/README.md`](human_eval/README.md)

## License

Code: **AGPL-3.0-or-later** ([`LICENSE`](LICENSE)). Data: see
[`DATA_LICENSE`](DATA_LICENSE) — our curated annotations, metadata, and
generated outputs are **CC BY 4.0**; embedded original GitHub issue text remains
the authors' IP, redistributed for research use with source URLs retained and
author @-mentions de-identified. Intellectual property for each referenced paper
remains with its respective authors.

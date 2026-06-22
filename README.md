# OpenRE-Bench

> A reproducible evaluation harness for comparing multi-agent Requirements
> Engineering frameworks under identical experimental conditions.

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License: AGPL v3](https://img.shields.io/badge/license-AGPLv3-green)

OpenRE-Bench is the artifact for the RE 2026 paper **"QUARE:
Quality-Aware Requirements Analysis through Multi-Agent Dialectical
Negotiation"**. It evaluates QUARE against reimplementations of MARE and iReDev
using the same case-study inputs, model configuration, phase contracts,
provenance records, and evaluation scripts.

The repository includes both executable code and pre-generated outputs, so the
artifact can be reviewed either by inspecting the packaged evidence or by
rerunning selected experiments.

## Artifact Location

- Source repository: <https://github.com/haowei614/OpenRE-Bench-Artifact>
- Archival DOI: <https://doi.org/10.5281/zenodo.20482393>
- Archival URL: <https://zenodo.org/records/20482393>
- Software citation: [`CITATION.cff`](CITATION.cff)

## Reviewer Quick Start

The locked environment targets Python 3.13. Python 3.14/free-threaded builds are
not recommended because transitive dependencies such as `tokenizers` may not
provide compatible wheels.

```bash
uv sync --all-groups
uv run openre_bench --version
uv run ruff check .
HF_HOME=/tmp/openre-bench-hf-cache uv run pytest
```

If local Python setup is inconvenient, use Docker:

```bash
docker build -t openre-bench .
docker run --rm openre-bench
docker run --rm -e HF_HOME=/tmp/openre-bench-hf-cache openre-bench uv run pytest
```

Verified checks for this revision:

- `uv run openre_bench --version` outputs `0.1.0`
- `uv run ruff check .` passes
- `HF_HOME=/tmp/openre-bench-hf-cache uv run pytest` passes with 651 tests
- Docker build and containerized pytest pass on Python 3.13

## Review Paths

No API key is required to inspect the submitted evidence:

- Paper-table CSVs: [`experiment_outputs/mare-iredev-quare/paper_tables/`](experiment_outputs/mare-iredev-quare/paper_tables/)
- Per-run phase artifacts: [`experiment_outputs/mare-iredev-quare/runs/`](experiment_outputs/mare-iredev-quare/runs/)
- Human-evaluation outputs: [`human_eval/analysis/`](human_eval/analysis/)
- Human-evaluation navigation guide: [`human_eval/README.md`](human_eval/README.md)

This is a non-execution verification path. The pre-generated artifacts are
paper evidence, not an alternative mechanism for generating new LLM outputs.

To exercise the harness with a local LLM, use an OpenAI-compatible endpoint such
as Ollama, vLLM, or LM Studio:

```bash
OPENAI_API_KEY=dummy \
OPENAI_MODEL='openai/qwen3.5:9b' \
OPENAI_BASE_URL='http://localhost:11434/v1' \
OPENAI_TIMEOUT_SECONDS=180 \
uv run openre_bench --llm-ping
```

This path was tested with Ollama and returned `pong`. Local model outputs are
useful for smoke testing the harness, but they are not expected to exactly match
the paper's `gpt-4o-mini-2024-07-18` results.

To use a hosted OpenAI-compatible provider:

```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY, OPENAI_MODEL, and optional OPENAI_BASE_URL
uv run openre_bench --check-openai
uv run openre_bench --llm-ping
```

## Reproduction Commands

Run one framework on one case:

```bash
uv run openre_bench --run-case \
  --case-input data/case_studies/ATM_input.json \
  --artifacts-dir artifacts/atm-quare \
  --run-record artifacts/atm-quare/run_record.json \
  --system quare
```

Replace `quare` with `mare` or `iredev` to exercise another adapter.

Run the full paper matrix:

```bash
uv run openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir experiment_outputs/mare-iredev-quare \
  --matrix-seeds 101,202,303 \
  --systems mare,iredev,quare \
  --matrix-workers 3
```

The matrix contains 5 case studies, 3 frameworks, 4 settings, and 3 seeds, for
180 runs. Use the pre-generated outputs for artifact review when time, quota, or
budget is limited.

## Cost Estimate

The packaged `run_record.json` files do not store provider-reported token usage
or billed cost, so cost values are estimates rather than audited billing records.
Using `gpt-4o-mini-2024-07-18` pricing at the time of packaging as an example:

```text
cost_usd = input_tokens / 1_000_000 * 0.15
         + output_tokens / 1_000_000 * 0.60
```

A single smoke run is expected to stay below USD 0.10. For the full 180-run
matrix, budget roughly USD 5-25 depending on retries, provider rounding, and
output length. Recompute with current provider prices before rerunning.

## Repository Map

| Path | Purpose |
|------|---------|
| [`src/openre_bench/`](src/openre_bench/) | Shared CLI, pipeline, schemas, LLM client, validators, and comparison harness. |
| [`openre_bench_quare/`](openre_bench_quare/) | QUARE adapter for the accompanying paper. |
| [`openre_bench_mare/`](openre_bench_mare/) | MARE baseline reimplementation. |
| [`openre_bench_iredev/`](openre_bench_iredev/) | iReDev baseline reimplementation. |
| [`data/case_studies/`](data/case_studies/) | Shared case-study input specifications. |
| [`data/ground_truth/`](data/ground_truth/) | Empty by design; see its README for why no gold-standard outputs are claimed. |
| [`experiment_outputs/mare-iredev-quare/`](experiment_outputs/mare-iredev-quare/) | Pre-generated run artifacts, comparison metrics, and paper-table CSV files. |
| [`human_eval/`](human_eval/) | Human-evaluation traceability files, item-level scores, and agreement analyses. |
| [`docs/`](docs/) | Detailed CLI, artifact-format, methodology, and guide documentation. |
| [`tests/`](tests/) | Regression and smoke tests for the harness and adapters. |

## Paper Claims to Artifact Mapping

| Paper Reference | Artifact Location |
|-----------------|-------------------|
| Table II, Benchmark Systems | [`experiment_outputs/mare-iredev-quare/paper_tables/table_ii_benchmark_systems.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_ii_benchmark_systems.csv) |
| Table III, Coverage and Diversity | [`experiment_outputs/mare-iredev-quare/paper_tables/table_iii_coverage_diversity.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_iii_coverage_diversity.csv) |
| Table IV, Semantic Preservation | [`experiment_outputs/mare-iredev-quare/paper_tables/table_iv_semantic_preservation.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_iv_semantic_preservation.csv) |
| Table V, Negotiation Summary | [`experiment_outputs/mare-iredev-quare/paper_tables/table_v_negotiation_summary.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_v_negotiation_summary.csv) |
| Table VI(A), Structural Correctness | [`experiment_outputs/mare-iredev-quare/paper_tables/table_vi_a_structural_compliance.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_vi_a_structural_compliance.csv) |
| Table VI(B), LLM-Judge Quality Scores | [`experiment_outputs/mare-iredev-quare/paper_tables/table_vi_b_llm_judge_quality.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_vi_b_llm_judge_quality.csv) |
| Table VI(C), Human Evaluation | [`human_eval/analysis/`](human_eval/analysis/) and [`experiment_outputs/mare-iredev-quare/paper_tables/table_vi_c_human_llm_validation.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_vi_c_human_llm_validation.csv) |
| Negotiation traces | `experiment_outputs/mare-iredev-quare/runs/quare-*/phase2_negotiation_trace.json` |

The summary report
[`MARE-iReDev-QUARE-comparison-report.md`](MARE-iReDev-QUARE-comparison-report.md)
provides a compact overview of the paper-reported experimental results.

## Further Documentation

- CLI and runtime configuration: [`docs/reference/cli.md`](docs/reference/cli.md)
- Artifact formats and run records: [`docs/reference/artifacts.md`](docs/reference/artifacts.md)
- NLP4RE ID-Card summary: [`docs/NLP4RE-ID-CARD.md`](docs/NLP4RE-ID-CARD.md)
- Human-evaluation navigation: [`human_eval/README.md`](human_eval/README.md)
- Ground-truth scope: [`data/ground_truth/README.md`](data/ground_truth/README.md)
- Precision/F1 scope: [`docs/guides/precision-f1.md`](docs/guides/precision-f1.md)

## License

AGPL-3.0-or-later. Intellectual property for each referenced paper remains with
its respective authors.

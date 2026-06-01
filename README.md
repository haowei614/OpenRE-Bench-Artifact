# OpenRE-Bench

> A unified replication harness for evaluating multi-agent Requirements
> Engineering frameworks under identical experimental conditions.

![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)
![License: AGPL v3](https://img.shields.io/badge/license-AGPLv3-green)

## Overview

OpenRE-Bench (**Open** **R**equirements **E**ngineering **Bench**mark) is the
artifact accompanying the paper:

> **QUARE: Quality-Aware Requirements Analysis through Multi-Agent Dialectical
> Negotiation**  
> *Conditionally accepted at IEEE RE 2026*

It provides faithful reimplementations of three multi-agent RE frameworks:
QUARE, MARE, and iReDev. All frameworks are executed under a shared pipeline
with identical model configuration, case inputs, phase contracts, provenance
metadata, and evaluation harness. This design isolates architectural differences
from confounding variables.

The repository is named OpenRE-Bench because the artifact is not only the QUARE
implementation; it is also the common replication and comparison harness used to
evaluate QUARE against baseline frameworks.

## Summary of Artifact

This artifact provides a reusable benchmark harness for evaluating multi-agent
Requirements Engineering frameworks under controlled experimental conditions. It
implements QUARE and reimplements two baseline frameworks, MARE and iReDev, so
that all systems can be executed over the same case-study inputs, LLM
configuration, phase contracts, provenance records, and evaluation scripts.

Expected inputs are JSON case-study descriptions under
[`data/case_studies/`](data/case_studies/). The primary outputs are structured
JSON phase artifacts, run records, comparison CSV files, paper-table CSV files,
and human-evaluation analysis outputs. Pre-generated outputs for the paper
experiments are included under
[`experiment_outputs/mare-iredev-quare/`](experiment_outputs/mare-iredev-quare/)
so that reviewers can inspect and verify the results without rerunning the full
LLM experiment matrix.

The motivation for the artifact is to support transparent comparison of QUARE
against related multi-agent RE approaches while minimizing confounds introduced
by different prompts, data formats, execution environments, or evaluation code.

## Authors Information

The artifact accompanies the paper:

> **QUARE: Quality-Aware Requirements Analysis through Multi-Agent Dialectical
> Negotiation**  
> *Conditionally accepted at IEEE RE 2026*

Authors should cite the paper and this archived artifact when reusing the
software, datasets, generated outputs, or evaluation materials.

**Author list:**

- Haowei Cheng, Waseda University, Tokyo, Japan
  (<haowei.cheng@fuji.waseda.jp>)
- Milhan Kim, Waseda University, Tokyo, Japan
  (<milhan.kim@ruri.waseda.jp>)
- Foutse Khomh, Polytechnique Montréal, Montréal, Canada
  (<foutse.khomh@polymtl.ca>)
- Teeradaj Racharak, Tohoku University, Miyagi, Japan
  (<racharak.teeradaj.c3@tohoku.ac.jp>)
- Nobukazu Yoshioka, Waseda University, Tokyo, Japan
  (<nobukazu@aoni.waseda.jp>)
- Naoyasu Ubayashi, Waseda University, Tokyo, Japan
  (<ubayashi@aoni.waseda.jp>)
- Hironori Washizaki, Waseda University, Tokyo, Japan
  (<washizaki@waseda.jp>)

**Software citation:** A machine-readable citation template is provided in
[`CITATION.cff`](CITATION.cff).

## Artifact Location

- Source repository:
  <https://github.com/haowei614/OpenRE-Bench-Artifact>
- Archival DOI: <https://doi.org/10.5281/zenodo.20482393>
- Archival URL: <https://zenodo.org/records/20482393>

The artifact should be evaluated from the archived DOI version for the final
submission. The GitHub repository is provided as a development mirror and source
code landing page.

## Implemented Frameworks

| Framework | Adapter Package | Agents | Strategy | Paper |
|-----------|----------------|--------|----------|-------|
| **QUARE** | [`openre_bench_quare/`](openre_bench_quare/) | 5 quality-specialized + orchestrator | Dialectical negotiation by quality dimension | This paper |
| **MARE** | [`openre_bench_mare/`](openre_bench_mare/) | 5 task-specialized | 9-action pipeline, single-turn negotiation | MARE paper, 2024 |
| **iReDev** | [`openre_bench_iredev/`](openre_bench_iredev/) | 6 knowledge-driven | 17-action workflow, surrogate human-in-the-loop | iReDev paper, 2025 |

Each adapter is registered by the shared pipeline in
[`src/openre_bench/pipeline/_core.py`](src/openre_bench/pipeline/_core.py) and
is selected through `--system quare`, `--system mare`, or `--system iredev`.

## Repository Structure

```text
OpenRE-Bench/
├── openre_bench_quare/     # QUARE adapter (this paper)
├── openre_bench_mare/      # MARE reimplementation
├── openre_bench_iredev/    # iReDev reimplementation
├── src/openre_bench/       # Shared harness: CLI, pipeline, schemas,
│                           #   LLM client, validator, comparison matrix
├── data/
│   ├── case_studies/       # 5 benchmark inputs (AD, ATM, Library,
│   │                       #   RollCall, Bookkeeping)
│   └── knowledge_base/     # Domain standards for RAG verification
├── experiment_outputs/     # Pre-generated outputs, run artifacts, paper tables
├── human_eval/             # Human evaluation data and agreement analysis
├── scripts/                # Evaluation and export utilities
├── tests/                  # Regression tests for harness and adapters
└── docs/                   # Supplementary documentation
```

## Description of Artifact

The top-level directories have the following roles:

| Path | Description |
|------|-------------|
| [`openre_bench_quare/`](openre_bench_quare/) | QUARE implementation for the accompanying paper. |
| [`openre_bench_mare/`](openre_bench_mare/) | Reimplementation of the MARE baseline adapter. |
| [`openre_bench_iredev/`](openre_bench_iredev/) | Reimplementation of the iReDev baseline adapter. |
| [`src/openre_bench/`](src/openre_bench/) | Shared CLI, pipeline, schemas, LLM client, validators, and comparison harness. |
| [`data/case_studies/`](data/case_studies/) | Five benchmark inputs used by all framework adapters. |
| [`data/knowledge_base/`](data/knowledge_base/) | Domain and standards material used by verification steps. |
| [`experiment_outputs/`](experiment_outputs/) | Pre-generated run artifacts, comparison metrics, and paper-table CSV files. |
| [`human_eval/`](human_eval/) | Human-evaluation workbook, traceability files, and agreement analyses. |
| [`scripts/`](scripts/) | Utilities for exporting requirements, building human-evaluation materials, and analyzing results. |
| [`tests/`](tests/) | Regression and smoke tests for the harness, metrics, runtime routing, and adapters. |
| [`docs/`](docs/) | Supplementary methodology, CLI, artifact-format, and target-adapter documentation. |

Each run directory contains phase outputs and a provenance record documenting
the case, framework, configuration, seed, and aggregate metrics.

## System Requirements

- Operating system: tested on macOS and expected to run on Linux or other POSIX
  environments with Python support.
- Python: `>=3.14`.
- Package manager: [`uv`](https://docs.astral.sh/uv/) is recommended for
  reproducible installation from `pyproject.toml` and `uv.lock`.
- LLM access: an OpenAI-compatible API key is required for commands that
  generate new LLM outputs.
- Optional cache setting: set `HF_HOME=/tmp/openre-bench-hf-cache` if the local
  Hugging Face cache location is unavailable or not writable.

Reviewers can inspect the included paper outputs and human-evaluation evidence
without model credentials.

## Quick Start

1. Install dependencies:

   ```bash
   uv sync --all-groups
   ```

2. Check the command-line entry point:

   ```bash
   uv run openre_bench --version
   ```

3. Configure an OpenAI-compatible key, either by environment variable or local
   key file:

   ```bash
   cp .env.example .env
   # edit .env, or create .api_key with OPENAI_API_KEY=...
   ```

4. Verify model access:

   ```bash
   uv run openre_bench --check-openai
   uv run openre_bench --llm-ping
   ```

5. Run local checks:

   ```bash
   uv run ruff check .
   HF_HOME=/tmp/openre-bench-hf-cache uv run pytest
   ```

## Installation Instructions

Start from a fresh checkout or the archived artifact directory:

```bash
uv sync --all-groups
uv run openre_bench --version
```

For commands that call an LLM, configure an OpenAI-compatible API key:

```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY and any provider-specific settings
uv run openre_bench --check-openai
uv run openre_bench --llm-ping
```

If model credentials are unavailable, skip LLM execution and use the
pre-generated artifacts under
[`experiment_outputs/mare-iredev-quare/`](experiment_outputs/mare-iredev-quare/).

## Usage Instructions

The main entry point is the `openre_bench` command installed by the Python
package. The most useful reviewer commands are:

```bash
uv run openre_bench --version
uv run openre_bench --check-openai
uv run openre_bench --llm-ping
```

To run one case through one adapter:

```bash
uv run openre_bench --run-case \
  --case-input data/case_studies/ATM_input.json \
  --artifacts-dir artifacts/atm-quare \
  --run-record artifacts/atm-quare/run_record.json \
  --system quare
```

Replace `quare` with `mare` or `iredev` to run the same case through another
adapter. Generated artifacts are written to the selected `--artifacts-dir`, and
the run-level provenance and metrics are written to `--run-record`.

For detailed CLI reference, see [`docs/reference/cli.md`](docs/reference/cli.md).

## Artifact Evaluation Guide

For artifact evaluation, the package can be inspected at several levels
depending on available time and compute budget.

### Recommended Reviewer Workflow

| Goal | Location or Command | Expected Cost |
|------|---------------------|---------------|
| Inspect pre-generated experiment outputs | [`experiment_outputs/mare-iredev-quare/`](experiment_outputs/mare-iredev-quare/) | No execution required |
| Check paper-table values | [`experiment_outputs/mare-iredev-quare/paper_tables/`](experiment_outputs/mare-iredev-quare/paper_tables/) | No execution required |
| Inspect human-evaluation evidence | [`human_eval/`](human_eval/) | No execution required |
| Run one framework on one case | See "Single-system smoke test" below | Short run |
| Reproduce the full comparison matrix | See "Full 180-run comparison matrix" below | Long-running |

### Reproducibility Levels

| Level | Purpose | What to Inspect or Run |
|-------|---------|------------------------|
| 0 | Inspect artifact without executing code | Paper tables, pre-generated run artifacts, and human-evaluation outputs |
| 1 | Confirm the CLI and one adapter execute locally | Run the single-system smoke test with `--system quare`, `mare`, or `iredev` |
| 2 | Reproduce the three-framework experiment matrix | Run the 180-run comparison matrix |
| 3 | Audit individual runs | Open per-run JSON artifacts under `experiment_outputs/mare-iredev-quare/runs/` |

### Main Output Files

Each run directory contains the phase artifacts produced by the shared harness.

| File | Meaning |
|------|---------|
| `phase1_initial_models.json` | Initial framework-specific generated requirements, goals, or model elements |
| `phase2_negotiation_trace.json` | Negotiation, interaction, or workflow trace when applicable |
| `phase3_integrated_kaos_model.json` | Integrated KAOS-style requirement model used for downstream evaluation |
| `phase4_verification_report.json` | Structural, compliance, and quality-checking outputs when verification is executed |
| `run_record.json` | Run provenance, configuration, comparability flags, and aggregate metrics |

### Troubleshooting

| Issue | Suggested Action |
|-------|------------------|
| Missing model credentials | Create `.env` from `.env.example`, or create `.api_key` with `OPENAI_API_KEY=...` |
| Hugging Face cache or permission errors | Set `HF_HOME=/tmp/openre-bench-hf-cache` before running tests |
| Full matrix takes too long | Run the single-system smoke test first and inspect pre-generated outputs |
| LLM access is unavailable | Use the pre-generated outputs and paper-table CSVs for inspection |

## Steps to Reproduce

The recommended reproduction path is staged so that reviewers can verify the
artifact within the conference time budget and still have a path to full
re-execution when compute and API access are available.

1. Inspect the machine-readable paper tables under
   [`experiment_outputs/mare-iredev-quare/paper_tables/`](experiment_outputs/mare-iredev-quare/paper_tables/).
2. Inspect the full pre-generated comparison outputs under
   [`experiment_outputs/mare-iredev-quare/`](experiment_outputs/mare-iredev-quare/).
3. Inspect the human-evaluation workbook, traceability files, and agreement
   outputs under [`human_eval/`](human_eval/).
4. Run the local test suite with
   `HF_HOME=/tmp/openre-bench-hf-cache uv run pytest`.
5. If LLM credentials are available, run the single-system smoke test below.
6. If sufficient time, budget, and API quota are available, rerun the full
   180-run comparison matrix below.

Known runtime deviation: the full matrix involves 180 LLM-backed runs and may
exceed 60 minutes depending on provider latency, quota, and local parallelism.
For artifact review within 60 minutes, use the smoke test plus the included
pre-generated outputs and paper-table CSV files.

### Reproducing Paper Results

### Full 180-run comparison matrix

This command reproduces the three-framework matrix used for the main paper
tables.

```bash
uv run openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir experiment_outputs/mare-iredev-quare \
  --matrix-seeds 101,202,303 \
  --systems mare,iredev,quare \
  --matrix-workers 3
```

The matrix contains 5 case studies, 3 frameworks, 4 settings, and 3 random
seeds, for 180 runs.

### Single-system smoke test

```bash
uv run openre_bench --run-case \
  --case-input data/case_studies/ATM_input.json \
  --artifacts-dir artifacts/atm-quare \
  --run-record artifacts/atm-quare/run_record.json \
  --system quare
```

Replace `quare` with `mare` or `iredev` to run the same case through another
adapter.

### Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Model | `gpt-4o-mini-2024-07-18` |
| Temperature | `0.7` |
| Round cap | `3` |
| Random seeds | `101, 202, 303` |
| Case studies | AD, ATM, Library, RollCall, Bookkeeping |
| Matrix settings | `single_agent`, `multi_agent_without_negotiation`, `multi_agent_with_negotiation`, `negotiation_integration_verification` |

## Paper Claims to Artifact Mapping

| Paper Reference | Artifact Location |
|-----------------|-------------------|
| Table II (Benchmark Systems) | [`experiment_outputs/mare-iredev-quare/paper_tables/table_ii_benchmark_systems.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_ii_benchmark_systems.csv) |
| Table III (Coverage and Diversity) | [`experiment_outputs/mare-iredev-quare/paper_tables/table_iii_coverage_diversity.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_iii_coverage_diversity.csv) |
| Table IV (Semantic Preservation) | [`experiment_outputs/mare-iredev-quare/paper_tables/table_iv_semantic_preservation.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_iv_semantic_preservation.csv) |
| Table V (Negotiation Summary) | [`experiment_outputs/mare-iredev-quare/paper_tables/table_v_negotiation_summary.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_v_negotiation_summary.csv) |
| Table VI(A) (Structural Correctness) | [`experiment_outputs/mare-iredev-quare/paper_tables/table_vi_a_structural_compliance.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_vi_a_structural_compliance.csv) |
| Table VI(B) (LLM-Judge Quality Scores) | [`experiment_outputs/mare-iredev-quare/paper_tables/table_vi_b_llm_judge_quality.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_vi_b_llm_judge_quality.csv) |
| Table VI(C) (Human Evaluation) | [`human_eval/analysis/`](human_eval/analysis/) and [`experiment_outputs/mare-iredev-quare/paper_tables/table_vi_c_human_llm_validation.csv`](experiment_outputs/mare-iredev-quare/paper_tables/table_vi_c_human_llm_validation.csv) |
| Negotiation traces (Section V-B) | `experiment_outputs/mare-iredev-quare/runs/quare-*/phase2_negotiation_trace.json` |

The summary report
[`MARE-iReDev-QUARE-comparison-report.md`](MARE-iReDev-QUARE-comparison-report.md)
provides a compact overview of the paper-reported experimental results. The
machine-readable paper tables are indexed in
[`experiment_outputs/mare-iredev-quare/paper_tables/`](experiment_outputs/mare-iredev-quare/paper_tables/).
The corresponding experiment outputs are available at
[`experiment_outputs/mare-iredev-quare/`](experiment_outputs/mare-iredev-quare/).

## Human Evaluation

The human-evaluation package is located in [`human_eval/`](human_eval/). It
contains the annotation workbook, traceability mapping, item-level scores,
human-human agreement metrics, and human-vs-LLM agreement outputs used for the
LLM-as-a-judge validation.

## License

AGPL-3.0-or-later. Intellectual property for each referenced paper remains with
its respective authors.

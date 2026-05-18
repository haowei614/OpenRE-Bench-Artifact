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

## Implemented Frameworks

| Framework | Adapter Package | Agents | Strategy | Paper |
|-----------|----------------|--------|----------|-------|
| **QUARE** | [`openre_bench_quare/`](openre_bench_quare/) | 5 quality-specialized + orchestrator | Dialectical negotiation by quality dimension | This paper |
| **MARE** | [`openre_bench_mare/`](openre_bench_mare/) | 5 task-specialized | 9-action pipeline, single-turn negotiation | Jin et al., 2024 |
| **iReDev** | [`openre_bench_iredev/`](openre_bench_iredev/) | 6 knowledge-driven | 17-action workflow, surrogate human-in-the-loop | Jin et al., 2025 |

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

## Reproducing Paper Results

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

## Human Evaluation

The human-evaluation package is located in [`human_eval/`](human_eval/). It
contains the annotation workbook, traceability mapping, item-level scores,
human-human agreement metrics, and human-vs-LLM agreement outputs used for the
LLM-as-a-judge validation.

## License

AGPL-3.0-or-later. Intellectual property for each referenced paper remains with
its respective authors.

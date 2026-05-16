# OpenRE-Bench

OpenRE-Bench is independent benchmark software for complex LLM-debate based requirements engineering stages. It runs
equal paper-grounded target adapters with the same case inputs, phase artifact contract, reproducibility metadata, and
report gates.

## License

- OpenRE-Bench is licensed under the AGPLv3+.
- Intellectual property for each referenced paper and implemented idea remains with its owner.
  **Do not upload paper directly.**

## Summary

OpenRE-Bench's current supported benchmark targets are [**MARE**](./openre_bench_mare/__init__.py) , [**iReDev**](./openre_bench_iredev/__init__.py), and [**QUARE**](./openre_bench_quare/__init__.py).

| Question                              | Where to look                                             |
|---------------------------------------|-----------------------------------------------------------|
| How are targets kept comparable?      | [Shared Benchmark Contract](#shared-benchmark-contract)   |
| How do I reproduce the N-way matrix?  | [Reviewer Reproduction Path](#reviewer-reproduction-path) |
| How is this codebase laid out?        | [Repository Layout](#repository-layout)                   |
| Where are current results summarized? | `MARE-iReDev-QUARE-comparison-report.md`                  |

## Fair use guide

The benchmarks **must** be paper-grounded, but this repository doesn't rely on or provide local paper (i.e., PDF) paths
as entry points.

### How to add your idea/methodology/paper

Treat each existing methodologies (e.g., MARE, iReDev, QUARE, and more papers) and yours, equal, and equally harness
OpenRE-Bench comparison protocol as external method anchors, and treat this bench/repository as the **executable** artifact.

Do not adjust OpenRE-Bench's source code or the comparison harness only for your own verification; if modification is needed, mostly expansion, it should be applied to all idea/methodology/papers.

### Before/During peer review

We do not recommend including author's name/journal/conference name in the benchmark adaptor package names, source code,
or in the result for peer review. Feel free attaching this codebase and your executable as an artifact/reproducibility
in your paper but acknoledge the license model we have. e.g., be ready to share your OpenRE-Bench executable (more
importantly, its modification) with the reviewers.

Also, note LLM-based models are not guaranteed to be stable across runs, therefore they don't yield the same results.
100% reproducibility is not guaranteed, or you should not guarantee it either.

### After peer review

Please include author's name/journal/conference name in the benchmark adaptor package so future readers can easily find
your work.

## Current Implementation Scope

- `src/openre_bench/` contains the independent benchmark harness, schemas, LLM client, validation, and reporting code.
- `openre_bench_METHOD/`(currently `openre_bench_mare/`, `openre_bench_iredev/`, and `openre_bench_quare/`) is
  target-adapter packages outside `src/`. They model external method semantics while importing shared OpenRE-Bench
  support APIs.
- `--system METHOD` (currently `--system mare`, `--system iredev`, and `--system quare`) route one target adapter
  through the same benchmark harness
  with strict provenance/comparability metadata.
- `--systems mare,iredev,quare` runs the three target adapters as one equal matrix axis.
- Runtime packages emit the shared phase contract while keeping each target's debate, negotiation, and artifact policy
  in its own package.
- QUARE can execute live LLM turns in phase 2 negotiation when negotiation is enabled and OpenAI credentials are
  available; fallback state is recorded in execution flags.
- Where possible, `/auto` provides resumable end-to-end orchestration with finality gates and per-agent conversation
  logs under `report/logs/<run_key>/`.

## Supported Benchmark Targets

OpenRE-Bench currently supports three paper-grounded benchmark targets:

| Target                                          | Runtime identity  | Agent / role model            | Workflow surface                                             |
|-------------------------------------------------|-------------------|-------------------------------|--------------------------------------------------------------|
| [**MARE**](./openre_bench_mare/__init__.py)     | `--system mare`   | 5 task-specialized roles      | 9-action requirements engineering workflow                   |
| [**iReDev**](./openre_bench_iredev/__init__.py) | `--system iredev` | 6 knowledge/stakeholder roles | 17-action knowledge-driven requirements development workflow |
| [**QUARE**](./openre_bench_quare/__init__.py)   | `--system quare`  | 5 quality-specialized agents  | Dialectic negotiation and quality-axis KAOS workflow         |

MARE, iReDev, and QUARE all run through the same case input shape, phase artifact filenames, provenance metadata,
comparability flags, matrix harness, and report gates.

## Shared Benchmark Contract

Every supported target must:

- Accept the same case input shape: `case_name`, `case_description`, and `requirement`.
- Emit the same required phase artifacts:
    - `phase1_initial_models.json`
    - `phase2_negotiation_trace.json`
    - `phase3_integrated_kaos_model.json`
    - `phase4_verification_report.json`
- Record reproducibility controls: model, temperature, seed, round cap, RAG state, artifact hashes, fallback flags, and
  runtime semantics.
- Preserve explicit comparability metadata instead of silently treating partial or fallback output as equivalent
  evidence.

## Reviewer Reproduction Path

1. `uv sync --all-groups`
2. `uv run ruff check .`
3. `HF_HOME=/tmp/openre-bench-hf-cache uv run pytest`
4.

`uv run openre_bench --run-comparison-matrix --cases-dir data/case_studies --output-dir experiment_outputs/mare-iredev-quare --matrix-seeds 101,202,303 --systems mare,iredev,quare --matrix-workers 3`

5. Review `comparison_runs.jsonl`, `comparison_metrics_by_case.csv`, `comparison_metrics_summary.csv`,
   `comparison_ablation_table.csv`, and `comparison_validity_log.md` under the selected output directory.

For a single-run smoke check:

`uv run openre_bench --run-case --case-input data/case_studies/ATM_input.json --artifacts-dir artifacts/atm-mare --run-record artifacts/atm-mare/run_record.json --system mare`

Replace `mare` with `iredev` or `quare` to run the same case through another target.

## Tech Stack

- Language/runtime: Python 3.14 free-threaded (`3.14t`; package metadata requires Python 3.14+)
- Package and environment manager: `uv`
- Packaging: `pyproject.toml` (`hatchling`)
- LLM client layer: `litellm`
- LLM inference provider: OpenAI API key (`OPENAI_API_KEY`) or local `.api_key`
- Dev tools: `ruff`, `pytest` (managed through `uv`)

## Quick Start (uv)

1. `uv sync --all-groups`
2. `uv run openre_bench --version`
3. `cp .env.example .env` and set `OPENAI_API_KEY`, or create `.api_key` with `OPENAI_API_KEY=...`
4. `uv run openre_bench --check-openai`
5. `uv run ruff check .`
6. `uv run pytest`

## LLM Configuration

OpenRE-Bench uses `litellm` to be model-agnostic but default routes inference to OpenAI through:

- Key precedence: `.api_key` -> environment variables
- `OPENAI_API_KEY` (required when `.api_key` is absent)
- `OPENAI_KEY` (optional fallback env key name)
- `.api_key` (optional dotenv-style key file; highest precedence)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `OPENAI_BASE_URL` (optional custom endpoint)

Note, while you can use arbitrary LLM models, execution other than gpt-4 series isn't guaranteed.

Useful checks:

- `uv run openre_bench --check-openai`
- `uv run openre_bench --llm-ping`

## Core Commands

Validate one run:

`uv run openre_bench --validate-comparison --case-input <case.json> --run-record <run.json> --artifacts-dir <artifacts_dir>`

Run one case:

`uv run openre_bench --run-case --case-input data/case_studies/ATM_input.json --artifacts-dir artifacts/atm-run --run-record artifacts/atm-run/run_record.json --system mare`

Runtime identities:

- `--system mare`
- `--system iredev`
- `--system quare`

## Comparison Matrix Harness

Run matrix experiments:

`uv run openre_bench --run-comparison-matrix --cases-dir data/case_studies --output-dir artifacts/smoke-matrix --matrix-seeds 101,202,303 --systems mare,iredev,quare --matrix-workers 3`

`--matrix-workers` parallelizes independent run cells only. Per-run target logic, artifact validation, metric
computation, and final report writing remain unchanged and ordered by the matrix definition.

Default settings:

- `single_agent`
- `multi_agent_without_negotiation`
- `multi_agent_with_negotiation`
- `negotiation_integration_verification`

Key outputs:

- `comparison_runs.jsonl`
- `comparison_metrics_by_case.csv`
- `comparison_metrics_summary.csv`
- `comparison_ablation_table.csv`
- `comparison_validity_log.md`

## Trace Audit Export

`uv run openre_bench --export-trace-audit --matrix-output-dir artifacts/smoke-matrix`

Output:

- `comparison_trace_audit.md`

## `/auto` Strict Report Workflow

`/auto` is a strict resumable report workflow for the current MARE/QUARE paper-claim proof path. Use the comparison
matrix for the primary MARE/iReDev/QUARE three-way benchmark.

`uv run openre_bench /auto --cases-dir data/case_studies --rag-corpus-dir data/knowledge_base --matrix-seeds 101,202,303`

Important outputs:

- `report/logs/<run_key>/` execution logs
- `report/logs/<run_key>/conversation_index.jsonl`
- `report/logs/<run_key>/conversation_coverage.json`
- `report/logs/<run_key>/conversation_coverage.md`
- `report/logs/<run_key>/conversations/<system>/<case>/<setting>/seed-<seed>/<run_id>/`
- `report/runs/<run_key>/proofs/finality_threshold_verdict.json`
- `report/runs/<run_key>/proofs/conversation_log_evidence.json`
- `report/runs/<run_key>/proofs/quare_vs_mare_deltas.json`
- `report/README.md`, `report/analysis.md`, `report/proofs/*.json` (latest run mirror)

## Blind Evaluation Preparation

`uv run openre_bench --blind-eval-prepare --matrix-output-dir artifacts/smoke-matrix --blind-output-dir artifacts/smoke-matrix/blind-eval --judge-script src/openre_bench/comparison_harness.py`

Outputs:

- `blinded_comparison_runs.jsonl`
- `blinded_comparison_metrics_by_case.csv`
- `blind_mapping_private.json`
- `blind_eval_protocol.md`

## Repository Layout

- `src/openre_bench/` shared benchmark harness, schemas, validator, matrix harness, and `/auto`
- `openre_bench_mare/`, `openre_bench_iredev/`, `openre_bench_quare/` root-level benchmark target-adapter packages
- `tests/` regression tests
- `report/` generated `/auto` run and report evidence

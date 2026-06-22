# Artifact Reference

This page documents the main files reviewers can inspect without re-running the
LLM experiment matrix.

## Case Inputs

Benchmark inputs live under `data/case_studies/`. Each JSON file describes one
case study and is used unchanged by the QUARE, MARE, and iReDev adapters.

## Per-Run Artifacts

Each run directory under `experiment_outputs/mare-iredev-quare/runs/` contains:

- `phase1_initial_models.json`: initial generated requirements, goals, or model
  elements.
- `phase2_negotiation_trace.json`: negotiation or workflow trace when enabled.
- `phase3_integrated_kaos_model.json`: integrated KAOS-style requirement model.
- `phase4_verification_report.json`: structural, compliance, and quality checks
  when verification is executed.
- `run_record.json`: run provenance and comparability metadata.

Some adapters emit additional framework-specific files, such as QUARE conflict
maps or software-material exports. Their paths are listed in each
`run_record.json`.

## Run Record Fields

Important `run_record.json` fields include:

- `run_id`, `case_id`, `system`, `setting`, and `seed`: identify the matrix
  cell.
- `model`, `temperature`, `round_cap`, and `max_tokens`: record the LLM
  configuration used for the run.
- `system_identity`: records implementation version and execution platform.
- `provenance`: records prompt/corpus hashes and model settings.
- `execution_flags`: records fallback, retry, and taint flags.
- `comparability`: explains whether a run is directly comparable across the
  protocol settings.
- `artifact_paths`: maps expected phase filenames to their stored paths.

The packaged records do not include provider-reported `prompt_tokens`,
`completion_tokens`, `total_tokens`, or `cost_usd`. Cost figures in the README
are therefore estimates based on model pricing and expected token ranges, not
audited billing records.

## Matrix Outputs

The top-level files under `experiment_outputs/mare-iredev-quare/` summarize the
180-run matrix:

- `comparison_runs.jsonl`: line-delimited run records.
- `comparison_metrics_by_case.csv`: per-run metrics by case.
- `comparison_metrics_summary.csv`: aggregated comparison metrics.
- `comparison_ablation_table.csv`: ablation-oriented summary.
- `paper_tables/`: CSV versions of paper tables.

## Human Evaluation

The human-evaluation package under `human_eval/` contains the Phase 3 sample
export, blind mapping, item-level human scores, human-human agreement, and
human-vs-LLM agreement outputs. See `human_eval/README.md` for the navigation
recipe from `Sample_ID` to the source Phase 3 artifact.

# Phase 3 Human Evaluation Package

This directory contains the human-evaluation workbook, source traceability files,
and derived analysis outputs for the QUARE LLM-as-a-judge validation.

## Primary Workbook

- `Phase3_Human_Evaluation_Annotators.xlsx`
  - `ANNOT-001` and `ANNOT-002`: human evaluator scores.
  - `LLM-Judge(Hidden to Annotator)`: framework labels and LLM judge scores.
  - `Instructions` and `Case_Context`: evaluation rubric and case context.

## Traceability Files

- `phase3_requirements_all.json` / `phase3_requirements_all.csv`
  - Full export of requirement-like Phase 3 outputs from
    `experiment_outputs/mare-iredev-quare/runs`.
- `phase3_requirements_summary.json`
  - Summary of the full Phase 3 export.
- `phase3_human_eval_mapping.json`
  - Traceability mapping from `Sample_ID` to framework, case study, source file,
    run id, seed, source requirement id, raw text, and cleaned text.
- `phase3_human_eval_sample_summary.json`
  - Sampling summary for the representative human-evaluation sample.

## Analysis Outputs

Files under `analysis/` are derived from `Phase3_Human_Evaluation_Annotators.xlsx`.

- `table_c_human_validation.tex`
  - LaTeX version of Table C.
- `table_c_human_validation.csv`
  - Table C values in CSV form.
- `human_eval_agreement_summary.csv` / `.json`
  - Human-human agreement between `ANNOT-001` and `ANNOT-002`.
- `human_eval_framework_summary.csv`
  - Human-score means by framework and criterion.
- `human_eval_item_level_scores.csv`
  - Per-requirement human scores and human means.
- `human_vs_llm_alignment.csv` / `.json`
  - Human-vs-LLM agreement metrics.
- `human_vs_llm_framework_summary.csv`
  - Human and LLM means by framework and criterion.

The traceability mapping file is intended for audit/reproducibility, not for annotators.

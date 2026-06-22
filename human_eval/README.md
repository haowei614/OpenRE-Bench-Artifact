# Phase 3 Human Evaluation Package

This directory contains the Phase 3 requirement export, blind traceability
mapping, and derived analysis outputs for the QUARE LLM-as-a-judge validation.

## Quick Navigation

Start with these files:

- `analysis/table_c_human_validation.csv`: compact CSV version of the
  human-validation table reported in the paper.
- `analysis/human_eval_item_level_scores.csv`: per-sample human scores and
  human means.
- `analysis/human_eval_agreement_summary.csv` / `.json`: human-human agreement
  between `ANNOT-001` and `ANNOT-002`.
- `analysis/human_vs_llm_alignment.csv` / `.json`: human-vs-LLM agreement
  metrics.
- `phase3_human_eval_sample_summary.json`: sampling counts by framework and
  case study.

## Trace a Sample Back to the Source Run

Use this recipe to audit any row in the human-evaluation analysis:

1. Pick a `Sample_ID` from `analysis/human_eval_item_level_scores.csv`.
2. Look up the same `Sample_ID` in `phase3_human_eval_mapping.json`.
3. Use the mapping fields for framework, case study, run id, seed, source
   requirement id, raw text, and cleaned text.
4. Open the corresponding
   `experiment_outputs/mare-iredev-quare/runs/<run_id>/phase3_integrated_kaos_model.json`
   file to inspect the original Phase 3 requirement context.
5. Compare aggregate values with
   `experiment_outputs/mare-iredev-quare/paper_tables/table_vi_c_human_llm_validation.csv`.

The mapping file is intended for audit and reproducibility. It was not part of
the blind information shown to annotators.

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

Files under `analysis/` are derived from the annotated human-evaluation data and
the traceability files above.

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

# Ground Truth Data

This directory is intentionally empty in the archived artifact.

OpenRE-Bench evaluates generated requirements with controlled case inputs,
structural validators, cross-system comparison metrics, LLM-judge scores, and a
separate human-evaluation package. The submitted study does not claim an
authoritative human gold-standard KAOS model for each case study, so no
gold-standard requirement set is packaged here.

In other words, the benchmark uses shared case-study input specifications, but
it does not include a single human-authored "correct" output model or
requirement set for each case. The reported evaluation is therefore not based on
ground-truth precision/recall; it is based on controlled cross-system
comparison, structural validation, LLM-judge assessment, and human-evaluation
agreement.

Use these locations instead:

- `data/case_studies/`: benchmark input specifications shared by all systems.
- `experiment_outputs/mare-iredev-quare/`: pre-generated system outputs and
  comparison metrics used for the paper.
- `human_eval/`: human-evaluation sample, traceability mapping, item-level
  scores, and agreement analysis.

The evaluation scripts tolerate this empty directory because ground-truth-based
precision/recall is optional and is not the basis for the reported paper tables.


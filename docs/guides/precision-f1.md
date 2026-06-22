# Precision and F1 Guide

OpenRE-Bench includes utility support for token-overlap and comparison metrics,
but the submitted paper tables do not depend on an authoritative
ground-truth-requirement set.

## Current Artifact Scope

The archived package provides:

- Shared case-study inputs under `data/case_studies/`.
- Generated outputs for QUARE, MARE, and iReDev under
  `experiment_outputs/mare-iredev-quare/`.
- Human-evaluation evidence under `human_eval/`.

The `data/ground_truth/` directory is intentionally empty because the study does
not claim a manually curated gold-standard KAOS model for each case. Reported
results therefore use controlled cross-system comparisons, structural checks,
LLM-judge outputs, and human-evaluation agreement rather than
ground-truth-based precision/recall.

## If a Future Gold Standard Is Added

Future users can add manually curated reference requirements to
`data/ground_truth/` and compute precision, recall, and F1 with the usual
definitions:

```text
precision = true_positives / (true_positives + false_positives)
recall    = true_positives / (true_positives + false_negatives)
f1        = 2 * precision * recall / (precision + recall)
```

Any such extension should document the labeling protocol, annotator agreement,
and matching rules before comparing systems against the gold standard.

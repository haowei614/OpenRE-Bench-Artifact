# MARE, iReDev, and QUARE: Three-Way Comparison Summary

This report summarizes the current OpenRE-Bench three-way matrix output for MARE, iReDev, and QUARE.
Detailed per-run and per-case values are kept in the generated CSV and JSONL artifacts under
`experiment_outputs/mare-iredev-quare/`.

## Run Profile

| Field | Value |
|---|---|
| Generated at | `2026-05-11T18:16:03Z` |
| Output directory | `experiment_outputs/mare-iredev-quare` |
| Systems | `mare`, `iredev`, `quare` |
| Cases | AD, ATM, Bookkeeping, Library, RollCall |
| Settings | `single_agent`, `multi_agent_without_negotiation`, `multi_agent_with_negotiation`, `negotiation_integration_verification` |
| Seeds | 101, 202, 303 |
| Model | `gpt-4o-mini` |
| Temperature | 0.7 |
| Round cap | 3 |
| Matrix workers | 3 |

The matrix contains 180 completed runs: 60 per system.

## Reproduction

Reproduce the benchmark matrix into the standard output directory:

```bash
uv sync --all-groups
uv run openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir experiment_outputs/mare-iredev-quare \
  --matrix-seeds 101,202,303 \
  --systems mare,iredev,quare \
  --matrix-workers 3
```

Primary generated artifacts:

- `experiment_outputs/mare-iredev-quare/comparison_runs.jsonl`
- `experiment_outputs/mare-iredev-quare/comparison_metrics_by_case.csv`
- `experiment_outputs/mare-iredev-quare/comparison_metrics_summary.csv`
- `experiment_outputs/mare-iredev-quare/comparison_ablation_table.csv`
- `experiment_outputs/mare-iredev-quare/comparison_validity_log.md`

## Validity

The current validity log reports:

| Check | Value |
|---|---:|
| Expected runs | 180 |
| Actual runs | 180 |
| Errors | 0 |
| Warnings | 0 |

The generated CSV schemas include `system` as a first-class axis in by-case, summary, and ablation outputs.

## Full Pipeline Means

These values are means over the `negotiation_integration_verification` setting across 5 cases and 3 seeds.
They are a compact orientation only; use `comparison_metrics_by_case.csv` for detailed analysis.

| Metric | MARE | iReDev | QUARE | Leading value |
|---|---:|---:|---:|---|
| Phase 1 elements | 24.4 | 28.1 | 35.0 | QUARE |
| Phase 2 steps | 10.0 | 12.0 | 10.8 | iReDev |
| Phase 3 elements | 24.4 | 28.1 | 35.0 | QUARE |
| Conflict resolution rate | 0.600 | 0.467 | 0.200 | MARE |
| CHV | 0.004726 | 0.005796 | 0.004295 | iReDev |
| MDC | 0.827819 | 0.702794 | 0.675223 | MARE |
| Semantic preservation F1 | 0.893161 | 0.922200 | 0.948567 | QUARE |
| P2 vs P1 semantic F1 | 0.980792 | 0.993798 | 1.000000 | QUARE |
| Logic score | 1.000 | 1.000 | 1.000 | Tie |
| Topology valid | 1.000 | 1.000 | 1.000 | Tie |
| Deterministic validation | 1.000 | 1.000 | 1.000 | Tie |
| Compliance coverage | 0.484444 | 0.562222 | 0.982222 | QUARE |
| Terminology consistency | 0.752857 | 0.611587 | 0.655238 | MARE |
| ISO 29148 unambiguous | 4.456286 | 4.145492 | 4.241524 | MARE |
| ISO 29148 correctness | 5.000 | 5.000 | 5.000 | Tie |
| ISO 29148 verifiability | 3.968889 | 4.124444 | 4.964444 | QUARE |
| ISO 29148 set consistency | 5.000 | 5.000 | 5.000 | Tie |
| ISO 29148 set feasibility | 3.762667 | 3.949333 | 4.957333 | QUARE |
| Runtime seconds | 32.747 | 161.020 | 93.177 | MARE |

## Runtime Means

| Setting | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| `single_agent` | 0.007 | 0.007 | 0.007 |
| `multi_agent_without_negotiation` | 31.649 | 173.098 | 0.022 |
| `multi_agent_with_negotiation` | 32.238 | 161.649 | 91.584 |
| `negotiation_integration_verification` | 32.747 | 161.020 | 93.177 |

## Observations

- QUARE has the highest phase element counts, semantic preservation values, compliance coverage, verifiability, and set feasibility in the full-pipeline means.
- iReDev has the highest CHV value and the highest phase 2 step count in the full-pipeline means.
- MARE has the highest MDC, conflict resolution rate, terminology consistency, unambiguous score, and lowest full-pipeline runtime in the full-pipeline means.
- All three systems have equal topology validity, deterministic validation, logic score, correctness, and set consistency in the full-pipeline means.

These observations describe the generated benchmark output. They are not a general claim that one target is universally preferable.

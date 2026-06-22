# MARE-iReDev-QUARE Experiment Outputs

This directory contains the pre-generated outputs for the 180-run comparison
matrix used by the replication package.

## Contents

| Path | Description |
|------|-------------|
| `comparison_metrics_by_case.csv` | Full per-run metrics table for the comparison matrix. |
| `comparison_metrics_summary.csv` | Aggregated metrics exported by the experiment scripts. |
| `comparison_ablation_table.csv` | Ablation-oriented summary exported by the comparison pipeline. |
| `comparison_runs.jsonl` | Run-level provenance records for the comparison matrix. |
| `runs/` | Per-run artifacts, including intermediate phase outputs and negotiation traces. |
| `paper_tables/` | Machine-readable CSV versions of the tables reported in the paper PDF. |

The `paper_tables/` directory is intended as the reviewer-facing index for
paper table values. The `comparison_*` files and `runs/` directory preserve the
experiment-output layout produced by the harness.

The `system_identity.python_version` values inside historical `run_record.json`
files are provenance for the already-generated paper outputs. They are not the
recommended setup version for re-exercising the artifact; use the Python 3.13
environment documented in the root README or the Dockerfile.

## Cost Notes

These pre-generated outputs can be inspected without an API key and without
incurring new model cost. The stored `run_record.json` files include model
labels, seeds, `max_tokens`, and run provenance, but they do not include
provider-reported token usage or billed cost.

For review planning, estimate hosted-model cost with:

```text
cost_usd = input_tokens / 1_000_000 * input_price_per_1m
         + output_tokens / 1_000_000 * output_price_per_1m
```

Using `gpt-4o-mini-2024-07-18` pricing at the time of packaging as a reference
(`$0.15 / 1M` input tokens and `$0.60 / 1M` output tokens), a single smoke run
is expected to stay below USD 0.10. The full directory corresponds to a 180-run
matrix, for which USD 5-25 is a conservative planning range depending on retry
behavior, provider rounding, and response length.

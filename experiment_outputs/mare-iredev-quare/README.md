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

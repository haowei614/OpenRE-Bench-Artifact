# Paper-Reported Experimental Results

This file records the numerical values reported in the current paper PDF
(`QUARE (48).pdf`). It is the source of truth for paper-table values in the
replication package.

Machine-readable CSV versions of these tables are stored in this directory with
`table_*` filenames.

## CSV Index

| File | Content |
|------|---------|
| `table_ii_benchmark_systems.csv` | Table II values used in the paper. |
| `table_ii_case_input_stats_detailed.csv` | Additional input word-count and requirement-count details for Table II. |
| `table_iii_coverage_diversity.csv` | Table III coverage and diversity values. |
| `table_iv_semantic_preservation.csv` | Table IV semantic-preservation values. |
| `table_v_negotiation_summary.csv` | Table V negotiation-process values. |
| `table_vi_a_structural_compliance.csv` | Table VI(A) structural and compliance values. |
| `table_vi_b_llm_judge_quality.csv` | Table VI(B) LLM-judge quality-score values. |
| `table_vi_c_human_llm_validation.csv` | Table VI(C) human-vs-LLM validation values. |

## Experimental Configuration

| Parameter | Value |
|---|---|
| Total runs | 180 |
| Frameworks | MARE, iReDev, QUARE |
| Model | `gpt-4o-mini-2024-07-18` |
| Temperature | 0.7 |
| Round cap | 3 |
| Seeds | 101, 202, 303 |
| Case studies | AD, ATM, Library, RollCall, Bookkeeping |

## Table III: Requirement Coverage and Diversity

Values are averaged across the five benchmark systems. CHV values are reported
as `x10^-3`.

| Framework | Req. Count | CHV | MDC | CU | MAC |
|---|---:|---:|---:|---:|---:|
| Single-agent | 14.2 | 2.8 | 0.715 | - | - |
| MARE | 24.4 | 4.8 | 0.835 | 0.30 | 4.6 |
| iReDev | 28.1 | 6.4 | 0.705 | 0.45 | 5.1 |
| QUARE | 35.0 | 4.3 | 0.673 | 0.20 | 6.7 |

## Table IV: Semantic Preservation

BERTScore F1 between Phase 3 and Phase 1 outputs, reported as percentages.

| Benchmark System | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | 88.4 | 92.7 | 94.8 |
| ATM | 89.6 | 92.0 | 95.5 |
| Library | 87.7 | 92.6 | 94.8 |
| RollCall | 88.8 | 93.0 | 94.5 |
| Bookkeeping | 90.5 | 92.9 | 94.8 |
| Average | 89.0 | 92.6 | 94.9 |

## Table V: Negotiation Process Summary

| Metric | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| Avg. negotiation steps | 10.0 | 12.0 | 16.5 |
| Conflict resolution rate (%) | 66.7 | 46.7 | 25.0 |
| Phase 2 vs. Phase 1 BERTScore (%) | 97.0 | 99.4 | 100.0 |

## Table VI(A): Structural Correctness

| Metric | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| DAG topology valid | yes | yes | yes |
| Logical consistency (`Slogic`) | 1.000 | 1.000 | 1.000 |
| Compliance coverage (%) | 47.6 | 47.8 | 98.2 |

## Table VI(B): ISO/IEC/IEEE 29148 Quality Scores

Scores use a 1-5 scale with an LLM judge.

| Criterion | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| Unambiguous | 4.41 | 4.19 | 4.24 |
| Correctness | 5.00 | 5.00 | 5.00 |
| Verifiability | 3.95 | 3.96 | 4.96 |
| Set Consistency | 5.00 | 5.00 | 5.00 |
| Set Feasibility | 3.74 | 3.75 | 4.96 |

## Table VI(C): Human vs. LLM-Judge Validation

| Criterion | Human MARE | Human iReDev | Human QUARE | LLM MARE | LLM iReDev | LLM QUARE |
|---|---:|---:|---:|---:|---:|---:|
| Unambiguous | 3.68 | 3.70 | 3.72 | 4.41 | 4.18 | 4.23 |
| Correctness | 4.95 | 4.93 | 4.97 | 5.00 | 5.00 | 5.00 |
| Verifiability | 4.00 | 4.05 | 4.27 | 3.95 | 3.95 | 4.97 |
| Set Consistency | 4.89 | 4.91 | 4.92 | 5.00 | 5.00 | 5.00 |
| Set Feasibility | 4.27 | 4.27 | 4.30 | 3.73 | 3.73 | 4.97 |

Human-evaluation source files and agreement metrics are available under
`human_eval/analysis/`.

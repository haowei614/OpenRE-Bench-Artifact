# OpenRE-Bench Documentation

This directory contains the project documentation for OpenRE-Bench. Each page has one primary responsibility so that benchmark concepts, command usage, artifact contracts, and evaluation guidance do not repeat the same material.

## Documents

| Document | Purpose |
|---|---|
| [NLP4RE-ID-CARD.md](NLP4RE-ID-CARD.md) | Filled-in NLP4RE ID-Card summary for the artifact. |
| [methodology.md](methodology.md) | Benchmark scope, supported target adapters, shared phases, KAOS conventions, and comparability principles. |
| [reference/cli.md](reference/cli.md) | Installation checks, CLI commands, runtime configuration, and reproducibility commands. |
| [reference/artifacts.md](reference/artifacts.md) | Case input shape, required phase artifacts, run records, matrix outputs, and evaluation fields. |
| [targets/quare-orchestrator.md](targets/quare-orchestrator.md) | QUARE orchestration behavior and its relationship to the shared benchmark phases. |
| [guides/incose-format.md](guides/incose-format.md) | INCOSE-style requirement statement guidance for interpreting or post-processing KAOS elements. |
| [guides/precision-f1.md](guides/precision-f1.md) | Manual labeling procedure and Precision, Recall, and F1 calculations. |

## Reading Order

Start with [methodology.md](methodology.md) for the benchmark model. Use [reference/cli.md](reference/cli.md) when running experiments, and [reference/artifacts.md](reference/artifacts.md) when validating outputs or consuming generated files.

The guide pages are supplemental. They describe evaluation and formatting practices; they are not substitutes for the runtime schemas in `src/openre_bench/schemas.py`.

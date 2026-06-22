# NLP4RE ID-Card for OpenRE-Bench

This page provides a filled-in summary following the seven dimensions of the
NLP4RE ID-Card. The ID-Card is intended to make NLP for Requirements Engineering
artifacts easier to understand, compare, and replicate.

Reference template: [The NLP4RE ID-Card](https://doi.org/10.5281/zenodo.14197338).

## I. RE Task

- Study/artifact: OpenRE-Bench, the replication artifact for **QUARE:
  Quality-Aware Requirements Analysis through Multi-Agent Dialectical
  Negotiation**.
- RE task: requirements analysis and comparison of multi-agent RE frameworks.
- Supported frameworks: QUARE, MARE, and iReDev.
- RE output type: structured phase artifacts, KAOS-style requirement models,
  negotiation traces, verification reports, run records, and comparison tables.
- Domains/cases: AD, ATM, Library, RollCall, and Bookkeeping case studies.

## II. NLP Tasks

- LLM-backed generation of requirements and intermediate analysis artifacts.
- Structured transformation of natural-language case descriptions into
  framework-specific phase artifacts.
- Multi-agent negotiation and integration of generated requirements.
- Retrieval-assisted verification using the local knowledge base.
- LLM-as-a-judge scoring, validated through a separate human-evaluation package.

## III. Input and Output Details

- Initial input: JSON case-study descriptions under `data/case_studies/`.
- Optional auxiliary input: domain and standards material under
  `data/knowledge_base/`.
- Runtime configuration: model, temperature, seed, round cap, selected system,
  RAG settings, and output directories.
- Main output: per-run JSON artifacts under
  `experiment_outputs/mare-iredev-quare/runs/`.
- Aggregated output: comparison CSVs and paper-table CSVs under
  `experiment_outputs/mare-iredev-quare/`.
- Human-evaluation output: item-level scores, agreement summaries, and
  human-vs-LLM alignment files under `human_eval/analysis/`.

## IV. Raw Data and Dataset

- Raw case inputs: five benchmark case-study specifications shared across all
  systems.
- Generated dataset: 180 pre-generated comparison runs covering 5 cases, 3
  systems, 4 settings, and 3 seeds.
- Gold-standard output data: not claimed. The artifact does not include a
  human-authored "correct" KAOS model or requirement set for each case.
- Ground-truth scope: `data/ground_truth/` is intentionally empty; see
  `data/ground_truth/README.md`.
- License: AGPL-3.0-or-later for the artifact code and packaged materials,
  subject to the intellectual-property rights of referenced papers.
- Availability: archived with DOI `10.5281/zenodo.20482393` and mirrored on
  GitHub.

## V. Annotation Process

- Human evaluation scope: validation of Phase 3 requirement-like outputs and
  LLM-as-a-judge alignment, not creation of gold-standard requirements.
- Human-evaluation materials: traceability mapping, item-level scores,
  human-human agreement, and human-vs-LLM agreement outputs under `human_eval/`.
- Navigation: `human_eval/README.md` gives a `Sample_ID`-level recipe from
  item-level scores to source run IDs and original Phase 3 JSON artifacts.
- Annotator-facing data: framework identities were separated from the
  audit/reproducibility mapping.

## VI. Tool Implementation

- Language/runtime: Python 3.13.
- Package manager: `uv`.
- Main CLI: `openre_bench`.
- LLM access: LiteLLM with hosted or local OpenAI-compatible endpoints.
- Local LLM path: tested with Ollama at `http://localhost:11434/v1` using
  `qwen3.5:9b`; `uv run openre_bench --llm-ping` returned `pong`.
- Containerization: `Dockerfile` based on `python:3.13-slim`.
- Validation commands:

```bash
uv sync --all-groups
uv run openre_bench --version
uv run ruff check .
HF_HOME=/tmp/openre-bench-hf-cache uv run pytest
```

## VII. Evaluation

- Primary comparison evidence: paper-table CSVs and pre-generated run artifacts
  under `experiment_outputs/mare-iredev-quare/`.
- Reproduction levels: non-execution inspection, single-case smoke test, and
  full 180-run matrix.
- Structural validation: phase artifacts and run records are checked by the
  shared validator/test suite.
- Human validation: human-human agreement and human-vs-LLM alignment are stored
  under `human_eval/analysis/`.
- Cost estimate: the artifact provides a formula and rough planning range, but
  historical run records do not contain provider-billed token counts.
- Known limitation: local LLMs can exercise the harness without a commercial API
  key, but exact reproduction of paper outputs uses the documented
  `gpt-4o-mini-2024-07-18` configuration.

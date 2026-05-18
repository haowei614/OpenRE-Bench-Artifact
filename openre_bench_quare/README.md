# QUARE Adapter

Implementation of the QUARE framework as described in the accompanying paper.

## Architecture

- **Agents**: Safety, Efficiency, Green (Sustainability), Trustworthiness,
  Responsibility, plus Orchestrator
- **Decomposition**: By ISO/IEC 25010 quality dimension
- **Negotiation**: Dialectical protocol (thesis-antithesis-synthesis), max 3
  rounds, BERTScore convergence
- **Integration**: KAOS three-level goal model with DAG validation
- **Verification**: RAG-augmented compliance checking (ISO 26262, ISO 27001)

## Entry Points

| File | Role |
|------|------|
| `__init__.py` | Adapter implementation and phase orchestration |

Registered in the shared pipeline via `src/openre_bench/pipeline/_core.py`.

## Usage

```bash
uv run openre_bench --run-case \
  --case-input data/case_studies/AD_input.json \
  --system quare \
  --artifacts-dir artifacts/ad-quare \
  --run-record artifacts/ad-quare/run_record.json
```

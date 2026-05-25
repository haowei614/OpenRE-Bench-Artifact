# MARE Adapter

Faithful reimplementation of the MARE framework (2024;
arXiv:2405.03256) for cross-framework comparison.

## Architecture

- **Agents**: 5 task-specialized roles (elicitation, modeling, verification,
  specification, coordination)
- **Decomposition**: By RE engineering task
- **Negotiation**: Single-turn via shared workspace
- **Pipeline**: 9-action requirements engineering workflow

## Reimplementation Notes

- Based on published paper description; no public implementation available
- Uses shared model configuration (`gpt-4o-mini-2024-07-18`, temperature `0.7`)
- Conservative interpretation adopted where details were underspecified

## Entry Points

| File | Role |
|------|------|
| `__init__.py` | Adapter implementation |

## Usage

```bash
uv run openre_bench --run-case \
  --case-input data/case_studies/AD_input.json \
  --system mare \
  --artifacts-dir artifacts/ad-mare \
  --run-record artifacts/ad-mare/run_record.json
```

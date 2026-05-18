# iReDev Adapter

Faithful reimplementation of the iReDev framework (Jin et al., 2025;
arXiv:2507.13081) for cross-framework comparison.

## Architecture

- **Agents**: 6 knowledge-driven agents with stakeholder roles
- **Decomposition**: By knowledge role
- **Negotiation**: Single-turn review and feedback via shared artifact pool
- **Pipeline**: 17-action knowledge-driven requirements workflow

## Reimplementation Notes

- Based on published paper description; no public implementation available
- Human-in-the-loop behavior is operationalized via an LLM-based surrogate with
  deterministic fallback
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
  --system iredev \
  --artifacts-dir artifacts/ad-iredev \
  --run-record artifacts/ad-iredev/run_record.json
```

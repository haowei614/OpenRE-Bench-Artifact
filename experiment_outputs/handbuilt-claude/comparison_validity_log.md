# Comparison Validity Log

- Generated at: 2026-09-30T05:16:01Z
- Cases dir: `data/_handbuilt`
- Output dir: `experiment_outputs/handbuilt-claude`
- Seeds: `[101, 202, 303]`
- Settings: `['negotiation_integration_verification']`
- Systems: `['mare', 'iredev', 'quare']`
- Matrix workers: `1`
- Model: `claude-haiku-4-5-20251001`
- Temperature: `0.7`
- Round cap: `3`
- Max tokens: `4000`
- RAG enabled: `True`
- RAG backend: `local_tfidf`
- RAG corpus dir: `data/knowledge_base`

## Completeness
- Expected runs: 45
- Actual runs: 45

## Strict Fail Conditions
- Missing required deliverables.
- Run count mismatch versus expected matrix cardinality.
- Missing required keys in `comparison_runs.jsonl`.
- Missing or malformed provenance metadata (hash contracts).
- Missing or malformed execution/comparability metadata contracts.
- Missing required columns in required CSV outputs.
- Any run with `validation_passed=false`.
- Any run with `rag_enabled=false` or missing `rag_backend`.
- Any run with `rag_fallback_used=true`.
- Any run with `fallback_tainted=true` or retry-tainted metadata.

## Errors
- None

## Warnings
- mare-ad-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-ad-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-ad-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-ad-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-ad-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-ad-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-ad-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-ad-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-ad-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-atm-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-atm-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-atm-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-atm-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-atm-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-atm-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-atm-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-atm-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-atm-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-bookkeeping-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-bookkeeping-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-bookkeeping-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-bookkeeping-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-bookkeeping-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-bookkeeping-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-bookkeeping-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-bookkeeping-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-bookkeeping-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-library-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-library-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-library-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-library-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-library-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-library-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-library-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-library-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-library-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-rollcall-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-rollcall-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-rollcall-negotiation_integration_verification-s101: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-rollcall-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-rollcall-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-rollcall-negotiation_integration_verification-s202: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- mare-rollcall-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- iredev-rollcall-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'
- quare-rollcall-negotiation_integration_verification-s303: Model drift: expected 'gpt-4o-mini', got 'claude-haiku-4-5-20251001'

# CLI Reference

This page summarizes the runtime configuration and reviewer-facing commands.
See the root README for the shortest quick-start path.

## Environment

The locked artifact environment targets Python 3.13.

```bash
uv sync --all-groups
uv run openre_bench --version
```

The Docker path uses the same Python version:

```bash
docker build -t openre-bench .
docker run --rm openre-bench
```

## LLM Configuration

LLM-backed commands use LiteLLM. The following settings can be placed in `.env`
or exported in the shell. A local `.api_key` file with `OPENAI_API_KEY=...` or
`OPENAI_KEY=...` is also supported and takes precedence for the key value.

- `OPENAI_API_KEY`: hosted provider key, or a dummy value for local servers that
  do not validate keys.
- `OPENAI_KEY`: fallback key name supported for compatibility.
- `OPENAI_MODEL`: model label passed to LiteLLM, defaulting to `gpt-4o-mini`.
- `OPENAI_BASE_URL`: optional OpenAI-compatible endpoint such as a local
  Ollama, vLLM, or LM Studio `/v1` URL.
- `OPENAI_TIMEOUT_SECONDS`: request timeout, defaulting to 180 seconds.
- `OPENAI_REQUEST_RETRIES`: LiteLLM retry count, defaulting to 2.

Example hosted configuration:

```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY=..., or create .api_key with OPENAI_API_KEY=...
uv run openre_bench --check-openai
uv run openre_bench --llm-ping
```

Example local OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY=dummy \
OPENAI_MODEL=ollama_chat/llama3.1 \
OPENAI_BASE_URL=http://localhost:11434/v1 \
uv run openre_bench --llm-ping
```

Local models are useful for exercising the harness without a commercial API.
They are not expected to reproduce the exact paper results generated with
`gpt-4o-mini-2024-07-18`.

## Core Commands

Check installation:

```bash
uv run openre_bench --version
uv run ruff check .
HF_HOME=/tmp/openre-bench-hf-cache uv run pytest
```

Run a no-key single-case smoke test:

```bash
uv run openre_bench --run-case \
  --case-input data/case_studies/ATM_input.json \
  --artifacts-dir artifacts/atm-quare-single \
  --run-record artifacts/atm-quare-single/run_record.json \
  --system quare \
  --setting single_agent
```

LLM-backed settings require either hosted model credentials or a local
OpenAI-compatible endpoint. Replace `quare` with `mare` or `iredev` to exercise
another adapter under the same no-key smoke-test setting.

Run the full paper matrix:

```bash
uv run openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir experiment_outputs/mare-iredev-quare \
  --matrix-seeds 101,202,303 \
  --systems mare,iredev,quare \
  --matrix-workers 3
```

## Non-Execution Review Path

When API access is unavailable, inspect:

- `experiment_outputs/mare-iredev-quare/paper_tables/` for paper-table CSVs.
- `experiment_outputs/mare-iredev-quare/runs/` for per-run phase artifacts.
- `human_eval/analysis/` for human-human and human-vs-LLM agreement outputs.

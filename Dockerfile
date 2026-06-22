FROM python:3.13-slim

ENV HF_HOME=/tmp/openre-bench-hf-cache \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md .python-version ./
COPY data ./data
COPY docs ./docs
COPY experiment_outputs ./experiment_outputs
COPY human_eval ./human_eval
COPY openre_bench_iredev ./openre_bench_iredev
COPY openre_bench_mare ./openre_bench_mare
COPY openre_bench_quare ./openre_bench_quare
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests
COPY LICENSE LICENSE.md CITATION.cff ./

RUN uv sync --frozen --all-groups

CMD ["uv", "run", "openre_bench", "--version"]

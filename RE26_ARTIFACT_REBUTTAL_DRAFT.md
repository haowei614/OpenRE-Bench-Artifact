We thank the reviewers for the careful artifact evaluation and for identifying
several places where the replication package could be made easier to exercise.
All issues discussed below have been addressed in `v1.0.2-artifact` (DOI:
https://doi.org/10.5281/zenodo.20790383; release notes:
https://github.com/haowei614/OpenRE-Bench-Artifact/releases/tag/v1.0.2-artifact).

First, we fixed the installation issue reported by Reviewers 1 and 3. The
artifact previously declared Python >=3.14 and also pinned a free-threaded
3.14t interpreter locally, which could cause transitive dependency failures in
`tokenizers`. We now target Python 3.13 in `pyproject.toml`, `.python-version`,
the lock file, and the README. We verified that dependency resolution succeeds
with `uv lock --python 3.13`. We also added a Dockerfile that fixes the runtime
environment to Python 3.13 and runs `uv sync --frozen --all-groups`, giving
reviewers a containerized path when local setup is inconvenient, including on
Windows hosts via Docker or WSL.

Second, we addressed the linting failure. We reproduced the `uv run ruff check .`
errors and fixed them: one unused import was removed, and the few scripts that
need a directory-bootstrap import pattern now have explicit per-file Ruff
exceptions for `E402`. The reviewer command `uv run ruff check .` now passes.

Third, we clarified the reviewer workflow. The README now separates three paths:
(1) no API key, where reviewers inspect the pre-generated paper tables, per-run
artifacts, and human-evaluation outputs; (2) an API-key smoke test for one
framework and one case; and (3) the full 180-run matrix for reviewers with
sufficient time, quota, and budget. This is intended to make it clear that API
access is only required for generating new LLM outputs, not for inspecting the
submitted evidence. We also revised the earlier ambiguous wording: pre-generated
artifacts are now described as a non-execution verification path, not as an
alternative way to generate new outputs. To reduce cognitive load, we also
shortened the top-level README into a reviewer landing page and moved detailed
CLI/artifact-format guidance to the `docs/` pages and subdirectory READMEs. The
README and CLI reference now also explicitly document both `.env` and local
`.api_key` configuration for `OPENAI_API_KEY`/`OPENAI_KEY`.

Fourth, we added cost and token guidance. The README and experiment-output
README now explain that the packaged run records do not contain provider-billed
token counts, so costs are estimates rather than audited billing records. We
therefore do not retroactively invent exact token usage. Instead, we provide a
recomputable formula based on input/output token counts and provider prices, with
`gpt-4o-mini-2024-07-18` pricing as the documented example. Using that estimate,
a single smoke run is expected to stay below USD 0.10, while the full 180-run
matrix should be budgeted at roughly USD 5-25 depending on retries, provider
rounding, and output length.

Fifth, we documented local LLM execution. The harness already supported
OpenAI-compatible endpoints through LiteLLM and `OPENAI_BASE_URL`; the README,
`.env.example`, and CLI reference now include examples for local endpoints such
as Ollama, vLLM, or LM Studio. We also verified this path locally with Ollama's
OpenAI-compatible endpoint (`http://localhost:11434/v1`) and `qwen3.5:9b`:
`OPENAI_API_KEY=dummy OPENAI_MODEL='openai/qwen3.5:9b'
OPENAI_BASE_URL='http://localhost:11434/v1' uv run openre_bench --llm-ping`
returned `pong`. We explicitly state that local models are useful for exercising
the harness when a commercial OpenAI key is unavailable, while exact
reproduction of the reported results uses the paper model configuration.

Finally, we improved evaluation-material navigation. The `human_eval/README.md`
now starts with a reviewer-facing quick navigation section for the main
human-validation CSV/JSON files and gives a `Sample_ID`-level recipe from
item-level scores to the traceability mapping, source run id, and original Phase
3 JSON artifacts. We also clarified why `data/ground_truth/` is empty: the paper
does not claim an authoritative gold-standard KAOS model or requirement set per
case. The artifact includes shared case-study inputs, but the reported evidence
instead relies on controlled cross-system comparisons, structural checks,
LLM-judge outputs, and human-evaluation agreement. Following Reviewer 2's
suggestion, we also added a filled-in NLP4RE ID-Card summary under
`docs/NLP4RE-ID-CARD.md`.

We hope these changes address the main reproducibility concerns while preserving
the original artifact content and reported results.

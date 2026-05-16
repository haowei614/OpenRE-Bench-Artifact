"""Compatibility shim; prefer importing OpenAI settings from ``openre_bench.llm``."""

from openre_bench.llm import MissingAPIKeyError  # noqa: F401
from openre_bench.llm import OpenAISettings  # noqa: F401
from openre_bench.llm import load_openai_settings  # noqa: F401

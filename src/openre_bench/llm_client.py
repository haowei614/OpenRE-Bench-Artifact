"""Compatibility shim; prefer importing LLM client classes from ``openre_bench.llm``."""

from openre_bench.llm import LLMClient  # noqa: F401
from openre_bench.llm import LLMClientError  # noqa: F401

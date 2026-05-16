"""Consolidated LLM configuration, client, and resolution for OpenRE-Bench.

This module owns every concern related to LLM inference:
  - Configuration loading (``OpenAISettings``, ``.api_key`` file)
  - Protocol contract (``LLMContract``)
  - LiteLLM-backed client (``LLMClient``)
  - Client resolution helpers used by the pipeline orchestrator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from openre_bench.schemas import (
    SETTING_MULTI_AGENT_WITH_NEGOTIATION,
    SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    SETTING_SINGLE_AGENT,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MissingAPIKeyError(RuntimeError):
    """Raised when required API credentials are missing."""


class LLMClientError(RuntimeError):
    """Raised when LLM requests fail or responses are malformed."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class OpenAISettings(BaseSettings):
    """Runtime settings for OpenAI-backed inference."""

    api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    api_key_fallback: str | None = Field(default=None, alias="OPENAI_KEY")
    model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    timeout_seconds: float = Field(default=180.0, alias="OPENAI_TIMEOUT_SECONDS")
    request_retries: int = Field(default=2, alias="OPENAI_REQUEST_RETRIES")

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def resolved_api_key(self) -> str:
        """Return the configured OpenAI key from supported variable names."""

        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        if self.api_key_fallback and self.api_key_fallback.strip():
            return self.api_key_fallback.strip()
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is required for LLM inference. "
            "Set it in your shell, .env, or .api_key as OPENAI_API_KEY=... or OPENAI_KEY=..."
        )

    @property
    def has_key(self) -> bool:
        """Whether a usable API key value is configured."""

        return bool(self.api_key or self.api_key_fallback)


def load_openai_settings() -> OpenAISettings:
    """Load OpenAI settings with ``.api_key`` taking highest priority."""

    settings = OpenAISettings()
    file_key = _load_key_from_file(".api_key")

    if file_key:
        return OpenAISettings(
            api_key=file_key,
            api_key_fallback=None,
            model=settings.model.strip() or "gpt-4o-mini",
            base_url=(settings.base_url or "").strip() or None,
            timeout_seconds=max(30.0, float(settings.timeout_seconds)),
            request_retries=max(0, int(settings.request_retries)),
        )

    if not settings.has_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is required for LLM inference. "
            "Set it in your shell, .env, or .api_key as OPENAI_API_KEY=..."
        )

    return OpenAISettings(
        api_key=settings.resolved_api_key,
        api_key_fallback=None,
        model=settings.model.strip() or "gpt-4o-mini",
        base_url=(settings.base_url or "").strip() or None,
        timeout_seconds=max(30.0, float(settings.timeout_seconds)),
        request_retries=max(0, int(settings.request_retries)),
    )


def _load_key_from_file(path: str) -> str:
    """Load the highest-priority API key from a KEY=VALUE local file."""

    key_file = Path(path)
    if not key_file.exists():
        return ""

    try:
        content = key_file.read_text(encoding="utf-8")
    except OSError:
        return ""

    fallback = ""
    for line in content.replace("\r", "\n").split("\n"):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith("export "):
            value = value[7:].strip()
        if "=" not in value:
            continue
        key_name, key_value = value.split("=", 1)
        key_name = key_name.strip().upper()
        key_value = key_value.strip().strip('"').strip("'")
        if key_name == "OPENAI_API_KEY":
            return key_value
        if key_name == "OPENAI_KEY" and not fallback:
            fallback = key_value

    return fallback


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMContract(Protocol):
    """System-agnostic chat contract for LLM-backed pipeline turns."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Minimal chat client using LiteLLM with OpenAI credentials."""

    def __init__(self, settings: OpenAISettings) -> None:
        self._settings = settings

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        from litellm import completion  # lazy import to keep module fast

        request: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": temperature,
            "api_key": self._settings.api_key,
            "timeout": float(self._settings.timeout_seconds),
            "num_retries": int(self._settings.request_retries),
        }
        if self._settings.base_url:
            request["api_base"] = self._settings.base_url
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if seed is not None:
            request["seed"] = int(seed)

        try:
            response = completion(**request)
        except Exception as exc:  # pragma: no cover
            raise LLMClientError(f"LiteLLM request failed: {exc}") from exc

        text = _extract_text(response)
        if not text:
            raise LLMClientError("LiteLLM response did not include assistant text.")
        return text


def _extract_text(response: Any) -> str:
    """Extract assistant text from LiteLLM response payload."""

    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""


# ---------------------------------------------------------------------------
# Chat helper
# ---------------------------------------------------------------------------


def chat_with_seed(
    *,
    llm_client: LLMContract,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    seed: int,
) -> tuple[str, bool]:
    """Call chat with seed and report that seed was passed to the client."""

    dynamic_client: Any = llm_client
    return (
        dynamic_client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        ),
        True,
    )


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def _negotiation_enabled(setting: str) -> bool:
    """Whether phase 2 negotiation should run for the setting."""

    return setting in {
        SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    }


def resolve_phase2_llm_client(
    *,
    setting: str,
    phase2_uses_llm: bool,
    model: str,
    llm_client: LLMContract | None = None,
) -> tuple[LLMContract | None, str]:
    """Resolve the client for phase 2 when the selected system declares LLM use."""

    if not phase2_uses_llm or not _negotiation_enabled(setting):
        return None, "disabled"

    if llm_client is not None:
        return llm_client, "injected"

    try:
        settings = load_openai_settings()
    except MissingAPIKeyError:
        return None, "missing_api_key"

    settings.model = model
    return LLMClient(settings), "openai"


def resolve_runtime_llm_client(
    *,
    setting: str,
    runtime_uses_llm: bool,
    model: str,
    llm_client: LLMContract | None = None,
) -> tuple[LLMContract | None, str]:
    """Resolve the LLM client for systems that declare runtime LLM actions."""

    if not runtime_uses_llm or setting == SETTING_SINGLE_AGENT:
        return None, "disabled"

    if llm_client is not None:
        return llm_client, "injected"

    try:
        settings = load_openai_settings()
    except MissingAPIKeyError:
        return None, "missing_api_key"

    settings.model = model
    return LLMClient(settings), "openai"

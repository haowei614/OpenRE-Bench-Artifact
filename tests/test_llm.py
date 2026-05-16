"""Tests for openre_bench.llm — settings, _extract_text, _load_key_from_file, resolvers."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from openre_bench.llm import (
    LLMClient,
    LLMClientError,
    LLMContract,
    MissingAPIKeyError,
    OpenAISettings,
    _extract_text,
    _load_key_from_file,
    chat_with_seed,
    load_openai_settings,
    resolve_phase2_llm_client,
    resolve_runtime_llm_client,
)


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_object_response():
    """Object-style response (real LiteLLM)."""

    class Msg:
        content = "hello"

    class Choice:
        message = Msg()

    class Response:
        choices = [Choice()]

    assert _extract_text(Response()) == "hello"


def test_extract_text_dict_response():
    resp = {"choices": [{"message": {"content": "dict content"}}]}
    assert _extract_text(resp) == "dict content"


def test_extract_text_no_choices():
    assert _extract_text({}) == ""
    assert _extract_text(object()) == ""


def test_extract_text_no_message():
    assert _extract_text({"choices": [{}]}) == ""


def test_extract_text_content_is_list():
    resp = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"text": "part1"},
                        {"text": "part2"},
                        {"image": "ignored"},
                    ]
                }
            }
        ]
    }
    assert _extract_text(resp) == "part1part2"


def test_extract_text_content_none():
    resp = {"choices": [{"message": {"content": None}}]}
    assert _extract_text(resp) == ""


def test_extract_text_content_whitespace():
    resp = {"choices": [{"message": {"content": "  trimmed  "}}]}
    assert _extract_text(resp) == "trimmed"


def test_extract_text_choices_none():
    resp = {"choices": None}
    assert _extract_text(resp) == ""


def test_extract_text_message_none_in_dict():
    resp = {"choices": [{"message": None}]}
    assert _extract_text(resp) == ""


def test_extract_text_object_choice_with_dict_message():
    """Object choice but dict message."""

    class Choice:
        message = {"content": "mixed"}

    class Response:
        choices = [Choice()]

    assert _extract_text(Response()) == "mixed"


def test_extract_text_content_list_with_non_dict():
    """Content list that contains non-dict items."""
    resp = {"choices": [{"message": {"content": [42, "string"]}}]}
    assert _extract_text(resp) == ""


def test_extract_text_content_is_int():
    resp = {"choices": [{"message": {"content": 42}}]}
    assert _extract_text(resp) == ""


# ---------------------------------------------------------------------------
# _load_key_from_file
# ---------------------------------------------------------------------------


def test_load_key_from_file_not_found(tmp_path: Path):
    assert _load_key_from_file(str(tmp_path / "missing")) == ""


def test_load_key_from_file_openai_api_key(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text("OPENAI_API_KEY=dummy-test-123\n")
    assert _load_key_from_file(str(f)) == "dummy-test-123"


def test_load_key_from_file_openai_key_fallback(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text("OPENAI_KEY=dummy-fallback\n")
    assert _load_key_from_file(str(f)) == "dummy-fallback"


def test_load_key_from_file_primary_wins(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text("OPENAI_KEY=fallback\nOPENAI_API_KEY=primary\n")
    assert _load_key_from_file(str(f)) == "primary"


def test_load_key_from_file_comments_and_blanks(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text("# comment\n\nOPENAI_API_KEY=dummy-real\n")
    assert _load_key_from_file(str(f)) == "dummy-real"


def test_load_key_from_file_export_prefix(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text("export OPENAI_API_KEY=dummy-exported\n")
    assert _load_key_from_file(str(f)) == "dummy-exported"


def test_load_key_from_file_quoted_values(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text('OPENAI_API_KEY="dummy-quoted"\n')
    assert _load_key_from_file(str(f)) == "dummy-quoted"


def test_load_key_from_file_no_equals(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text("just-a-line-without-equals\n")
    assert _load_key_from_file(str(f)) == ""


def test_load_key_from_file_irrelevant_keys(tmp_path: Path):
    f = tmp_path / ".api_key"
    f.write_text("OTHER_KEY=value\nFOO=bar\n")
    assert _load_key_from_file(str(f)) == ""


# ---------------------------------------------------------------------------
# OpenAISettings
# ---------------------------------------------------------------------------


def test_openai_settings_defaults():
    s = OpenAISettings(api_key=None, api_key_fallback=None)
    assert s.model == "gpt-4o-mini"
    assert s.has_key is False


def test_openai_settings_resolved_api_key_primary():
    s = OpenAISettings(api_key="dummy-primary")
    assert s.resolved_api_key == "dummy-primary"


def test_openai_settings_resolved_api_key_fallback():
    s = OpenAISettings(api_key=None, api_key_fallback="dummy-fallback")
    assert s.resolved_api_key == "dummy-fallback"


def test_openai_settings_resolved_api_key_missing():
    s = OpenAISettings(api_key=None, api_key_fallback=None)
    with pytest.raises(MissingAPIKeyError):
        _ = s.resolved_api_key


def test_openai_settings_has_key():
    assert OpenAISettings(api_key="x").has_key is True
    assert OpenAISettings(api_key_fallback="y").has_key is True


def test_openai_settings_blank_key_not_resolved():
    s = OpenAISettings(api_key="  ", api_key_fallback="  ")
    with pytest.raises(MissingAPIKeyError):
        _ = s.resolved_api_key


# ---------------------------------------------------------------------------
# load_openai_settings
# ---------------------------------------------------------------------------


def test_load_openai_settings_with_file_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key_file = tmp_path / ".api_key"
    key_file.write_text("OPENAI_API_KEY=dummy-from-file\n")
    monkeypatch.chdir(tmp_path)
    # Clear env vars
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    settings = load_openai_settings()
    assert settings.resolved_api_key == "dummy-from-file"


def test_load_openai_settings_without_any_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        load_openai_settings()


def test_load_openai_settings_with_env_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-env-key")
    settings = load_openai_settings()
    assert settings.resolved_api_key == "dummy-env-key"


# ---------------------------------------------------------------------------
# chat_with_seed
# ---------------------------------------------------------------------------


def test_llm_contract_chat_declares_seed_parameter():
    signature = inspect.signature(LLMContract.chat)
    client_signature = inspect.signature(LLMClient.chat)

    assert "seed" in signature.parameters
    assert signature.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["seed"].default is None
    assert signature.parameters["seed"].annotation == client_signature.parameters["seed"].annotation


def test_chat_with_seed_client_accepting_seed():
    class SeedClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return f"seeded-{seed}"

    text, used_seed = chat_with_seed(
        llm_client=SeedClient(),
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=100,
        seed=42,
    )
    assert text == "seeded-42"
    assert used_seed is True


def test_chat_with_seed_type_error_propagates():
    class BrokenClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            raise TypeError("unrelated error")

    with pytest.raises(TypeError, match="unrelated"):
        chat_with_seed(
            llm_client=BrokenClient(),
            messages=[],
            temperature=0.7,
            max_tokens=100,
            seed=42,
        )


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def test_resolve_phase2_llm_client_disabled_capability():
    client, source = resolve_phase2_llm_client(
        setting="multi_agent_with_negotiation",
        phase2_uses_llm=False,
        model="gpt-4o-mini",
    )
    assert client is None
    assert source == "disabled"


def test_resolve_phase2_llm_client_quare_single_agent_disabled():
    client, source = resolve_phase2_llm_client(
        setting="single_agent",
        phase2_uses_llm=True,
        model="gpt-4o-mini",
    )
    assert client is None
    assert source == "disabled"


def test_resolve_phase2_llm_client_quare_injected():
    class FakeClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return ""

    client, source = resolve_phase2_llm_client(
        setting="multi_agent_with_negotiation",
        phase2_uses_llm=True,
        model="gpt-4o-mini",
        llm_client=FakeClient(),
    )
    assert client is not None
    assert source == "injected"


def test_resolve_phase2_llm_client_quare_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    client, source = resolve_phase2_llm_client(
        setting="multi_agent_with_negotiation",
        phase2_uses_llm=True,
        model="gpt-4o-mini",
    )
    assert client is None
    assert source == "missing_api_key"


def test_resolve_runtime_llm_client_disabled_capability():
    client, source = resolve_runtime_llm_client(
        setting="multi_agent_with_negotiation",
        runtime_uses_llm=False,
        model="gpt-4o-mini",
    )
    assert client is None
    assert source == "disabled"


def test_resolve_runtime_llm_client_mare_single_agent_disabled():
    client, source = resolve_runtime_llm_client(
        setting="single_agent",
        runtime_uses_llm=True,
        model="gpt-4o-mini",
    )
    assert client is None
    assert source == "disabled"


def test_resolve_runtime_llm_client_mare_injected():
    class FakeClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return ""

    client, source = resolve_runtime_llm_client(
        setting="multi_agent_with_negotiation",
        runtime_uses_llm=True,
        model="gpt-4o-mini",
        llm_client=FakeClient(),
    )
    assert client is not None
    assert source == "injected"


def test_resolve_runtime_llm_client_iredev_injected():
    class FakeClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return ""

    client, source = resolve_runtime_llm_client(
        setting="multi_agent_with_negotiation",
        runtime_uses_llm=True,
        model="gpt-4o-mini",
        llm_client=FakeClient(),
    )
    assert source == "injected"


def test_resolve_runtime_llm_client_mare_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    client, source = resolve_runtime_llm_client(
        setting="multi_agent_with_negotiation",
        runtime_uses_llm=True,
        model="gpt-4o-mini",
    )
    assert client is None
    assert source == "missing_api_key"


# ---------------------------------------------------------------------------
# LLMClient (unit-level, without real LiteLLM)
# ---------------------------------------------------------------------------


def test_llm_client_init():
    settings = OpenAISettings(api_key="dummy-test")
    client = LLMClient(settings)
    assert isinstance(client, LLMContract)


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


def test_errors_missing_api_key_error():
    with pytest.raises(MissingAPIKeyError):
        raise MissingAPIKeyError("test")


def test_errors_llm_client_error():
    with pytest.raises(LLMClientError):
        raise LLMClientError("test")

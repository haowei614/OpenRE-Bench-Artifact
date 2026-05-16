"""Coverage tests for openre_bench.pipeline._core — pure utility functions and ActionRunResult."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openre_bench.pipeline._core import (
    ActionRunResult,
    RuntimeExecutionMeta,
    Phase2ExecutionMeta,
    _CORPUS_HASH_CACHE,
    _RAG_CHUNK_CACHE,
    _coerce_action_items,
    _chunk_overlap_score,
    _chunk_text,
    _hash_corpus_dir,
    _load_rag_chunks,
    _prepare_rag_context,
    _prompt_contract_hash,
    _rag_payload,
    _run_llm_action_with_fallback,
    _runtime_semantics_mode,
    _sha256_payload,
    default_run_id,
)


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_empty():
    assert _chunk_text("") == []


def test_chunk_text_short_paragraphs_filtered():
    text = "Short paragraph.\n\nAnother short one."
    assert _chunk_text(text) == []


def test_chunk_text_long_paragraph():
    text = "A" * 100 + "\n\n" + "B" * 100
    chunks = _chunk_text(text)
    assert len(chunks) == 2


def test_chunk_text_very_long_paragraph_split():
    text = "X" * 1600
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_carriage_return_normalized():
    text = "A" * 100 + "\r\n\r\n" + "B" * 100
    chunks = _chunk_text(text)
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# _chunk_overlap_score
# ---------------------------------------------------------------------------


def test_chunk_overlap_score_full_overlap():
    # _tokens("hello world") → ["ello", "orld"] because regex is [a-z0-9_]+
    score = _chunk_overlap_score(query="abc def", chunk_tokens={"abc", "def"})
    assert score == pytest.approx(1.0)


def test_chunk_overlap_score_no_overlap():
    score = _chunk_overlap_score(query="hello world", chunk_tokens={"xyz"})
    assert score == pytest.approx(0.0)


def test_chunk_overlap_score_empty_query():
    score = _chunk_overlap_score(query="", chunk_tokens={"hello"})
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _rag_payload
# ---------------------------------------------------------------------------


def test_rag_payload_disabled():
    result = _rag_payload(query="test", rag_context={"rag_enabled": False})
    assert result["source"] == "openre_bench.phase1"
    assert result["retrieved_chunks"] == []


def test_rag_payload_no_chunks():
    result = _rag_payload(query="test", rag_context={"rag_enabled": True, "chunks": []})
    assert result["source"] == "openre_bench.phase1.rag_fallback"


def test_rag_payload_with_chunks():
    chunks = [
        {
            "chunk_id": "c1",
            "document": "doc1.txt",
            "text": "The system shall authenticate users via secure protocols.",
            "tokens": {"system", "authenticate", "users", "secure", "protocols"},
        },
        {
            "chunk_id": "c2",
            "document": "doc2.txt",
            "text": "Performance requirements specify response latency targets for services.",
            "tokens": {"performance", "requirements", "latency", "targets"},
        },
    ]
    result = _rag_payload(
        query="authenticate users",
        rag_context={"rag_enabled": True, "chunks": chunks},
    )
    assert result["source"] == "openre_bench.phase1.rag"
    assert result["source_chunk_id"] is not None
    assert len(result["retrieved_chunks"]) <= 3


# ---------------------------------------------------------------------------
# _runtime_semantics_mode
# ---------------------------------------------------------------------------


def test_runtime_semantics_mode_mare_multi():
    mode = _runtime_semantics_mode(system="mare", setting="multi_agent_with_negotiation")
    assert mode == "mare_paper_workflow_v1"


def test_runtime_semantics_mode_mare_single():
    mode = _runtime_semantics_mode(system="mare", setting="single_agent")
    assert mode == "mare_single_agent_baseline"


def test_runtime_semantics_mode_iredev_multi():
    mode = _runtime_semantics_mode(system="iredev", setting="multi_agent_with_negotiation")
    assert mode == "iredev_knowledge_driven_v1"


def test_runtime_semantics_mode_iredev_single():
    mode = _runtime_semantics_mode(system="iredev", setting="single_agent")
    assert mode == "iredev_single_agent_baseline"


def test_runtime_semantics_mode_quare():
    mode = _runtime_semantics_mode(system="quare", setting="multi_agent_with_negotiation")
    assert mode == "quare_dialectic_scaffold_v1"


# ---------------------------------------------------------------------------
# _prompt_contract_hash
# ---------------------------------------------------------------------------


def test_prompt_contract_hash_deterministic():
    h1 = _prompt_contract_hash(system="mare", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    h2 = _prompt_contract_hash(system="mare", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    assert h1 == h2
    assert len(h1) == 64


def test_prompt_contract_hash_different_systems():
    h_mare = _prompt_contract_hash(system="mare", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    h_iredev = _prompt_contract_hash(system="iredev", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    h_quare = _prompt_contract_hash(system="quare", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    assert h_mare != h_iredev
    assert h_mare != h_quare


def test_prompt_contract_hash_single_agent():
    h = _prompt_contract_hash(system="mare", setting="single_agent", round_cap=3, max_tokens=500)
    assert len(h) == 64


# ---------------------------------------------------------------------------
# _sha256_payload
# ---------------------------------------------------------------------------


def test_sha256_payload_deterministic():
    payload = {"key": "value"}
    h1 = _sha256_payload(payload)
    h2 = _sha256_payload(payload)
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_payload_matches_utils_for_non_ascii_payload():
    from openre_bench.pipeline._utils import _sha256_payload as utils_sha256_payload

    payload = {
        "korean": "정합성",
        "items": ["MARE", "iReDev", "QUARE"],
        "nested": {"symbols": ["α", "β"], "enabled": True},
    }
    assert _sha256_payload(payload) == utils_sha256_payload(payload)


# ---------------------------------------------------------------------------
# default_run_id
# ---------------------------------------------------------------------------


def test_default_run_id_format():
    rid = default_run_id("ATM System", 42)
    assert "atm-system" in rid
    assert "s042" in rid


def test_default_run_id_special_chars():
    rid = default_run_id("Test!@#$", 1)
    assert "s001" in rid


# ---------------------------------------------------------------------------
# ActionRunResult
# ---------------------------------------------------------------------------


def test_action_run_result_dataclass():
    r = ActionRunResult(
        items=["a", "b"],
        summary="test",
        execution_mode="llm",
        llm_generated=True,
        fallback_reason="",
    )
    assert r.items == ["a", "b"]
    assert r.llm_generated is True


# ---------------------------------------------------------------------------
# _coerce_action_items
# ---------------------------------------------------------------------------


def test_coerce_action_items_accepts_dict_items():
    result = _coerce_action_items(
        [
            {
                "id": "UR-1",
                "statement": "The system shall keep generated requirements non-empty.",
                "priority": "Must",
            }
        ],
        limit=3,
    )

    assert result == ["The system shall keep generated requirements non-empty."]


def test_run_llm_action_with_fallback_accepts_action_specific_list_key():
    class ActionKeyClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return json.dumps(
                {
                    "user_requirements": [
                        {"statement": "The system shall preserve iReDev output items."}
                    ],
                    "summary": "Action-specific key returned.",
                }
            )

    meta = RuntimeExecutionMeta()
    result = _run_llm_action_with_fallback(
        llm_client=ActionKeyClient(),
        messages=[{"role": "user", "content": "test"}],
        llm_temperature=0.7,
        llm_max_tokens=500,
        llm_seed=42,
        llm_source="test",
        llm_meta=meta,
        coerce_items_fn=_coerce_action_items,
        max_items=5,
        fallback_items=["fallback"],
    )

    assert result.execution_mode == "llm"
    assert result.llm_generated is True
    assert result.items == ["The system shall preserve iReDev output items."]
    assert meta.llm_turns == 1
    assert meta.llm_fallback_turns == 0


def test_run_llm_action_with_fallback_recovers_malformed_items_block():
    class MalformedItemsClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return """
            {
              "items": [
                @startuml
                actor User
                User --> System : request
                @enduml
              ],
              "summary": "PlantUML"
            }
            """

    meta = RuntimeExecutionMeta()
    result = _run_llm_action_with_fallback(
        llm_client=MalformedItemsClient(),
        messages=[{"role": "user", "content": "test"}],
        llm_temperature=0.7,
        llm_max_tokens=500,
        llm_seed=42,
        llm_source="test",
        llm_meta=meta,
        coerce_items_fn=_coerce_action_items,
        max_items=5,
        fallback_items=["fallback"],
    )

    assert result.execution_mode == "llm"
    assert result.llm_generated is True
    assert result.items == [
        "@startuml",
        "actor User",
        "User --> System : request",
        "@enduml",
    ]
    assert result.fallback_reason == ""
    assert meta.llm_turns == 1
    assert meta.llm_parse_recoveries == 1
    assert meta.llm_retry_count == 0
    assert meta.llm_fallback_turns == 0


# ---------------------------------------------------------------------------
# _run_llm_action_with_fallback
# ---------------------------------------------------------------------------


def test_run_llm_action_with_fallback_no_client_fallback():
    meta = RuntimeExecutionMeta()
    result = _run_llm_action_with_fallback(
        llm_client=None,
        messages=[],
        llm_temperature=0.7,
        llm_max_tokens=500,
        llm_seed=42,
        llm_source="disabled",
        llm_meta=meta,
        coerce_items_fn=lambda items, limit: items or [],
        max_items=5,
        fallback_items=["fallback_item"],
    )
    assert result.execution_mode == "fallback"
    assert result.items == ["fallback_item"]
    assert result.llm_generated is False
    assert meta.llm_fallback_turns == 1


def test_run_llm_action_with_fallback_client_raises_llm_error():
    from openre_bench.llm import LLMClientError

    class FailingClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            raise LLMClientError("API error")

    meta = RuntimeExecutionMeta()
    result = _run_llm_action_with_fallback(
        llm_client=FailingClient(),
        messages=[{"role": "user", "content": "test"}],
        llm_temperature=0.7,
        llm_max_tokens=500,
        llm_seed=42,
        llm_source="test",
        llm_meta=meta,
        coerce_items_fn=lambda items, limit: items or [],
        max_items=5,
        fallback_items=["fallback"],
    )
    assert result.execution_mode == "fallback"
    assert "request_failed" in result.fallback_reason


def test_run_llm_action_with_fallback_client_returns_valid_json():
    class GoodClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return json.dumps({"items": ["item1", "item2"], "summary": "LLM summary"})

    meta = RuntimeExecutionMeta()
    result = _run_llm_action_with_fallback(
        llm_client=GoodClient(),
        messages=[{"role": "user", "content": "test"}],
        llm_temperature=0.7,
        llm_max_tokens=500,
        llm_seed=42,
        llm_source="test",
        llm_meta=meta,
        coerce_items_fn=lambda items, limit: [str(i) for i in (items or [])[:limit]],
        max_items=5,
        fallback_items=["fallback"],
    )
    assert result.execution_mode == "llm"
    assert result.llm_generated is True
    assert "item1" in result.items
    assert meta.llm_turns == 1


def test_run_llm_action_with_fallback_client_returns_empty_items_then_fallback():
    class EmptyClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return json.dumps({"items": [], "summary": "empty"})

    meta = RuntimeExecutionMeta()
    result = _run_llm_action_with_fallback(
        llm_client=EmptyClient(),
        messages=[{"role": "user", "content": "test"}],
        llm_temperature=0.7,
        llm_max_tokens=500,
        llm_seed=42,
        llm_source="test",
        llm_meta=meta,
        coerce_items_fn=lambda items, limit: [str(i) for i in (items or [])[:limit]],
        max_items=5,
        fallback_items=["fallback"],
    )
    assert result.execution_mode == "fallback"
    assert "empty_items" in result.fallback_reason


def test_run_llm_action_with_fallback_client_returns_bad_json():
    class BadJsonClient:
        def chat(self, messages, *, temperature=0.0, max_tokens=None, seed=None):
            return "not json at all"

    meta = RuntimeExecutionMeta()
    result = _run_llm_action_with_fallback(
        llm_client=BadJsonClient(),
        messages=[{"role": "user", "content": "test"}],
        llm_temperature=0.7,
        llm_max_tokens=500,
        llm_seed=42,
        llm_source="test",
        llm_meta=meta,
        coerce_items_fn=lambda items, limit: items or [],
        max_items=5,
        fallback_items=["fallback"],
    )
    assert result.execution_mode == "fallback"
    assert "parse_failed" in result.fallback_reason


# ---------------------------------------------------------------------------
# RuntimeExecutionMeta / Phase2ExecutionMeta
# ---------------------------------------------------------------------------


def test_execution_meta_mare_meta_defaults():
    meta = RuntimeExecutionMeta()
    assert meta.llm_enabled is False
    assert meta.execution_mode == "deterministic_emulation"
    assert meta.llm_turns == 0


def test_execution_meta_phase2_meta_defaults():
    meta = Phase2ExecutionMeta()
    assert meta.llm_source == "disabled"
    assert meta.llm_enabled is False


def test_execution_meta_accumulation():
    meta = RuntimeExecutionMeta()
    meta.llm_turns += 1
    meta.llm_fallback_turns += 2
    assert meta.llm_turns == 1
    assert meta.llm_fallback_turns == 2


# ---------------------------------------------------------------------------
# _load_rag_chunks
# ---------------------------------------------------------------------------


def test_load_rag_chunks_missing_dir(tmp_path: Path):
    _RAG_CHUNK_CACHE.clear()
    result = _load_rag_chunks(tmp_path / "nonexistent")
    assert result == []


def test_load_rag_chunks_empty_dir(tmp_path: Path):
    _RAG_CHUNK_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    result = _load_rag_chunks(corpus)
    assert result == []


def test_load_rag_chunks_load_text_file(tmp_path: Path):
    _RAG_CHUNK_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Create a long enough text file to generate chunks
    content = "A" * 120 + "\n\n" + "B" * 120
    (corpus / "doc.txt").write_text(content)
    result = _load_rag_chunks(corpus)
    assert len(result) >= 1
    assert result[0]["document"] == "doc.txt"


def test_load_rag_chunks_skips_non_allowed_suffixes(tmp_path: Path):
    _RAG_CHUNK_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "image.png").write_bytes(b"\x89PNG")
    result = _load_rag_chunks(corpus)
    assert result == []


def test_load_rag_chunks_caching(tmp_path: Path):
    _RAG_CHUNK_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text("X" * 120)
    r1 = _load_rag_chunks(corpus)
    r2 = _load_rag_chunks(corpus)
    assert r1 is r2  # Same object from cache


# ---------------------------------------------------------------------------
# _hash_corpus_dir
# ---------------------------------------------------------------------------


def test_hash_corpus_dir_missing_dir(tmp_path: Path):
    _CORPUS_HASH_CACHE.clear()
    result = _hash_corpus_dir(tmp_path / "nonexistent")
    assert result == ""


def test_hash_corpus_dir_empty_dir(tmp_path: Path):
    _CORPUS_HASH_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    result = _hash_corpus_dir(corpus)
    assert len(result) == 64


def test_hash_corpus_dir_deterministic(tmp_path: Path):
    _CORPUS_HASH_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Hello world")
    h1 = _hash_corpus_dir(corpus)
    _CORPUS_HASH_CACHE.clear()
    h2 = _hash_corpus_dir(corpus)
    assert h1 == h2


def test_hash_corpus_dir_caching(tmp_path: Path):
    _CORPUS_HASH_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    h1 = _hash_corpus_dir(corpus)
    h2 = _hash_corpus_dir(corpus)
    assert h1 == h2


# ---------------------------------------------------------------------------
# _prepare_rag_context
# ---------------------------------------------------------------------------


def test_prepare_rag_context_disabled():
    _RAG_CHUNK_CACHE.clear()
    _CORPUS_HASH_CACHE.clear()
    result = _prepare_rag_context(rag_enabled=False, rag_backend="local_tfidf", rag_corpus_dir=None)
    assert result["rag_enabled"] is False
    assert result["chunks"] == []


def test_prepare_rag_context_uses_packaged_default_when_repo_data_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from openre_bench.pipeline import _core as core

    _RAG_CHUNK_CACHE.clear()
    _CORPUS_HASH_CACHE.clear()
    cwd = tmp_path / "cwd"
    packaged_root = tmp_path / "packaged"
    packaged_corpus = packaged_root / "data" / "knowledge_base"
    cwd.mkdir()
    packaged_corpus.mkdir(parents=True)
    (packaged_corpus / "doc.txt").write_text("Packaged corpus text " * 20)

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(core.resources, "files", lambda package: packaged_root)

    result = _prepare_rag_context(rag_enabled=True, rag_backend="local_tfidf", rag_corpus_dir=None)

    assert result["rag_corpus_dir"] == str(packaged_corpus.resolve())
    assert result["chunk_count"] >= 1
    assert result["fallback_used"] is False


def test_prepare_rag_context_enabled_with_corpus(tmp_path: Path):
    _RAG_CHUNK_CACHE.clear()
    _CORPUS_HASH_CACHE.clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("X" * 120)
    result = _prepare_rag_context(rag_enabled=True, rag_backend="local_tfidf", rag_corpus_dir=corpus)
    assert result["rag_enabled"] is True
    assert result["chunk_count"] >= 1
    assert result["corpus_hash"] != ""


def test_prepare_rag_context_enabled_empty_corpus(tmp_path: Path):
    _RAG_CHUNK_CACHE.clear()
    _CORPUS_HASH_CACHE.clear()
    corpus = tmp_path / "empty_corpus"
    corpus.mkdir()
    result = _prepare_rag_context(rag_enabled=True, rag_backend="", rag_corpus_dir=corpus)
    assert result["rag_enabled"] is True
    assert result["fallback_used"] is True
    assert result["rag_backend"] == "local_tfidf"  # default

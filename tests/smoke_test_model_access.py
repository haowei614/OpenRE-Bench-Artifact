"""Smoke tests: verify QUARE, MARE, and iReDev can access models via .api_key.

Run with:  uv run python -m pytest tests/smoke_test_model_access.py -v -s
"""

from __future__ import annotations

from pathlib import Path

from openre_bench.llm import (
    LLMClient,
    LLMContract,
    load_openai_settings,
)
from openre_bench.schemas import (
    SETTING_MULTI_AGENT_WITH_NEGOTIATION,
    SYSTEM_IREDEV,
    SYSTEM_MARE,
    SYSTEM_QUARE,
)
from openre_bench.pipeline._core import (
    PipelineConfig,
    _resolve_phase2_llm_client,
    _resolve_runtime_llm_client,
    run_case_pipeline,
)


CASE_INPUT = Path(__file__).resolve().parent.parent / "data" / "case_studies" / "ATM_input.json"
PROMPT = [{"role": "user", "content": "Say exactly one word: OK"}]


# ---------------------------------------------------------------------------
# 1. Raw LLMClient smoke (model reachable at all?)
# ---------------------------------------------------------------------------


def test_raw_llm_client_can_reach_model():
    """Load settings from .api_key and make one tiny LLM call."""
    settings = load_openai_settings()
    client = LLMClient(settings)
    assert isinstance(client, LLMContract)
    reply = client.chat(PROMPT, temperature=0.0, max_tokens=10)
    assert isinstance(reply, str) and len(reply) > 0
    print(f"  [raw]   model={settings.model}  reply={reply!r}")


# ---------------------------------------------------------------------------
# 2. Resolver smoke — each system gets a live client with correct model
# ---------------------------------------------------------------------------


def _make_config(system: str, tmp_path: Path) -> PipelineConfig:
    artifacts = tmp_path / system
    artifacts.mkdir(parents=True, exist_ok=True)
    return PipelineConfig(
        case_input=CASE_INPUT,
        artifacts_dir=artifacts,
        run_record_path=artifacts / "run_record.json",
        run_id=f"smoke-{system}",
        setting=SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        seed=42,
        model="gpt-4o-mini",
        temperature=0.0,
        round_cap=2,
        max_tokens=512,
        system=system,
    )


def test_quare_resolver_returns_live_client(tmp_path: Path):
    """QUARE phase-2 resolver builds an LLMClient with model set."""
    cfg = _make_config(SYSTEM_QUARE, tmp_path)
    client, source = _resolve_phase2_llm_client(config=cfg, phase2_uses_llm=True)
    assert source == "openai", f"expected 'openai' source, got '{source}'"
    assert client is not None
    reply = client.chat(PROMPT, temperature=0.0, max_tokens=10)
    assert isinstance(reply, str) and len(reply) > 0
    print(f"  [QUARE] source={source}  reply={reply!r}")


def test_mare_resolver_returns_live_client(tmp_path: Path):
    """MARE runtime resolver builds an LLMClient with model set."""
    cfg = _make_config(SYSTEM_MARE, tmp_path)
    client, source = _resolve_runtime_llm_client(config=cfg, runtime_uses_llm=True)
    assert source == "openai", f"expected 'openai' source, got '{source}'"
    assert client is not None
    reply = client.chat(PROMPT, temperature=0.0, max_tokens=10)
    assert isinstance(reply, str) and len(reply) > 0
    print(f"  [MARE]  source={source}  reply={reply!r}")


def test_iredev_resolver_returns_live_client(tmp_path: Path):
    """iReDev runtime resolver builds an LLMClient with model set."""
    cfg = _make_config(SYSTEM_IREDEV, tmp_path)
    client, source = _resolve_runtime_llm_client(config=cfg, runtime_uses_llm=True)
    assert source == "openai", f"expected 'openai' source, got '{source}'"
    assert client is not None
    reply = client.chat(PROMPT, temperature=0.0, max_tokens=10)
    assert isinstance(reply, str) and len(reply) > 0
    print(f"  [iReDev] source={source}  reply={reply!r}")


# ---------------------------------------------------------------------------
# 3. Full pipeline smoke — each system can run a case end-to-end
# ---------------------------------------------------------------------------


def test_quare_pipeline_smoke(tmp_path: Path):
    """QUARE full pipeline produces a run record with LLM notes."""
    cfg = _make_config(SYSTEM_QUARE, tmp_path)
    record = run_case_pipeline(cfg)
    assert record.system == SYSTEM_QUARE
    phase2_llm = record.notes.get("phase2_llm", {})
    print(f"  [QUARE pipeline] phase2_llm.source={phase2_llm.get('source')}"
          f"  turns={phase2_llm.get('turns')}")
    assert phase2_llm.get("source") == "openai"
    assert phase2_llm.get("turns", 0) > 0


def test_mare_pipeline_smoke(tmp_path: Path):
    """MARE full pipeline produces a run record with runtime LLM notes."""
    cfg = _make_config(SYSTEM_MARE, tmp_path)
    record = run_case_pipeline(cfg)
    assert record.system == SYSTEM_MARE
    runtime_llm = record.notes.get("runtime_llm", {})
    print(f"  [MARE pipeline]  runtime_llm.source={runtime_llm.get('source')}"
          f"  turns={runtime_llm.get('turns')}"
          f"  mode={runtime_llm.get('execution_mode')}")
    assert runtime_llm.get("source") == "openai"
    assert runtime_llm.get("turns", 0) > 0


def test_iredev_pipeline_smoke(tmp_path: Path):
    """iReDev full pipeline produces a run record with runtime LLM notes."""
    cfg = _make_config(SYSTEM_IREDEV, tmp_path)
    record = run_case_pipeline(cfg)
    assert record.system == SYSTEM_IREDEV
    runtime_llm = record.notes.get("runtime_llm", {})
    print(f"  [iReDev pipeline] runtime_llm.source={runtime_llm.get('source')}"
          f"  turns={runtime_llm.get('turns')}"
          f"  mode={runtime_llm.get('execution_mode')}")
    assert runtime_llm.get("source") == "openai"
    assert runtime_llm.get("turns", 0) > 0

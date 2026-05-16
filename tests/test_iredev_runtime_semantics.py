"""Regression tests for paper-faithful iReDev runtime semantics guardrails."""

from __future__ import annotations

from pathlib import Path

from openre_bench.comparison_validator import validate_run_record
from openre_bench.comparison_validator import validate_system_behavior_contract
from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline import run_case_pipeline
from openre_bench.schemas import IREDEV_ACTIONS
from openre_bench.schemas import IREDEV_AGENT_ROLES
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SYSTEM_IREDEV
from openre_bench.schemas import load_json_file
from openre_bench.schemas import write_json_file
from tests.fake_iredev_llm import ScriptedIredevLLMClient


def _write_case(path: Path) -> None:
    write_json_file(
        path,
        {
            "case_name": "ATM",
            "case_description": "ATM requirements with explicit stakeholder conflict language.",
            "requirement": (
                "The ATM shall enforce strong fraud detection while preserving low-latency response. "
                "If tradeoff conflict appears, reviewers shall document resolution rationale and "
                "verification evidence."
            ),
        },
    )


def _write_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "guidance.md").write_text(
        (
            "ATM guidance requires explicit user stories, requirement drafts, model extraction, "
            "checker reports, and final SRS sections with traceable rationale."
        ),
        encoding="utf-8",
    )


def _run_iredev_case(tmp_path: Path) -> Path:
    case_path = tmp_path / "iredev-case.json"
    corpus_dir = tmp_path / "iredev-corpus"
    artifacts_dir = tmp_path / "iredev-artifacts"
    run_record_path = artifacts_dir / "run_record.json"

    _write_case(case_path)
    _write_corpus(corpus_dir)

    run_case_pipeline(
        PipelineConfig(
            case_input=case_path,
            artifacts_dir=artifacts_dir,
            run_record_path=run_record_path,
            run_id="iredev-atm-niv-s101",
            setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
            seed=101,
            model="gpt-4o-mini",
            temperature=0.7,
            round_cap=3,
            max_tokens=4000,
            system=SYSTEM_IREDEV,
            rag_enabled=True,
            rag_backend="local_tfidf",
            rag_corpus_dir=corpus_dir,
            llm_client=ScriptedIredevLLMClient(),
        )
    )
    return artifacts_dir


def test_iredev_multi_agent_run_emits_roles_and_actions_semantics(tmp_path: Path) -> None:
    artifacts_dir = _run_iredev_case(tmp_path)

    phase1 = load_json_file(artifacts_dir / "phase1_initial_models.json")
    assert sorted(phase1.keys()) == sorted(IREDEV_AGENT_ROLES)

    run_record = load_json_file(artifacts_dir / "run_record.json")
    runtime_semantics = run_record["notes"]["runtime_semantics"]
    assert runtime_semantics["mode"] == "iredev_knowledge_driven_v1"
    assert runtime_semantics["execution_mode"] == "llm_driven"
    assert int(runtime_semantics["llm_seed"]) == 101
    assert int(runtime_semantics["llm_seed_applied_turns"]) >= 0
    assert int(runtime_semantics["llm_seed_applied_turns"]) <= int(runtime_semantics["llm_turns"])
    assert int(runtime_semantics["llm_turns"]) >= len(IREDEV_ACTIONS)
    assert int(runtime_semantics["llm_fallback_turns"]) == 0
    assert sorted(runtime_semantics["roles_executed"]) == sorted(IREDEV_AGENT_ROLES)
    assert sorted(runtime_semantics["actions_executed"]) == sorted(IREDEV_ACTIONS)
    assert set(runtime_semantics["llm_actions"]) == set(IREDEV_ACTIONS)
    assert runtime_semantics["fallback_actions"] == []
    assert len(runtime_semantics["workspace_digest"]) == 64

    run_report = validate_run_record(artifacts_dir / "run_record.json")
    behavior_report = validate_system_behavior_contract(system=SYSTEM_IREDEV, artifacts_dir=artifacts_dir)
    assert run_report.errors == []
    assert behavior_report.errors == []


def test_iredev_runtime_guardrail_rejects_missing_action_trace(tmp_path: Path) -> None:
    artifacts_dir = _run_iredev_case(tmp_path)
    run_record_path = artifacts_dir / "run_record.json"
    payload = load_json_file(run_record_path)

    payload["notes"]["runtime_semantics"]["actions_executed"] = list(IREDEV_ACTIONS[:-1])
    write_json_file(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("actions_executed" in error for error in report.errors)


def test_iredev_behavior_guardrail_rejects_missing_phase1_role(tmp_path: Path) -> None:
    artifacts_dir = _run_iredev_case(tmp_path)
    phase1_path = artifacts_dir / "phase1_initial_models.json"
    payload = load_json_file(phase1_path)
    payload.pop("Reviewer")
    write_json_file(phase1_path, payload)

    report = validate_system_behavior_contract(system=SYSTEM_IREDEV, artifacts_dir=artifacts_dir)
    assert any("missing required roles" in error for error in report.errors)


def test_iredev_phase1_has_six_agents_with_kaos_elements(tmp_path: Path) -> None:
    artifacts_dir = _run_iredev_case(tmp_path)
    phase1 = load_json_file(artifacts_dir / "phase1_initial_models.json")

    assert len(phase1) == 6
    for role in IREDEV_AGENT_ROLES:
        assert role in phase1
        elements = phase1[role]
        assert isinstance(elements, list)
        assert len(elements) >= 1
        root = elements[0]
        assert root["element_type"] == "Goal"
        assert root["hierarchy_level"] == 1
        assert root["stakeholder"] == role


def test_iredev_system_identity_in_run_record(tmp_path: Path) -> None:
    artifacts_dir = _run_iredev_case(tmp_path)
    run_record = load_json_file(artifacts_dir / "run_record.json")
    assert run_record["system"] == "iredev"
    assert run_record["system_identity"]["system_name"] == "iredev"

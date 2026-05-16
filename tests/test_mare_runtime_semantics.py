"""Regression tests for paper-faithful MARE runtime semantics guardrails."""

from __future__ import annotations

from pathlib import Path

from openre_bench.comparison_validator import validate_run_record
from openre_bench.comparison_validator import validate_system_behavior_contract
from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline import run_case_pipeline
from openre_bench.schemas import MARE_ACTIONS
from openre_bench.schemas import MARE_AGENT_ROLES
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import load_json_file
from openre_bench.schemas import write_json_file
from tests.fake_mare_llm import ScriptedMareLLMClient


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


def _run_mare_case(tmp_path: Path) -> Path:
    case_path = tmp_path / "mare-case.json"
    corpus_dir = tmp_path / "mare-corpus"
    artifacts_dir = tmp_path / "mare-artifacts"
    run_record_path = artifacts_dir / "run_record.json"

    _write_case(case_path)
    _write_corpus(corpus_dir)

    run_case_pipeline(
        PipelineConfig(
            case_input=case_path,
            artifacts_dir=artifacts_dir,
            run_record_path=run_record_path,
            run_id="mare-atm-niv-s101",
            setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
            seed=101,
            model="gpt-4o-mini",
            temperature=0.7,
            round_cap=3,
            max_tokens=4000,
            system=SYSTEM_MARE,
            rag_enabled=True,
            rag_backend="local_tfidf",
            rag_corpus_dir=corpus_dir,
            llm_client=ScriptedMareLLMClient(),
        )
    )
    return artifacts_dir


def test_mare_multi_agent_run_emits_roles_and_actions_semantics(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)

    phase1 = load_json_file(artifacts_dir / "phase1_initial_models.json")
    assert sorted(phase1.keys()) == sorted(MARE_AGENT_ROLES)

    run_record = load_json_file(artifacts_dir / "run_record.json")
    runtime_semantics = run_record["notes"]["runtime_semantics"]
    assert runtime_semantics["mode"] == "mare_paper_workflow_v1"
    assert runtime_semantics["execution_mode"] == "llm_driven"
    assert int(runtime_semantics["llm_seed"]) == 101
    assert int(runtime_semantics["llm_seed_applied_turns"]) >= 0
    assert int(runtime_semantics["llm_seed_applied_turns"]) <= int(runtime_semantics["llm_turns"])
    assert int(runtime_semantics["llm_turns"]) == len(MARE_ACTIONS)
    assert int(runtime_semantics["llm_fallback_turns"]) == 0
    assert sorted(runtime_semantics["roles_executed"]) == sorted(MARE_AGENT_ROLES)
    assert sorted(runtime_semantics["actions_executed"]) == sorted(MARE_ACTIONS)
    assert sorted(runtime_semantics["llm_actions"]) == sorted(MARE_ACTIONS)
    assert runtime_semantics["fallback_actions"] == []
    assert len(runtime_semantics["workspace_digest"]) == 64

    run_report = validate_run_record(artifacts_dir / "run_record.json")
    behavior_report = validate_system_behavior_contract(system=SYSTEM_MARE, artifacts_dir=artifacts_dir)
    assert run_report.errors == []
    assert behavior_report.errors == []


def test_mare_runtime_guardrail_rejects_missing_action_trace(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)
    run_record_path = artifacts_dir / "run_record.json"
    payload = load_json_file(run_record_path)

    payload["notes"]["runtime_semantics"]["actions_executed"] = list(MARE_ACTIONS[:-1])
    write_json_file(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("actions_executed" in error for error in report.errors)


def test_mare_runtime_guardrail_rejects_emulated_execution_mode(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)
    run_record_path = artifacts_dir / "run_record.json"
    payload = load_json_file(run_record_path)

    runtime = payload["notes"]["runtime_semantics"]
    runtime["execution_mode"] = "deterministic_emulation"
    runtime["llm_fallback_turns"] = 1
    runtime["fallback_actions"] = ["WriteSRS"]
    runtime["llm_actions"] = list(MARE_ACTIONS[:-1])
    write_json_file(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("execution_mode" in error for error in report.errors)


def test_mare_runtime_guardrail_handles_malformed_list_fields(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)
    run_record_path = artifacts_dir / "run_record.json"
    payload = load_json_file(run_record_path)

    runtime = payload["notes"]["runtime_semantics"]
    runtime["roles_executed"] = None
    runtime["actions_executed"] = None
    runtime["llm_actions"] = None
    runtime["fallback_actions"] = None
    write_json_file(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("roles_executed must be a list" in error for error in report.errors)
    assert any("actions_executed must be a list" in error for error in report.errors)
    assert any("llm_actions must be a list" in error for error in report.errors)
    assert any("fallback_actions must be a list" in error for error in report.errors)


def test_mare_runtime_guardrail_requires_boolean_llm_required(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)
    run_record_path = artifacts_dir / "run_record.json"
    payload = load_json_file(run_record_path)

    runtime = payload["notes"]["runtime_semantics"]
    runtime["llm_required"] = "true"
    write_json_file(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("llm_required must be a boolean" in error for error in report.errors)


def test_mare_runtime_guardrail_rejects_non_object_action_trace_step(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)
    run_record_path = artifacts_dir / "run_record.json"
    payload = load_json_file(run_record_path)

    runtime = payload["notes"]["runtime_semantics"]
    runtime["action_trace"][0] = "invalid"
    write_json_file(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("action_trace step must be an object" in error for error in report.errors)


def test_mare_behavior_guardrail_rejects_missing_phase1_role(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)
    phase1_path = artifacts_dir / "phase1_initial_models.json"
    payload = load_json_file(phase1_path)
    payload.pop("Documenter")
    write_json_file(phase1_path, payload)

    report = validate_system_behavior_contract(system=SYSTEM_MARE, artifacts_dir=artifacts_dir)
    assert any("missing required roles" in error for error in report.errors)


def test_mare_conflict_detection_uses_paper_role_mapping(tmp_path: Path) -> None:
    artifacts_dir = _run_mare_case(tmp_path)
    phase2 = load_json_file(artifacts_dir / "phase2_negotiation_trace.json")
    summary = phase2.get("summary_stats", {}) if isinstance(phase2, dict) else {}
    assert int(summary.get("detected_conflicts", 0)) > 0

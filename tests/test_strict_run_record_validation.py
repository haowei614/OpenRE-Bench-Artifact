"""Strict run-record metadata validation tests for MVP comparability gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openre_bench.comparison_validator import validate_run_record
from openre_bench.schemas import MARE_ACTIONS
from openre_bench.schemas import MARE_AGENT_ROLES
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SETTING_SINGLE_AGENT
from openre_bench.schemas import SYSTEM_MARE


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_run_record() -> dict[str, Any]:
    return {
        "run_id": "atm-negotiation-integration-verification-s101",
        "case_id": "ATM",
        "system": SYSTEM_MARE,
        "setting": SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
        "seed": 101,
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "round_cap": 3,
        "system_identity": {
            "system_name": SYSTEM_MARE,
            "implementation": "deterministic-parity-pipeline",
            "implementation_version": "0.1.0",
            "python_version": "3.11.0",
            "platform": "Linux",
            "machine": "x86_64",
        },
        "provenance": {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "seed": 101,
            "prompt_hash": _sha256("prompt"),
            "corpus_hash": _sha256("corpus"),
            "corpus_path": "/tmp/corpus",
        },
        "execution_flags": {
            "rag_fallback_used": False,
            "llm_fallback_used": False,
            "fallback_tainted": False,
            "retry_used": False,
            "retry_count": 0,
        },
        "comparability": {
            "is_comparable": True,
            "non_comparable_reasons": [],
        },
        "max_tokens": 4000,
        "rag_enabled": True,
        "rag_backend": "local_tfidf",
        "rag_fallback_used": False,
        "start_timestamp": "2026-02-15T00:00:00Z",
        "end_timestamp": "2026-02-15T00:00:01Z",
        "runtime_seconds": 1.0,
        "artifacts_dir": "/tmp/artifacts",
        "artifact_paths": {
            "phase1_initial_models.json": "/tmp/phase1_initial_models.json",
            "phase2_negotiation_trace.json": "/tmp/phase2_negotiation_trace.json",
            "phase3_integrated_kaos_model.json": "/tmp/phase3_integrated_kaos_model.json",
            "phase4_verification_report.json": "/tmp/phase4_verification_report.json",
        },
        "notes": {
            "runtime_semantics": {
                "mode": "mare_paper_workflow_v1",
                "llm_required": True,
                "execution_mode": "llm_driven",
                "llm_turns": len(MARE_ACTIONS),
                "llm_fallback_turns": 0,
                "roles_executed": list(MARE_AGENT_ROLES),
                "actions_executed": list(MARE_ACTIONS),
                "llm_actions": list(MARE_ACTIONS),
                "fallback_actions": [],
                "action_trace": [
                    {
                        "role": "role",
                        "action": action,
                        "llm_generated": True,
                    }
                    for action in MARE_ACTIONS
                ],
                "workspace_digest": _sha256("workspace"),
            }
        },
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_run_record_strict_pass(tmp_path: Path) -> None:
    run_record_path = tmp_path / "run_record.json"
    _write_payload(run_record_path, _base_run_record())

    report = validate_run_record(run_record_path)
    assert report.errors == []


def test_validate_run_record_missing_metadata_section_fails(tmp_path: Path) -> None:
    payload = _base_run_record()
    payload.pop("provenance")
    run_record_path = tmp_path / "run_record.json"
    _write_payload(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("missing required metadata sections" in error for error in report.errors)


def test_validate_run_record_fallback_tainted_fails(tmp_path: Path) -> None:
    payload = _base_run_record()
    payload["execution_flags"]["rag_fallback_used"] = True
    payload["execution_flags"]["fallback_tainted"] = True
    payload["rag_fallback_used"] = True
    run_record_path = tmp_path / "run_record.json"
    _write_payload(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("Fallback-tainted metadata detected" in error for error in report.errors)


def test_validate_run_record_retry_tainted_fails(tmp_path: Path) -> None:
    payload = _base_run_record()
    payload["execution_flags"]["retry_used"] = True
    payload["execution_flags"]["retry_count"] = 1
    run_record_path = tmp_path / "run_record.json"
    _write_payload(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("Retry-tainted metadata detected" in error for error in report.errors)


def test_validate_run_record_requires_explicit_non_comparable_reason(tmp_path: Path) -> None:
    payload = _base_run_record()
    payload["setting"] = SETTING_SINGLE_AGENT
    payload["comparability"] = {
        "is_comparable": False,
        "non_comparable_reasons": [],
    }
    run_record_path = tmp_path / "run_record.json"
    _write_payload(run_record_path, payload)

    report = validate_run_record(run_record_path)
    assert any("non-comparable runs must include explicit reasons" in error for error in report.errors)

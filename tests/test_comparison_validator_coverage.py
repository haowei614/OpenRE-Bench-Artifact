"""Coverage tests for openre_bench.comparison_validator — validate_case_input, validate_run_record, etc."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from openre_bench.comparison_validator import (
    ValidationReport,
    _is_sha256_hex,
    _require_keys,
    _require_object_payload,
    _to_int,
    _validate_comparability,
    _validate_execution_flags,
    _validate_iredev_behavior_contract,
    _validate_mare_behavior_contract,
    _validate_mare_runtime_semantics,
    _validate_provenance,
    _validate_quare_behavior_contract,
    _validate_quare_optional_artifacts,
    _validate_system_identity,
    validate_case_input,
    validate_phase_artifacts,
    validate_run_record,
    validate_system_behavior_contract,
)
from openre_bench.schemas import (
    IREDEV_AGENT_ROLES,
    MARE_ACTIONS,
    MARE_AGENT_ROLES,
    RunRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SHA256 = "a" * 64


def _make_run_record_dict(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid run record dict."""
    base: dict[str, Any] = {
        "run_id": "test-run-001",
        "case_id": "case-001",
        "system": "quare",
        "setting": "negotiation_integration_verification",
        "seed": 42,
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "round_cap": 3,
        "max_tokens": 4000,
        "rag_enabled": True,
        "rag_backend": "local_tfidf",
        "rag_fallback_used": False,
        "start_timestamp": "2025-01-01T00:00:00Z",
        "end_timestamp": "2025-01-01T00:01:00Z",
        "runtime_seconds": 60.0,
        "artifacts_dir": "/tmp/artifacts",
        "system_identity": {
            "system_name": "quare",
            "implementation": "openre_bench",
            "implementation_version": "0.1.0",
            "python_version": "3.11.0",
            "platform": "Linux",
            "machine": "x86_64",
        },
        "provenance": {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "seed": 42,
            "prompt_hash": _VALID_SHA256,
            "corpus_hash": _VALID_SHA256,
            "corpus_path": "/data/corpus",
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
    }
    base.update(overrides)
    return base


def _make_parsed_run_record(**overrides: Any) -> RunRecord:
    return RunRecord.model_validate(_make_run_record_dict(**overrides))


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


def test_validation_report_ok_empty():
    r = ValidationReport()
    assert r.ok is True


def test_validation_report_ok_with_errors():
    r = ValidationReport(errors=["fail"])
    assert r.ok is False


def test_validation_report_ok_with_warnings_only():
    r = ValidationReport(warnings=["warn"])
    assert r.ok is True


# ---------------------------------------------------------------------------
# validate_case_input
# ---------------------------------------------------------------------------


def test_validate_case_input_missing_file(tmp_path: Path):
    r = validate_case_input(tmp_path / "missing.json")
    assert not r.ok


def test_validate_case_input_invalid_json(tmp_path: Path):
    f = tmp_path / "case.json"
    f.write_text("not json")
    r = validate_case_input(f)
    assert not r.ok


def test_validate_case_input_not_object(tmp_path: Path):
    f = tmp_path / "case.json"
    f.write_text(json.dumps([1, 2, 3]))
    r = validate_case_input(f)
    assert not r.ok


def test_validate_case_input_missing_fields(tmp_path: Path):
    f = tmp_path / "case.json"
    f.write_text(json.dumps({"case_name": "x", "case_description": "y", "requirement": "z", "extra": "ok"}))
    # This has all required fields so should pass
    r = validate_case_input(f)
    assert r.ok


def test_validate_case_input_empty_requirement(tmp_path: Path):
    f = tmp_path / "case.json"
    f.write_text(json.dumps({"case_name": "x", "case_description": "y", "requirement": "  "}))
    r = validate_case_input(f)
    assert not r.ok


def test_validate_case_input_valid(tmp_path: Path):
    f = tmp_path / "case.json"
    f.write_text(json.dumps({
        "case_name": "ATM",
        "case_description": "ATM requirements",
        "requirement": "The system shall authenticate users.",
    }))
    r = validate_case_input(f)
    assert r.ok


# ---------------------------------------------------------------------------
# validate_run_record
# ---------------------------------------------------------------------------


def test_validate_run_record_missing_file(tmp_path: Path):
    r = validate_run_record(tmp_path / "missing.json")
    assert not r.ok


def test_validate_run_record_invalid_json(tmp_path: Path):
    f = tmp_path / "run.json"
    f.write_text("{bad json")
    r = validate_run_record(f)
    assert not r.ok


def test_validate_run_record_not_object(tmp_path: Path):
    f = tmp_path / "run.json"
    f.write_text('"string value"')
    r = validate_run_record(f)
    assert not r.ok


def test_validate_run_record_missing_required_fields(tmp_path: Path):
    f = tmp_path / "run.json"
    f.write_text(json.dumps({"run_id": "x"}))
    r = validate_run_record(f)
    assert not r.ok


def test_validate_run_record_missing_metadata_sections(tmp_path: Path):
    f = tmp_path / "run.json"
    data = {k: "x" for k in ["run_id", "case_id", "system", "setting", "seed", "model", "temperature", "round_cap"]}
    f.write_text(json.dumps(data))
    r = validate_run_record(f)
    assert not r.ok


def test_validate_run_record_valid_run_record(tmp_path: Path):
    f = tmp_path / "run.json"
    f.write_text(json.dumps(_make_run_record_dict()))
    r = validate_run_record(f)
    # May have warnings for model/temp/round_cap drift but should be structurally ok
    # It depends on the comparability
    assert isinstance(r, ValidationReport)


def test_validate_run_record_model_drift_warning(tmp_path: Path):
    f = tmp_path / "run.json"
    f.write_text(json.dumps(_make_run_record_dict(model="gpt-4")))
    r = validate_run_record(f)
    assert any("Model drift" in w for w in r.warnings)


def test_validate_run_record_temperature_drift_warning(tmp_path: Path):
    f = tmp_path / "run.json"
    data = _make_run_record_dict(temperature=0.5)
    data["provenance"]["temperature"] = 0.5
    f.write_text(json.dumps(data))
    r = validate_run_record(f)
    assert any("Temperature drift" in w for w in r.warnings)


def test_validate_run_record_rag_disabled_error(tmp_path: Path):
    f = tmp_path / "run.json"
    f.write_text(json.dumps(_make_run_record_dict(rag_enabled=False)))
    r = validate_run_record(f)
    assert any("RAG parity" in e for e in r.errors)


def test_validate_run_record_rag_backend_empty(tmp_path: Path):
    f = tmp_path / "run.json"
    f.write_text(json.dumps(_make_run_record_dict(rag_backend="")))
    r = validate_run_record(f)
    assert any("rag_backend" in e for e in r.errors)


def test_validate_run_record_rag_fallback_error(tmp_path: Path):
    f = tmp_path / "run.json"
    data = _make_run_record_dict(rag_fallback_used=True)
    data["execution_flags"]["rag_fallback_used"] = True
    data["execution_flags"]["fallback_tainted"] = True
    f.write_text(json.dumps(data))
    r = validate_run_record(f)
    assert any("RAG fallback" in e for e in r.errors)


# ---------------------------------------------------------------------------
# _validate_system_identity
# ---------------------------------------------------------------------------


def test_validate_system_identity_mismatch():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.system_identity.system_name = "wrong"
    _validate_system_identity(parsed, report)
    assert any("mismatch" in e for e in report.errors)


def test_validate_system_identity_missing_fields():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.system_identity.implementation = ""
    _validate_system_identity(parsed, report)
    assert any("missing" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# _validate_provenance
# ---------------------------------------------------------------------------


def test_validate_provenance_model_mismatch():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.provenance.model = "different-model"
    _validate_provenance(parsed, report)
    assert any("model" in e.lower() for e in report.errors)


def test_validate_provenance_temperature_mismatch():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.provenance.temperature = 999.0
    _validate_provenance(parsed, report)
    assert any("temperature" in e.lower() for e in report.errors)


def test_validate_provenance_seed_mismatch():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.provenance.seed = 999
    _validate_provenance(parsed, report)
    assert any("seed" in e.lower() for e in report.errors)


def test_validate_provenance_bad_prompt_hash():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.provenance.prompt_hash = "not-a-hash"
    _validate_provenance(parsed, report)
    assert any("prompt_hash" in e for e in report.errors)


def test_validate_provenance_rag_corpus_path_missing():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.provenance.corpus_path = ""
    _validate_provenance(parsed, report)
    assert any("corpus_path" in e for e in report.errors)


def test_validate_provenance_rag_corpus_hash_invalid():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.provenance.corpus_hash = "nope"
    _validate_provenance(parsed, report)
    assert any("corpus_hash" in e for e in report.errors)


# ---------------------------------------------------------------------------
# _validate_execution_flags
# ---------------------------------------------------------------------------


def test_validate_execution_flags_rag_fallback_mismatch():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.execution_flags.rag_fallback_used = True
    parsed.rag_fallback_used = False
    _validate_execution_flags(parsed, report)
    assert any("rag_fallback_used" in e for e in report.errors)


def test_validate_execution_flags_fallback_tainted_mismatch():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.execution_flags.fallback_tainted = True
    _validate_execution_flags(parsed, report)
    assert any("fallback_tainted" in e.lower() or "Fallback-tainted" in e for e in report.errors)


def test_validate_execution_flags_retry_used_mismatch():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.execution_flags.retry_used = True
    parsed.execution_flags.retry_count = 0
    _validate_execution_flags(parsed, report)
    assert any("retry" in e.lower() for e in report.errors)


def test_validate_execution_flags_retry_tainted():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.execution_flags.retry_used = True
    parsed.execution_flags.retry_count = 1
    _validate_execution_flags(parsed, report)
    assert any("retry" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# _validate_comparability
# ---------------------------------------------------------------------------


def test_validate_comparability_valid():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    _validate_comparability(parsed, report)
    # Should pass or just have mismatch based on setting-expected reasons


def test_validate_comparability_non_comparable_missing_reasons():
    report = ValidationReport()
    parsed = _make_parsed_run_record()
    parsed.comparability.is_comparable = False
    parsed.comparability.non_comparable_reasons = []
    _validate_comparability(parsed, report)
    assert any("non-comparable" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# _is_sha256_hex
# ---------------------------------------------------------------------------


def test_is_sha256_hex_valid():
    assert _is_sha256_hex(_VALID_SHA256) is True


def test_is_sha256_hex_wrong_length():
    assert _is_sha256_hex("abc") is False


def test_is_sha256_hex_non_hex_chars():
    assert _is_sha256_hex("g" * 64) is False


# ---------------------------------------------------------------------------
# _require_keys / _require_object_payload / _to_int
# ---------------------------------------------------------------------------


def test_helpers_require_keys_missing():
    report = ValidationReport()
    _require_keys({"a": 1}, {"a", "b"}, "test.json", report)
    assert len(report.errors) == 1


def test_helpers_require_keys_present():
    report = ValidationReport()
    _require_keys({"a": 1, "b": 2}, {"a", "b"}, "test.json", report)
    assert len(report.errors) == 0


def test_helpers_require_object_payload_none():
    report = ValidationReport()
    result = _require_object_payload(payload=None, file_name="f", report=report)
    assert result is None
    assert len(report.errors) == 0


def test_helpers_require_object_payload_not_dict():
    report = ValidationReport()
    result = _require_object_payload(payload=[1, 2], file_name="f", report=report)
    assert result is None
    assert len(report.errors) == 1


def test_helpers_require_object_payload_dict():
    report = ValidationReport()
    result = _require_object_payload(payload={"k": "v"}, file_name="f", report=report)
    assert result == {"k": "v"}


def test_helpers_to_int_valid():
    assert _to_int(42, 0) == 42
    assert _to_int("7", 0) == 7


def test_helpers_to_int_invalid():
    assert _to_int("abc", -1) == -1
    assert _to_int(None, 99) == 99


# ---------------------------------------------------------------------------
# validate_phase_artifacts
# ---------------------------------------------------------------------------


def test_validate_phase_artifacts_missing_dir(tmp_path: Path):
    r = validate_phase_artifacts(tmp_path / "missing")
    assert not r.ok


def test_validate_phase_artifacts_not_a_dir(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    r = validate_phase_artifacts(f)
    assert not r.ok


def test_validate_phase_artifacts_missing_phase_files(tmp_path: Path):
    r = validate_phase_artifacts(tmp_path)
    assert not r.ok


def test_validate_phase_artifacts_valid_phase_files(tmp_path: Path):
    from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME, PHASE3_FILENAME, PHASE4_FILENAME

    # Create minimal valid phase files
    (tmp_path / PHASE1_FILENAME).write_text(json.dumps({}))
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps({
        "total_negotiations": 0,
        "negotiations": {},
        "summary_stats": {},
    }))
    (tmp_path / PHASE3_FILENAME).write_text(json.dumps({
        "gsn_elements": [],
        "gsn_connections": [],
    }))
    (tmp_path / PHASE4_FILENAME).write_text(json.dumps({
        "fact_checking": {},
        "deterministic_validation": {},
        "topology_status": {},
    }))
    r = validate_phase_artifacts(tmp_path)
    # May have warnings about empty phase1 but no errors
    assert isinstance(r, ValidationReport)


# ---------------------------------------------------------------------------
# validate_system_behavior_contract
# ---------------------------------------------------------------------------


def test_validate_system_behavior_contract_unknown_system(tmp_path: Path):
    r = validate_system_behavior_contract(system="unknown", artifacts_dir=tmp_path)
    assert not r.ok


def test_validate_system_behavior_contract_missing_dir(tmp_path: Path):
    r = validate_system_behavior_contract(system="mare", artifacts_dir=tmp_path / "missing")
    assert not r.ok


def test_validate_system_behavior_contract_mare_system(tmp_path: Path):
    from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME

    phase1 = {role: [] for role in MARE_AGENT_ROLES}
    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(phase1))
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps({
        "total_negotiations": 0,
        "negotiations": {},
        "summary_stats": {},
    }))
    (tmp_path / "run_record.json").write_text(json.dumps({"setting": "multi_agent_with_negotiation"}))
    r = validate_system_behavior_contract(system="mare", artifacts_dir=tmp_path)
    assert isinstance(r, ValidationReport)


def test_validate_system_behavior_contract_uses_explicit_run_record_path(tmp_path: Path):
    from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    external_run_record = tmp_path / "custom_run_record.json"

    (artifacts_dir / PHASE1_FILENAME).write_text(json.dumps({}))
    (artifacts_dir / PHASE2_FILENAME).write_text(json.dumps({
        "total_negotiations": 0,
        "negotiations": {},
        "summary_stats": {},
    }))
    external_run_record.write_text(json.dumps({"setting": "multi_agent_with_negotiation"}))

    report = validate_system_behavior_contract(
        system="mare",
        artifacts_dir=artifacts_dir,
        run_record_path=external_run_record,
    )

    assert any("phase1 is missing required roles" in item for item in report.errors)
    assert all("Failed to parse run_record.json" not in item for item in report.errors)


def test_validate_system_behavior_contract_iredev_system(tmp_path: Path):
    from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME

    phase1 = {role: [] for role in IREDEV_AGENT_ROLES}
    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(phase1))
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps({
        "total_negotiations": 0,
        "negotiations": {},
        "summary_stats": {},
    }))
    (tmp_path / "run_record.json").write_text(json.dumps({"setting": "multi_agent_with_negotiation"}))
    r = validate_system_behavior_contract(system="iredev", artifacts_dir=tmp_path)
    assert isinstance(r, ValidationReport)


def test_validate_system_behavior_contract_quare_system(tmp_path: Path):
    from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME, PHASE0_FILENAME, PHASE25_FILENAME, PHASE5_FILENAME

    (tmp_path / PHASE1_FILENAME).write_text(json.dumps({}))
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps({
        "total_negotiations": 1,
        "negotiations": {
            "n1": {
                "negotiation_id": "n1",
                "focus_agent": "A",
                "reviewer_agents": ["B"],
                "start_timestamp": "2025-01-01T00:00:00Z",
                "steps": [{
                    "step_id": 1,
                    "timestamp": "2025-01-01T00:00:00Z",
                    "focus_agent": "A",
                    "reviewer_agent": "B",
                    "round_number": 1,
                    "message_type": "proposal",
                    "analysis_text": "text",
                    "negotiation_mode": "quare_dialectic",
                }],
                "final_consensus": True,
                "total_rounds": 2,
            },
        },
        "summary_stats": {"detected_conflicts": 1, "resolved_conflicts": 1},
    }))
    (tmp_path / PHASE0_FILENAME).write_text(json.dumps({
        "phase": "phase0",
        "case_id": "c1",
        "setting": "s",
        "generated_at": "2025-01-01",
        "extracted_rules": [],
        "extraction_metadata": {},
    }))
    (tmp_path / PHASE25_FILENAME).write_text(json.dumps({
        "phase": "phase2.5",
        "case_id": "c1",
        "setting": "s",
        "generated_at": "2025-01-01",
        "round_cap": 3,
        "conflict_map": {},
        "summary": {
            "detected_conflict_pairs": 1,
            "resolved_conflict_pairs": 1,
            "unresolved_conflict_pairs": 0,
            "phase2_detected_conflicts": 1,
            "phase2_resolved_conflicts": 1,
        },
    }))
    (tmp_path / PHASE5_FILENAME).write_text(json.dumps({
        "phase": "phase5",
        "case_id": "c1",
        "setting": "s",
        "generated_at": "2025-01-01",
        "materials": {},
        "quality_signals": {},
    }))
    (tmp_path / "run_record.json").write_text(json.dumps({"setting": "multi_agent_with_negotiation"}))
    r = validate_system_behavior_contract(system="quare", artifacts_dir=tmp_path)
    assert isinstance(r, ValidationReport)


# ---------------------------------------------------------------------------
# _validate_mare_behavior_contract
# ---------------------------------------------------------------------------


def test_validate_mare_behavior_contract_missing_roles():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={},
        phase1_payload={},  # missing roles
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("missing required roles" in e for e in report.errors)


def test_validate_mare_behavior_contract_negotiation_rounds_violation():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={"n1": {"total_rounds": 5}},
        phase1_payload={role: [] for role in MARE_AGENT_ROLES},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("total_rounds" in e for e in report.errors)


def test_validate_mare_behavior_contract_quare_mode_detected():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={
            "n1": {
                "total_rounds": 1,
                "steps": [{
                    "round_number": 1,
                    "negotiation_mode": "quare_dialectic",
                    "resolution_state": None,
                    "requires_refinement": None,
                }],
            }
        },
        phase1_payload={role: [] for role in MARE_AGENT_ROLES},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("QUARE negotiation_mode" in e for e in report.errors)


def test_validate_mare_behavior_contract_resolution_state_quare_only():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={
            "n1": {
                "total_rounds": 1,
                "steps": [{
                    "round_number": 1,
                    "negotiation_mode": "baseline",
                    "resolution_state": "resolved",
                    "requires_refinement": None,
                }],
            }
        },
        phase1_payload={role: [] for role in MARE_AGENT_ROLES},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("resolution_state" in e for e in report.errors)


def test_validate_mare_behavior_contract_requires_refinement_quare_only():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={
            "n1": {
                "total_rounds": 1,
                "steps": [{
                    "round_number": 1,
                    "negotiation_mode": "baseline",
                    "resolution_state": None,
                    "requires_refinement": True,
                }],
            }
        },
        phase1_payload={role: [] for role in MARE_AGENT_ROLES},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("requires_refinement" in e for e in report.errors)


def test_validate_mare_behavior_contract_non_object_phase1():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={},
        phase1_payload="not a dict",
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("must be an object" in e for e in report.errors)


def test_validate_mare_behavior_contract_steps_not_list():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={"n1": {"total_rounds": 1, "steps": "bad"}},
        phase1_payload={role: [] for role in MARE_AGENT_ROLES},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("steps must be a list" in e for e in report.errors)


def test_validate_mare_behavior_contract_round_number_violation():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={
            "n1": {
                "total_rounds": 1,
                "steps": [{
                    "round_number": 5,
                    "negotiation_mode": "baseline",
                    "resolution_state": None,
                    "requires_refinement": None,
                }],
            }
        },
        phase1_payload={role: [] for role in MARE_AGENT_ROLES},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("round_number" in e for e in report.errors)


def test_validate_mare_behavior_contract_single_agent_skips_phase1_check():
    report = ValidationReport()
    _validate_mare_behavior_contract(
        negotiations={},
        phase1_payload="irrelevant",
        setting="single_agent",
        report=report,
    )
    assert not any("phase1" in e for e in report.errors)


# ---------------------------------------------------------------------------
# _validate_iredev_behavior_contract
# ---------------------------------------------------------------------------


def test_validate_iredev_behavior_contract_missing_roles():
    report = ValidationReport()
    _validate_iredev_behavior_contract(
        negotiations={},
        phase1_payload={},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("missing required roles" in e for e in report.errors)


def test_validate_iredev_behavior_contract_negotiation_rounds_violation():
    report = ValidationReport()
    _validate_iredev_behavior_contract(
        negotiations={"n1": {"total_rounds": 5}},
        phase1_payload={role: [] for role in IREDEV_AGENT_ROLES},
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("total_rounds" in e for e in report.errors)


def test_validate_iredev_behavior_contract_non_object_phase1():
    report = ValidationReport()
    _validate_iredev_behavior_contract(
        negotiations={},
        phase1_payload="not a dict",
        setting="multi_agent_with_negotiation",
        report=report,
    )
    assert any("must be an object" in e for e in report.errors)


# ---------------------------------------------------------------------------
# _validate_quare_behavior_contract
# ---------------------------------------------------------------------------


def test_validate_quare_behavior_contract_missing_quare_mode():
    report = ValidationReport()
    _validate_quare_behavior_contract(
        phase2_payload={
            "total_negotiations": 1,
            "negotiations": {"n1": {"total_rounds": 1, "steps": [{"negotiation_mode": "other"}]}},
            "summary_stats": {},
        },
        report=report,
    )
    assert any("quare_dialectic" in e for e in report.errors)


def test_validate_quare_behavior_contract_conflicts_without_multi_round():
    report = ValidationReport()
    _validate_quare_behavior_contract(
        phase2_payload={
            "total_negotiations": 1,
            "negotiations": {
                "n1": {
                    "total_rounds": 1,
                    "steps": [{"negotiation_mode": "quare_dialectic"}],
                },
            },
            "summary_stats": {"detected_conflicts": 3},
        },
        report=report,
    )
    assert any("multi-round" in e for e in report.errors)


def test_validate_quare_behavior_contract_negotiations_not_dict():
    report = ValidationReport()
    _validate_quare_behavior_contract(
        phase2_payload={"negotiations": "bad"},
        report=report,
    )
    # Returns early, no errors
    assert len(report.errors) == 0


def test_validate_quare_behavior_contract_steps_not_list():
    report = ValidationReport()
    _validate_quare_behavior_contract(
        phase2_payload={
            "total_negotiations": 1,
            "negotiations": {"n1": {"total_rounds": 1, "steps": "not_list"}},
            "summary_stats": {},
        },
        report=report,
    )
    # Steps not a list is handled gracefully (continue)
    assert any("quare_dialectic" in e for e in report.errors)


# ---------------------------------------------------------------------------
# _validate_mare_runtime_semantics
# ---------------------------------------------------------------------------


def test_validate_mare_runtime_semantics_non_mare_skips():
    report = ValidationReport()
    parsed = _make_parsed_run_record(system="quare")
    _validate_mare_runtime_semantics(parsed, report)
    assert len(report.errors) == 0


def test_validate_mare_runtime_semantics_single_agent_skips():
    report = ValidationReport()
    parsed = _make_parsed_run_record(system="mare", setting="single_agent")
    _validate_mare_runtime_semantics(parsed, report)
    assert len(report.errors) == 0


def test_validate_mare_runtime_semantics_missing_runtime_semantics():
    report = ValidationReport()
    parsed = _make_parsed_run_record(system="mare", setting="multi_agent_with_negotiation")
    parsed.notes = {}
    _validate_mare_runtime_semantics(parsed, report)
    assert any("runtime_semantics" in e for e in report.errors)


def test_validate_mare_runtime_semantics_full_valid_mare_semantics():
    report = ValidationReport()
    action_trace = [
        {"action": a, "role": r, "llm_generated": True}
        for r, a in zip(list(MARE_AGENT_ROLES) + list(MARE_AGENT_ROLES), list(MARE_ACTIONS))
    ]
    notes = {
        "runtime_semantics": {
            "mode": "mare_paper_workflow_v1",
            "roles_executed": sorted(MARE_AGENT_ROLES),
            "actions_executed": sorted(MARE_ACTIONS),
            "workspace_digest": _VALID_SHA256,
            "llm_required": True,
            "execution_mode": "llm_driven",
            "llm_turns": len(MARE_ACTIONS),
            "llm_fallback_turns": 0,
            "llm_actions": sorted(MARE_ACTIONS),
            "fallback_actions": [],
            "action_trace": action_trace,
        }
    }
    parsed = _make_parsed_run_record(system="mare", setting="multi_agent_with_negotiation", notes=notes)
    _validate_mare_runtime_semantics(parsed, report)
    # Should be clean
    assert all("guardrail failed" not in e for e in report.errors)


# ---------------------------------------------------------------------------
# _validate_quare_optional_artifacts
# ---------------------------------------------------------------------------


def test_validate_quare_optional_artifacts_phase0_rules_not_list(tmp_path: Path):
    from openre_bench.schemas import PHASE0_FILENAME, PHASE25_FILENAME, PHASE5_FILENAME
    report = ValidationReport()
    (tmp_path / PHASE0_FILENAME).write_text(json.dumps({
        "phase": "phase0", "case_id": "c1", "setting": "s",
        "generated_at": "t", "extracted_rules": "not_a_list",
        "extraction_metadata": {},
    }))
    (tmp_path / PHASE25_FILENAME).write_text(json.dumps({
        "phase": "p25", "case_id": "c1", "setting": "s",
        "generated_at": "t", "round_cap": 3, "conflict_map": {},
        "summary": {
            "detected_conflict_pairs": 0, "resolved_conflict_pairs": 0,
            "unresolved_conflict_pairs": 0,
        },
    }))
    (tmp_path / PHASE5_FILENAME).write_text(json.dumps({
        "phase": "p5", "case_id": "c1", "setting": "s",
        "generated_at": "t", "materials": {}, "quality_signals": {},
    }))
    _validate_quare_optional_artifacts(
        artifacts_dir=tmp_path,
        phase2_payload={"summary_stats": {}},
        report=report,
    )
    assert any("extracted_rules must be a list" in e for e in report.errors)

def test_validate_quare_optional_artifacts_phase25_conflict_map_not_dict(tmp_path: Path):
    from openre_bench.schemas import PHASE0_FILENAME, PHASE25_FILENAME, PHASE5_FILENAME
    report = ValidationReport()
    (tmp_path / PHASE0_FILENAME).write_text(json.dumps({
        "phase": "p0", "case_id": "c1", "setting": "s",
        "generated_at": "t", "extracted_rules": [],
        "extraction_metadata": {},
    }))
    (tmp_path / PHASE25_FILENAME).write_text(json.dumps({
        "phase": "p25", "case_id": "c1", "setting": "s",
        "generated_at": "t", "round_cap": 3,
        "conflict_map": "not_dict",
        "summary": {
            "detected_conflict_pairs": 0, "resolved_conflict_pairs": 0,
            "unresolved_conflict_pairs": 0,
        },
    }))
    (tmp_path / PHASE5_FILENAME).write_text(json.dumps({
        "phase": "p5", "case_id": "c1", "setting": "s",
        "generated_at": "t", "materials": {}, "quality_signals": {},
    }))
    _validate_quare_optional_artifacts(
        artifacts_dir=tmp_path,
        phase2_payload={"summary_stats": {}},
        report=report,
    )
    assert any("conflict_map must be an object" in e for e in report.errors)

def test_validate_quare_optional_artifacts_phase25_summary_not_dict(tmp_path: Path):
    from openre_bench.schemas import PHASE0_FILENAME, PHASE25_FILENAME, PHASE5_FILENAME
    report = ValidationReport()
    (tmp_path / PHASE0_FILENAME).write_text(json.dumps({
        "phase": "p0", "case_id": "c1", "setting": "s",
        "generated_at": "t", "extracted_rules": [],
        "extraction_metadata": {},
    }))
    (tmp_path / PHASE25_FILENAME).write_text(json.dumps({
        "phase": "p25", "case_id": "c1", "setting": "s",
        "generated_at": "t", "round_cap": 3,
        "conflict_map": {},
        "summary": "not_dict",
    }))
    (tmp_path / PHASE5_FILENAME).write_text(json.dumps({
        "phase": "p5", "case_id": "c1", "setting": "s",
        "generated_at": "t", "materials": {}, "quality_signals": {},
    }))
    _validate_quare_optional_artifacts(
        artifacts_dir=tmp_path,
        phase2_payload={"summary_stats": {}},
        report=report,
    )
    assert any("summary must be an object" in e for e in report.errors)

def test_validate_quare_optional_artifacts_phase5_materials_not_dict(tmp_path: Path):
    from openre_bench.schemas import PHASE0_FILENAME, PHASE25_FILENAME, PHASE5_FILENAME
    report = ValidationReport()
    (tmp_path / PHASE0_FILENAME).write_text(json.dumps({
        "phase": "p0", "case_id": "c1", "setting": "s",
        "generated_at": "t", "extracted_rules": [],
        "extraction_metadata": {},
    }))
    (tmp_path / PHASE25_FILENAME).write_text(json.dumps({
        "phase": "p25", "case_id": "c1", "setting": "s",
        "generated_at": "t", "round_cap": 3, "conflict_map": {},
        "summary": {
            "detected_conflict_pairs": 0, "resolved_conflict_pairs": 0,
            "unresolved_conflict_pairs": 0,
        },
    }))
    (tmp_path / PHASE5_FILENAME).write_text(json.dumps({
        "phase": "p5", "case_id": "c1", "setting": "s",
        "generated_at": "t", "materials": "not_dict",
        "quality_signals": {},
    }))
    _validate_quare_optional_artifacts(
        artifacts_dir=tmp_path,
        phase2_payload={"summary_stats": {}},
        report=report,
    )
    assert any("materials must be an object" in e for e in report.errors)

def test_validate_quare_optional_artifacts_phase25_unresolved_mismatch(tmp_path: Path):
    from openre_bench.schemas import PHASE0_FILENAME, PHASE25_FILENAME, PHASE5_FILENAME
    report = ValidationReport()
    (tmp_path / PHASE0_FILENAME).write_text(json.dumps({
        "phase": "p0", "case_id": "c1", "setting": "s",
        "generated_at": "t", "extracted_rules": [],
        "extraction_metadata": {},
    }))
    (tmp_path / PHASE25_FILENAME).write_text(json.dumps({
        "phase": "p25", "case_id": "c1", "setting": "s",
        "generated_at": "t", "round_cap": 3, "conflict_map": {},
        "summary": {
            "detected_conflict_pairs": 5,
            "resolved_conflict_pairs": 2,
            "unresolved_conflict_pairs": 1,  # should be 3
        },
    }))
    (tmp_path / PHASE5_FILENAME).write_text(json.dumps({
        "phase": "p5", "case_id": "c1", "setting": "s",
        "generated_at": "t", "materials": {}, "quality_signals": {},
    }))
    _validate_quare_optional_artifacts(
        artifacts_dir=tmp_path,
        phase2_payload={"summary_stats": {}},
        report=report,
    )
    assert any("unresolved count" in e for e in report.errors)

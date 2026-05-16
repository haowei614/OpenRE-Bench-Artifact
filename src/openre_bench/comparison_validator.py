"""Validation helpers for OpenRE-Bench anchored comparison runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openre_bench.schemas import CaseInput
from openre_bench.schemas import IREDEV_ACTIONS
from openre_bench.schemas import IREDEV_AGENT_ROLES
from openre_bench.schemas import MARE_ACTIONS
from openre_bench.schemas import MARE_AGENT_ROLES
from openre_bench.schemas import PHASE0_FILENAME
from openre_bench.schemas import PHASE1_FILENAME
from openre_bench.schemas import PHASE2_FILENAME
from openre_bench.schemas import PHASE25_FILENAME
from openre_bench.schemas import PHASE3_FILENAME
from openre_bench.schemas import PHASE4_FILENAME
from openre_bench.schemas import PHASE5_FILENAME
from openre_bench.schemas import Phase1Artifact
from openre_bench.schemas import Phase2Artifact
from openre_bench.schemas import Phase3Artifact
from openre_bench.schemas import Phase4Artifact
from openre_bench.schemas import RunRecord
from openre_bench.schemas import SETTING_SINGLE_AGENT
from openre_bench.schemas import non_comparable_reasons_for_setting
from openre_bench.schemas import SUPPORTED_SYSTEMS

REQUIRED_CASE_FIELDS = {"case_name", "case_description", "requirement"}

REQUIRED_RUN_FIELDS = {
    "run_id",
    "case_id",
    "system",
    "setting",
    "seed",
    "model",
    "temperature",
    "round_cap",
}

REQUIRED_METADATA_SECTIONS = {
    "system_identity",
    "provenance",
    "execution_flags",
    "comparability",
}

EXPECTED_MODEL = "gpt-4o-mini"
EXPECTED_TEMPERATURE = 0.7
EXPECTED_ROUND_CAP = 3
EXPECTED_RAG_ENABLED = True

REQUIRED_PHASE_FILES = {
    PHASE1_FILENAME,
    PHASE2_FILENAME,
    PHASE3_FILENAME,
    PHASE4_FILENAME,
}

SHA256_HEX_LENGTH = 64


@dataclass
class ValidationReport:
    """Collects validation errors and warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_case_input(path: Path) -> ValidationReport:
    report = ValidationReport()
    if not path.exists():
        report.errors.append(f"Case input not found: {path}")
        return report

    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"Failed to parse case input JSON: {path} ({exc})")
        return report

    if not isinstance(payload, dict):
        report.errors.append(f"Case input must be a JSON object: {path}")
        return report

    try:
        parsed = CaseInput.model_validate(payload)
    except ValidationError as exc:
        report.errors.append(f"Case input schema validation failed: {exc}")
        return report

    missing = sorted(REQUIRED_CASE_FIELDS - set(payload.keys()))
    if missing:
        report.errors.append(
            f"Case input missing required fields {missing}: {path}"
        )

    if not parsed.requirement.strip():
        report.errors.append(f"Case input has empty requirement: {path}")

    return report


def validate_run_record(path: Path) -> ValidationReport:
    report = ValidationReport()
    if not path.exists():
        report.errors.append(f"Run record not found: {path}")
        return report

    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"Failed to parse run record JSON: {path} ({exc})")
        return report

    if not isinstance(payload, dict):
        report.errors.append(f"Run record must be a JSON object: {path}")
        return report

    missing = sorted(REQUIRED_RUN_FIELDS - set(payload.keys()))
    if missing:
        report.errors.append(
            f"Run record missing required fields {missing}: {path}"
        )
        return report

    missing_metadata = sorted(REQUIRED_METADATA_SECTIONS - set(payload.keys()))
    if missing_metadata:
        report.errors.append(
            "Run record missing required metadata sections "
            f"{missing_metadata}: {path}"
        )
        return report

    try:
        parsed = RunRecord.model_validate(payload)
    except ValidationError as exc:
        report.errors.append(f"Run record schema validation failed: {exc}")
        return report

    _validate_system_identity(parsed, report)
    _validate_provenance(parsed, report)
    _validate_execution_flags(parsed, report)
    _validate_comparability(parsed, report)
    _validate_mare_runtime_semantics(parsed, report)
    _validate_iredev_runtime_semantics(parsed, report)

    if parsed.system not in SUPPORTED_SYSTEMS:
        report.errors.append(
            f"System identity must be one of {SUPPORTED_SYSTEMS}, got '{parsed.system}'"
        )

    model = parsed.model.strip()
    if model != EXPECTED_MODEL:
        report.warnings.append(
            f"Model drift: expected '{EXPECTED_MODEL}', got '{model}'"
        )

    temperature = float(parsed.temperature)

    if temperature != EXPECTED_TEMPERATURE:
        report.warnings.append(
            "Temperature drift: expected "
            f"{EXPECTED_TEMPERATURE}, got {temperature}"
        )

    round_cap = int(parsed.round_cap)

    if round_cap != EXPECTED_ROUND_CAP:
        report.warnings.append(
            f"Round cap drift: expected {EXPECTED_ROUND_CAP}, got {round_cap}"
        )

    if bool(parsed.rag_enabled) != EXPECTED_RAG_ENABLED:
        report.errors.append(
            "RAG parity failure: expected rag_enabled=true for protocol-comparable runs"
        )

    rag_backend = parsed.rag_backend.strip().lower()
    if rag_backend in {"", "none"}:
        report.errors.append("RAG parity failure: rag_backend must be set for comparable runs")

    if bool(parsed.rag_fallback_used):
        report.errors.append(
            "RAG fallback used; run must be rerun with stable retrieval behavior for parity"
        )

    return report


def validate_phase_artifacts(path: Path) -> ValidationReport:
    report = ValidationReport()
    if not path.exists():
        report.errors.append(f"Artifacts directory not found: {path}")
        return report
    if not path.is_dir():
        report.errors.append(f"Artifacts path is not a directory: {path}")
        return report

    existing_files = {p.name for p in path.iterdir() if p.is_file()}
    missing_files = sorted(REQUIRED_PHASE_FILES - existing_files)
    if missing_files:
        report.errors.append(
            f"Artifacts directory missing required phase files: {missing_files}"
        )
        return report

    phase1 = _read_artifact_json(path, PHASE1_FILENAME, report)
    if isinstance(phase1, dict) and not phase1:
        report.warnings.append(f"{PHASE1_FILENAME} is empty")
    if isinstance(phase1, dict):
        try:
            Phase1Artifact.model_validate(phase1)
        except ValidationError as exc:
            report.errors.append(f"{PHASE1_FILENAME} schema validation failed: {exc}")

    phase2 = _read_artifact_json(path, PHASE2_FILENAME, report)
    if isinstance(phase2, dict):
        try:
            Phase2Artifact.model_validate(phase2)
        except ValidationError as exc:
            report.errors.append(f"{PHASE2_FILENAME} schema validation failed: {exc}")
        _require_keys(
            phase2,
            {"total_negotiations", "negotiations", "summary_stats"},
            PHASE2_FILENAME,
            report,
        )

    phase3 = _read_artifact_json(path, PHASE3_FILENAME, report)
    if isinstance(phase3, dict):
        try:
            Phase3Artifact.model_validate(phase3)
        except ValidationError as exc:
            report.errors.append(f"{PHASE3_FILENAME} schema validation failed: {exc}")
        _require_keys(
            phase3,
            {"gsn_elements", "gsn_connections"},
            PHASE3_FILENAME,
            report,
        )

    phase4 = _read_artifact_json(path, PHASE4_FILENAME, report)
    if isinstance(phase4, dict):
        try:
            Phase4Artifact.model_validate(phase4)
        except ValidationError as exc:
            report.errors.append(f"{PHASE4_FILENAME} schema validation failed: {exc}")
        _require_keys(
            phase4,
            {"fact_checking", "deterministic_validation", "topology_status"},
            PHASE4_FILENAME,
            report,
        )

    return report


def validate_system_behavior_contract(
    *,
    system: str,
    artifacts_dir: Path,
    run_record_path: Path | None = None,
) -> ValidationReport:
    """Validate system-specific phase-2 behavior separation contracts."""

    report = ValidationReport()
    normalized_system = system.strip().lower()
    if normalized_system not in SUPPORTED_SYSTEMS:
        report.errors.append(
            f"System behavior contract check failed: unknown system '{system}'"
        )
        return report

    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        report.errors.append(f"Artifacts directory not found: {artifacts_dir}")
        return report
    phase1_payload = _read_artifact_json(artifacts_dir, PHASE1_FILENAME, report)
    phase2_payload = _read_artifact_json(artifacts_dir, PHASE2_FILENAME, report)
    run_record_payload = _read_json_path(
        run_record_path or artifacts_dir / "run_record.json",
        "run_record.json",
        report,
    )
    if not isinstance(phase2_payload, dict):
        return report
    setting = ""
    if isinstance(run_record_payload, dict):
        setting = str(run_record_payload.get("setting", "")).strip()

    negotiations = phase2_payload.get("negotiations", {})
    if not isinstance(negotiations, dict):
        report.errors.append("Phase2 contract failure: negotiations must be a JSON object")
        return report

    if normalized_system == "mare":
        _validate_mare_behavior_contract(
            negotiations=negotiations,
            phase1_payload=phase1_payload,
            setting=setting,
            report=report,
        )
        return report

    if normalized_system == "iredev":
        _validate_iredev_behavior_contract(
            negotiations=negotiations,
            phase1_payload=phase1_payload,
            setting=setting,
            report=report,
        )
        return report

    _validate_quare_behavior_contract(phase2_payload=phase2_payload, report=report)
    _validate_quare_optional_artifacts(
        artifacts_dir=artifacts_dir,
        phase2_payload=phase2_payload,
        report=report,
    )
    return report


def _read_artifact_json(path: Path, name: str, report: ValidationReport) -> Any:
    return _read_json_path(path / name, name, report)


def _read_json_path(file_path: Path, display_name: str, report: ValidationReport) -> Any:
    try:
        return _load_json(file_path)
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"Failed to parse {display_name}: {exc}")
        return None


def _require_keys(
    payload: dict[str, Any],
    required_keys: set[str],
    file_name: str,
    report: ValidationReport,
) -> None:
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        report.errors.append(f"{file_name} missing required keys: {missing}")


def _validate_system_identity(parsed: RunRecord, report: ValidationReport) -> None:
    """Validate machine-checkable runtime identity metadata."""

    identity = parsed.system_identity
    if identity.system_name.strip().lower() != parsed.system.strip().lower():
        report.errors.append(
            "System identity mismatch: "
            f"system='{parsed.system}' vs system_identity.system_name='{identity.system_name}'"
        )

    required_fields = {
        "implementation": identity.implementation,
        "implementation_version": identity.implementation_version,
        "python_version": identity.python_version,
        "platform": identity.platform,
        "machine": identity.machine,
    }
    missing = sorted(name for name, value in required_fields.items() if not value.strip())
    if missing:
        report.errors.append(f"System identity missing non-empty fields: {missing}")


def _validate_provenance(parsed: RunRecord, report: ValidationReport) -> None:
    """Validate controlled settings and hashes for strict provenance checks."""

    provenance = parsed.provenance
    if provenance.model.strip() != parsed.model.strip():
        report.errors.append(
            "Provenance mismatch: provenance.model does not match top-level model"
        )
    if float(provenance.temperature) != float(parsed.temperature):
        report.errors.append(
            "Provenance mismatch: provenance.temperature does not match top-level temperature"
        )
    if int(provenance.seed) != int(parsed.seed):
        report.errors.append("Provenance mismatch: provenance.seed does not match top-level seed")

    if not _is_sha256_hex(provenance.prompt_hash):
        report.errors.append("Provenance failure: prompt_hash must be a SHA256 hex digest")

    if bool(parsed.rag_enabled):
        if not provenance.corpus_path.strip():
            report.errors.append("Provenance failure: corpus_path must be set when rag_enabled=true")
        if not _is_sha256_hex(provenance.corpus_hash):
            report.errors.append(
                "Provenance failure: corpus_hash must be a SHA256 hex digest when rag_enabled=true"
            )


def _validate_execution_flags(parsed: RunRecord, report: ValidationReport) -> None:
    """Invalidate fallback-tainted and retry-tainted runs."""

    flags = parsed.execution_flags
    if bool(flags.rag_fallback_used) != bool(parsed.rag_fallback_used):
        report.errors.append(
            "Execution flag mismatch: execution_flags.rag_fallback_used "
            "must match top-level rag_fallback_used"
        )

    fallback_tainted = bool(flags.rag_fallback_used or flags.llm_fallback_used)
    if bool(flags.fallback_tainted) != fallback_tainted:
        report.errors.append(
            "Execution flag mismatch: fallback_tainted must equal "
            "(rag_fallback_used OR llm_fallback_used)"
        )
    if bool(flags.fallback_tainted):
        report.errors.append(
            "Fallback-tainted metadata detected; run is invalid for strict comparability"
        )

    if bool(flags.retry_used) != bool(flags.retry_count > 0):
        report.errors.append(
            "Execution flag mismatch: retry_used must equal (retry_count > 0)"
        )
    if bool(flags.retry_used or flags.retry_count > 0):
        report.errors.append(
            "Retry-tainted metadata detected; run is invalid for strict comparability"
        )


def _validate_comparability(parsed: RunRecord, report: ValidationReport) -> None:
    """Validate explicit comparability semantics and reasons by setting."""

    comparability = parsed.comparability
    reasons = comparability.non_comparable_reasons
    expected_reasons = non_comparable_reasons_for_setting(parsed.setting)
    expected_comparable = not expected_reasons

    if comparability.is_comparable != expected_comparable:
        report.errors.append(
            "Comparability mismatch: is_comparable does not match setting semantics"
        )
    if not comparability.is_comparable and not reasons:
        report.errors.append(
            "Comparability failure: non-comparable runs must include explicit reasons"
        )
    if comparability.is_comparable and reasons:
        report.errors.append(
            "Comparability failure: comparable runs must not include non-comparable reasons"
        )
    if sorted(reasons) != sorted(expected_reasons):
        report.errors.append(
            "Comparability mismatch: non_comparable_reasons do not match protocol "
            f"expectation for setting '{parsed.setting}'"
        )


def _validate_mare_runtime_semantics(parsed: RunRecord, report: ValidationReport) -> None:
    """Require paper-faithful runtime semantics evidence for MARE multi-agent settings."""

    if parsed.system not in ("mare", "iredev"):
        return
    if parsed.setting == SETTING_SINGLE_AGENT:
        return

    notes = parsed.notes if isinstance(parsed.notes, dict) else {}
    runtime_semantics = notes.get("runtime_semantics")
    if not isinstance(runtime_semantics, dict):
        report.errors.append(
            "MARE semantics guardrail failed: notes.runtime_semantics must be present"
        )
        return

    mode = str(runtime_semantics.get("mode", "")).strip()
    expected_modes = {"mare_paper_workflow_v1", "iredev_knowledge_driven_v1"}
    if mode not in expected_modes:
        report.errors.append(
            "MARE semantics guardrail failed: runtime_semantics.mode must be "
            f"one of {sorted(expected_modes)} for multi-agent settings"
        )

    # Determine expected roles/actions based on system
    if parsed.system == "iredev":
        expected_roles = IREDEV_AGENT_ROLES
        expected_actions = IREDEV_ACTIONS
        role_label = "six iReDev"
        action_label = "seventeen iReDev"
        action_count = len(IREDEV_ACTIONS)
    else:
        expected_roles = MARE_AGENT_ROLES
        expected_actions = MARE_ACTIONS
        role_label = "five paper"
        action_label = "nine paper"
        action_count = len(MARE_ACTIONS)

    roles = _runtime_semantics_list_strings(
        runtime_semantics=runtime_semantics,
        field_name="roles_executed",
        report=report,
    )
    if roles is not None and sorted(roles) != sorted(expected_roles):
        report.errors.append(
            f"MARE semantics guardrail failed: roles_executed must include all {role_label} roles"
        )

    actions = _runtime_semantics_list_strings(
        runtime_semantics=runtime_semantics,
        field_name="actions_executed",
        report=report,
    )
    if actions is not None and sorted(actions) != sorted(expected_actions):
        report.errors.append(
            f"MARE semantics guardrail failed: actions_executed must include all {action_label} actions"
        )

    digest = str(runtime_semantics.get("workspace_digest", "")).strip()
    if not _is_sha256_hex(digest):
        report.errors.append(
            "MARE semantics guardrail failed: workspace_digest must be a SHA256 hex digest"
        )

    llm_required_raw = runtime_semantics.get("llm_required")
    if not isinstance(llm_required_raw, bool):
        report.errors.append(
            "MARE semantics guardrail failed: llm_required must be a boolean"
        )
    llm_required = llm_required_raw if isinstance(llm_required_raw, bool) else False
    if not llm_required:
        report.errors.append(
            "MARE semantics guardrail failed: llm_required must be true for multi-agent settings"
        )

    execution_mode = str(runtime_semantics.get("execution_mode", "")).strip()
    if execution_mode != "llm_driven":
        report.errors.append(
            "MARE semantics guardrail failed: execution_mode must be 'llm_driven' "
            "for multi-agent settings"
        )

    llm_turns = _to_int(runtime_semantics.get("llm_turns"), default=0)
    if llm_turns < action_count:
        report.errors.append(
            f"MARE semantics guardrail failed: llm_turns must cover all {action_label} actions"
        )

    llm_fallback_turns = _to_int(runtime_semantics.get("llm_fallback_turns"), default=0)
    if llm_fallback_turns != 0:
        report.errors.append(
            "MARE semantics guardrail failed: llm_fallback_turns must be zero for strict comparability"
        )

    llm_actions = _runtime_semantics_list_strings(
        runtime_semantics=runtime_semantics,
        field_name="llm_actions",
        report=report,
    )
    if llm_actions is not None and set(llm_actions) != set(expected_actions):
        report.errors.append(
            f"MARE semantics guardrail failed: llm_actions must cover all {action_label} actions"
        )

    fallback_actions = _runtime_semantics_list_strings(
        runtime_semantics=runtime_semantics,
        field_name="fallback_actions",
        report=report,
    )
    if fallback_actions:
        report.errors.append(
            "MARE semantics guardrail failed: fallback_actions must be empty for strict comparability"
        )

    action_trace = runtime_semantics.get("action_trace", [])
    if not isinstance(action_trace, list) or len(action_trace) < action_count:
        report.errors.append(
            f"MARE semantics guardrail failed: action_trace must include at least {action_count} action entries"
        )
    else:
        invalid_trace_entry = any(not isinstance(step, dict) for step in action_trace)
        if invalid_trace_entry:
            report.errors.append(
                "MARE semantics guardrail failed: every action_trace step must be an object"
            )
        elif any(
            not isinstance(step.get("llm_generated"), bool) or not step.get("llm_generated")
            for step in action_trace
        ):
            report.errors.append(
                "MARE semantics guardrail failed: every action_trace step must be LLM-generated"
            )


def _validate_iredev_runtime_semantics(parsed: RunRecord, report: ValidationReport) -> None:
    """iReDev runtime semantics are validated by the shared _validate_mare_runtime_semantics."""
    pass


def _validate_iredev_behavior_contract(
    *,
    negotiations: dict[str, Any],
    phase1_payload: Any,
    setting: str,
    report: ValidationReport,
) -> None:
    """Ensure iReDev traces contain all six agent roles in phase1."""

    if setting and setting != SETTING_SINGLE_AGENT:
        if not isinstance(phase1_payload, dict):
            report.errors.append(
                "iReDev semantics guardrail failed: phase1 payload must be an object"
            )
        else:
            missing_roles = [role for role in IREDEV_AGENT_ROLES if role not in phase1_payload]
            if missing_roles:
                report.errors.append(
                    "iReDev semantics guardrail failed: phase1 is missing required roles "
                    f"{missing_roles}"
                )

    for negotiation_key, negotiation in negotiations.items():
        if not isinstance(negotiation, dict):
            continue
        rounds = _to_int(negotiation.get("total_rounds"), default=0)
        if rounds not in {0, 1}:
            report.errors.append(
                "iReDev baseline guardrail failed: total_rounds must be 1 for "
                f"negotiation '{negotiation_key}', got {rounds}"
            )


def _is_sha256_hex(value: str) -> bool:
    """Return True when value is a SHA256 hex string."""

    text = value.strip()
    if len(text) != SHA256_HEX_LENGTH:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in text)


def _runtime_semantics_list_strings(
    *,
    runtime_semantics: dict[str, Any],
    field_name: str,
    report: ValidationReport,
) -> list[str] | None:
    """Read runtime_semantics list fields safely without raising on malformed payloads."""

    raw_values = runtime_semantics.get(field_name)
    if not isinstance(raw_values, list):
        report.errors.append(
            "MARE semantics guardrail failed: "
            f"runtime_semantics.{field_name} must be a list"
        )
        return None
    return [str(item) for item in raw_values]


def _to_int(value: Any, default: int) -> int:
    """Safely parse integer-like values used by validation checks."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _validate_mare_behavior_contract(
    *,
    negotiations: dict[str, Any],
    phase1_payload: Any,
    setting: str,
    report: ValidationReport,
) -> None:
    """Ensure MARE traces stay baseline-faithful and QUARE-free."""

    if setting and setting != SETTING_SINGLE_AGENT:
        if not isinstance(phase1_payload, dict):
            report.errors.append(
                "MARE semantics guardrail failed: phase1 payload must be an object"
            )
        else:
            missing_roles = [role for role in MARE_AGENT_ROLES if role not in phase1_payload]
            if missing_roles:
                report.errors.append(
                    "MARE semantics guardrail failed: phase1 is missing required roles "
                    f"{missing_roles}"
                )

    for negotiation_key, negotiation in negotiations.items():
        if not isinstance(negotiation, dict):
            continue
        rounds = _to_int(negotiation.get("total_rounds"), default=0)
        if rounds not in {0, 1}:
            report.errors.append(
                "MARE baseline guardrail failed: total_rounds must be 1 for "
                f"negotiation '{negotiation_key}', got {rounds}"
            )

        steps = negotiation.get("steps", [])
        if not isinstance(steps, list):
            report.errors.append(
                "MARE baseline guardrail failed: steps must be a list for "
                f"negotiation '{negotiation_key}'"
            )
            continue

        for step in steps:
            if not isinstance(step, dict):
                continue

            round_number = _to_int(step.get("round_number"), default=0)
            if round_number not in {0, 1}:
                report.errors.append(
                    "MARE baseline guardrail failed: round_number must remain 1, got "
                    f"{round_number} in negotiation '{negotiation_key}'"
                )

            mode = str(step.get("negotiation_mode", "")).strip().lower()
            if mode == "quare_dialectic":
                report.errors.append(
                    "MARE baseline guardrail failed: QUARE negotiation_mode marker "
                    f"detected in negotiation '{negotiation_key}'"
                )

            if step.get("resolution_state") not in {None, ""}:
                report.errors.append(
                    "MARE baseline guardrail failed: resolution_state is QUARE-only and "
                    f"must be absent in negotiation '{negotiation_key}'"
                )

            if step.get("requires_refinement") is not None:
                report.errors.append(
                    "MARE baseline guardrail failed: requires_refinement is QUARE-only "
                    f"and must be absent in negotiation '{negotiation_key}'"
                )


def _validate_quare_behavior_contract(
    *,
    phase2_payload: dict[str, Any],
    report: ValidationReport,
) -> None:
    """Ensure QUARE traces contain explicit dialectic semantics when conflicts exist."""

    negotiations = phase2_payload.get("negotiations", {})
    if not isinstance(negotiations, dict):
        return

    has_quare_mode = False
    has_multi_round = False
    for negotiation in negotiations.values():
        if not isinstance(negotiation, dict):
            continue

        total_rounds = _to_int(negotiation.get("total_rounds"), default=0)
        has_multi_round = has_multi_round or total_rounds > 1

        steps = negotiation.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            mode = str(step.get("negotiation_mode", "")).strip().lower()
            if mode == "quare_dialectic":
                has_quare_mode = True

    summary_stats = phase2_payload.get("summary_stats", {})
    detected_conflicts = 0
    if isinstance(summary_stats, dict):
        detected_conflicts = _to_int(summary_stats.get("detected_conflicts"), default=0)

    if phase2_payload.get("total_negotiations", 0) and not has_quare_mode:
        report.errors.append(
            "QUARE behavior contract failed: phase2 trace is missing "
            "negotiation_mode='quare_dialectic' markers"
        )
    if detected_conflicts > 0 and not has_multi_round:
        report.errors.append(
            "QUARE behavior contract failed: conflicts were detected but no multi-round "
            "negotiation was recorded"
        )


def _validate_quare_optional_artifacts(
    *,
    artifacts_dir: Path,
    phase2_payload: dict[str, Any],
    report: ValidationReport,
) -> None:
    """Validate QUARE-only protocol artifacts beyond canonical phase1-4 outputs."""

    phase0_payload = _read_artifact_json(artifacts_dir, PHASE0_FILENAME, report)
    phase0_payload = _require_object_payload(
        payload=phase0_payload,
        file_name=PHASE0_FILENAME,
        report=report,
    )
    if phase0_payload is not None:
        _require_keys(
            phase0_payload,
            {"phase", "case_id", "setting", "generated_at", "extracted_rules", "extraction_metadata"},
            PHASE0_FILENAME,
            report,
        )
        rules = phase0_payload.get("extracted_rules", [])
        if not isinstance(rules, list):
            report.errors.append(f"{PHASE0_FILENAME} contract failure: extracted_rules must be a list")

    phase25_payload = _read_artifact_json(artifacts_dir, PHASE25_FILENAME, report)
    phase25_payload = _require_object_payload(
        payload=phase25_payload,
        file_name=PHASE25_FILENAME,
        report=report,
    )
    if phase25_payload is not None:
        _require_keys(
            phase25_payload,
            {"phase", "case_id", "setting", "generated_at", "round_cap", "conflict_map", "summary"},
            PHASE25_FILENAME,
            report,
        )
        conflict_map = phase25_payload.get("conflict_map", {})
        if not isinstance(conflict_map, dict):
            report.errors.append(f"{PHASE25_FILENAME} contract failure: conflict_map must be an object")
        summary = phase25_payload.get("summary", {})
        if not isinstance(summary, dict):
            report.errors.append(f"{PHASE25_FILENAME} contract failure: summary must be an object")
        else:
            detected_pairs = _to_int(summary.get("detected_conflict_pairs"), default=-1)
            resolved_pairs = _to_int(summary.get("resolved_conflict_pairs"), default=-1)
            unresolved_pairs = _to_int(summary.get("unresolved_conflict_pairs"), default=-1)
            if detected_pairs < 0 or resolved_pairs < 0 or unresolved_pairs < 0:
                report.errors.append(
                    f"{PHASE25_FILENAME} contract failure: summary counts must be non-negative integers"
                )
            if detected_pairs - resolved_pairs != unresolved_pairs:
                report.errors.append(
                    f"{PHASE25_FILENAME} contract failure: unresolved count must equal "
                    "detected_conflict_pairs - resolved_conflict_pairs"
                )

            phase2_summary = phase2_payload.get("summary_stats", {})
            phase2_detected = _to_int(
                phase2_summary.get("detected_conflicts") if isinstance(phase2_summary, dict) else 0,
                default=0,
            )
            phase2_resolved = _to_int(
                phase2_summary.get("resolved_conflicts") if isinstance(phase2_summary, dict) else 0,
                default=0,
            )
            aligned_detected = _to_int(summary.get("phase2_detected_conflicts"), default=phase2_detected)
            aligned_resolved = _to_int(summary.get("phase2_resolved_conflicts"), default=phase2_resolved)
            if aligned_detected != phase2_detected:
                report.errors.append(
                    f"{PHASE25_FILENAME} contract failure: phase2_detected_conflicts does not match "
                    "phase2 summary_stats.detected_conflicts"
                )
            if aligned_resolved != phase2_resolved:
                report.errors.append(
                    f"{PHASE25_FILENAME} contract failure: phase2_resolved_conflicts does not match "
                    "phase2 summary_stats.resolved_conflicts"
                )

    phase5_payload = _read_artifact_json(artifacts_dir, PHASE5_FILENAME, report)
    phase5_payload = _require_object_payload(
        payload=phase5_payload,
        file_name=PHASE5_FILENAME,
        report=report,
    )
    if phase5_payload is not None:
        _require_keys(
            phase5_payload,
            {"phase", "case_id", "setting", "generated_at", "materials", "quality_signals"},
            PHASE5_FILENAME,
            report,
        )
        materials = phase5_payload.get("materials", {})
        if not isinstance(materials, dict):
            report.errors.append(f"{PHASE5_FILENAME} contract failure: materials must be an object")
        quality = phase5_payload.get("quality_signals", {})
        if not isinstance(quality, dict):
            report.errors.append(f"{PHASE5_FILENAME} contract failure: quality_signals must be an object")


def _require_object_payload(
    *,
    payload: Any,
    file_name: str,
    report: ValidationReport,
) -> dict[str, Any] | None:
    """Require optional artifact top-level JSON payloads to be objects."""

    if payload is None:
        return None
    if not isinstance(payload, dict):
        report.errors.append(
            f"{file_name} contract failure: top-level JSON payload must be an object"
        )
        return None
    return payload

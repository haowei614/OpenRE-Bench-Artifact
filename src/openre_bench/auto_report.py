"""Deterministic /auto orchestration for strict MARE/QUARE claim reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from openre_bench.comparison_harness import ABLATION_CSV_NAME
from openre_bench.comparison_harness import BY_CASE_CSV_NAME
from openre_bench.comparison_harness import MatrixConfig
from openre_bench.comparison_harness import RUNS_JSONL_NAME
from openre_bench.comparison_harness import SUMMARY_CSV_NAME
from openre_bench.comparison_harness import VALIDITY_MD_NAME
from openre_bench.comparison_harness import export_trace_audit
from openre_bench.comparison_harness import run_comparison_matrix
from openre_bench.comparison_validator import validate_phase_artifacts
from openre_bench.comparison_validator import validate_run_record
from openre_bench.comparison_validator import validate_system_behavior_contract
from openre_bench.schemas import PHASE1_FILENAME
from openre_bench.schemas import PHASE2_FILENAME
from openre_bench.schemas import PHASE0_FILENAME
from openre_bench.schemas import PHASE25_FILENAME
from openre_bench.schemas import PHASE5_FILENAME
from openre_bench.schemas import MARE_ACTIONS
from openre_bench.schemas import MARE_AGENT_ROLES
from openre_bench.schemas import SETTING_MULTI_AGENT_WITH_NEGOTIATION
from openre_bench.schemas import SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SUPPORTED_SYSTEMS
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import SYSTEM_QUARE
from openre_bench.schemas import utc_timestamp
from openre_bench.schemas import write_json_file


REQUIRED_MATRIX_FILES = (
    RUNS_JSONL_NAME,
    BY_CASE_CSV_NAME,
    SUMMARY_CSV_NAME,
    ABLATION_CSV_NAME,
    VALIDITY_MD_NAME,
)

KEY_DELTA_METRICS = (
    "runtime_seconds",
    "semantic_preservation_f1",
    "conflict_resolution_rate",
    "compliance_coverage",
    "mdc",
    "chv",
)

PAPER_MODEL = "gpt-4o-mini"
PAPER_TEMPERATURE = 0.7
PAPER_SEEDS = (101, 202, 303)
PAPER_SETTINGS = (
    "single_agent",
    "multi_agent_without_negotiation",
    "multi_agent_with_negotiation",
    "negotiation_integration_verification",
)
PAPER_CLAIMS_FILE = "paper_claim_validation.json"

LITERAL_SETTING_TO_PHASE = {
    SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION: "phase1",
    SETTING_MULTI_AGENT_WITH_NEGOTIATION: "phase2",
}

UNSPECIFIED_SYSTEM = "_unspecified"

CONVERSATION_INDEX_JSONL = "conversation_index.jsonl"
CONVERSATION_COVERAGE_JSON = "conversation_coverage.json"
CONVERSATION_COVERAGE_MD = "conversation_coverage.md"


@dataclass
class AutoReportConfig:
    """Configuration for strict /auto report generation."""

    report_dir: Path
    cases_dir: Path
    seeds: list[int]
    settings: list[str]
    model: str
    temperature: float
    round_cap: int
    max_tokens: int
    rag_enabled: bool
    rag_backend: str
    rag_corpus_dir: Path
    judge_pipeline_path: Path


@dataclass
class AutoReportResult:
    """Result bundle for /auto execution."""

    run_key: str
    run_dir: Path
    logs_dir: Path
    mare_dir: Path
    quare_dir: Path
    report_readme: Path
    report_analysis: Path
    proofs_dir: Path
    verdict_path: Path
    hard_failures: list[str]
    warnings: list[str]


class _Logger:
    """Append-only timestamped logger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def _write(self, level: str, message: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_timestamp()}] [{level}] {message}\n")


def run_auto_report(config: AutoReportConfig) -> AutoReportResult:
    """Run strict controlled MARE/QUARE matrix and generate report/proofs."""

    case_paths = sorted(config.cases_dir.glob("*_input.json"))
    if not case_paths:
        raise FileNotFoundError(f"No case input files found in {config.cases_dir}")

    expected_runs = len(case_paths) * len(config.seeds) * len(config.settings)
    controls = {
        "auto_contract_version": "2026-02-15-paper-literal-validity-v1",
        "cases_dir": str(config.cases_dir.resolve()),
        "case_files": [path.name for path in case_paths],
        "expected_runs": expected_runs,
        "seeds": list(config.seeds),
        "settings": list(config.settings),
        "model": config.model,
        "temperature": config.temperature,
        "round_cap": config.round_cap,
        "max_tokens": config.max_tokens,
        "rag_enabled": config.rag_enabled,
        "rag_backend": config.rag_backend,
        "rag_corpus_dir": str(config.rag_corpus_dir.resolve()),
        "judge_pipeline_path": str(config.judge_pipeline_path.resolve()),
        "paper_claim_source": "external MARE/QUARE benchmark claim profile",
        "paper_control_profile": {
            "model": PAPER_MODEL,
            "temperature": PAPER_TEMPERATURE,
            "seeds": list(PAPER_SEEDS),
            "settings": list(PAPER_SETTINGS),
        },
    }

    run_key = _build_run_key(controls)
    run_dir = config.report_dir / "runs" / run_key
    logs_dir = config.report_dir / "logs" / run_key
    mare_dir = run_dir / SYSTEM_MARE
    quare_dir = run_dir / SYSTEM_QUARE
    proofs_dir = run_dir / "proofs"
    report_readme = run_dir / "README.md"
    report_analysis = run_dir / "analysis.md"
    verdict_path = proofs_dir / "finality_threshold_verdict.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    proofs_dir.mkdir(parents=True, exist_ok=True)

    logger = _Logger(logs_dir / "auto.log")
    warnings: list[str] = []
    hard_failures: list[str] = []

    logger.info(f"Starting /auto run {run_key}")
    logger.info(f"Expected runs per system: {expected_runs}")
    write_json_file(run_dir / "controls.json", controls)

    mare_ok = _run_or_resume_system(
        system=SYSTEM_MARE,
        output_dir=mare_dir,
        expected_runs=expected_runs,
        config=config,
        logger=logger,
        log_path=logs_dir / "mare-matrix.log",
        warnings=warnings,
        hard_failures=hard_failures,
    )
    quare_ok = _run_or_resume_system(
        system=SYSTEM_QUARE,
        output_dir=quare_dir,
        expected_runs=expected_runs,
        config=config,
        logger=logger,
        log_path=logs_dir / "quare-matrix.log",
        warnings=warnings,
        hard_failures=hard_failures,
    )

    if mare_ok and quare_ok:
        _write_proofs_and_reports(
            run_key=run_key,
            mare_dir=mare_dir,
            quare_dir=quare_dir,
            run_dir=run_dir,
            proofs_dir=proofs_dir,
            logs_dir=logs_dir,
            expected_runs=expected_runs,
            settings=config.settings,
            controls=controls,
            logger=logger,
            warnings=warnings,
        )
    else:
        warning = "Skipped proof/report generation because one or more system matrices failed."
        warnings.append(warning)
        logger.warning(warning)

    _mirror_latest_outputs(
        report_dir=config.report_dir,
        run_key=run_key,
        run_dir=run_dir,
        report_readme=report_readme,
        report_analysis=report_analysis,
        proofs_dir=proofs_dir,
    )

    status_payload = {
        "run_key": run_key,
        "run_dir": str(run_dir),
        "logs_dir": str(logs_dir),
        "mare_dir": str(mare_dir),
        "quare_dir": str(quare_dir),
        "warnings": warnings,
        "hard_failures": hard_failures,
        "completed_at": utc_timestamp(),
    }
    write_json_file(run_dir / "run_status.json", status_payload)
    write_json_file(config.report_dir / "latest_run.json", status_payload)

    return AutoReportResult(
        run_key=run_key,
        run_dir=run_dir,
        logs_dir=logs_dir,
        mare_dir=mare_dir,
        quare_dir=quare_dir,
        report_readme=report_readme,
        report_analysis=report_analysis,
        proofs_dir=proofs_dir,
        verdict_path=verdict_path,
        hard_failures=hard_failures,
        warnings=warnings,
    )


def _run_or_resume_system(
    *,
    system: str,
    output_dir: Path,
    expected_runs: int,
    config: AutoReportConfig,
    logger: _Logger,
    log_path: Path,
    warnings: list[str],
    hard_failures: list[str],
) -> bool:
    """Run matrix for one system, reusing complete outputs when available."""

    complete, reason = _matrix_outputs_complete(
        output_dir=output_dir,
        expected_runs=expected_runs,
        system=system,
    )
    if complete:
        logger.info(f"{system}: matrix already complete ({reason}); resuming from {output_dir}")
        _log_matrix_snapshot(output_dir=output_dir, system=system, log_path=log_path)
        return True

    logger.info(f"{system}: running strict matrix in {output_dir}")
    matrix_config = MatrixConfig(
        cases_dir=config.cases_dir,
        output_dir=output_dir,
        seeds=config.seeds,
        settings=config.settings,
        model=config.model,
        temperature=config.temperature,
        round_cap=config.round_cap,
        max_tokens=config.max_tokens,
        system=system,
        rag_enabled=config.rag_enabled,
        rag_backend=config.rag_backend,
        rag_corpus_dir=config.rag_corpus_dir,
        judge_pipeline_path=config.judge_pipeline_path,
    )

    try:
        result = run_comparison_matrix(matrix_config)
    except Exception as exc:
        message = f"{system}: matrix execution failed: {exc}"
        hard_failures.append(message)
        logger.error(message)
        return False

    if result.warnings:
        for item in result.warnings:
            scoped = f"{system}: {item}"
            warnings.append(scoped)
            logger.warning(scoped)

    if result.errors:
        for item in result.errors:
            scoped = f"{system}: {item}"
            hard_failures.append(scoped)
            logger.error(scoped)
        return False

    complete, reason = _matrix_outputs_complete(
        output_dir=output_dir,
        expected_runs=expected_runs,
        system=system,
    )
    if not complete:
        message = f"{system}: matrix incomplete after run ({reason})"
        hard_failures.append(message)
        logger.error(message)
        return False

    try:
        trace_audit = export_trace_audit(matrix_output_dir=output_dir)
        logger.info(f"{system}: trace audit written to {trace_audit.output_path}")
    except Exception as exc:
        scoped = f"{system}: trace audit export failed: {exc}"
        warnings.append(scoped)
        logger.warning(scoped)

    _log_matrix_snapshot(output_dir=output_dir, system=system, log_path=log_path)
    return True


def _write_proofs_and_reports(
    *,
    run_key: str,
    mare_dir: Path,
    quare_dir: Path,
    run_dir: Path,
    proofs_dir: Path,
    logs_dir: Path,
    expected_runs: int,
    settings: list[str],
    controls: dict[str, Any],
    logger: _Logger,
    warnings: list[str],
) -> None:
    """Generate proofs, verdict, and human-readable report outputs."""

    mare_rows = _read_jsonl_rows(mare_dir / RUNS_JSONL_NAME)
    quare_rows = _read_jsonl_rows(quare_dir / RUNS_JSONL_NAME)

    conversation_summary = _generate_conversation_logs(
        run_key=run_key,
        logs_dir=logs_dir,
        mare_rows=mare_rows,
        quare_rows=quare_rows,
    )
    write_json_file(proofs_dir / "conversation_log_evidence.json", conversation_summary)
    if not bool(conversation_summary.get("is_complete", False)):
        warning = (
            "Conversation logs are incomplete; finality gate will fail until logs are regenerated."
        )
        warnings.append(warning)
        logger.warning(warning)

    mare_stats = _collect_system_stats(system=SYSTEM_MARE, rows=mare_rows)
    quare_stats = _collect_system_stats(system=SYSTEM_QUARE, rows=quare_rows)

    replay = _run_independent_validator_replay(
        mare_rows=mare_rows,
        quare_rows=quare_rows,
        logs_dir=logs_dir,
    )

    deltas = _compute_key_deltas(
        mare_csv=mare_dir / BY_CASE_CSV_NAME,
        quare_csv=quare_dir / BY_CASE_CSV_NAME,
    )
    paper_control_check = _evaluate_paper_control_profile(controls)
    paper_claims = _build_paper_claim_validation(
        quare_csv=quare_dir / BY_CASE_CSV_NAME,
        controls=controls,
        control_check=paper_control_check,
    )

    validation_evidence = {
        "generated_at": utc_timestamp(),
        "expected_runs_per_system": expected_runs,
        "systems": {
            SYSTEM_MARE: mare_stats,
            SYSTEM_QUARE: quare_stats,
        },
    }
    write_json_file(proofs_dir / "final_validation_evidence.json", validation_evidence)
    write_json_file(proofs_dir / "independent_validator_replay.json", replay)
    write_json_file(proofs_dir / "quare_vs_mare_deltas.json", deltas)
    write_json_file(proofs_dir / PAPER_CLAIMS_FILE, paper_claims)

    verdict = _build_verdict(
        expected_runs=expected_runs,
        settings=settings,
        controls=controls,
        mare_stats=mare_stats,
        quare_stats=quare_stats,
        replay=replay,
        deltas=deltas,
        paper_control_check=paper_control_check,
        paper_claims=paper_claims,
        conversation_summary=conversation_summary,
    )
    write_json_file(proofs_dir / "finality_threshold_verdict.json", verdict)

    manifest = _build_manifest(
        key_paths=[
            mare_dir / RUNS_JSONL_NAME,
            mare_dir / BY_CASE_CSV_NAME,
            mare_dir / SUMMARY_CSV_NAME,
            mare_dir / ABLATION_CSV_NAME,
            mare_dir / VALIDITY_MD_NAME,
            quare_dir / RUNS_JSONL_NAME,
            quare_dir / BY_CASE_CSV_NAME,
            quare_dir / SUMMARY_CSV_NAME,
            quare_dir / ABLATION_CSV_NAME,
            quare_dir / VALIDITY_MD_NAME,
            proofs_dir / "final_validation_evidence.json",
            proofs_dir / "independent_validator_replay.json",
            proofs_dir / "quare_vs_mare_deltas.json",
            proofs_dir / PAPER_CLAIMS_FILE,
            proofs_dir / "conversation_log_evidence.json",
            proofs_dir / "finality_threshold_verdict.json",
        ],
        root_dir=run_dir,
    )
    write_json_file(proofs_dir / "manifest.json", manifest)

    _write_report_readme(
        path=run_dir / "README.md",
        run_dir=run_dir,
        verdict=verdict,
        validation_evidence=validation_evidence,
        conversation_summary=conversation_summary,
        paper_claims=paper_claims,
    )
    _write_analysis_md(
        path=run_dir / "analysis.md",
        deltas=deltas,
        replay=replay,
        verdict=verdict,
        conversation_summary=conversation_summary,
        paper_claims=paper_claims,
        warnings=warnings,
    )
    logger.info("Proof bundle and report documents generated.")


def _build_verdict(
    *,
    expected_runs: int,
    settings: list[str],
    controls: dict[str, Any],
    mare_stats: dict[str, Any],
    quare_stats: dict[str, Any],
    replay: dict[str, Any],
    deltas: dict[str, Any],
    paper_control_check: dict[str, Any],
    paper_claims: dict[str, Any],
    conversation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build strict GO/NO-GO verdict payload with explicit thresholds."""

    def _strict_validity(stats: dict[str, Any], replay_stats: dict[str, Any]) -> bool:
        return (
            int(stats["total_runs"]) == expected_runs
            and int(stats["validation_passed_runs"]) == expected_runs
            and int(stats["validation_error_items"]) == 0
            and int(stats["validation_warning_items"]) == 0
            and int(replay_stats["error_items"]) == 0
            and int(replay_stats["warning_items"]) == 0
        )

    def _taint_clean(stats: dict[str, Any]) -> bool:
        return (
            int(stats["fallback_tainted_runs"]) == 0
            and int(stats["retry_used_runs"]) == 0
            and int(stats["rag_fallback_used_runs"]) == 0
            and int(stats["llm_fallback_used_runs"]) == 0
        )

    replay_systems = replay.get("systems", {})
    mare_replay = replay_systems.get(SYSTEM_MARE, {})
    quare_replay = replay_systems.get(SYSTEM_QUARE, {})

    by_setting = deltas.get("by_setting", {})
    niv_delta = _to_float(
        by_setting.get(SETTING_NEGOTIATION_INTEGRATION_VERIFICATION, {})
        .get("semantic_preservation_f1", {})
        .get("quare_minus_mare")
    )

    checks = {
        "mare_strict_validity": _strict_validity(mare_stats, mare_replay),
        "quare_strict_validity": _strict_validity(quare_stats, quare_replay),
        "mare_taint_cleanliness": _taint_clean(mare_stats),
        "quare_taint_cleanliness": _taint_clean(quare_stats),
        "behavioral_separation": (
            int(mare_stats["quare_mode_markers"]) == 0
            and int(quare_stats["quare_mode_markers"]) > 0
            and int(quare_stats["llm_turns_sum"]) > 0
        ),
        "comparability_metadata_complete": (
            int(mare_stats["metadata_incomplete_runs"]) == 0
            and int(quare_stats["metadata_incomplete_runs"]) == 0
        ),
        "runtime_semantics_complete": int(
            mare_stats.get("mare_semantics_incomplete_runs", 0)
        )
        == 0
        and int(mare_stats.get("mare_llm_emulation_runs", 0)) == 0,
        "llm_seed_reproducibility": (
            int(mare_stats.get("seed_reproducibility_incomplete_runs", 0)) == 0
            and int(quare_stats.get("seed_reproducibility_incomplete_runs", 0)) == 0
        ),
        "paper_controls_matched": bool(paper_control_check.get("is_paper_matched", False)),
        "paper_claims_non_contradictory": bool(
            paper_claims.get("claim_summary", {}).get("hard_fail_count", 1) == 0
        ),
        "niv_semantic_non_regression": niv_delta is not None and niv_delta >= -0.005,
        "conversation_logs_complete": bool(conversation_summary.get("is_complete", False)),
    }
    final_verdict = "GO" if all(bool(value) for value in checks.values()) else "NO-GO"

    return {
        "generated_at": utc_timestamp(),
        "thresholds": {
            "strict_validity": "all runs valid, no validator warnings/errors, and clean replay",
            "taint_cleanliness": "fallback/retry taint counters are zero",
            "behavioral_separation": "MARE has no QUARE markers; QUARE has dialectic markers",
            "comparability_metadata_complete": "system_identity/provenance/execution/comparability present",
            "runtime_semantics_complete": (
                "MARE multi-agent runs include LLM-driven 5-role/9-action semantics evidence"
            ),
            "llm_seed_reproducibility": (
                "all runs with LLM turns record seed_applied_turns matching llm turns"
            ),
            "paper_controls_matched": (
                "model/temperature/seeds/settings match paper-defined profile"
            ),
            "paper_claims_non_contradictory": (
                "paper claim mapping contains no hard FAIL contradictions"
            ),
            "niv_semantic_non_regression": "QUARE-MARE NIV semantic_preservation_f1 >= -0.005",
            "conversation_logs_complete": "all runs have complete timeline + per-agent markdown logs",
        },
        "check_results": checks,
        "key_metrics": {
            "niv_semantic_preservation_delta_quare_minus_mare": niv_delta,
            "mare_llm_turns_sum": mare_stats["llm_turns_sum"],
            "quare_llm_turns_sum": quare_stats["llm_turns_sum"],
            "conversation_coverage_ratio": conversation_summary.get("coverage_ratio", 0.0),
            "conversation_complete_runs": conversation_summary.get("complete_runs", 0),
            "conversation_expected_runs": conversation_summary.get("expected_runs", 0),
            "settings": settings,
            "paper_controls": {
                "input": {
                    "model": controls.get("model"),
                    "temperature": controls.get("temperature"),
                    "seeds": controls.get("seeds"),
                    "settings": controls.get("settings"),
                },
                "is_paper_matched": paper_control_check.get("is_paper_matched", False),
                "mismatches": paper_control_check.get("mismatches", []),
            },
            "paper_claim_summary": paper_claims.get("claim_summary", {}),
        },
        "final_completion_verdict": final_verdict,
    }


def _run_independent_validator_replay(
    *,
    mare_rows: list[dict[str, Any]],
    quare_rows: list[dict[str, Any]],
    logs_dir: Path,
) -> dict[str, Any]:
    """Replay run-record/artifact/system validations for all rows."""

    replay_log = _Logger(logs_dir / "validator-replay.log")
    systems = {
        SYSTEM_MARE: mare_rows,
        SYSTEM_QUARE: quare_rows,
    }
    payload: dict[str, Any] = {
        "generated_at": utc_timestamp(),
        "systems": {},
    }

    for system, rows in systems.items():
        error_items = 0
        warning_items = 0
        sampled_errors: list[str] = []
        sampled_warnings: list[str] = []

        for row in rows:
            artifacts_dir = Path(str(row.get("artifacts_dir", "")))
            run_record_path = artifacts_dir / "run_record.json"
            run_report = validate_run_record(run_record_path)
            artifact_report = validate_phase_artifacts(artifacts_dir)
            behavior_report = validate_system_behavior_contract(
                system=system,
                artifacts_dir=artifacts_dir,
                run_record_path=run_record_path,
            )

            reports = (run_report, artifact_report, behavior_report)
            for report in reports:
                error_items += len(report.errors)
                warning_items += len(report.warnings)
                if report.errors and len(sampled_errors) < 10:
                    sampled_errors.extend(report.errors[: max(0, 10 - len(sampled_errors))])
                if report.warnings and len(sampled_warnings) < 10:
                    sampled_warnings.extend(report.warnings[: max(0, 10 - len(sampled_warnings))])

        replay_log.info(
            f"{system}: replayed {len(rows)} runs, errors={error_items}, warnings={warning_items}"
        )
        payload["systems"][system] = {
            "total_runs": len(rows),
            "error_items": error_items,
            "warning_items": warning_items,
            "sampled_errors": sampled_errors,
            "sampled_warnings": sampled_warnings,
        }

    return payload


def _collect_system_stats(system: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect strict validity and behavior separation counters for one system."""

    validation_error_items = 0
    validation_warning_items = 0
    validation_passed_runs = 0
    fallback_tainted_runs = 0
    retry_used_runs = 0
    rag_fallback_used_runs = 0
    llm_fallback_used_runs = 0
    quare_mode_markers = 0
    llm_turns_sum = 0
    metadata_incomplete_runs = 0
    seed_reproducibility_incomplete_runs = 0
    mare_semantics_incomplete_runs = 0
    mare_llm_emulation_runs = 0
    optional_artifact_presence = {
        PHASE0_FILENAME: 0,
        PHASE25_FILENAME: 0,
        PHASE5_FILENAME: 0,
    }

    for row in rows:
        if bool(row.get("validation_passed", False)):
            validation_passed_runs += 1
        validation_error_items += len(row.get("validation_errors", []))
        validation_warning_items += len(row.get("validation_warnings", []))

        notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
        execution_flags = row.get("execution_flags")
        comparability = row.get("comparability")
        if not isinstance(row.get("system_identity"), dict) or not isinstance(
            row.get("provenance"), dict
        ) or not isinstance(execution_flags, dict) or not isinstance(comparability, dict):
            metadata_incomplete_runs += 1

        phase2_llm = notes.get("phase2_llm") if isinstance(notes, dict) else None
        runtime_llm = notes.get("runtime_llm") if isinstance(notes, dict) else None
        if _llm_seed_reproducibility_incomplete(phase2_llm) or _llm_seed_reproducibility_incomplete(
            runtime_llm
        ):
            seed_reproducibility_incomplete_runs += 1

        if system == SYSTEM_MARE and str(row.get("setting", "")).strip() != "single_agent":
            runtime_raw = notes.get("runtime_semantics") if isinstance(notes, dict) else None
            runtime_ok = isinstance(runtime_raw, dict)
            runtime_llm_ok = isinstance(runtime_raw, dict)
            if runtime_ok and isinstance(runtime_raw, dict):
                roles = _runtime_semantics_string_list(runtime_raw.get("roles_executed"))
                actions = _runtime_semantics_string_list(runtime_raw.get("actions_executed"))
                llm_actions = _runtime_semantics_string_list(runtime_raw.get("llm_actions"))
                digest = str(runtime_raw.get("workspace_digest", "")).strip()
                runtime_ok = (
                    roles is not None
                    and actions is not None
                    and sorted(roles) == sorted(MARE_AGENT_ROLES)
                    and sorted(actions) == sorted(MARE_ACTIONS)
                    and len(digest) == 64
                )
                runtime_llm_ok = (
                    llm_actions is not None
                    and
                    str(runtime_raw.get("execution_mode", "")).strip() == "llm_driven"
                    and _to_int(runtime_raw.get("llm_turns"), default=0) >= len(MARE_ACTIONS)
                    and _to_int(runtime_raw.get("llm_fallback_turns"), default=0) == 0
                    and sorted(llm_actions) == sorted(MARE_ACTIONS)
                )
            if not runtime_ok:
                mare_semantics_incomplete_runs += 1
            if not runtime_llm_ok:
                mare_llm_emulation_runs += 1

            if isinstance(runtime_raw, dict):
                llm_turns_sum += _to_int(runtime_raw.get("llm_turns"), default=0)

        if isinstance(execution_flags, dict):
            if bool(execution_flags.get("fallback_tainted", False)):
                fallback_tainted_runs += 1
            if bool(execution_flags.get("retry_used", False)):
                retry_used_runs += 1
            if bool(execution_flags.get("rag_fallback_used", False)):
                rag_fallback_used_runs += 1
            if bool(execution_flags.get("llm_fallback_used", False)):
                llm_fallback_used_runs += 1

        artifacts_dir = Path(str(row.get("artifacts_dir", "")))
        for file_name in optional_artifact_presence:
            if (artifacts_dir / file_name).exists():
                optional_artifact_presence[file_name] += 1

        phase2_path = artifacts_dir / "phase2_negotiation_trace.json"
        if phase2_path.exists():
            try:
                phase2_payload = json.loads(phase2_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            summary_stats = phase2_payload.get("summary_stats", {})
            if isinstance(summary_stats, dict):
                llm_turns_sum += _to_int(summary_stats.get("llm_turns"), default=0)
            negotiations = phase2_payload.get("negotiations", {})
            if isinstance(negotiations, dict):
                for negotiation in negotiations.values():
                    if not isinstance(negotiation, dict):
                        continue
                    steps = negotiation.get("steps", [])
                    if not isinstance(steps, list):
                        continue
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                        if str(step.get("negotiation_mode", "")).strip() == "quare_dialectic":
                            quare_mode_markers += 1

    return {
        "system": system,
        "total_runs": len(rows),
        "validation_passed_runs": validation_passed_runs,
        "validation_error_items": validation_error_items,
        "validation_warning_items": validation_warning_items,
        "fallback_tainted_runs": fallback_tainted_runs,
        "retry_used_runs": retry_used_runs,
        "rag_fallback_used_runs": rag_fallback_used_runs,
        "llm_fallback_used_runs": llm_fallback_used_runs,
        "quare_mode_markers": quare_mode_markers,
        "llm_turns_sum": llm_turns_sum,
        "metadata_incomplete_runs": metadata_incomplete_runs,
        "seed_reproducibility_incomplete_runs": seed_reproducibility_incomplete_runs,
        "mare_semantics_incomplete_runs": mare_semantics_incomplete_runs,
        "mare_llm_emulation_runs": mare_llm_emulation_runs,
        "optional_artifact_presence": optional_artifact_presence,
    }


def _llm_seed_reproducibility_incomplete(payload: Any) -> bool:
    """Return True when LLM turns are present but seed application evidence is missing."""

    if not isinstance(payload, dict):
        return False
    if not bool(payload.get("enabled", False)):
        return False
    turns = _to_int(payload.get("turns"), default=0)
    if turns <= 0:
        return False
    seed_turns = _to_int(payload.get("seed_applied_turns"), default=-1)
    return seed_turns < turns


def _evaluate_paper_control_profile(controls: dict[str, Any]) -> dict[str, Any]:
    """Check whether run controls match the paper-declared profile exactly."""

    mismatches: list[str] = []
    model = str(controls.get("model", "")).strip()
    if model != PAPER_MODEL:
        mismatches.append(f"model expected '{PAPER_MODEL}' got '{model}'")

    temperature = _to_float(controls.get("temperature"))
    if temperature is None or abs(temperature - PAPER_TEMPERATURE) > 1e-9:
        mismatches.append(
            f"temperature expected {PAPER_TEMPERATURE} got {controls.get('temperature')}"
        )

    seeds_raw = controls.get("seeds")
    seeds = []
    if isinstance(seeds_raw, list):
        seeds = sorted(_to_int(item, default=-1) for item in seeds_raw)
    paper_seeds = sorted(PAPER_SEEDS)
    if seeds != paper_seeds:
        mismatches.append(f"seeds expected {paper_seeds} got {seeds}")

    settings_raw = controls.get("settings")
    settings = sorted(str(item).strip() for item in settings_raw) if isinstance(settings_raw, list) else []
    paper_settings = sorted(PAPER_SETTINGS)
    if settings != paper_settings:
        mismatches.append(f"settings expected {paper_settings} got {settings}")

    return {
        "is_paper_matched": len(mismatches) == 0,
        "mismatches": mismatches,
        "paper_profile": {
            "model": PAPER_MODEL,
            "temperature": PAPER_TEMPERATURE,
            "seeds": list(PAPER_SEEDS),
            "settings": list(PAPER_SETTINGS),
        },
    }


def _discover_repo_root(*, run_dir: Path) -> Path:
    """Return a stable repository root for contract discovery.

    Prefer anchors near the run directory, then fall back to the module path,
    then the current working directory.
    """

    anchor_candidates = [run_dir.resolve(), Path(__file__).resolve().parent, Path.cwd().resolve()]
    for anchor in anchor_candidates:
        for candidate in (anchor, *anchor.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src/openre_bench").is_dir():
                return candidate
    return Path.cwd().resolve()


def _audit_precision_f1_contract(*, run_dir: Path) -> dict[str, Any]:
    """Audit whether literal RQ1 precision/F1 inputs are available."""

    repo_root = _discover_repo_root(run_dir=run_dir)
    local_gt_dir = repo_root / "data/ground_truth"

    contract_roots = [repo_root]

    external_gt_dirs = [root / "data/ground_truth" for root in contract_roots]
    script_candidates = [
        root / "scripts/evaluate_precision_recall_f1.py" for root in contract_roots
    ] + [root / "scripts/calculate_all_phases_precision_f1.py" for root in contract_roots]

    local_gt_files = (
        sorted(
            str(path)
            for path in local_gt_dir.glob("*.json")
            if path.is_file() and path.name.lower() != "readme.json"
        )
        if local_gt_dir.exists()
        else []
    )
    external_gt_files = sorted(
        str(path)
        for gt_dir in external_gt_dirs
        if gt_dir.exists()
        for path in gt_dir.glob("*.json")
        if path.is_file() and path.name.lower() != "readme.json"
    )

    label_candidates: set[Path] = set()
    for pattern in (
        "**/precision_evaluation_progress.json",
        "**/manual_precision_labels.json",
        "**/all_phases_precision_f1_results.json",
    ):
        for path in run_dir.glob(pattern):
            if path.is_file():
                label_candidates.add(path)
    label_files = sorted(str(path) for path in label_candidates)
    script_availability = {
        str(path): path.exists() and path.is_file() for path in script_candidates
    }

    has_gt = len(local_gt_files) > 0 or len(external_gt_files) > 0
    has_labels = len(label_files) > 0
    # Either script path is sufficient; these are alternate calculators in the
    # companion repository and may not coexist in every checkout.
    has_scripts = any(script_availability.values()) if script_availability else False

    missing_requirements: list[str] = []
    if not has_gt:
        missing_requirements.append(
            "ground truth case-level precision/F1 benchmark files (*.json)"
        )
    if not has_labels:
        missing_requirements.append(
            "manual TP/FP label artifacts (precision_evaluation_progress.json)"
        )
    # Script presence is tracked for provenance but not required to mark the
    # contract as available because OpenRE-Bench computes literal gains internally.

    return {
        "contract_available": has_gt and has_labels,
        "checked_paths": [
            str(repo_root),
            str(local_gt_dir),
            *(str(path) for path in external_gt_dirs),
            str(run_dir),
            *(str(path) for path in script_candidates),
        ],
        "found": {
            "local_ground_truth_files": local_gt_files,
            "external_ground_truth_files": external_gt_files,
            "label_files": label_files,
            "scripts": script_availability,
            "script_hint_available": has_scripts,
        },
        "missing_requirements": missing_requirements,
    }


def _split_requirement_units(text: str) -> list[str]:
    """Split case requirement text into deterministic comparison units."""

    chunks = re.split(r"[\n\r\.!?;]+", text)
    units: list[str] = []
    for chunk in chunks:
        normalized = " ".join(chunk.strip().split())
        if len(normalized) >= 20:
            units.append(normalized)
    return units


def _load_case_requirement_units(*, run_dir: Path) -> dict[str, list[str]]:
    """Load requirement-source units from /auto controls case inputs."""

    controls_path = run_dir / "controls.json"
    if not controls_path.exists() or not controls_path.is_file():
        return {}
    try:
        controls_payload = json.loads(controls_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(controls_payload, dict):
        return {}

    cases_dir_raw = controls_payload.get("cases_dir")
    cases_dir = Path(str(cases_dir_raw)) if isinstance(cases_dir_raw, str) else None
    if cases_dir is None or not cases_dir.exists() or not cases_dir.is_dir():
        return {}

    units_by_case: dict[str, list[str]] = {}
    case_files = controls_payload.get("case_files")
    candidate_files: list[Path]
    if isinstance(case_files, list) and all(isinstance(item, str) for item in case_files):
        candidate_files = [cases_dir / str(item) for item in case_files]
    else:
        candidate_files = sorted(cases_dir.glob("*_input.json"))

    for case_path in candidate_files:
        if not case_path.exists() or not case_path.is_file():
            continue
        try:
            payload = json.loads(case_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        case_name = str(payload.get("case_name", "")).strip()
        requirement_text = str(payload.get("requirement", "")).strip()
        if not case_name or not requirement_text:
            continue
        units = _split_requirement_units(requirement_text)
        if not units:
            continue
        units_by_case[case_name] = units
    return units_by_case


def _token_overlap_f1(*, left: str, right: str) -> float:
    """Compute token-overlap F1 for one pair of text strings."""

    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _relative_gain(current: float | None, baseline: float | None) -> float | None:
    """Return relative gain `(current - baseline) / baseline` when defined."""

    if current is None or baseline is None or baseline == 0:
        return None
    return round((current - baseline) / baseline, 6)


def _semantic_precision_recall_f1(*, candidates: list[str], references: list[str]) -> tuple[float, float, float] | None:
    """Compute deterministic precision/recall/F1 proxies over text units."""

    if not candidates or not references:
        return None

    candidate_best = [
        max(_token_overlap_f1(left=candidate, right=reference) for reference in references)
        for candidate in candidates
    ]
    reference_best = [
        max(_token_overlap_f1(left=candidate, right=reference) for candidate in candidates)
        for reference in references
    ]

    precision = sum(candidate_best) / len(candidate_best)
    recall = sum(reference_best) / len(reference_best)
    f1 = 0.0
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _artifact_phase1_requirement_texts(*, artifacts_dir: Path) -> list[str]:
    """Extract phase-1 requirement-like descriptions from one run artifacts dir."""

    phase1_path = artifacts_dir / PHASE1_FILENAME
    if not phase1_path.exists() or not phase1_path.is_file():
        return []
    try:
        payload = json.loads(phase1_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []

    texts: list[str] = []
    for elements in payload.values():
        if not isinstance(elements, list):
            continue
        for element in elements:
            if not isinstance(element, dict):
                continue
            description = str(element.get("description", "")).strip()
            if description:
                texts.append(description)
    return texts


def _artifact_phase2_requirement_texts(*, artifacts_dir: Path) -> list[str]:
    """Extract phase-2 negotiated requirement-like descriptions from one run."""

    phase2_path = artifacts_dir / PHASE2_FILENAME
    if not phase2_path.exists() or not phase2_path.is_file():
        return []
    try:
        payload = json.loads(phase2_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []

    texts: list[str] = []
    seen: set[str] = set()
    negotiations = payload.get("negotiations", {})
    if not isinstance(negotiations, dict):
        return texts

    for negotiation in negotiations.values():
        if not isinstance(negotiation, dict):
            continue
        steps = negotiation.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue

        selected_step: dict[str, Any] | None = None
        for step in reversed(steps):
            if isinstance(step, dict) and str(step.get("message_type", "")).strip() == "backward":
                selected_step = step
                break
        if selected_step is None and isinstance(steps[-1], dict):
            selected_step = steps[-1]
        if selected_step is None:
            continue

        elements = selected_step.get("kaos_elements", [])
        if not isinstance(elements, list):
            continue
        for element in elements:
            if not isinstance(element, dict):
                continue
            description = str(element.get("description", "")).strip()
            name = str(element.get("name", "")).strip()
            text = description or name
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def _derive_precision_f1_gains_from_artifacts(*, run_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Derive precision/F1 gains from run artifacts when CSV columns are absent."""

    requirement_units_by_case = _load_case_requirement_units(run_dir=run_dir)
    if not requirement_units_by_case:
        return {
            "method": "token_overlap_requirement_alignment",
            "pair_count": 0,
            "precision_gain": None,
            "f1_gain": None,
        }

    setting_map: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        setting = str(row.get("setting", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        seed = _to_int(row.get("seed"), default=-1)
        if not case_id or not setting or not run_id:
            continue
        case_settings = setting_map.setdefault((case_id, seed), {})
        case_settings[setting] = run_id

    precision_gains: list[float] = []
    f1_gains: list[float] = []
    details: list[dict[str, Any]] = []
    for (case_id, seed), settings in sorted(setting_map.items()):
        baseline_run = settings.get("multi_agent_without_negotiation")
        negotiated_run = settings.get("multi_agent_with_negotiation")
        if not baseline_run or not negotiated_run:
            continue

        references = requirement_units_by_case.get(case_id)
        if not references:
            continue

        baseline_artifacts = run_dir / SYSTEM_QUARE / "runs" / baseline_run
        negotiated_artifacts = run_dir / SYSTEM_QUARE / "runs" / negotiated_run
        baseline_candidates = _artifact_phase1_requirement_texts(artifacts_dir=baseline_artifacts)
        negotiated_candidates = _artifact_phase2_requirement_texts(artifacts_dir=negotiated_artifacts)
        baseline_scores = _semantic_precision_recall_f1(
            candidates=baseline_candidates,
            references=references,
        )
        negotiated_scores = _semantic_precision_recall_f1(
            candidates=negotiated_candidates,
            references=references,
        )
        if baseline_scores is None or negotiated_scores is None:
            continue

        baseline_precision, _, baseline_f1 = baseline_scores
        negotiated_precision, _, negotiated_f1 = negotiated_scores
        precision_gain = _relative_gain(negotiated_precision, baseline_precision)
        f1_gain = _relative_gain(negotiated_f1, baseline_f1)
        if precision_gain is not None:
            precision_gains.append(precision_gain)
        if f1_gain is not None:
            f1_gains.append(f1_gain)
        details.append(
            {
                "case_id": case_id,
                "seed": seed,
                "phase1_precision": round(baseline_precision, 6),
                "phase2_precision": round(negotiated_precision, 6),
                "phase1_f1": round(baseline_f1, 6),
                "phase2_f1": round(negotiated_f1, 6),
                "precision_gain": precision_gain,
                "f1_gain": f1_gain,
            }
        )

    return {
        "method": "token_overlap_requirement_alignment",
        "pair_count": len(details),
        "precision_gain": round(max(precision_gains), 6) if precision_gains else None,
        "f1_gain": round(max(f1_gains), 6) if f1_gains else None,
        "details": details,
    }


def _normalize_case_id(value: Any) -> str | None:
    """Normalize case identifiers to uppercase alphanumeric tokens."""

    text = str(value or "").strip()
    if not text:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", text)
    if not cleaned:
        return None
    return cleaned.upper()


def _canonical_phase_label(value: Any) -> str | None:
    """Normalize heterogeneous phase labels to phase1/phase2."""

    text = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    if compact in {"phase1", "p1", "1", "initial", "generation"}:
        return "phase1"
    if compact in {"phase2", "p2", "2", "negotiation", "dialectic"}:
        return "phase2"
    return None


def _canonical_system_label(value: Any) -> str | None:
    """Normalize heterogeneous system labels to canonical IDs."""

    text = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    if compact in {SYSTEM_MARE}:
        return SYSTEM_MARE
    if compact in {SYSTEM_QUARE}:
        return SYSTEM_QUARE
    return None


def _canonical_setting_label(value: Any) -> str | None:
    """Normalize heterogeneous setting labels to canonical setting IDs."""

    text = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    aliases = {
        "singleagent": "single_agent",
        "multiagentwithoutnegotiation": SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
        "multiagentwithnegotiation": SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        "negotiationintegrationverification": SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    }
    return aliases.get(compact)


def _infer_system_from_run_id(value: Any) -> str | None:
    """Infer run system identity from run identifiers."""

    text = str(value or "").strip().lower()
    if not text:
        return None
    for system in SUPPORTED_SYSTEMS:
        if text.startswith(f"{system}-") or f"-{system}-" in text:
            return system
    return None


def _infer_setting_from_text(value: Any) -> str | None:
    """Infer canonical setting from IDs, labels, or path fragments."""

    text = str(value or "").strip().lower()
    if not text:
        return None
    for setting in PAPER_SETTINGS:
        if setting in text:
            return setting
    return _canonical_setting_label(text)


def _infer_system_hint_from_path(path: Path) -> str | None:
    """Infer system from a label file path."""

    for part in path.parts:
        system = _canonical_system_label(part)
        if system is not None:
            return system
    return _infer_system_from_run_id(path.name)


def _write_literal_phase_metrics(
    *,
    output: dict[tuple[str, str, int], dict[str, dict[str, float]]],
    system: str,
    case_id: str,
    seed: int,
    phase: str,
    metrics: dict[str, float],
) -> None:
    """Write phase metrics deterministically without cross-label overwrites."""

    if seed < 0 or phase not in {"phase1", "phase2"}:
        return
    existing = output.setdefault((system, case_id, seed), {})
    current = existing.setdefault(phase, {})
    for metric_name, metric_value in metrics.items():
        if metric_name not in {"precision", "f1"}:
            continue
        if metric_name not in current:
            current[metric_name] = float(metric_value)
            continue
        # Keep the lower duplicate value to preserve fail-closed behavior.
        current[metric_name] = min(float(current[metric_name]), float(metric_value))


def _literal_phase_setting_compatible(*, phase: str, setting: str | None) -> bool:
    """Allow literal phase pairing only for the paper baseline and negotiated settings."""

    if setting is None:
        return True
    if phase == "phase1":
        return setting == SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION
    if phase == "phase2":
        return setting == SETTING_MULTI_AGENT_WITH_NEGOTIATION
    return False


def _extract_phase_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Extract precision/F1 metrics from one JSON object if present."""

    def _pick_metric_value(*keys: str) -> Any:
        for key in keys:
            if key in metric_candidates:
                return metric_candidates[key]
        return None

    metric_candidates: dict[str, Any] = {}
    for key, raw_value in payload.items():
        key_text = str(key).strip().lower()
        metric_candidates[key_text] = raw_value
        compact_key = re.sub(r"[^a-z0-9]", "", key_text)
        if compact_key:
            metric_candidates[compact_key] = raw_value

    nested_metrics = payload.get("metrics")
    if isinstance(nested_metrics, dict):
        for key, raw_value in nested_metrics.items():
            key_text = str(key).strip().lower()
            metric_candidates[key_text] = raw_value
            compact_key = re.sub(r"[^a-z0-9]", "", key_text)
            if compact_key:
                metric_candidates[compact_key] = raw_value

    precision = _to_float(_pick_metric_value("precision", "prec", "phaseprecision"))
    recall = _to_float(_pick_metric_value("recall"))
    f1 = _to_float(_pick_metric_value("f1", "f1score", "f1_score"))

    tp = _to_float(_pick_metric_value("tp", "truepositive", "truepositives"))
    fp = _to_float(_pick_metric_value("fp", "falsepositive", "falsepositives"))
    fn = _to_float(_pick_metric_value("fn", "falsenegative", "falsenegatives"))

    if precision is None and tp is not None and fp is not None and (tp + fp) > 0:
        precision = tp / (tp + fp)
    if recall is None and tp is not None and fn is not None and (tp + fn) > 0:
        recall = tp / (tp + fn)
    if f1 is None and precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)

    metrics: dict[str, float] = {}
    if precision is not None:
        metrics["precision"] = float(precision)
    if f1 is not None:
        metrics["f1"] = float(f1)
    return metrics


def _ingest_literal_metric_node(
    *,
    node: Any,
    output: dict[tuple[str, str, int], dict[str, dict[str, float]]],
    case_hint: str | None = None,
    seed_hint: int = -1,
    system_hint: str | None = None,
    setting_hint: str | None = None,
) -> None:
    """Collect literal phase1/phase2 metrics from heterogeneous JSON nodes."""

    if isinstance(node, list):
        for item in node:
            _ingest_literal_metric_node(
                node=item,
                output=output,
                case_hint=case_hint,
                seed_hint=seed_hint,
                system_hint=system_hint,
                setting_hint=setting_hint,
            )
        return

    if not isinstance(node, dict):
        return

    case_id = _normalize_case_id(
        node.get("case_id")
        or node.get("case")
        or node.get("case_name")
        or case_hint
    )
    seed = _to_int(node.get("seed"), default=seed_hint)
    run_id = node.get("run_id") or node.get("record_id") or node.get("id")

    system = _canonical_system_label(
        node.get("system") or node.get("system_name") or node.get("runtime_system") or system_hint
    )
    if system is None:
        system = _infer_system_from_run_id(run_id)
    system_key = system or UNSPECIFIED_SYSTEM

    setting = _canonical_setting_label(
        node.get("setting")
        or node.get("setting_name")
        or node.get("runtime_setting")
        or node.get("mode")
        or setting_hint
    )
    if setting is None:
        setting = _infer_setting_from_text(run_id)

    phase = _canonical_phase_label(node.get("phase") or node.get("phase_name") or node.get("stage"))
    if phase is None and setting in LITERAL_SETTING_TO_PHASE:
        phase = LITERAL_SETTING_TO_PHASE[setting]

    if case_id is not None and phase is not None and _literal_phase_setting_compatible(
        phase=phase,
        setting=setting,
    ):
        metrics = _extract_phase_metrics(node)
        if metrics:
            _write_literal_phase_metrics(
                output=output,
                system=system_key,
                case_id=case_id,
                seed=seed,
                phase=phase,
                metrics=metrics,
            )

    phase_blocks = {
        "phase1": "phase1",
        "phase_1": "phase1",
        "p1": "phase1",
        "phase2": "phase2",
        "phase_2": "phase2",
        "p2": "phase2",
    }

    for key, candidate in node.items():
        if not isinstance(candidate, dict):
            continue
        setting_name = _canonical_setting_label(key)
        if setting_name not in LITERAL_SETTING_TO_PHASE:
            continue
        if case_id is None:
            continue
        metrics = _extract_phase_metrics(candidate)
        if not metrics:
            continue
        phase_name = LITERAL_SETTING_TO_PHASE[setting_name]
        if not _literal_phase_setting_compatible(phase=phase_name, setting=setting_name):
            continue
        _write_literal_phase_metrics(
            output=output,
            system=system_key,
            case_id=case_id,
            seed=seed,
            phase=phase_name,
            metrics=metrics,
        )

    for key, phase_name in phase_blocks.items():
        candidate = node.get(key)
        if isinstance(candidate, dict) and case_id is not None:
            if not _literal_phase_setting_compatible(phase=phase_name, setting=setting):
                continue
            metrics = _extract_phase_metrics(candidate)
            if metrics:
                _write_literal_phase_metrics(
                    output=output,
                    system=system_key,
                    case_id=case_id,
                    seed=seed,
                    phase=phase_name,
                    metrics=metrics,
                )

    for key in ("case_results", "cases", "results", "items", "records"):
        nested = node.get(key)
        if isinstance(nested, list):
            for item in nested:
                _ingest_literal_metric_node(
                    node=item,
                    output=output,
                    case_hint=case_id,
                    seed_hint=seed,
                    system_hint=system,
                    setting_hint=setting,
                )


def _load_ground_truth_case_ids(contract: dict[str, Any]) -> set[str]:
    """Load normalized case IDs declared by available ground-truth files."""

    case_ids: set[str] = set()
    found = contract.get("found", {}) if isinstance(contract, dict) else {}
    gt_files: list[str] = []
    if isinstance(found, dict):
        for key in ("local_ground_truth_files", "external_ground_truth_files"):
            values = found.get(key)
            if isinstance(values, list):
                gt_files.extend(str(item) for item in values)

    for file_path in gt_files:
        path = Path(file_path)
        case_from_name = _normalize_case_id(path.stem)
        if case_from_name is not None:
            case_ids.add(case_from_name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            for key in ("case_id", "case_name", "case"):
                value = _normalize_case_id(payload.get(key))
                if value is not None:
                    case_ids.add(value)
    return case_ids


def _compute_literal_precision_f1_gains(
    *,
    run_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute literal precision/F1 gains from manual labels + ground-truth contract."""

    if not bool(contract.get("contract_available", False)):
        return None

    found = contract.get("found", {}) if isinstance(contract, dict) else {}
    label_paths: list[Path] = []
    if isinstance(found, dict):
        for value in found.get("label_files", []):
            path = Path(str(value))
            if path.exists() and path.is_file():
                label_paths.append(path)
    if not label_paths:
        return None

    case_allowlist = _load_ground_truth_case_ids(contract)
    literal_by_case_seed: dict[tuple[str, str, int], dict[str, dict[str, float]]] = {}
    parsed_files: list[str] = []
    parse_errors: list[str] = []

    for label_path in label_paths:
        system_hint = _infer_system_hint_from_path(label_path)
        setting_hint = _infer_setting_from_text(label_path.name)
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_errors.append(str(label_path))
            continue
        parsed_files.append(str(label_path))
        _ingest_literal_metric_node(
            node=payload,
            output=literal_by_case_seed,
            system_hint=system_hint,
            setting_hint=setting_hint,
        )

    precision_gains: list[float] = []
    f1_gains: list[float] = []
    details: list[dict[str, Any]] = []

    explicit_systems: dict[tuple[str, int], set[str]] = {}
    for system, case_id, seed in literal_by_case_seed:
        if system == UNSPECIFIED_SYSTEM:
            continue
        explicit_systems.setdefault((case_id, seed), set()).add(system)

    for (system, case_id, seed), phase_map in sorted(literal_by_case_seed.items()):
        if system != SYSTEM_QUARE:
            if system != UNSPECIFIED_SYSTEM:
                continue
            seen = explicit_systems.get((case_id, seed), set())
            if seen:
                continue
        if case_allowlist and case_id not in case_allowlist:
            continue
        phase1 = phase_map.get("phase1", {})
        phase2 = phase_map.get("phase2", {})
        baseline_precision = _to_float(phase1.get("precision"))
        negotiated_precision = _to_float(phase2.get("precision"))
        baseline_f1 = _to_float(phase1.get("f1"))
        negotiated_f1 = _to_float(phase2.get("f1"))
        precision_gain = _relative_gain(negotiated_precision, baseline_precision)
        f1_gain = _relative_gain(negotiated_f1, baseline_f1)
        if precision_gain is not None:
            precision_gains.append(precision_gain)
        if f1_gain is not None:
            f1_gains.append(f1_gain)
        details.append(
            {
                "case_id": case_id,
                "seed": seed,
                "phase1_precision": baseline_precision,
                "phase2_precision": negotiated_precision,
                "phase1_f1": baseline_f1,
                "phase2_f1": negotiated_f1,
                "precision_gain": precision_gain,
                "f1_gain": f1_gain,
            }
        )

    return {
        "method": "literal_manual_label_ground_truth_contract",
        "pair_count": len(details),
        "precision_gain": round(max(precision_gains), 6) if precision_gains else None,
        "f1_gain": round(max(f1_gains), 6) if f1_gains else None,
        "details": details,
        "parsed_label_files": parsed_files,
        "parse_error_files": parse_errors,
        "ground_truth_cases": sorted(case_allowlist),
        "run_dir": str(run_dir),
    }


def _build_paper_claim_validation(
    *,
    quare_csv: Path,
    controls: dict[str, Any],
    control_check: dict[str, Any],
) -> dict[str, Any]:
    """Map paper claims to concrete artifact values with strict blocker semantics."""

    rows = _read_csv_rows(quare_csv)

    def _rows(setting: str, case_id: str | None = None) -> list[dict[str, Any]]:
        selected = [row for row in rows if str(row.get("setting", "")).strip() == setting]
        if case_id is not None:
            selected = [row for row in selected if str(row.get("case_id", "")).strip() == case_id]
        return selected

    def _mean(rows_subset: list[dict[str, Any]], field: str) -> float | None:
        values = [_to_float(row.get(field)) for row in rows_subset]
        numeric = [value for value in values if value is not None]
        if not numeric:
            return None
        return round(mean(numeric), 6)

    def _gain(observed: float | None, baseline: float | None) -> float | None:
        if observed is None or baseline is None or baseline == 0:
            return None
        return round((observed - baseline) / baseline, 6)

    def _status(observed: float | None, threshold: float, *, comparator: str = ">=") -> str:
        if observed is None:
            return "NON-COMPARABLE"
        if comparator == ">=":
            return "PASS" if observed >= threshold else "FAIL"
        return "PASS" if observed <= threshold else "FAIL"

    def _max_gain_from_columns(column_candidates: tuple[str, ...]) -> tuple[float | None, str | None]:
        if not rows:
            return None, None
        available = {str(key) for key in rows[0].keys()}
        metric_column = next((name for name in column_candidates if name in available), None)
        if metric_column is None:
            return None, None

        # Preserve per-seed comparability: do not collapse all rows in a case
        # into one setting map, otherwise CSV row order can change the gain.
        per_case_seed: dict[tuple[str, int], dict[str, float]] = {}
        for row in rows:
            case_id = str(row.get("case_id", "")).strip()
            setting_name = str(row.get("setting", "")).strip()
            seed = _to_int(row.get("seed"), default=-1)
            value = _to_float(row.get(metric_column))
            if not case_id or value is None:
                continue
            case_map = per_case_seed.setdefault((case_id, seed), {})
            case_map[setting_name] = value

        gains: list[float] = []
        for case_map in per_case_seed.values():
            baseline = case_map.get("multi_agent_without_negotiation")
            negotiated = case_map.get("multi_agent_with_negotiation")
            if baseline is None or negotiated is None or baseline == 0:
                continue
            gains.append((negotiated - baseline) / baseline)

        if not gains:
            return None, metric_column
        return round(max(gains), 6), metric_column

    run_dir = quare_csv.parent.parent
    precision_f1_contract = _audit_precision_f1_contract(run_dir=run_dir)

    single_rows = _rows("single_agent")
    multi_without_rows = _rows("multi_agent_without_negotiation")
    multi_with_rows = _rows("multi_agent_with_negotiation")
    niv_rows = _rows("negotiation_integration_verification")

    single_chv = _mean(single_rows, "chv")
    multi_without_chv = _mean(multi_without_rows, "chv")
    single_mdc = _mean(single_rows, "mdc")
    multi_without_mdc = _mean(multi_without_rows, "mdc")
    single_p1 = _mean(single_rows, "n_phase1_elements")
    multi_without_p1 = _mean(multi_without_rows, "n_phase1_elements")

    ad_phase1 = _mean(_rows("multi_agent_without_negotiation", case_id="AD"), "n_phase1_elements")
    ad_phase3 = _mean(_rows("multi_agent_with_negotiation", case_id="AD"), "n_phase3_elements")
    ad_filtering = None
    if ad_phase1 is not None and ad_phase1 > 0 and ad_phase3 is not None:
        ad_filtering = round((ad_phase1 - ad_phase3) / ad_phase1, 6)

    conflict_rate = _mean(multi_with_rows, "conflict_resolution_rate")
    semantic_max = max(
        [value for value in [_to_float(row.get("semantic_preservation_f1")) for row in niv_rows] if value is not None],
        default=None,
    )
    topology_mean = _mean(multi_with_rows, "topology_is_valid")

    precision_gain, precision_column = _max_gain_from_columns(
        (
            "precision",
            "phase2_precision",
            "precision_phase2",
            "rq1_precision",
            "precision_conflict",
        )
    )
    f1_gain, f1_column = _max_gain_from_columns(
        (
            "f1",
            "phase2_f1",
            "f1_phase2",
            "rq1_f1",
            "conflict_f1",
        )
    )

    literal_precision_f1: dict[str, Any] | None = None
    if bool(precision_f1_contract.get("contract_available", False)):
        literal_precision_f1 = _compute_literal_precision_f1_gains(
            run_dir=run_dir,
            contract=precision_f1_contract,
        )

    literal_method = "literal_manual_label_ground_truth_contract"
    if isinstance(literal_precision_f1, dict):
        literal_method = str(
            literal_precision_f1.get("method", literal_method)
        ).strip() or literal_method
    literal_precision_gain = (
        _to_float(literal_precision_f1.get("precision_gain"))
        if isinstance(literal_precision_f1, dict)
        else None
    )
    literal_f1_gain = (
        _to_float(literal_precision_f1.get("f1_gain"))
        if isinstance(literal_precision_f1, dict)
        else None
    )
    literal_precision_f1_available = bool(
        literal_precision_gain is not None and literal_f1_gain is not None
    )
    if literal_precision_f1_available:
        precision_gain = literal_precision_gain
        f1_gain = literal_f1_gain
        precision_column = f"literal:{literal_method}:precision_gain"
        f1_column = f"literal:{literal_method}:f1_gain"

    derived_precision_f1: dict[str, Any] | None = None
    if not literal_precision_f1_available and (precision_gain is None or f1_gain is None):
        derived_precision_f1 = _derive_precision_f1_gains_from_artifacts(run_dir=run_dir, rows=rows)
        derived_method = str(derived_precision_f1.get("method", "artifact_derivation")).strip()
        if precision_gain is None:
            precision_gain = _to_float(derived_precision_f1.get("precision_gain"))
            if precision_gain is not None:
                precision_column = f"derived:{derived_method}"
        if f1_gain is None:
            f1_gain = _to_float(derived_precision_f1.get("f1_gain"))
            if f1_gain is not None:
                f1_column = f"derived:{derived_method}"

    precision_f1_signal_available = bool(precision_gain is not None and f1_gain is not None)
    source_columns = [str(precision_column or ""), str(f1_column or "")]
    derived_proxy_used = any(value.startswith("derived:") for value in source_columns if value)
    precision_f1_literal_comparable = bool(
        literal_precision_f1_available
        and precision_f1_contract.get("contract_available", False)
    )

    blocker_reason = (
        "RQ1 precision/F1 literal validation requires manual-labeled ground-truth"
        " contract and non-derived precision/F1 evidence; current run only provides"
        " proxy-level derivation or incomplete contract"
    )
    if bool(precision_f1_contract.get("contract_available", False)) and not literal_precision_f1_available:
        blocker_reason = (
            "RQ1 precision/F1 contract artifacts were discovered, but literal"
            " phase1/phase2 precision/F1 gains could not be computed from the"
            " available label payloads"
        )

    blocker_payload = {
        "blocker_code": "missing_rq1_precision_f1_data_contract",
        "severity": "hard_fail",
        "reason": blocker_reason,
        "provenance": {
            "precision_f1_contract": precision_f1_contract,
            "literal_contract_derivation": literal_precision_f1,
            "artifact_derivation": derived_precision_f1,
            "source_columns": {
                "precision": precision_column,
                "f1": f1_column,
            },
            "derived_proxy_used": derived_proxy_used,
            "precision_f1_signal_available": precision_f1_signal_available,
        },
        "closure_criteria": [
            "provide manual TP/FP labeling artifacts for evaluated runs",
            "populate data/ground_truth with case-level benchmark references",
            "ensure label artifacts expose phase1/phase2 precision/F1 per case and seed",
        ],
    }

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "RQ1_CHV_gain",
            "paper_target": "multi-agent CHV gain >= 72% over single-agent",
            "formula": "(CHV_multi_without_neg - CHV_single) / CHV_single",
            "observed": {
                "single_agent": single_chv,
                "multi_agent_without_negotiation": multi_without_chv,
                "gain": _gain(multi_without_chv, single_chv),
            },
            "status": _status(_gain(multi_without_chv, single_chv), 0.72),
            "hard_fail": _status(_gain(multi_without_chv, single_chv), 0.72) == "FAIL",
        },
        {
            "claim_id": "RQ1_MDC_gain",
            "paper_target": "multi-agent MDC gain >= 17% over single-agent",
            "formula": "(MDC_multi_without_neg - MDC_single) / MDC_single",
            "observed": {
                "single_agent": single_mdc,
                "multi_agent_without_negotiation": multi_without_mdc,
                "gain": _gain(multi_without_mdc, single_mdc),
            },
            "status": _status(_gain(multi_without_mdc, single_mdc), 0.17),
            "hard_fail": _status(_gain(multi_without_mdc, single_mdc), 0.17) == "FAIL",
        },
        {
            "claim_id": "RQ1_requirement_volume",
            "paper_target": "multi-agent requirement count multiplier >= 2.4x over single-agent",
            "formula": "n_phase1_multi_without_neg / n_phase1_single",
            "observed": {
                "single_agent": single_p1,
                "multi_agent_without_negotiation": multi_without_p1,
                "multiplier": (
                    round(multi_without_p1 / single_p1, 6)
                    if single_p1 is not None and single_p1 > 0 and multi_without_p1 is not None
                    else None
                ),
            },
            "status": _status(
                (
                    round(multi_without_p1 / single_p1, 6)
                    if single_p1 is not None and single_p1 > 0 and multi_without_p1 is not None
                    else None
                ),
                2.4,
            ),
            "hard_fail": _status(
                (
                    round(multi_without_p1 / single_p1, 6)
                    if single_p1 is not None and single_p1 > 0 and multi_without_p1 is not None
                    else None
                ),
                2.4,
            )
            == "FAIL",
        },
        {
            "claim_id": "RQ1_filtering_AD",
            "paper_target": "AD filtering rate >= 70.4% (81 -> 24)",
            "formula": "(AD_phase1_without_neg - AD_phase3_with_neg) / AD_phase1_without_neg",
            "observed": {
                "ad_phase1_without_negotiation": ad_phase1,
                "ad_phase3_with_negotiation": ad_phase3,
                "filtering_rate": ad_filtering,
            },
            "status": _status(ad_filtering, 0.704),
            "hard_fail": _status(ad_filtering, 0.704) == "FAIL",
        },
        {
            "claim_id": "RQ1_precision_gain",
            "paper_target": "precision gain up to 92.9%",
            "formula": "(precision_phase2 - precision_phase1) / precision_phase1",
            "observed": {
                "gain": precision_gain,
                "source_column": precision_column,
            },
            "status": (
                _status(precision_gain, 0.929)
                if precision_f1_literal_comparable
                else "BLOCKED"
            ),
            "hard_fail": (
                _status(precision_gain, 0.929) == "FAIL"
                if precision_f1_literal_comparable
                else True
            ),
            "blocker": blocker_payload if not precision_f1_literal_comparable else None,
        },
        {
            "claim_id": "RQ1_f1_gain",
            "paper_target": "F1 gain up to 43.5%",
            "formula": "(f1_phase2 - f1_phase1) / f1_phase1",
            "observed": {
                "gain": f1_gain,
                "source_column": f1_column,
            },
            "status": (
                _status(f1_gain, 0.435)
                if precision_f1_literal_comparable
                else "BLOCKED"
            ),
            "hard_fail": (
                _status(f1_gain, 0.435) == "FAIL"
                if precision_f1_literal_comparable
                else True
            ),
            "blocker": blocker_payload if not precision_f1_literal_comparable else None,
        },
        {
            "claim_id": "RQ2_conflict_resolution",
            "paper_target": "conflict resolution rate == 1.0",
            "formula": "mean(conflict_resolution_rate in multi_agent_with_negotiation)",
            "observed": {"value": conflict_rate},
            "status": _status(conflict_rate, 1.0),
            "hard_fail": _status(conflict_rate, 1.0) == "FAIL",
        },
        {
            "claim_id": "RQ2_semantic_preservation",
            "paper_target": "max semantic_preservation_f1 (NIV) >= 0.933",
            "formula": "max(semantic_preservation_f1 in negotiation_integration_verification)",
            "observed": {"value": round(semantic_max, 6) if semantic_max is not None else None},
            "status": _status(semantic_max, 0.933),
            "hard_fail": _status(semantic_max, 0.933) == "FAIL",
        },
        {
            "claim_id": "RQ3_topology_validity",
            "paper_target": "topology_is_valid == 1.0 in negotiation setting",
            "formula": "mean(topology_is_valid in multi_agent_with_negotiation)",
            "observed": {"value": topology_mean},
            "status": _status(topology_mean, 1.0),
            "hard_fail": _status(topology_mean, 1.0) == "FAIL",
        },
    ]

    hard_fail_count = sum(1 for claim in claims if bool(claim.get("hard_fail", False)))
    blocked_hard_fail_count = sum(1 for claim in claims if claim.get("status") == "BLOCKED")
    non_comparable_count = sum(1 for claim in claims if claim.get("status") == "NON-COMPARABLE")

    return {
        "generated_at": utc_timestamp(),
        "paper_source": "external MARE/QUARE benchmark claim profile",
        "artifact_source": str(quare_csv),
        "controls": {
            "model": controls.get("model"),
            "temperature": controls.get("temperature"),
            "seeds": controls.get("seeds"),
            "settings": controls.get("settings"),
        },
        "paper_control_check": control_check,
        "precision_f1_data_contract": precision_f1_contract,
        "precision_f1_literal_derivation": literal_precision_f1,
        "precision_f1_derivation": derived_precision_f1,
        "claims": claims,
        "claim_summary": {
            "total_claims": len(claims),
            "pass_count": sum(1 for claim in claims if claim.get("status") == "PASS"),
            "hard_fail_count": hard_fail_count,
            "blocked_hard_fail_count": blocked_hard_fail_count,
            "non_comparable_count": non_comparable_count,
        },
    }


def _compute_key_deltas(*, mare_csv: Path, quare_csv: Path) -> dict[str, Any]:
    """Compute key metric deltas (QUARE minus MARE) by setting and overall."""

    mare_rows = _read_csv_rows(mare_csv)
    quare_rows = _read_csv_rows(quare_csv)

    by_setting: dict[str, dict[str, dict[str, float | None]]] = {}
    settings = sorted(
        set(str(row.get("setting", "")).strip() for row in mare_rows)
        | set(str(row.get("setting", "")).strip() for row in quare_rows)
    )
    for setting in settings:
        mare_setting = [row for row in mare_rows if str(row.get("setting", "")).strip() == setting]
        quare_setting = [
            row for row in quare_rows if str(row.get("setting", "")).strip() == setting
        ]
        metric_map: dict[str, dict[str, float | None]] = {}
        for metric in KEY_DELTA_METRICS:
            mare_mean = _mean_metric(mare_setting, metric)
            quare_mean = _mean_metric(quare_setting, metric)
            metric_map[metric] = {
                "mare_mean": mare_mean,
                "quare_mean": quare_mean,
                "quare_minus_mare": _delta(quare_mean, mare_mean),
            }
        by_setting[setting] = metric_map

    overall: dict[str, dict[str, float | None]] = {}
    for metric in KEY_DELTA_METRICS:
        mare_mean = _mean_metric(mare_rows, metric)
        quare_mean = _mean_metric(quare_rows, metric)
        overall[metric] = {
            "mare_mean": mare_mean,
            "quare_mean": quare_mean,
            "quare_minus_mare": _delta(quare_mean, mare_mean),
        }

    return {
        "generated_at": utc_timestamp(),
        "metrics": list(KEY_DELTA_METRICS),
        "by_setting": by_setting,
        "overall": overall,
    }


def _write_report_readme(
    *,
    path: Path,
    run_dir: Path,
    verdict: dict[str, Any],
    validation_evidence: dict[str, Any],
    conversation_summary: dict[str, Any],
    paper_claims: dict[str, Any],
) -> None:
    """Write top-level human-readable report summary."""

    systems = validation_evidence.get("systems", {})
    mare = systems.get(SYSTEM_MARE, {})
    quare = systems.get(SYSTEM_QUARE, {})
    final_verdict = verdict.get("final_completion_verdict", "NO-GO")
    claim_summary = paper_claims.get("claim_summary", {})

    lines = [
        "# MARE/QUARE Strict Auto Report",
        "",
        f"- Generated at: {utc_timestamp()}",
        f"- Run directory: `{run_dir}`",
        f"- Final verdict: **{final_verdict}**",
        "",
        "## Strict Summary",
        f"- MARE validation passed: {mare.get('validation_passed_runs', 0)}/{mare.get('total_runs', 0)}",
        f"- QUARE validation passed: {quare.get('validation_passed_runs', 0)}/{quare.get('total_runs', 0)}",
        f"- MARE llm_turns_sum: {mare.get('llm_turns_sum', 0)}",
        f"- QUARE llm_turns_sum: {quare.get('llm_turns_sum', 0)}",
        "- Conversation log coverage: "
        f"{conversation_summary.get('complete_runs', 0)}/"
        f"{conversation_summary.get('expected_runs', 0)} "
        f"({_fmt(_to_float(conversation_summary.get('coverage_ratio')))})",
        "",
        "## Proof Artifacts",
        "- `proofs/final_validation_evidence.json`",
        "- `proofs/independent_validator_replay.json`",
        "- `proofs/quare_vs_mare_deltas.json`",
        f"- `proofs/{PAPER_CLAIMS_FILE}`",
        "- `proofs/conversation_log_evidence.json`",
        "- `proofs/finality_threshold_verdict.json`",
        "- `proofs/manifest.json`",
        "",
        "## Paper Claim Summary",
        f"- pass_count: {claim_summary.get('pass_count', 0)}",
        f"- hard_fail_count: {claim_summary.get('hard_fail_count', 0)}",
        f"- blocked_hard_fail_count: {claim_summary.get('blocked_hard_fail_count', 0)}",
        f"- non_comparable_count: {claim_summary.get('non_comparable_count', 0)}",
        "",
        "## Conversation Artifacts",
        f"- `../../logs/{run_dir.name}/{CONVERSATION_INDEX_JSONL}`",
        f"- `../../logs/{run_dir.name}/{CONVERSATION_COVERAGE_JSON}`",
        f"- `../../logs/{run_dir.name}/{CONVERSATION_COVERAGE_MD}`",
        f"- `../../logs/{run_dir.name}/conversations/...`",
        "",
        "See `analysis.md` for detailed threshold checks, metric deltas, and log coverage.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_analysis_md(
    *,
    path: Path,
    deltas: dict[str, Any],
    replay: dict[str, Any],
    verdict: dict[str, Any],
    conversation_summary: dict[str, Any],
    paper_claims: dict[str, Any],
    warnings: list[str],
) -> None:
    """Write detailed analysis markdown for reviewer sign-off."""

    lines = [
        "# MARE/QUARE Strict Analysis",
        "",
        f"- Generated at: {utc_timestamp()}",
        f"- Final verdict: **{verdict.get('final_completion_verdict', 'NO-GO')}**",
        "",
        "## Threshold Results",
    ]

    checks = verdict.get("check_results", {})
    for key in sorted(checks):
        lines.append(f"- {key}: `{checks[key]}`")

    lines.extend(
        [
            "",
            "## Key Deltas (QUARE - MARE)",
            "| setting | metric | mare_mean | quare_mean | quare_minus_mare |",
            "|---|---|---:|---:|---:|",
        ]
    )
    by_setting = deltas.get("by_setting", {})
    for setting in sorted(by_setting):
        metric_map = by_setting.get(setting, {})
        for metric in KEY_DELTA_METRICS:
            values = metric_map.get(metric, {})
            lines.append(
                "| "
                f"{setting} | {metric} | {_fmt(values.get('mare_mean'))} | "
                f"{_fmt(values.get('quare_mean'))} | {_fmt(values.get('quare_minus_mare'))} |"
            )

    lines.extend(["", "## Independent Replay", ""])
    replay_systems = replay.get("systems", {})
    for system in (SYSTEM_MARE, SYSTEM_QUARE):
        payload = replay_systems.get(system, {})
        lines.append(
            f"- {system}: runs={payload.get('total_runs', 0)}, "
            f"errors={payload.get('error_items', 0)}, warnings={payload.get('warning_items', 0)}"
        )

    lines.extend(["", "## Paper Claim Mapping", ""])
    claim_summary = paper_claims.get("claim_summary", {})
    lines.append(
        f"- pass={claim_summary.get('pass_count', 0)}, "
        f"fail={claim_summary.get('hard_fail_count', 0)}, "
        f"blocked={claim_summary.get('blocked_hard_fail_count', 0)}, "
        f"non_comparable={claim_summary.get('non_comparable_count', 0)}"
    )
    lines.extend(
        [
            "",
            "| claim_id | status | paper_target | observed |",
            "|---|---|---|---|",
        ]
    )
    for claim in paper_claims.get("claims", []):
        if not isinstance(claim, dict):
            continue
        observed = claim.get("observed")
        if isinstance(observed, dict):
            observed_text = json.dumps(observed, sort_keys=True, ensure_ascii=True)
        else:
            observed_text = str(observed)
        lines.append(
            "| "
            f"{claim.get('claim_id', '')} | {claim.get('status', '')} | "
            f"{claim.get('paper_target', '')} | {observed_text} |"
        )

    lines.extend(
        [
            "",
            "## Conversation Log Coverage",
            "",
            f"- expected_runs: {conversation_summary.get('expected_runs', 0)}",
            f"- complete_runs: {conversation_summary.get('complete_runs', 0)}",
            f"- coverage_ratio: {_fmt(_to_float(conversation_summary.get('coverage_ratio')))}",
            f"- complete: `{conversation_summary.get('is_complete', False)}`",
            f"- missing_runs: {len(conversation_summary.get('missing_run_ids', []))}",
            f"- missing_agent_log_runs: {conversation_summary.get('runs_with_missing_agent_logs', 0)}",
        ]
    )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for item in warnings:
            lines.append(f"- {item}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_manifest(*, key_paths: list[Path], root_dir: Path) -> dict[str, Any]:
    """Build hash manifest for key evidence files."""

    files: dict[str, str] = {}
    for path in key_paths:
        if not path.exists():
            continue
        relative = str(path.resolve().relative_to(root_dir.resolve()))
        files[relative] = _file_sha256(path)
    return {
        "generated_at": utc_timestamp(),
        "root_dir": str(root_dir),
        "files": files,
    }


def _mirror_latest_outputs(
    *,
    report_dir: Path,
    run_key: str,
    run_dir: Path,
    report_readme: Path,
    report_analysis: Path,
    proofs_dir: Path,
) -> None:
    """Copy latest report/proof files to stable top-level paths under report/."""

    report_dir.mkdir(parents=True, exist_ok=True)
    if report_readme.exists():
        latest_readme = _rewrite_readme_for_report_root(
            readme=report_readme.read_text(encoding="utf-8"),
            run_key=run_key,
        )
        (report_dir / "README.md").write_text(latest_readme, encoding="utf-8")
    if report_analysis.exists():
        (report_dir / "analysis.md").write_text(
            report_analysis.read_text(encoding="utf-8"), encoding="utf-8"
        )

    latest_proofs = report_dir / "proofs"
    latest_proofs.mkdir(parents=True, exist_ok=True)
    for artifact in proofs_dir.glob("*.json"):
        (latest_proofs / artifact.name).write_text(artifact.read_text(encoding="utf-8"), encoding="utf-8")

    write_json_file(
        report_dir / "latest_run_pointer.json",
        {
            "run_key": run_key,
            "run_dir": str(run_dir),
            "readme": str(report_readme),
            "analysis": str(report_analysis),
            "proofs_dir": str(proofs_dir),
            "updated_at": utc_timestamp(),
        },
    )


def _rewrite_readme_for_report_root(*, readme: str, run_key: str) -> str:
    """Adjust run README log links so they resolve from report/README.md."""

    run_relative_prefix = f"../../logs/{run_key}/"
    legacy_prefix = f"../logs/{run_key}/"
    latest_prefix = f"logs/{run_key}/"
    updated = readme.replace(run_relative_prefix, latest_prefix)
    return updated.replace(legacy_prefix, latest_prefix)


def _log_matrix_snapshot(*, output_dir: Path, system: str, log_path: Path) -> None:
    """Write per-run matrix snapshot log for reproducibility."""

    rows = _read_jsonl_rows(output_dir / RUNS_JSONL_NAME)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_timestamp()}] system={system} total_rows={len(rows)}\n")
        for row in rows:
            handle.write(
                f"[{utc_timestamp()}] run_id={row.get('run_id')} "
                f"setting={row.get('setting')} validation_passed={row.get('validation_passed')}\n"
            )


def _generate_conversation_logs(
    *,
    run_key: str,
    logs_dir: Path,
    mare_rows: list[dict[str, Any]],
    quare_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build deterministic human-readable conversation logs for every matrix run."""

    conversations_root = logs_dir / "conversations"
    conversations_root.mkdir(parents=True, exist_ok=True)

    indexed_rows: list[dict[str, Any]] = []
    for row in mare_rows:
        indexed_rows.append({**row, "system": SYSTEM_MARE})
    for row in quare_rows:
        indexed_rows.append({**row, "system": SYSTEM_QUARE})

    indexed_rows.sort(
        key=lambda row: (
            str(row.get("system", "")).strip(),
            str(row.get("case_id", "")).strip(),
            str(row.get("setting", "")).strip(),
            _to_int(row.get("seed"), default=0),
            str(row.get("run_id", "")).strip(),
        )
    )

    index_rows: list[dict[str, Any]] = []
    complete_runs = 0
    runs_with_missing_agent_logs = 0
    regenerated_runs = 0
    reused_runs = 0
    missing_run_ids: list[str] = []
    by_system: dict[str, dict[str, int]] = {
        SYSTEM_MARE: {"runs": 0, "complete_runs": 0, "regenerated_runs": 0},
        SYSTEM_QUARE: {"runs": 0, "complete_runs": 0, "regenerated_runs": 0},
    }

    for row in indexed_rows:
        bundle = _render_conversation_bundle(
            run_key=run_key,
            logs_dir=logs_dir,
            conversations_root=conversations_root,
            row=row,
        )
        index_rows.append(bundle)

        system = str(bundle.get("system", "")).strip()
        system_counts = by_system.setdefault(system, {"runs": 0, "complete_runs": 0, "regenerated_runs": 0})
        system_counts["runs"] += 1

        if bool(bundle.get("regenerated", False)):
            regenerated_runs += 1
            system_counts["regenerated_runs"] += 1
        if bool(bundle.get("reused", False)):
            reused_runs += 1

        if bool(bundle.get("log_complete", False)):
            complete_runs += 1
            system_counts["complete_runs"] += 1
        else:
            missing_run_ids.append(str(bundle.get("run_id", "")))

        if bool(bundle.get("missing_expected_agents", False)) or bundle.get("missing_agent_logs"):
            runs_with_missing_agent_logs += 1

    index_path = logs_dir / CONVERSATION_INDEX_JSONL
    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    expected_runs = len(indexed_rows)
    coverage_ratio = 1.0 if expected_runs == 0 else round(complete_runs / expected_runs, 6)
    is_complete = expected_runs > 0 and complete_runs == expected_runs and runs_with_missing_agent_logs == 0

    coverage_payload = {
        "generated_at": utc_timestamp(),
        "run_key": run_key,
        "expected_runs": expected_runs,
        "complete_runs": complete_runs,
        "coverage_ratio": coverage_ratio,
        "is_complete": is_complete,
        "runs_with_missing_agent_logs": runs_with_missing_agent_logs,
        "missing_run_ids": missing_run_ids,
        "regenerated_runs": regenerated_runs,
        "reused_runs": reused_runs,
        "by_system": by_system,
        "index_path": str(index_path),
        "conversations_root": str(conversations_root),
    }

    coverage_json_path = logs_dir / CONVERSATION_COVERAGE_JSON
    write_json_file(coverage_json_path, coverage_payload)

    coverage_md_path = logs_dir / CONVERSATION_COVERAGE_MD
    coverage_md_lines = [
        "# Conversation Coverage",
        "",
        f"- Run key: `{run_key}`",
        f"- Expected runs: {expected_runs}",
        f"- Complete runs: {complete_runs}",
        f"- Coverage ratio: {_fmt(coverage_ratio)}",
        f"- Complete: `{is_complete}`",
        f"- Regenerated runs: {regenerated_runs}",
        f"- Reused runs: {reused_runs}",
        "",
        "## By System",
    ]
    for system in (SYSTEM_MARE, SYSTEM_QUARE):
        payload = by_system.get(system, {"runs": 0, "complete_runs": 0, "regenerated_runs": 0})
        coverage_md_lines.append(
            f"- {system}: runs={payload['runs']}, complete={payload['complete_runs']}, "
            f"regenerated={payload['regenerated_runs']}"
        )
    if missing_run_ids:
        coverage_md_lines.extend(["", "## Missing Runs"])
        for run_id in missing_run_ids:
            coverage_md_lines.append(f"- `{run_id}`")
    coverage_md_path.write_text("\n".join(coverage_md_lines) + "\n", encoding="utf-8")

    return coverage_payload


def _render_conversation_bundle(
    *,
    run_key: str,
    logs_dir: Path,
    conversations_root: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Render one run's timeline and per-agent markdown logs."""

    run_id = str(row.get("run_id", "")).strip()
    system = str(row.get("system", "")).strip() or "unknown"
    case_id = str(row.get("case_id", "")).strip() or "unknown"
    setting = str(row.get("setting", "")).strip() or "unknown"
    seed = _to_int(row.get("seed"), default=0)
    artifacts_dir = Path(str(row.get("artifacts_dir", "")).strip())

    bundle_dir = (
        conversations_root
        / _sanitize_component(system)
        / _sanitize_component(case_id)
        / _sanitize_component(setting)
        / f"seed-{seed}"
        / _sanitize_component(run_id)
    )
    timeline_path = bundle_dir / "timeline.md"
    agents_dir = bundle_dir / "agents"
    meta_path = bundle_dir / "meta.json"

    phase2_path = artifacts_dir / "phase2_negotiation_trace.json"
    if not phase2_path.exists() or not phase2_path.is_file():
        return {
            "run_key": run_key,
            "run_id": run_id,
            "system": system,
            "case_id": case_id,
            "setting": setting,
            "seed": seed,
            "status": "missing_phase2_artifact",
            "log_complete": False,
            "missing_agent_logs": [],
            "reused": False,
            "regenerated": False,
        }

    try:
        phase2_payload = json.loads(phase2_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "run_key": run_key,
            "run_id": run_id,
            "system": system,
            "case_id": case_id,
            "setting": setting,
            "seed": seed,
            "status": "invalid_phase2_artifact",
            "log_complete": False,
            "missing_agent_logs": [],
            "reused": False,
            "regenerated": False,
        }

    if not isinstance(phase2_payload, dict):
        return {
            "run_key": run_key,
            "run_id": run_id,
            "system": system,
            "case_id": case_id,
            "setting": setting,
            "seed": seed,
            "status": "phase2_not_object",
            "log_complete": False,
            "missing_agent_logs": [],
            "reused": False,
            "regenerated": False,
        }

    steps = _phase2_steps_for_logging(phase2_payload)
    expected_agents = _phase2_agents_for_logging(phase2_payload=phase2_payload, steps=steps)
    expected_agents_source = "phase2"
    if not expected_agents:
        expected_agents = _phase1_agents_for_logging(artifacts_dir=artifacts_dir)
        expected_agents_source = "phase1" if expected_agents else "none"

    agent_file_map = _agent_log_paths(agents_dir=agents_dir, expected_agents=expected_agents)
    agent_paths = list(agent_file_map.values())
    missing_expected_agents = len(expected_agents) == 0

    source_hash_payload = {
        "run_id": run_id,
        "system": system,
        "case_id": case_id,
        "setting": setting,
        "seed": seed,
        "phase2_sha256": _file_sha256(phase2_path),
        "expected_agents": expected_agents,
        "expected_agents_source": expected_agents_source,
        "step_count": len(steps),
    }
    source_hash = hashlib.sha256(
        json.dumps(source_hash_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    reused = _conversation_bundle_is_fresh(
        meta_path=meta_path,
        source_hash=source_hash,
        timeline_path=timeline_path,
        agent_paths=agent_paths,
    )
    regenerated = False
    if not reused:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        agents_dir.mkdir(parents=True, exist_ok=True)

        timeline_path.write_text(
            _render_timeline_markdown(
                run_id=run_id,
                system=system,
                case_id=case_id,
                setting=setting,
                seed=seed,
                source_hash=source_hash,
                steps=steps,
            ),
            encoding="utf-8",
        )

        for agent in expected_agents:
            agent_steps = [
                step
                for step in steps
                if step.get("focus_agent") == agent or step.get("reviewer_agent") == agent
            ]
            agent_file_map[agent].write_text(
                _render_agent_markdown(
                    run_id=run_id,
                    system=system,
                    case_id=case_id,
                    setting=setting,
                    seed=seed,
                    agent=agent,
                    source_hash=source_hash,
                    steps=agent_steps,
                ),
                encoding="utf-8",
            )

        meta_payload = {
            "generated_at": utc_timestamp(),
            "run_key": run_key,
            "run_id": run_id,
            "system": system,
            "case_id": case_id,
            "setting": setting,
            "seed": seed,
            "source_hash": source_hash,
            "phase2_path": str(phase2_path),
            "timeline_path": str(timeline_path),
            "agent_logs": {agent: str(path) for agent, path in agent_file_map.items()},
            "expected_agents": expected_agents,
            "expected_agents_source": expected_agents_source,
            "step_count": len(steps),
        }
        write_json_file(meta_path, meta_payload)
        regenerated = True

    missing_agent_logs = [
        agent for agent, path in agent_file_map.items() if not path.exists() or not path.is_file()
    ]
    missing_files: list[str] = []
    if not timeline_path.exists() or not timeline_path.is_file():
        missing_files.append("timeline.md")
    if not meta_path.exists() or not meta_path.is_file():
        missing_files.append("meta.json")
    if missing_expected_agents:
        missing_files.append("agents/*.md")

    relative_bundle = bundle_dir.resolve().relative_to(logs_dir.resolve())
    return {
        "run_key": run_key,
        "run_id": run_id,
        "system": system,
        "case_id": case_id,
        "setting": setting,
        "seed": seed,
        "status": "ok",
        "log_complete": (not missing_expected_agents)
        and len(missing_agent_logs) == 0
        and len(missing_files) == 0,
        "missing_agent_logs": missing_agent_logs,
        "missing_expected_agents": missing_expected_agents,
        "missing_files": missing_files,
        "expected_agents": expected_agents,
        "expected_agents_source": expected_agents_source,
        "step_count": len(steps),
        "source_hash": source_hash,
        "bundle_path": str(relative_bundle),
        "timeline_path": str(timeline_path.resolve().relative_to(logs_dir.resolve())),
        "meta_path": str(meta_path.resolve().relative_to(logs_dir.resolve())),
        "reused": reused,
        "regenerated": regenerated,
    }


def _conversation_bundle_is_fresh(
    *,
    meta_path: Path,
    source_hash: str,
    timeline_path: Path,
    agent_paths: list[Path],
) -> bool:
    """Return true when existing conversation files match current source hash."""

    if not meta_path.exists() or not timeline_path.exists():
        return False
    for path in agent_paths:
        if not path.exists():
            return False
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and str(payload.get("source_hash", "")).strip() == source_hash


def _phase2_steps_for_logging(phase2_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize phase-2 steps into a deterministic list for markdown rendering."""

    negotiations = phase2_payload.get("negotiations", {})
    if not isinstance(negotiations, dict):
        return []

    flattened: list[dict[str, Any]] = []
    for pair_key in sorted(negotiations):
        negotiation = negotiations.get(pair_key, {})
        if not isinstance(negotiation, dict):
            continue
        default_focus = str(negotiation.get("focus_agent", "")).strip()
        reviewer_agents = negotiation.get("reviewer_agents", [])
        default_reviewer = ""
        if isinstance(reviewer_agents, list) and reviewer_agents:
            default_reviewer = str(reviewer_agents[0]).strip()

        steps = negotiation.get("steps", [])
        if not isinstance(steps, list):
            continue
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            normalized = {
                "negotiation_pair": str(pair_key),
                "step_id": _to_int(step.get("step_id"), default=index),
                "timestamp": str(step.get("timestamp", "")).strip(),
                "round_number": _to_int(step.get("round_number"), default=1),
                "message_type": str(step.get("message_type", "")).strip() or "unknown",
                "focus_agent": str(step.get("focus_agent", "")).strip() or default_focus,
                "reviewer_agent": str(step.get("reviewer_agent", "")).strip() or default_reviewer,
                "analysis_text": _coerce_text(step.get("analysis_text") or step.get("analysis")),
                "feedback": _coerce_text(step.get("feedback")),
                "conflict_detected": step.get("conflict_detected"),
                "resolution_state": step.get("resolution_state"),
                "requires_refinement": step.get("requires_refinement"),
                "negotiation_mode": step.get("negotiation_mode"),
            }
            flattened.append(normalized)

    flattened.sort(
        key=lambda step: (
            _to_int(step.get("step_id"), default=0),
            _to_int(step.get("round_number"), default=1),
            str(step.get("timestamp", "")),
            str(step.get("negotiation_pair", "")),
        )
    )
    return flattened


def _phase2_agents_for_logging(
    *, phase2_payload: dict[str, Any], steps: list[dict[str, Any]]
) -> list[str]:
    """Extract deterministic per-run agent identities from phase-2 payload."""

    agents: set[str] = set()
    for step in steps:
        focus = str(step.get("focus_agent", "")).strip()
        reviewer = str(step.get("reviewer_agent", "")).strip()
        if focus:
            agents.add(focus)
        if reviewer:
            agents.add(reviewer)

    negotiations = phase2_payload.get("negotiations", {})
    if isinstance(negotiations, dict):
        for negotiation in negotiations.values():
            if not isinstance(negotiation, dict):
                continue
            focus = str(negotiation.get("focus_agent", "")).strip()
            if focus:
                agents.add(focus)
            reviewers = negotiation.get("reviewer_agents", [])
            if isinstance(reviewers, list):
                for reviewer in reviewers:
                    reviewer_text = str(reviewer).strip()
                    if reviewer_text:
                        agents.add(reviewer_text)
    return sorted(agents)


def _phase1_agents_for_logging(*, artifacts_dir: Path) -> list[str]:
    """Fallback agent discovery using phase-1 models when phase-2 has no negotiations."""

    phase1_path = artifacts_dir / PHASE1_FILENAME
    if not phase1_path.exists() or not phase1_path.is_file():
        return []
    try:
        phase1_payload = json.loads(phase1_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(phase1_payload, dict):
        return []

    agents = {str(agent).strip() for agent in phase1_payload.keys() if str(agent).strip()}
    return sorted(agents)


def _agent_log_paths(*, agents_dir: Path, expected_agents: list[str]) -> dict[str, Path]:
    """Build deterministic markdown file paths for every expected agent."""

    mapping: dict[str, Path] = {}
    used_names: set[str] = set()
    for agent in expected_agents:
        base = _sanitize_component(agent).lower()
        if not base:
            base = "agent"
        file_name = f"{base}.md"
        suffix = 2
        while file_name in used_names:
            file_name = f"{base}-{suffix}.md"
            suffix += 1
        used_names.add(file_name)
        mapping[agent] = agents_dir / file_name
    return mapping


def _render_timeline_markdown(
    *,
    run_id: str,
    system: str,
    case_id: str,
    setting: str,
    seed: int,
    source_hash: str,
    steps: list[dict[str, Any]],
) -> str:
    """Render one run-level timeline markdown transcript."""

    lines = [
        f"# Conversation Timeline: {run_id}",
        "",
        f"- System: `{system}`",
        f"- Case: `{case_id}`",
        f"- Setting: `{setting}`",
        f"- Seed: `{seed}`",
        f"- Source hash: `{source_hash}`",
        f"- Total steps: `{len(steps)}`",
        "",
    ]

    if not steps:
        lines.append("No phase-2 steps were recorded for this run.")
        return "\n".join(lines) + "\n"

    for step in steps:
        lines.extend(
            [
                "---",
                "",
                f"## Step {step.get('step_id')} ({step.get('message_type')})",
                f"- Negotiation pair: `{step.get('negotiation_pair')}`",
                f"- Round: `{step.get('round_number')}`",
                f"- Focus agent: `{step.get('focus_agent')}`",
                f"- Reviewer agent: `{step.get('reviewer_agent')}`",
                f"- Timestamp: `{step.get('timestamp')}`",
                f"- Negotiation mode: `{_coerce_text(step.get('negotiation_mode'), fallback='N/A')}`",
                f"- Conflict detected: `{_state_text(step.get('conflict_detected'))}`",
                f"- Requires refinement: `{_state_text(step.get('requires_refinement'))}`",
                f"- Resolution state: `{_coerce_text(step.get('resolution_state'), fallback='N/A')}`",
                "",
                "### Analysis",
                _coerce_text(step.get("analysis_text"), fallback="N/A"),
                "",
                "### Feedback",
                _coerce_text(step.get("feedback"), fallback="N/A"),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def _render_agent_markdown(
    *,
    run_id: str,
    system: str,
    case_id: str,
    setting: str,
    seed: int,
    agent: str,
    source_hash: str,
    steps: list[dict[str, Any]],
) -> str:
    """Render per-agent conversation markdown view."""

    lines = [
        f"# Agent Conversation: {agent}",
        "",
        f"- Run ID: `{run_id}`",
        f"- System: `{system}`",
        f"- Case: `{case_id}`",
        f"- Setting: `{setting}`",
        f"- Seed: `{seed}`",
        f"- Source hash: `{source_hash}`",
        f"- Total relevant steps: `{len(steps)}`",
        "",
    ]

    if not steps:
        lines.append("No phase-2 steps involve this agent.")
        return "\n".join(lines) + "\n"

    for step in steps:
        role = "focus"
        if str(step.get("reviewer_agent", "")).strip() == agent:
            role = "reviewer"
        lines.extend(
            [
                "---",
                "",
                f"## Step {step.get('step_id')} ({step.get('message_type')}, {role})",
                f"- Negotiation pair: `{step.get('negotiation_pair')}`",
                f"- Round: `{step.get('round_number')}`",
                f"- Peer: `{step.get('reviewer_agent') if role == 'focus' else step.get('focus_agent')}`",
                f"- Timestamp: `{step.get('timestamp')}`",
                "",
                "### Analysis",
                _coerce_text(step.get("analysis_text"), fallback="N/A"),
                "",
                "### Feedback",
                _coerce_text(step.get("feedback"), fallback="N/A"),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def _sanitize_component(value: str) -> str:
    """Sanitize arbitrary path component text into deterministic safe ASCII."""

    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip("-.")
    return text or "unknown"


def _coerce_text(value: Any, *, fallback: str = "") -> str:
    """Normalize optional scalar text for markdown rendering."""

    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _state_text(value: Any) -> str:
    """Render tri-state booleans in a consistent way."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return "N/A"


def _matrix_outputs_complete(*, output_dir: Path, expected_runs: int, system: str) -> tuple[bool, str]:
    """Return complete/incomplete with reason for one system matrix output directory."""

    if not output_dir.exists() or not output_dir.is_dir():
        return False, "missing output directory"

    for name in REQUIRED_MATRIX_FILES:
        if not (output_dir / name).exists():
            return False, f"missing {name}"

    rows = _read_jsonl_rows(output_dir / RUNS_JSONL_NAME)
    if len(rows) != expected_runs:
        return False, f"row count mismatch ({len(rows)} vs {expected_runs})"

    seen_ids: set[str] = set()
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            return False, "empty run_id detected"
        if run_id in seen_ids:
            return False, f"duplicate run_id detected ({run_id})"
        seen_ids.add(run_id)

        if str(row.get("system", "")).strip() != system:
            return False, "system mismatch in run rows"
        if not bool(row.get("validation_passed", False)):
            return False, f"validation failed for run {run_id}"

    return True, "all strict checks passed"


def _build_run_key(controls: dict[str, Any]) -> str:
    """Build deterministic run key from normalized controls payload."""

    payload = json.dumps(controls, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"auto-{digest}"


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from path."""

    rows: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows from path."""

    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    """Compute mean value for one metric across CSV row payloads."""

    values: list[float] = []
    for row in rows:
        parsed = _to_float(row.get(key))
        if parsed is None:
            continue
        values.append(parsed)
    if not values:
        return None
    return round(mean(values), 6)


def _delta(left: float | None, right: float | None) -> float | None:
    """Return rounded left-right delta if both values exist."""

    if left is None or right is None:
        return None
    return round(left - right, 6)


def _to_float(value: Any) -> float | None:
    """Coerce scalar metric to float where possible."""

    if value in (None, "", "N/A"):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _to_int(value: Any, *, default: int) -> int:
    """Coerce scalar to int with default fallback."""

    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _runtime_semantics_string_list(value: Any) -> list[str] | None:
    """Return runtime-semantics list as strings, or None for malformed payloads."""

    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _fmt(value: float | None) -> str:
    """Render optional float values for markdown tables."""

    if value is None:
        return "N/A"
    return f"{value:.6f}"


def _file_sha256(path: Path) -> str:
    """Compute SHA256 for one file path."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(65536)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

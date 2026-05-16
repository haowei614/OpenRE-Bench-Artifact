"""Run-matrix harness and protocol deliverable generation for comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import pstdev
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull  # type: ignore[import-untyped]
from scipy.spatial import QhullError  # type: ignore[import-untyped]
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-not-found]

from openre_bench.comparison_validator import validate_case_input
from openre_bench.comparison_validator import validate_phase_artifacts
from openre_bench.comparison_validator import validate_run_record
from openre_bench.comparison_validator import validate_system_behavior_contract
from openre_bench.pipeline import MARE_ROLE_QUALITY_ATTRIBUTES
from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline import run_case_pipeline
from openre_bench.schemas import DEFAULT_AGENT_QUALITY_ATTRIBUTES
from openre_bench.schemas import CaseInput
from openre_bench.schemas import DEFAULT_MATRIX_SETTINGS
from openre_bench.schemas import PHASE1_FILENAME
from openre_bench.schemas import PHASE2_FILENAME
from openre_bench.schemas import PHASE3_FILENAME
from openre_bench.schemas import PHASE4_FILENAME
from openre_bench.schemas import SETTING_MULTI_AGENT_WITH_NEGOTIATION
from openre_bench.schemas import SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SETTING_SINGLE_AGENT
from openre_bench.schemas import non_comparable_reasons_for_setting
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import SUPPORTED_SYSTEMS
from openre_bench.schemas import load_json_file
from openre_bench.schemas import utc_timestamp
from openre_bench.schemas import write_json_file

RUNS_JSONL_NAME = "comparison_runs.jsonl"
BY_CASE_CSV_NAME = "comparison_metrics_by_case.csv"
SUMMARY_CSV_NAME = "comparison_metrics_summary.csv"
ABLATION_CSV_NAME = "comparison_ablation_table.csv"
VALIDITY_MD_NAME = "comparison_validity_log.md"
TRACE_AUDIT_MD_NAME = "comparison_trace_audit.md"
BLINDED_RUNS_JSONL_NAME = "blinded_comparison_runs.jsonl"
BLINDED_BY_CASE_CSV_NAME = "blinded_comparison_metrics_by_case.csv"
BLIND_MAPPING_JSON_NAME = "blind_mapping_private.json"
BLIND_PROTOCOL_MD_NAME = "blind_eval_protocol.md"
BLINDING_SCHEME_VERSION = "blind-v1"

QUALITY_AXES: tuple[str, ...] = (
    "Safety",
    "Efficiency",
    "Sustainability",
    "Trustworthiness",
    "Responsibility",
)

QUALITY_AXIS_BY_TOKEN: dict[str, str] = {
    "safety": "Safety",
    "efficiency": "Efficiency",
    "sustainability": "Sustainability",
    "trustworthiness": "Trustworthiness",
    "responsibility": "Responsibility",
}

INTEGRATED_QUALITY_TOKENS = {
    "integrated",
    "overall",
    "combined",
    "hybrid",
    "general",
}

QUALITY_AXIS_RUBRICS: dict[str, str] = {
    "Safety": (
        "Requirements that prioritize hazard prevention, risk control, fault tolerance, and"
        " safe system behavior."
    ),
    "Efficiency": (
        "Requirements that improve throughput, latency, resource utilization, and"
        " performance efficiency."
    ),
    "Sustainability": (
        "Requirements that reduce energy consumption, environmental impact, waste, and"
        " lifecycle footprint."
    ),
    "Trustworthiness": (
        "Requirements that strengthen security, privacy, auditability, verification evidence,"
        " and dependable operation."
    ),
    "Responsibility": (
        "Requirements that ensure compliance, governance, accountability, transparency, and"
        " ethical obligations."
    ),
}

MAX_AXIS_TEXT_TOKENS = 96
METADATA_AXIS_WEIGHT = 0.7
SEMANTIC_AXIS_WEIGHT = 0.3

for role_name, axis in DEFAULT_AGENT_QUALITY_ATTRIBUTES.items():
    token = re.sub(r"[^a-z0-9]+", "", role_name.lower())
    if token:
        QUALITY_AXIS_BY_TOKEN[token] = axis

for role_name, axis in MARE_ROLE_QUALITY_ATTRIBUTES.items():
    token = re.sub(r"[^a-z0-9]+", "", role_name.lower())
    if token:
        QUALITY_AXIS_BY_TOKEN[token] = axis

_BERTSCORE_CACHE: dict[str, float] = {}
_BERTSCORER: Any | None = None

REQUIRED_RUN_KEYS = {
    "run_id",
    "case_id",
    "seed",
    "system",
    "setting",
    "model",
    "temperature",
    "max_tokens",
    "round_cap",
    "artifact_paths",
    "system_identity",
    "provenance",
    "execution_flags",
    "comparability",
    "validation_passed",
}

BY_CASE_COLUMNS = [
    "run_id",
    "case_id",
    "seed",
    "system",
    "setting",
    "model",
    "temperature",
    "max_tokens",
    "round_cap",
    "rag_enabled",
    "rag_fallback_used",
    "fallback_tainted",
    "retry_used",
    "retry_count",
    "runtime_seconds",
    "n_phase1_agents",
    "n_phase1_elements",
    "n_phase2_negotiations",
    "n_phase2_steps",
    "n_phase3_elements",
    "n_phase3_connections",
    "conflict_resolution_rate",
    "chv",
    "mdc",
    "semantic_preservation_f1",
    "semantic_p2_vs_p1_f1",
    "s_logic",
    "topology_is_valid",
    "deterministic_is_valid",
    "compliance_coverage",
    "s_term",
    "iso29148_unambiguous",
    "iso29148_correctness",
    "iso29148_verifiability",
    "iso29148_set_consistency",
    "iso29148_set_feasibility",
    "blind_eval_run_id",
    "judge_pipeline_hash",
    "validation_passed",
    "non_comparable_reason",
]

SUMMARY_COLUMNS = [
    "system",
    "case_id",
    "setting",
    "runs",
    "valid_runs",
    "invalid_runs",
    "mean_runtime_seconds",
    "std_runtime_seconds",
    "mean_phase1_elements",
    "std_phase1_elements",
    "mean_phase2_steps",
    "std_phase2_steps",
    "mean_phase3_elements",
    "std_phase3_elements",
    "mean_conflict_resolution_rate",
    "std_conflict_resolution_rate",
    "mean_topology_valid",
    "std_topology_valid",
]

ABLATION_COLUMNS = [
    "system",
    "case_id",
    "seed",
    "single_agent_phase1_elements",
    "multi_agent_without_negotiation_phase1_elements",
    "multi_agent_with_negotiation_phase2_steps",
    "negotiation_integration_verification_phase3_elements",
    "delta_multi_without_neg_vs_single_phase1_elements",
    "delta_multi_with_neg_vs_without_neg_phase2_steps",
    "delta_full_vs_with_neg_phase3_elements",
    "full_topology_valid",
    "notes",
]


@dataclass
class MatrixConfig:
    """Configuration for a comparison run matrix."""

    cases_dir: Path
    output_dir: Path
    seeds: list[int]
    settings: list[str]
    model: str
    temperature: float
    round_cap: int
    max_tokens: int
    rag_enabled: bool
    rag_backend: str
    rag_corpus_dir: Path
    system: str = SYSTEM_MARE
    systems: list[str] | None = None
    max_workers: int = 1
    judge_pipeline_path: Path | None = None


@dataclass
class MatrixResult:
    """Paths and validation status for generated deliverables."""

    output_dir: Path
    runs_jsonl: Path
    by_case_csv: Path
    summary_csv: Path
    ablation_csv: Path
    validity_md: Path
    total_runs: int
    expected_runs: int
    errors: list[str]
    warnings: list[str]


@dataclass
class TraceAuditResult:
    """Trace-audit output path and summary counters."""

    output_path: Path
    total_runs: int
    runs_with_loops: int
    runs_with_conflicts: int


@dataclass
class BlindPrepResult:
    """Blind-preparation artifact paths."""

    output_dir: Path
    blinded_runs_jsonl: Path
    blinded_by_case_csv: Path
    mapping_json: Path
    protocol_md: Path
    judge_pipeline_hash: str


@dataclass(frozen=True)
class _MatrixJob:
    """One independent matrix cell scheduled by the harness."""

    index: int
    case_path: Path
    case_input: CaseInput
    seed: int
    setting: str
    system: str
    run_id: str
    artifacts_dir: Path
    run_record_path: Path
    pipeline_config: PipelineConfig
    judge_pipeline_hash: str


@dataclass(frozen=True)
class _MatrixRunOutcome:
    """Completed run plus validation results for one matrix cell."""

    job: _MatrixJob
    run_record: Any
    local_errors: list[str]
    local_warnings: list[str]


def run_comparison_matrix(config: MatrixConfig) -> MatrixResult:
    """Run configured matrix and emit protocol deliverables."""

    case_paths = sorted(config.cases_dir.glob("*_input.json"))
    if not case_paths:
        raise FileNotFoundError(f"No case input files found in {config.cases_dir}")

    output_dir = config.output_dir
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    judge_pipeline_hash = _file_sha256(config.judge_pipeline_path)
    systems = _matrix_systems(config)

    jobs = _build_matrix_jobs(
        config=config,
        case_paths=case_paths,
        runs_dir=runs_dir,
        systems=systems,
        judge_pipeline_hash=judge_pipeline_hash,
    )
    outcomes = _run_matrix_jobs(jobs=jobs, max_workers=config.max_workers)

    for outcome in outcomes:
        job = outcome.job
        run_record = outcome.run_record
        errors.extend(f"{job.run_id}: {item}" for item in outcome.local_errors)
        warnings.extend(f"{job.run_id}: {item}" for item in outcome.local_warnings)

        metrics = _compute_run_metrics(job.artifacts_dir)
        validation_passed = not outcome.local_errors
        non_comparable_reason = _non_comparable_reason_text(
            run_record.comparability.non_comparable_reasons
        )

        run_rows.append(
            {
                **run_record.model_dump(mode="json"),
                "judge_pipeline_hash": judge_pipeline_hash,
                "blind_eval_run_id": "",
                "artifact_blinded": False,
                "blinding_scheme_version": "",
                "trace_audit_path": "",
                "validation_passed": validation_passed,
                "validation_errors": outcome.local_errors,
                "validation_warnings": outcome.local_warnings,
                "metrics": metrics,
                "non_comparable_reason": non_comparable_reason,
            }
        )
        metric_rows.append(
            _build_metric_row(
                run_record=run_record,
                metrics=metrics,
                judge_pipeline_hash=judge_pipeline_hash,
                validation_passed=validation_passed,
                non_comparable_reason=non_comparable_reason,
            )
        )

    runs_jsonl = output_dir / RUNS_JSONL_NAME
    by_case_csv = output_dir / BY_CASE_CSV_NAME
    summary_csv = output_dir / SUMMARY_CSV_NAME
    ablation_csv = output_dir / ABLATION_CSV_NAME
    validity_md = output_dir / VALIDITY_MD_NAME

    _write_jsonl(runs_jsonl, run_rows)
    _write_csv(by_case_csv, BY_CASE_COLUMNS, metric_rows)
    summary_rows = _build_summary_rows(metric_rows)
    _write_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)
    ablation_rows = _build_ablation_rows(metric_rows)
    _write_csv(ablation_csv, ABLATION_COLUMNS, ablation_rows)

    expected_runs = len(case_paths) * len(config.seeds) * len(config.settings) * len(systems)
    validity_errors, validity_warnings = _validate_deliverables(
        runs_jsonl=runs_jsonl,
        by_case_csv=by_case_csv,
        summary_csv=summary_csv,
        ablation_csv=ablation_csv,
        validity_md=validity_md,
        expected_runs=expected_runs,
        expected_cases=len(case_paths),
        expected_seeds=len(config.seeds),
        expected_settings=len(config.settings),
        expected_systems=len(systems),
        full_ablation_expected=_supports_full_ablation(config.settings),
    )
    errors.extend(validity_errors)
    warnings.extend(validity_warnings)

    _write_validity_log(
        path=validity_md,
        config=config,
        total_runs=len(run_rows),
        expected_runs=expected_runs,
        errors=errors,
        warnings=warnings,
    )

    return MatrixResult(
        output_dir=output_dir,
        runs_jsonl=runs_jsonl,
        by_case_csv=by_case_csv,
        summary_csv=summary_csv,
        ablation_csv=ablation_csv,
        validity_md=validity_md,
        total_runs=len(run_rows),
        expected_runs=expected_runs,
        errors=errors,
        warnings=warnings,
    )


def _matrix_systems(config: MatrixConfig) -> list[str]:
    """Return the ordered system axis for a matrix run."""

    raw_systems = config.systems if config.systems else [config.system]
    systems: list[str] = []
    for raw_system in raw_systems:
        system = str(raw_system).strip().lower()
        if system not in SUPPORTED_SYSTEMS:
            raise ValueError(f"Unknown system '{raw_system}'. Expected one of {SUPPORTED_SYSTEMS}")
        if system not in systems:
            systems.append(system)
    return systems


def _build_matrix_jobs(
    *,
    config: MatrixConfig,
    case_paths: list[Path],
    runs_dir: Path,
    systems: list[str],
    judge_pipeline_hash: str,
) -> list[_MatrixJob]:
    """Build independent jobs in the same deterministic matrix order used for output."""

    jobs: list[_MatrixJob] = []
    for case_path in case_paths:
        case_payload = load_json_file(case_path)
        case_input = CaseInput.model_validate(case_payload)

        for seed in config.seeds:
            for setting in config.settings:
                for system in systems:
                    run_id = _build_run_id(case_input.case_name, setting, seed, system)
                    artifacts_dir = runs_dir / run_id
                    run_record_path = artifacts_dir / "run_record.json"
                    pipeline_config = PipelineConfig(
                        case_input=case_path,
                        artifacts_dir=artifacts_dir,
                        run_record_path=run_record_path,
                        run_id=run_id,
                        setting=setting,
                        seed=seed,
                        model=config.model,
                        temperature=config.temperature,
                        round_cap=config.round_cap,
                        max_tokens=config.max_tokens,
                        system=system,
                        rag_enabled=config.rag_enabled,
                        rag_backend=config.rag_backend,
                        rag_corpus_dir=config.rag_corpus_dir,
                    )
                    jobs.append(
                        _MatrixJob(
                            index=len(jobs),
                            case_path=case_path,
                            case_input=case_input,
                            seed=seed,
                            setting=setting,
                            system=system,
                            run_id=run_id,
                            artifacts_dir=artifacts_dir,
                            run_record_path=run_record_path,
                            pipeline_config=pipeline_config,
                            judge_pipeline_hash=judge_pipeline_hash,
                        )
                    )
    return jobs


def _run_matrix_jobs(*, jobs: list[_MatrixJob], max_workers: int) -> list[_MatrixRunOutcome]:
    """Run matrix jobs with bounded concurrency while preserving input order."""

    worker_count = max(1, int(max_workers))
    if worker_count == 1 or len(jobs) <= 1:
        return [_run_matrix_job(job) for job in jobs]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_run_matrix_job, jobs))


def _run_matrix_job(job: _MatrixJob) -> _MatrixRunOutcome:
    """Execute one matrix job without touching matrix-level aggregate files."""

    run_record = run_case_pipeline(job.pipeline_config)
    _write_run_record_provenance(
        run_record_path=job.run_record_path,
        judge_pipeline_hash=job.judge_pipeline_hash,
    )

    case_report = validate_case_input(job.case_path)
    run_report = validate_run_record(job.run_record_path)
    artifact_report = validate_phase_artifacts(job.artifacts_dir)
    behavior_report = validate_system_behavior_contract(
        system=run_record.system,
        artifacts_dir=job.artifacts_dir,
        run_record_path=job.run_record_path,
    )

    local_errors = (
        case_report.errors
        + run_report.errors
        + artifact_report.errors
        + behavior_report.errors
    )
    local_warnings = (
        case_report.warnings
        + run_report.warnings
        + artifact_report.warnings
        + behavior_report.warnings
    )
    return _MatrixRunOutcome(
        job=job,
        run_record=run_record,
        local_errors=local_errors,
        local_warnings=local_warnings,
    )


def _build_metric_row(
    *,
    run_record: Any,
    metrics: dict[str, Any],
    judge_pipeline_hash: str,
    validation_passed: bool,
    non_comparable_reason: str,
) -> dict[str, Any]:
    """Build one row for comparison_metrics_by_case.csv."""

    return {
        "run_id": run_record.run_id,
        "case_id": run_record.case_id,
        "seed": run_record.seed,
        "system": run_record.system,
        "setting": run_record.setting,
        "model": run_record.model,
        "temperature": run_record.temperature,
        "max_tokens": run_record.max_tokens,
        "round_cap": run_record.round_cap,
        "rag_enabled": run_record.rag_enabled,
        "rag_fallback_used": run_record.rag_fallback_used,
        "fallback_tainted": run_record.execution_flags.fallback_tainted,
        "retry_used": run_record.execution_flags.retry_used,
        "retry_count": run_record.execution_flags.retry_count,
        "runtime_seconds": run_record.runtime_seconds,
        "n_phase1_agents": metrics["n_phase1_agents"],
        "n_phase1_elements": metrics["n_phase1_elements"],
        "n_phase2_negotiations": metrics["n_phase2_negotiations"],
        "n_phase2_steps": metrics["n_phase2_steps"],
        "n_phase3_elements": metrics["n_phase3_elements"],
        "n_phase3_connections": metrics["n_phase3_connections"],
        "conflict_resolution_rate": metrics["conflict_resolution_rate"],
        "chv": metrics["chv"],
        "mdc": metrics["mdc"],
        "semantic_preservation_f1": metrics["semantic_preservation_f1"],
        "semantic_p2_vs_p1_f1": metrics["semantic_p2_vs_p1_f1"],
        "s_logic": metrics["s_logic"],
        "topology_is_valid": metrics["topology_is_valid"],
        "deterministic_is_valid": metrics["deterministic_is_valid"],
        "compliance_coverage": metrics["compliance_coverage"],
        "s_term": metrics["s_term"],
        "iso29148_unambiguous": metrics["iso29148_unambiguous"],
        "iso29148_correctness": metrics["iso29148_correctness"],
        "iso29148_verifiability": metrics["iso29148_verifiability"],
        "iso29148_set_consistency": metrics["iso29148_set_consistency"],
        "iso29148_set_feasibility": metrics["iso29148_set_feasibility"],
        "blind_eval_run_id": "",
        "judge_pipeline_hash": judge_pipeline_hash,
        "validation_passed": validation_passed,
        "non_comparable_reason": non_comparable_reason,
    }


def _compute_run_metrics(artifacts_dir: Path) -> dict[str, Any]:
    """Compute protocol metrics from generated phase artifacts."""

    phase1 = load_json_file(artifacts_dir / PHASE1_FILENAME)
    phase2 = load_json_file(artifacts_dir / PHASE2_FILENAME)
    phase3 = load_json_file(artifacts_dir / PHASE3_FILENAME)
    phase4 = load_json_file(artifacts_dir / PHASE4_FILENAME)

    n_phase1_agents = len(phase1)
    n_phase1_elements = sum(len(elements) for elements in phase1.values())
    n_phase2_negotiations = int(phase2.get("total_negotiations", 0))
    n_phase2_steps = int(phase2.get("summary_stats", {}).get("total_steps", 0))
    n_phase3_elements = len(phase3.get("gsn_elements", []))
    n_phase3_connections = len(phase3.get("gsn_connections", []))

    detected_conflicts = int(phase2.get("summary_stats", {}).get("detected_conflicts", 0))
    resolved_conflicts = int(phase2.get("summary_stats", {}).get("resolved_conflicts", 0))

    # Paper-faithful RQ2 metric: resolved_conflicts / detected_conflicts.
    # When no conflicts are detected, keep the rate at 0.0 instead of using
    # consensus as a proxy denominator.
    conflict_resolution_rate = (
        resolved_conflicts / detected_conflicts if detected_conflicts > 0 else 0.0
    )

    phase3_elements = phase3.get("gsn_elements", [])
    if not isinstance(phase3_elements, list):
        phase3_elements = []
    phase3_requirements = _phase3_requirement_texts(phase3)
    phase1_requirements = _phase1_requirement_texts(phase1)
    phase2_requirements = _phase2_requirement_texts(phase2)
    chv, mdc = _compute_chv_mdc(phase3_elements)

    semantic_preservation_f1 = _semantic_preservation_f1(
        candidates=phase3_requirements,
        references=phase1_requirements,
    )
    semantic_p2_vs_p1_f1 = _semantic_preservation_f1(
        candidates=phase2_requirements,
        references=phase1_requirements,
    )

    topology_valid = int(bool(phase3.get("topology_status", {}).get("is_valid", False)))
    deterministic_valid = int(bool(phase4.get("deterministic_validation", {}).get("is_valid", False)))
    s_logic = _to_float(phase4.get("verification_results", {}).get("s_logic"), default=0.0)
    compliance_coverage = _to_float(
        phase4.get("verification_results", {})
        .get("compliance_coverage", {})
        .get("coverage_ratio"),
        default=0.0,
    )
    s_term = _to_float(
        phase4.get("verification_results", {})
        .get("terminology_consistency", {})
        .get("consistency_ratio"),
        default=0.0,
    )
    iso29148_scores = _iso29148_scores(
        s_logic=s_logic,
        s_term=s_term,
        topology_valid=topology_valid,
        deterministic_valid=deterministic_valid,
        compliance_coverage=compliance_coverage,
    )

    return {
        "n_phase1_agents": n_phase1_agents,
        "n_phase1_elements": n_phase1_elements,
        "n_phase2_negotiations": n_phase2_negotiations,
        "n_phase2_steps": n_phase2_steps,
        "n_phase3_elements": n_phase3_elements,
        "n_phase3_connections": n_phase3_connections,
        "conflict_resolution_rate": round(conflict_resolution_rate, 6),
        "chv": round(chv, 6),
        "mdc": round(mdc, 6),
        "semantic_preservation_f1": round(semantic_preservation_f1, 6),
        "semantic_p2_vs_p1_f1": round(semantic_p2_vs_p1_f1, 6),
        "s_logic": round(s_logic, 6),
        "topology_is_valid": topology_valid,
        "deterministic_is_valid": deterministic_valid,
        "compliance_coverage": round(compliance_coverage, 6),
        "s_term": round(s_term, 6),
        "iso29148_unambiguous": iso29148_scores["unambiguous"],
        "iso29148_correctness": iso29148_scores["correctness"],
        "iso29148_verifiability": iso29148_scores["verifiability"],
        "iso29148_set_consistency": iso29148_scores["set_consistency"],
        "iso29148_set_feasibility": iso29148_scores["set_feasibility"],
    }


def _phase1_requirement_texts(phase1: dict[str, list[dict[str, Any]]]) -> list[str]:
    texts: list[str] = []
    for elements in phase1.values():
        for element in elements:
            description = str(element.get("description", "")).strip()
            if description:
                texts.append(description)
    return texts


def _phase3_requirement_texts(phase3: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for element in phase3.get("gsn_elements", []):
        name = str(element.get("name", "")).strip()
        description = str(element.get("description", "")).strip()
        text = f"{name}. {description}".strip(" .")
        if text:
            texts.append(text)
    return texts


def _phase2_requirement_texts(phase2: dict[str, Any]) -> list[str]:
    """Extract negotiated requirement-like text for phase-trajectory scoring."""

    texts: list[str] = []
    for negotiation in phase2.get("negotiations", {}).values():
        steps = negotiation.get("steps", [])
        for step in steps:
            if step.get("message_type") != "backward":
                continue
            for element in step.get("kaos_elements", []):
                description = str(element.get("description", "")).strip()
                if description:
                    texts.append(description)
    return texts


def _quality_vector_from_element(
    element: dict[str, Any],
    *,
    semantic_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Project one requirement into R^5 from explicit role/quality metadata.

    This avoids text-keyword boosts and hash-bucket fallback behavior that can
    inflate CHV/MDC without evidence of true quality-axis coverage.
    """

    metadata_prior = np.zeros(len(QUALITY_AXES), dtype=float)
    inferred_axes: set[str] = set()
    has_integrated_hint = False

    for raw_value in _quality_metadata_candidates(element):
        axes, integrated = _infer_axes_from_value(raw_value)
        inferred_axes.update(axes)
        has_integrated_hint = has_integrated_hint or integrated

    if inferred_axes:
        for axis in inferred_axes:
            metadata_prior[QUALITY_AXES.index(axis)] = 1.0
    elif has_integrated_hint:
        # "Integrated" is not evidence of broad axis coverage by itself.
        # Keep metadata prior neutral and rely on text-axis projection.
        metadata_prior = np.zeros(len(QUALITY_AXES), dtype=float)

    text_scores = np.zeros(len(QUALITY_AXES), dtype=float)
    if semantic_scores is not None and semantic_scores.size == len(QUALITY_AXES):
        # Keep cosine similarities on their absolute scale. Per-element max
        # normalization can promote weak evidence to 1.0 and bias CHV/MDC.
        text_scores = np.clip(semantic_scores.astype(float), 0.0, 1.0)

    if float(np.sum(metadata_prior)) > 0.0 and float(np.sum(text_scores)) > 0.0:
        combined = (METADATA_AXIS_WEIGHT * metadata_prior) + (SEMANTIC_AXIS_WEIGHT * text_scores)
    elif float(np.sum(metadata_prior)) > 0.0:
        combined = metadata_prior
    else:
        combined = text_scores

    return np.clip(combined, 0.0, 1.0)


def _normalize_axis_text(text: str) -> str:
    """Normalize requirement text to reduce keyword-stuffing effects."""

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
        if len(deduped) >= MAX_AXIS_TEXT_TOKENS:
            break
    return " ".join(deduped)


def _axis_scoring_text(element: dict[str, Any]) -> str:
    """Build normalized text payload used by rubric projection."""

    raw = " ".join(
        str(element.get(key, ""))
        for key in ("name", "description", "measurable_criteria")
    ).strip()
    return _normalize_axis_text(raw)


def _project_semantic_axis_scores(elements: list[dict[str, Any]]) -> np.ndarray:
    """Project requirement texts onto quality rubrics using cosine similarity."""

    if not elements:
        return np.zeros((0, len(QUALITY_AXES)), dtype=float)

    texts = [_axis_scoring_text(item) for item in elements]
    if not any(texts):
        return np.zeros((len(elements), len(QUALITY_AXES)), dtype=float)

    rubric_texts = [_normalize_axis_text(QUALITY_AXIS_RUBRICS[axis]) for axis in QUALITY_AXES]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2048)
    try:
        tfidf = vectorizer.fit_transform(texts + rubric_texts)
    except ValueError:
        return np.zeros((len(elements), len(QUALITY_AXES)), dtype=float)

    text_vectors = tfidf[: len(texts)]
    rubric_vectors = tfidf[len(texts) :]
    similarities = cosine_similarity(text_vectors, rubric_vectors)
    return np.clip(similarities, 0.0, 1.0)


def _quality_metadata_candidates(element: dict[str, Any]) -> list[Any]:
    """Collect metadata values that can encode quality-axis ownership."""

    candidates: list[Any] = [
        element.get("quality_attribute"),
        element.get("stakeholder"),
        element.get("role"),
        element.get("agent"),
        element.get("agent_name"),
    ]
    properties = element.get("properties")
    if isinstance(properties, dict):
        candidates.extend(
            [
                properties.get("quality_attribute"),
                properties.get("role"),
                properties.get("agent"),
                properties.get("agent_name"),
            ]
        )
    return candidates


def _infer_axes_from_value(raw_value: Any) -> tuple[set[str], bool]:
    """Infer quality axes and integrated hints from one metadata value."""

    if raw_value is None:
        return set(), False

    parts: list[Any]
    if isinstance(raw_value, (list, tuple, set)):
        parts = list(raw_value)
    else:
        parts = [raw_value]

    axes: set[str] = set()
    integrated = False
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        for token in re.split(r"(?:,|/|\||;|\+|\band\b)", text, flags=re.IGNORECASE):
            compact = re.sub(r"[^a-z0-9]+", "", token.lower())
            if not compact:
                continue
            axis = QUALITY_AXIS_BY_TOKEN.get(compact)
            if axis:
                axes.add(axis)
                continue
            if compact in INTEGRATED_QUALITY_TOKENS:
                integrated = True

    return axes, integrated


def _compute_chv_mdc(elements: list[dict[str, Any]]) -> tuple[float, float]:
    """Compute CHV/MDC from paper-aligned quality-space projection in R^5."""

    normalized_elements = [item for item in elements if isinstance(item, dict)]
    semantic_scores = _project_semantic_axis_scores(normalized_elements)
    projected = np.array(
        [
            _quality_vector_from_element(item, semantic_scores=semantic_scores[index])
            for index, item in enumerate(normalized_elements)
        ],
        dtype=float,
    )
    if projected.shape[0] == 0:
        return 0.0, 0.0

    chv = 0.0
    hull_space = _intrinsic_hull_space(projected)
    if hull_space.shape[1] >= 2 and hull_space.shape[0] > hull_space.shape[1]:
        try:
            chv = float(ConvexHull(hull_space).volume)
        except (QhullError, ValueError):
            chv = 0.0

    centroid = np.mean(projected, axis=0).reshape(1, -1)
    distances = cdist(projected, centroid, metric="euclidean").flatten()
    mdc = float(np.mean(distances)) if len(distances) else 0.0
    return chv, mdc


def _intrinsic_hull_space(points: np.ndarray) -> np.ndarray:
    """Project to intrinsic rank before hull volume to avoid false degeneracy."""

    if points.ndim != 2 or points.shape[0] == 0:
        return points
    centered = points - np.mean(points, axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-9))
    if rank <= 0:
        return centered
    return centered @ vt[:rank].T


def _semantic_preservation_f1(candidates: list[str], references: list[str]) -> float:
    """Compute semantic preservation using QUARE-equivalent BERTScore flow."""

    if not candidates or not references:
        return 0.0

    cache_key = _semantic_cache_key(candidates, references)
    cached = _BERTSCORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scorer = _get_bertscorer()
    pair_candidates: list[str] = []
    pair_references: list[str] = []
    row_offsets: list[tuple[int, int]] = []
    for candidate in candidates:
        start = len(pair_candidates)
        for reference in references:
            pair_candidates.append(candidate)
            pair_references.append(reference)
        row_offsets.append((start, len(pair_candidates)))

    _, _, f1_values = scorer.score(
        pair_candidates,
        pair_references,
        batch_size=64,
        verbose=False,
    )

    f1_scores: list[float] = []
    for start, end in row_offsets:
        row = f1_values[start:end]
        best_index = int(row.argmax().item())
        f1_scores.append(float(row[best_index].item()))

    result = sum(f1_scores) / len(f1_scores)
    _BERTSCORE_CACHE[cache_key] = result
    return result


def _semantic_cache_key(candidates: list[str], references: list[str]) -> str:
    """Build stable cache key for semantic-preservation computation."""

    payload = json.dumps(
        {
            "candidates": candidates,
            "references": references,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_bertscorer() -> Any:
    """Load and cache a BERTScorer instance for semantic-preservation scoring."""

    global _BERTSCORER
    if _BERTSCORER is not None:
        return _BERTSCORER

    try:
        from bert_score import BERTScorer  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "BERTScore is required for strict semantic-preservation parity. "
            "Install `bert-score` and rerun."
        ) from exc

    _BERTSCORER = BERTScorer(
        model_type="bert-base-uncased",
        lang="en",
        rescale_with_baseline=False,
    )
    return _BERTSCORER


def _iso29148_scores(
    *,
    s_logic: float,
    s_term: float,
    topology_valid: int,
    deterministic_valid: int,
    compliance_coverage: float,
) -> dict[str, float]:
    """Build deterministic ISO29148-like Likert scores from available metrics."""

    normalized_logic = _clamp01(s_logic)
    normalized_term = _clamp01(s_term)
    normalized_topology = _clamp01(float(topology_valid))
    normalized_deterministic = _clamp01(float(deterministic_valid))
    normalized_compliance = _clamp01(compliance_coverage)

    unambiguous = _likert_from_ratio(0.55 * normalized_term + 0.45 * normalized_logic)
    correctness = _likert_from_ratio(0.70 * normalized_logic + 0.30 * normalized_deterministic)
    verifiability = _likert_from_ratio(
        0.50 * normalized_deterministic + 0.50 * normalized_compliance
    )
    set_consistency = _likert_from_ratio(0.50 * normalized_logic + 0.50 * normalized_topology)
    set_feasibility = _likert_from_ratio(
        0.60 * normalized_compliance + 0.40 * normalized_deterministic
    )
    return {
        "unambiguous": unambiguous,
        "correctness": correctness,
        "verifiability": verifiability,
        "set_consistency": set_consistency,
        "set_feasibility": set_feasibility,
    }


def _clamp01(value: float) -> float:
    """Clamp any numeric value into [0, 1]."""

    return max(0.0, min(1.0, value))


def _likert_from_ratio(value: float) -> float:
    """Map normalized ratio in [0,1] to a 1-5 Likert score."""

    return round(1.0 + 4.0 * _clamp01(value), 6)


def _file_sha256(path: Path | None) -> str:
    """Return SHA256 hash for a file path when available."""

    if path is None:
        return ""
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(65536)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_run_record_provenance(*, run_record_path: Path, judge_pipeline_hash: str) -> None:
    """Inject provenance fields into persisted run records."""

    payload = load_json_file(run_record_path)
    if not isinstance(payload, dict):
        return
    payload["artifact_blinded"] = False
    payload["blinding_scheme_version"] = ""
    payload["blind_eval_run_id"] = ""
    payload["judge_pipeline_hash"] = judge_pipeline_hash
    payload["trace_audit_path"] = ""
    write_json_file(run_record_path, payload)


def export_trace_audit(
    *,
    matrix_output_dir: Path,
    output_path: Path | None = None,
) -> TraceAuditResult:
    """Export a markdown sanity audit over phase2 negotiation traces."""

    runs_jsonl = matrix_output_dir / RUNS_JSONL_NAME
    if not runs_jsonl.exists():
        raise FileNotFoundError(f"Missing runs JSONL: {runs_jsonl}")

    run_rows = _read_jsonl(runs_jsonl)
    target_path = output_path or (matrix_output_dir / TRACE_AUDIT_MD_NAME)

    runs_with_loops = 0
    runs_with_conflicts = 0
    lines = [
        "# Comparison Trace Audit",
        "",
        f"- Generated at: {utc_timestamp()}",
        f"- Matrix output dir: `{matrix_output_dir}`",
        f"- Total runs scanned: {len(run_rows)}",
        "",
        "## Per-Run Negotiation Sanity",
        "",
        "| run_id | setting | total_negotiations | total_steps | backward_steps | loop_detected | conflicts_detected | conflicts_resolved |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]

    for row in run_rows:
        run_id = str(row.get("run_id", ""))
        setting = str(row.get("setting", ""))
        artifact_paths = row.get("artifact_paths", {})
        phase2_path = Path(str(artifact_paths.get(PHASE2_FILENAME, "")))
        if not phase2_path.exists():
            total_negotiations = 0
            total_steps = 0
            backward_steps = 0
            detected_conflicts = 0
            resolved_conflicts = 0
        else:
            phase2_payload = load_json_file(phase2_path)
            total_negotiations = int(phase2_payload.get("total_negotiations", 0))
            summary_stats = phase2_payload.get("summary_stats", {})
            total_steps = int(summary_stats.get("total_steps", 0))
            backward_steps = 0
            for negotiation in phase2_payload.get("negotiations", {}).values():
                for step in negotiation.get("steps", []):
                    if step.get("message_type") == "backward":
                        backward_steps += 1
            detected_conflicts = int(summary_stats.get("detected_conflicts", 0))
            resolved_conflicts = int(summary_stats.get("resolved_conflicts", 0))

        loop_detected = backward_steps > 0 and total_steps >= 2
        if loop_detected:
            runs_with_loops += 1
        if detected_conflicts > 0:
            runs_with_conflicts += 1

        lines.append(
            "| "
            f"{run_id} | {setting} | {total_negotiations} | {total_steps} | "
            f"{backward_steps} | {'yes' if loop_detected else 'no'} | "
            f"{detected_conflicts} | {resolved_conflicts} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            f"- Runs with feedback loops: {runs_with_loops}/{len(run_rows)}",
            f"- Runs with detected conflicts: {runs_with_conflicts}/{len(run_rows)}",
        ]
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return TraceAuditResult(
        output_path=target_path,
        total_runs=len(run_rows),
        runs_with_loops=runs_with_loops,
        runs_with_conflicts=runs_with_conflicts,
    )


def prepare_blind_evaluation(
    *,
    matrix_output_dir: Path,
    blind_output_dir: Path,
    judge_pipeline_path: Path,
) -> BlindPrepResult:
    """Build anonymized run artifacts for blind LLM-as-a-judge evaluation."""

    runs_jsonl = matrix_output_dir / RUNS_JSONL_NAME
    by_case_csv = matrix_output_dir / BY_CASE_CSV_NAME
    if not runs_jsonl.exists():
        raise FileNotFoundError(f"Missing runs JSONL: {runs_jsonl}")
    if not by_case_csv.exists():
        raise FileNotFoundError(f"Missing by-case CSV: {by_case_csv}")

    judge_pipeline_hash = _file_sha256(judge_pipeline_path)
    run_rows = _read_jsonl(runs_jsonl)
    systems = sorted({str(row.get("system", "System")) for row in run_rows})
    system_mapping = {
        system_name: f"System_{chr(ord('A') + index)}"
        for index, system_name in enumerate(systems)
    }

    artifacts_root = blind_output_dir / "blinded_artifacts"
    blinded_rows: list[dict[str, Any]] = []
    run_mapping: dict[str, str] = {}
    for index, row in enumerate(run_rows, start=1):
        original_run_id = str(row.get("run_id", f"run-{index:03d}"))
        blind_run_id = f"BLIND_RUN_{index:03d}"
        run_mapping[original_run_id] = blind_run_id

        blinded_row = json.loads(json.dumps(row))
        blinded_row["run_id"] = blind_run_id
        blinded_row["blind_eval_run_id"] = blind_run_id
        blinded_row["judge_pipeline_hash"] = judge_pipeline_hash
        blinded_row["artifact_blinded"] = True
        blinded_row["blinding_scheme_version"] = BLINDING_SCHEME_VERSION
        blinded_row["system"] = system_mapping.get(str(row.get("system", "")), "System_A")
        if isinstance(blinded_row.get("system_identity"), dict):
            blinded_row["system_identity"]["system_name"] = blinded_row["system"]
        blinded_row["notes"] = {"anonymized": True}

        destination_dir = artifacts_root / blind_run_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        original_artifacts = row.get("artifact_paths", {})
        blinded_artifacts: dict[str, str] = {}
        for filename in PHASE1_FILENAME, PHASE2_FILENAME, PHASE3_FILENAME, PHASE4_FILENAME:
            source = Path(str(original_artifacts.get(filename, "")))
            destination = destination_dir / filename
            if source.exists() and source.is_file():
                shutil.copy2(source, destination)
            blinded_artifacts[filename] = str(destination)

        blinded_row["artifacts_dir"] = str(destination_dir)
        blinded_row["artifact_paths"] = blinded_artifacts
        blinded_rows.append(blinded_row)

    blinded_runs_jsonl = blind_output_dir / BLINDED_RUNS_JSONL_NAME
    _write_jsonl(blinded_runs_jsonl, blinded_rows)

    with by_case_csv.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
        source_columns = list(source_rows[0].keys()) if source_rows else BY_CASE_COLUMNS

    extra_columns = ["blind_eval_run_id", "judge_pipeline_hash"]
    final_columns = list(source_columns)
    for column in extra_columns:
        if column not in final_columns:
            final_columns.append(column)

    blinded_metric_rows: list[dict[str, Any]] = []
    for row in source_rows:
        original_run_id = str(row.get("run_id", ""))
        blinded_row = dict(row)
        blinded_row["run_id"] = run_mapping.get(original_run_id, original_run_id)
        blinded_row["blind_eval_run_id"] = run_mapping.get(original_run_id, "")
        blinded_row["judge_pipeline_hash"] = judge_pipeline_hash
        if "system" in blinded_row:
            blinded_row["system"] = system_mapping.get(str(row.get("system", "")), "System_A")
        blinded_metric_rows.append(blinded_row)

    blinded_by_case_csv = blind_output_dir / BLINDED_BY_CASE_CSV_NAME
    _write_csv(blinded_by_case_csv, final_columns, blinded_metric_rows)

    mapping_json = blind_output_dir / BLIND_MAPPING_JSON_NAME
    write_json_file(
        mapping_json,
        {
            "generated_at": utc_timestamp(),
            "blinding_scheme_version": BLINDING_SCHEME_VERSION,
            "system_mapping": system_mapping,
            "run_mapping": run_mapping,
        },
    )

    protocol_md = blind_output_dir / BLIND_PROTOCOL_MD_NAME
    protocol_md.write_text(
        "\n".join(
            [
                "# Blind Evaluation Preparation",
                "",
                f"- Generated at: {utc_timestamp()}",
                f"- Source matrix dir: `{matrix_output_dir}`",
                f"- Output dir: `{blind_output_dir}`",
                f"- Blinding scheme: `{BLINDING_SCHEME_VERSION}`",
                f"- Judge pipeline path: `{judge_pipeline_path}`",
                f"- Judge pipeline hash (sha256): `{judge_pipeline_hash}`",
                "",
                "## Generated Files",
                f"- `{blinded_runs_jsonl}`",
                f"- `{blinded_by_case_csv}`",
                f"- `{mapping_json}` (traceability mapping; do not disclose to judges)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return BlindPrepResult(
        output_dir=blind_output_dir,
        blinded_runs_jsonl=blinded_runs_jsonl,
        blinded_by_case_csv=blinded_by_case_csv,
        mapping_json=mapping_json,
        protocol_md=protocol_md,
        judge_pipeline_hash=judge_pipeline_hash,
    )


def _build_summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-case summaries with means/std for numeric metrics."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row.get("system", SYSTEM_MARE)), row["case_id"], row["setting"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (system, case_id, setting), rows in sorted(grouped.items()):
        runtime_values = _numeric_values(rows, "runtime_seconds")
        p1_values = _numeric_values(rows, "n_phase1_elements")
        p2_values = _numeric_values(rows, "n_phase2_steps")
        p3_values = _numeric_values(rows, "n_phase3_elements")
        conflict_values = _numeric_values(rows, "conflict_resolution_rate")
        topology_values = _numeric_values(rows, "topology_is_valid")

        summary_rows.append(
            {
                "system": system,
                "case_id": case_id,
                "setting": setting,
                "runs": len(rows),
                "valid_runs": sum(1 for row in rows if row["validation_passed"]),
                "invalid_runs": sum(1 for row in rows if not row["validation_passed"]),
                "mean_runtime_seconds": _avg(runtime_values),
                "std_runtime_seconds": _std(runtime_values),
                "mean_phase1_elements": _avg(p1_values),
                "std_phase1_elements": _std(p1_values),
                "mean_phase2_steps": _avg(p2_values),
                "std_phase2_steps": _std(p2_values),
                "mean_phase3_elements": _avg(p3_values),
                "std_phase3_elements": _std(p3_values),
                "mean_conflict_resolution_rate": _avg(conflict_values),
                "std_conflict_resolution_rate": _std(conflict_values),
                "mean_topology_valid": _avg(topology_values),
                "std_topology_valid": _std(topology_values),
            }
        )

    return summary_rows


def _build_ablation_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build phase-wise delta table for ablation comparisons."""

    rows_by_system_case_seed: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in metric_rows:
        system = str(row.get("system", SYSTEM_MARE))
        rows_by_system_case_seed[(system, row["case_id"], int(row["seed"]))][row["setting"]] = row

    ablation_rows: list[dict[str, Any]] = []
    for (system, case_id, seed), by_setting in sorted(rows_by_system_case_seed.items()):
        single = by_setting.get(SETTING_SINGLE_AGENT)
        without_neg = by_setting.get(SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION)
        with_neg = by_setting.get(SETTING_MULTI_AGENT_WITH_NEGOTIATION)
        full = by_setting.get(SETTING_NEGOTIATION_INTEGRATION_VERIFICATION)
        if not (single and without_neg and with_neg and full):
            continue

        single_p1 = int(_to_float(single.get("n_phase1_elements"), 0.0))
        without_neg_p1 = int(_to_float(without_neg.get("n_phase1_elements"), 0.0))
        with_neg_p2_steps = int(_to_float(with_neg.get("n_phase2_steps"), 0.0))
        without_neg_p2_steps = int(_to_float(without_neg.get("n_phase2_steps"), 0.0))
        full_p3 = int(_to_float(full.get("n_phase3_elements"), 0.0))
        with_neg_p3 = int(_to_float(with_neg.get("n_phase3_elements"), 0.0))

        ablation_rows.append(
            {
                "system": system,
                "case_id": case_id,
                "seed": seed,
                "single_agent_phase1_elements": single_p1,
                "multi_agent_without_negotiation_phase1_elements": without_neg_p1,
                "multi_agent_with_negotiation_phase2_steps": with_neg_p2_steps,
                "negotiation_integration_verification_phase3_elements": full_p3,
                "delta_multi_without_neg_vs_single_phase1_elements": without_neg_p1 - single_p1,
                "delta_multi_with_neg_vs_without_neg_phase2_steps": with_neg_p2_steps
                - without_neg_p2_steps,
                "delta_full_vs_with_neg_phase3_elements": full_p3 - with_neg_p3,
                "full_topology_valid": int(_to_float(full.get("topology_is_valid"), 0.0)),
                "notes": "Protocol metrics computed from deterministic parity artifacts.",
            }
        )

    return ablation_rows


def _validate_deliverables(
    *,
    runs_jsonl: Path,
    by_case_csv: Path,
    summary_csv: Path,
    ablation_csv: Path,
    validity_md: Path,
    expected_runs: int,
    expected_cases: int,
    expected_seeds: int,
    expected_settings: int,
    full_ablation_expected: bool,
    expected_systems: int = 1,
) -> tuple[list[str], list[str]]:
    """Strict validation for required protocol deliverables."""

    errors: list[str] = []
    warnings: list[str] = []

    for path in (runs_jsonl, by_case_csv, summary_csv, ablation_csv):
        if not path.exists():
            errors.append(f"Missing deliverable: {path}")

    if not runs_jsonl.exists():
        return errors, warnings

    run_lines = list(_read_jsonl(runs_jsonl))
    if len(run_lines) != expected_runs:
        errors.append(
            f"Run count mismatch in {runs_jsonl.name}: expected {expected_runs}, got {len(run_lines)}"
        )

    for row in run_lines:
        missing = REQUIRED_RUN_KEYS - set(row.keys())
        if missing:
            errors.append(f"Run row missing required keys {sorted(missing)}: {row.get('run_id', '?')}")
        if not bool(row.get("validation_passed", False)):
            errors.append(f"Run marked invalid: {row.get('run_id', '?')}")
        if not bool(row.get("rag_enabled", False)):
            errors.append(f"RAG parity failure (rag_enabled=false): {row.get('run_id', '?')}")
        rag_backend = str(row.get("rag_backend", "")).strip().lower()
        if rag_backend in {"", "none"}:
            errors.append(f"RAG parity failure (rag_backend unset): {row.get('run_id', '?')}")
        if bool(row.get("rag_fallback_used", False)):
            errors.append(f"RAG fallback used: {row.get('run_id', '?')}")

        provenance = row.get("provenance", {})
        if not isinstance(provenance, dict):
            errors.append(f"Provenance metadata malformed: {row.get('run_id', '?')}")
        else:
            prompt_hash = str(provenance.get("prompt_hash", ""))
            if not _is_sha256_hex(prompt_hash):
                errors.append(f"Prompt hash missing/invalid: {row.get('run_id', '?')}")
            corpus_hash = str(provenance.get("corpus_hash", ""))
            if bool(row.get("rag_enabled", False)) and not _is_sha256_hex(corpus_hash):
                errors.append(f"Corpus hash missing/invalid: {row.get('run_id', '?')}")

        execution_flags = row.get("execution_flags", {})
        if not isinstance(execution_flags, dict):
            errors.append(f"Execution flags malformed: {row.get('run_id', '?')}")
        else:
            if bool(execution_flags.get("fallback_tainted", False)):
                errors.append(f"Fallback-tainted metadata: {row.get('run_id', '?')}")
            retry_used = bool(execution_flags.get("retry_used", False))
            retry_count = _to_int(execution_flags.get("retry_count", 0), default=0)
            if retry_used or retry_count > 0:
                errors.append(f"Retry-tainted metadata: {row.get('run_id', '?')}")

        comparability = row.get("comparability", {})
        if not isinstance(comparability, dict):
            errors.append(f"Comparability metadata malformed: {row.get('run_id', '?')}")
        else:
            reasons_raw = comparability.get("non_comparable_reasons", [])
            reasons = [str(item) for item in reasons_raw] if isinstance(reasons_raw, list) else []
            expected_reasons = non_comparable_reasons_for_setting(str(row.get("setting", "")))
            expected_reason_text = _non_comparable_reason_text(expected_reasons)
            row_reason_text = str(row.get("non_comparable_reason", ""))
            if row_reason_text != expected_reason_text:
                errors.append(f"Non-comparable reason mismatch: {row.get('run_id', '?')}")

            is_comparable = bool(comparability.get("is_comparable", False))
            if is_comparable != (not expected_reasons):
                errors.append(f"Comparability flag mismatch for setting: {row.get('run_id', '?')}")
            if sorted(reasons) != sorted(expected_reasons):
                errors.append(
                    f"Comparability reasons mismatch for setting: {row.get('run_id', '?')}"
                )

    _validate_csv_columns(by_case_csv, BY_CASE_COLUMNS, errors)
    _validate_csv_columns(summary_csv, SUMMARY_COLUMNS, errors)
    _validate_csv_columns(ablation_csv, ABLATION_COLUMNS, errors)

    if by_case_csv.exists():
        with by_case_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_runs:
            errors.append(
                f"By-case row count mismatch: expected {expected_runs}, got {len(rows)}"
            )

    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_summary_rows = expected_cases * expected_settings * expected_systems
        if len(rows) != expected_summary_rows:
            errors.append(
                "Summary row count mismatch: expected "
                f"{expected_summary_rows}, got {len(rows)}"
            )

    if ablation_csv.exists():
        with ablation_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_ablation_rows = expected_cases * expected_seeds * expected_systems
        if full_ablation_expected:
            if len(rows) != expected_ablation_rows:
                errors.append(
                    "Ablation row count mismatch: expected "
                    f"{expected_ablation_rows}, got {len(rows)}"
                )
        elif rows:
            warnings.append(
                "Ablation rows were generated from a partial setting list; treat deltas as partial."
            )

    return errors, warnings


def _write_validity_log(
    *,
    path: Path,
    config: MatrixConfig,
    total_runs: int,
    expected_runs: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Write protocol validity report for matrix run."""

    systems = _matrix_systems(config)
    lines = [
        "# Comparison Validity Log",
        "",
        f"- Generated at: {utc_timestamp()}",
        f"- Cases dir: `{config.cases_dir}`",
        f"- Output dir: `{config.output_dir}`",
        f"- Seeds: `{config.seeds}`",
        f"- Settings: `{config.settings}`",
        f"- Systems: `{systems}`",
        f"- Matrix workers: `{max(1, int(config.max_workers))}`",
        f"- Model: `{config.model}`",
        f"- Temperature: `{config.temperature}`",
        f"- Round cap: `{config.round_cap}`",
        f"- Max tokens: `{config.max_tokens}`",
        f"- RAG enabled: `{config.rag_enabled}`",
        f"- RAG backend: `{config.rag_backend}`",
        f"- RAG corpus dir: `{config.rag_corpus_dir}`",
        "",
        "## Completeness",
        f"- Expected runs: {expected_runs}",
        f"- Actual runs: {total_runs}",
        "",
        "## Strict Fail Conditions",
        "- Missing required deliverables.",
        "- Run count mismatch versus expected matrix cardinality.",
        "- Missing required keys in `comparison_runs.jsonl`.",
        "- Missing or malformed provenance metadata (hash contracts).",
        "- Missing or malformed execution/comparability metadata contracts.",
        "- Missing required columns in required CSV outputs.",
        "- Any run with `validation_passed=false`.",
        "- Any run with `rag_enabled=false` or missing `rag_backend`.",
        "- Any run with `rag_fallback_used=true`.",
        "- Any run with `fallback_tainted=true` or retry-tainted metadata.",
        "",
        "## Errors",
    ]
    if errors:
        lines.extend(f"- {item}" for item in errors)
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Warnings")
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_run_id(case_name: str, setting: str, seed: int, system: str) -> str:
    """Build deterministic run id for matrix execution."""

    normalized_system = system.strip().lower().replace(" ", "-")
    normalized_case = case_name.strip().lower().replace(" ", "-")
    normalized_setting = setting.strip().lower().replace(" ", "-")
    return f"{normalized_system}-{normalized_case}-{normalized_setting}-s{seed:03d}"


def _non_comparable_reason_text(reasons: list[str]) -> str:
    """Serialize non-comparable reasons into a stable CSV-friendly string."""

    return "|".join(sorted(reasons))


def _is_sha256_hex(value: str) -> bool:
    """Return True when value is a valid SHA256 hex digest."""

    text = value.strip()
    if len(text) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in text)


def _supports_full_ablation(settings: list[str]) -> bool:
    """Whether current matrix setting list supports complete ablation rows."""

    required = {
        SETTING_SINGLE_AGENT,
        SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
        SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    }
    return required.issubset(set(settings))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with explicit column ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _validate_csv_columns(path: Path, columns: list[str], errors: list[str]) -> None:
    """Validate required CSV columns."""

    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        header = reader.fieldnames or []
    missing = sorted(set(columns) - set(header))
    if missing:
        errors.append(f"CSV `{path.name}` missing required columns: {missing}")


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Extract numeric values from rows for aggregation."""

    values: list[float] = []
    for row in rows:
        value = _to_float_or_none(row.get(key))
        if value is None:
            continue
        values.append(value)
    return values


def _avg(values: list[float]) -> float | str:
    """Return mean or N/A."""

    if not values:
        return "N/A"
    return round(mean(values), 6)


def _std(values: list[float]) -> float | str:
    """Return population standard deviation or N/A."""

    if not values:
        return "N/A"
    if len(values) == 1:
        return 0.0
    return round(pstdev(values), 6)


def _to_float_or_none(value: Any) -> float | None:
    """Convert value to float when possible."""

    if value in (None, "", "N/A"):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float) -> float:
    """Convert value to float with default fallback."""

    parsed = _to_float_or_none(value)
    if parsed is None:
        return default
    return parsed


def _to_int(value: Any, default: int) -> int:
    """Convert value to int with default fallback."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_settings(raw_value: str | None) -> list[str]:
    """Parse comma-separated setting names."""

    if not raw_value:
        return list(DEFAULT_MATRIX_SETTINGS)
    settings = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not settings:
        return list(DEFAULT_MATRIX_SETTINGS)

    canonical = set(DEFAULT_MATRIX_SETTINGS)
    unknown = [item for item in settings if item not in canonical]
    if unknown:
        raise ValueError(f"Unknown setting names: {unknown}")
    return settings


def parse_systems(raw_value: str | None) -> list[str]:
    """Parse comma-separated benchmark target names."""

    if not raw_value:
        return []
    systems = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not systems:
        return []

    unknown = [item for item in systems if item not in SUPPORTED_SYSTEMS]
    if unknown:
        raise ValueError(f"Unknown system names: {unknown}")
    return list(dict.fromkeys(systems))


def parse_seeds(raw_value: str | None) -> list[int]:
    """Parse comma-separated seed list."""

    if not raw_value:
        return [101]
    seeds = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    return seeds or [101]

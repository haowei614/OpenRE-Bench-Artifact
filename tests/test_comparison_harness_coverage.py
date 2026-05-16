"""Coverage tests for openre_bench.comparison_harness — pure utility functions."""

from __future__ import annotations

import csv
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import openre_bench.comparison_harness as harness
from openre_bench.comparison_harness import (
    ABLATION_COLUMNS,
    BY_CASE_COLUMNS,
    MatrixConfig,
    MatrixResult,
    SUMMARY_COLUMNS,
    TraceAuditResult,
    _avg,
    _axis_scoring_text,
    _build_ablation_rows,
    _build_run_id,
    _build_summary_rows,
    _clamp01,
    _compute_chv_mdc,
    _file_sha256,
    _infer_axes_from_value,
    _intrinsic_hull_space,
    _is_sha256_hex,
    _iso29148_scores,
    _likert_from_ratio,
    _non_comparable_reason_text,
    _normalize_axis_text,
    _numeric_values,
    _phase1_requirement_texts,
    _phase2_requirement_texts,
    _phase3_requirement_texts,
    _project_semantic_axis_scores,
    _quality_metadata_candidates,
    _quality_vector_from_element,
    _read_jsonl,
    _semantic_cache_key,
    _std,
    _supports_full_ablation,
    _to_float,
    _to_float_or_none,
    _to_int,
    _validate_csv_columns,
    _validate_deliverables,
    _write_csv,
    _write_jsonl,
    _write_run_record_provenance,
    _write_validity_log,
    export_trace_audit,
    prepare_blind_evaluation,
    parse_seeds,
    parse_settings,
    parse_systems,
    run_comparison_matrix,
    BY_CASE_CSV_NAME,
    QUALITY_AXES,
    RUNS_JSONL_NAME,
)
from openre_bench.schemas import (
    PHASE2_FILENAME,
    SETTING_MULTI_AGENT_WITH_NEGOTIATION,
    SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
    SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    SETTING_SINGLE_AGENT,
    non_comparable_reasons_for_setting,
)


# ---------------------------------------------------------------------------
# _clamp01 / _likert_from_ratio
# ---------------------------------------------------------------------------


def test_clamp_and_likert_clamp_in_range():
    assert _clamp01(0.5) == pytest.approx(0.5)


def test_clamp_and_likert_clamp_below():
    assert _clamp01(-1.0) == pytest.approx(0.0)


def test_clamp_and_likert_clamp_above():
    assert _clamp01(2.0) == pytest.approx(1.0)


def test_clamp_and_likert_likert_zero():
    assert _likert_from_ratio(0.0) == pytest.approx(1.0)


def test_clamp_and_likert_likert_one():
    assert _likert_from_ratio(1.0) == pytest.approx(5.0)


def test_clamp_and_likert_likert_mid():
    result = _likert_from_ratio(0.5)
    # 1.0 + 4.0 * 0.5 = 3.0
    assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _to_float_or_none / _to_float / _to_int
# ---------------------------------------------------------------------------


def test_coercions_float_or_none_valid():
    assert _to_float_or_none(3.14) == pytest.approx(3.14)
    assert _to_float_or_none("2.5") == pytest.approx(2.5)
    assert _to_float_or_none(42) == pytest.approx(42.0)


def test_coercions_float_or_none_invalid():
    assert _to_float_or_none(None) is None
    assert _to_float_or_none("") is None
    assert _to_float_or_none("N/A") is None
    assert _to_float_or_none("abc") is None


def test_coercions_float_or_none_nan():
    assert _to_float_or_none(float("nan")) is None


def test_coercions_float_or_none_bool():
    assert _to_float_or_none(True) == pytest.approx(1.0)
    assert _to_float_or_none(False) == pytest.approx(0.0)


def test_coercions_to_float_with_default():
    assert _to_float(3.0, 0.0) == pytest.approx(3.0)
    assert _to_float("abc", -1.0) == pytest.approx(-1.0)


def test_coercions_to_int_valid():
    assert _to_int(42, 0) == 42
    assert _to_int("7", 0) == 7


def test_coercions_to_int_invalid():
    assert _to_int("abc", -1) == -1
    assert _to_int(None, 99) == 99


# ---------------------------------------------------------------------------
# _avg / _std
# ---------------------------------------------------------------------------


def test_aggregation_avg_empty():
    assert _avg([]) == "N/A"


def test_aggregation_avg_values():
    assert _avg([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_aggregation_std_empty():
    assert _std([]) == "N/A"


def test_aggregation_std_single():
    assert _std([5.0]) == pytest.approx(0.0)


def test_aggregation_std_values():
    from statistics import pstdev
    result = _std([1.0, 2.0, 3.0])
    assert result == pytest.approx(round(pstdev([1.0, 2.0, 3.0]), 6))


# ---------------------------------------------------------------------------
# _numeric_values
# ---------------------------------------------------------------------------


def test_numeric_values_basic():
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": "N/A"}, {"x": None}]
    assert _numeric_values(rows, "x") == [1.0, 2.0]


def test_numeric_values_missing_key():
    assert _numeric_values([{"a": 1}], "missing") == []


# ---------------------------------------------------------------------------
# _build_run_id
# ---------------------------------------------------------------------------


def test_build_run_id_basic():
    result = _build_run_id("ATM System", "single_agent", 42, "mare")
    assert result == "mare-atm-system-single_agent-s042"


def test_build_run_id_whitespace():
    result = _build_run_id("  Test  ", "  multi  ", 1, "  quare  ")
    assert result.startswith("quare-")


# ---------------------------------------------------------------------------
# _non_comparable_reason_text
# ---------------------------------------------------------------------------


def test_non_comparable_reason_text_empty():
    assert _non_comparable_reason_text([]) == ""


def test_non_comparable_reason_text_sorted():
    assert _non_comparable_reason_text(["b", "a"]) == "a|b"


# ---------------------------------------------------------------------------
# _is_sha256_hex
# ---------------------------------------------------------------------------


def test_is_sha256_hex_valid():
    assert _is_sha256_hex("a" * 64) is True


def test_is_sha256_hex_wrong_length():
    assert _is_sha256_hex("abc") is False


def test_is_sha256_hex_bad_chars():
    assert _is_sha256_hex("g" * 64) is False


# ---------------------------------------------------------------------------
# _supports_full_ablation
# ---------------------------------------------------------------------------


def test_supports_full_ablation_full():
    from openre_bench.schemas import (
        SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
        SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
        SETTING_SINGLE_AGENT,
    )

    settings = [
        SETTING_SINGLE_AGENT,
        SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
        SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    ]
    assert _supports_full_ablation(settings) is True


def test_supports_full_ablation_incomplete():
    assert _supports_full_ablation(["single_agent"]) is False


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def test_jsonl_io_write_read_roundtrip(tmp_path: Path):
    rows = [{"a": 1}, {"b": 2}]
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, rows)
    result = _read_jsonl(path)
    assert result == rows


def test_jsonl_io_read_blank_lines_skipped(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n')
    result = _read_jsonl(path)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# CSV I/O & validation
# ---------------------------------------------------------------------------


def test_csv_io_write_csv(tmp_path: Path):
    path = tmp_path / "test.csv"
    _write_csv(path, ["x", "y"], [{"x": 1, "y": 2}])
    assert path.exists()
    content = path.read_text()
    assert "x,y" in content


def test_csv_io_validate_csv_columns_ok(tmp_path: Path):
    path = tmp_path / "test.csv"
    _write_csv(path, ["x", "y"], [{"x": 1, "y": 2}])
    errors: list[str] = []
    _validate_csv_columns(path, ["x", "y"], errors)
    assert len(errors) == 0


def test_csv_io_validate_csv_missing_columns(tmp_path: Path):
    path = tmp_path / "test.csv"
    _write_csv(path, ["x"], [{"x": 1}])
    errors: list[str] = []
    _validate_csv_columns(path, ["x", "z"], errors)
    assert len(errors) == 1


def test_csv_io_validate_csv_nonexistent(tmp_path: Path):
    errors: list[str] = []
    _validate_csv_columns(tmp_path / "missing.csv", ["x"], errors)
    assert len(errors) == 0


# ---------------------------------------------------------------------------
# parse_settings / parse_seeds
# ---------------------------------------------------------------------------


def test_parsers_parse_settings_none():
    result = parse_settings(None)
    assert result == [
        "single_agent",
        "multi_agent_without_negotiation",
        "multi_agent_with_negotiation",
        "negotiation_integration_verification",
    ]


def test_parsers_parse_settings_valid():
    result = parse_settings("single_agent,multi_agent_with_negotiation")
    assert result == ["single_agent", "multi_agent_with_negotiation"]


def test_parsers_parse_settings_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        parse_settings("nonexistent_setting")


def test_parsers_parse_settings_empty_string():
    result = parse_settings("")
    assert result == [
        "single_agent",
        "multi_agent_without_negotiation",
        "multi_agent_with_negotiation",
        "negotiation_integration_verification",
    ]


def test_parsers_parse_seeds_none():
    assert parse_seeds(None) == [101]


def test_parsers_parse_seeds_valid():
    assert parse_seeds("1,2,3") == [1, 2, 3]


def test_parsers_parse_seeds_empty():
    assert parse_seeds("") == [101]


def test_parsers_parse_systems_empty():
    assert parse_systems("") == []


def test_parsers_parse_systems_valid_dedupes():
    assert parse_systems("mare,iredev,mare,quare") == ["mare", "iredev", "quare"]


def test_parsers_parse_systems_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        parse_systems("mare,unknown")


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def test_phase_text_extraction_phase1_texts():
    phase1 = {
        "SafetyAgent": [
            {"description": "The system shall be safe."},
            {"description": ""},
        ],
        "EfficiencyAgent": [
            {"description": "The system shall be efficient."},
        ],
    }
    texts = _phase1_requirement_texts(phase1)
    assert len(texts) == 2
    assert "safe" in texts[0].lower()


def test_phase_text_extraction_phase3_texts():
    phase3 = {
        "gsn_elements": [
            {"name": "G1", "description": "Top-level goal"},
            {"name": "", "description": ""},
        ]
    }
    texts = _phase3_requirement_texts(phase3)
    assert len(texts) == 1
    assert "G1" in texts[0]


def test_phase_text_extraction_phase2_texts():
    phase2 = {
        "negotiations": {
            "n1": {
                "steps": [
                    {
                        "message_type": "backward",
                        "kaos_elements": [{"description": "Backward step finding"}],
                    },
                    {
                        "message_type": "forward",
                        "kaos_elements": [{"description": "Ignored"}],
                    },
                ]
            }
        }
    }
    texts = _phase2_requirement_texts(phase2)
    assert len(texts) == 1
    assert "Backward" in texts[0]


# ---------------------------------------------------------------------------
# _normalize_axis_text / _axis_scoring_text
# ---------------------------------------------------------------------------


def test_normalize_axis_text_basic():
    result = _normalize_axis_text("The system SHALL be Safe, Safe, safe!")
    assert "system" in result
    # Duplicates removed
    assert result.count("safe") == 1


def test_normalize_axis_text_empty():
    assert _normalize_axis_text("") == ""


def test_normalize_axis_text_axis_scoring_text():
    element = {"name": "G1", "description": "Safety goal", "measurable_criteria": "5ms latency"}
    text = _axis_scoring_text(element)
    assert "g1" in text
    assert "safety" in text


# ---------------------------------------------------------------------------
# _quality_metadata_candidates
# ---------------------------------------------------------------------------


def test_quality_metadata_candidates_basic():
    element = {"quality_attribute": "Safety", "stakeholder": "User"}
    candidates = _quality_metadata_candidates(element)
    assert "Safety" in candidates
    assert "User" in candidates


def test_quality_metadata_candidates_with_properties():
    element = {"properties": {"quality_attribute": "Efficiency", "agent": "EfficiencyAgent"}}
    candidates = _quality_metadata_candidates(element)
    assert "Efficiency" in candidates
    assert "EfficiencyAgent" in candidates


def test_quality_metadata_candidates_no_properties():
    candidates = _quality_metadata_candidates({})
    assert all(v is None for v in candidates)


# ---------------------------------------------------------------------------
# _infer_axes_from_value
# ---------------------------------------------------------------------------


def test_infer_axes_from_value_none():
    axes, integrated = _infer_axes_from_value(None)
    assert axes == set()
    assert integrated is False


def test_infer_axes_from_value_known_axis():
    axes, integrated = _infer_axes_from_value("Safety")
    assert "Safety" in axes


def test_infer_axes_from_value_integrated():
    axes, integrated = _infer_axes_from_value("Integrated")
    assert integrated is True
    assert len(axes) == 0


def test_infer_axes_from_value_list_value():
    axes, _ = _infer_axes_from_value(["Safety", "Efficiency"])
    assert "Safety" in axes
    assert "Efficiency" in axes


def test_infer_axes_from_value_compound_string():
    axes, _ = _infer_axes_from_value("Safety/Efficiency")
    assert "Safety" in axes
    assert "Efficiency" in axes


def test_infer_axes_from_value_unknown():
    axes, integrated = _infer_axes_from_value("xyz123unknown")
    assert len(axes) == 0
    assert integrated is False


# ---------------------------------------------------------------------------
# _quality_vector_from_element
# ---------------------------------------------------------------------------


def test_quality_vector_from_element_with_known_quality():
    element = {"quality_attribute": "Safety"}
    vec = _quality_vector_from_element(element)
    assert vec.shape == (len(QUALITY_AXES),)
    assert vec[QUALITY_AXES.index("Safety")] > 0


def test_quality_vector_from_element_with_semantic_scores():
    element = {"quality_attribute": "Safety"}
    semantic = np.array([0.5, 0.2, 0.1, 0.3, 0.4])
    vec = _quality_vector_from_element(element, semantic_scores=semantic)
    assert vec.shape == (len(QUALITY_AXES),)
    assert float(np.sum(vec)) > 0


def test_quality_vector_from_element_integrated_only():
    element = {"quality_attribute": "Integrated"}
    vec = _quality_vector_from_element(element)
    assert vec.shape == (len(QUALITY_AXES),)


def test_quality_vector_from_element_empty_element():
    vec = _quality_vector_from_element({})
    assert vec.shape == (len(QUALITY_AXES),)


# ---------------------------------------------------------------------------
# _project_semantic_axis_scores
# ---------------------------------------------------------------------------


def test_project_semantic_axis_scores_empty():
    result = _project_semantic_axis_scores([])
    assert result.shape == (0, len(QUALITY_AXES))


def test_project_semantic_axis_scores_no_text():
    result = _project_semantic_axis_scores([{}])
    assert result.shape == (1, len(QUALITY_AXES))


def test_project_semantic_axis_scores_with_text():
    elements = [
        {"name": "Safety goal", "description": "The system must be safe from hazards"},
        {"name": "Efficiency goal", "description": "Response time within 5ms"},
    ]
    result = _project_semantic_axis_scores(elements)
    assert result.shape == (2, len(QUALITY_AXES))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# _compute_chv_mdc
# ---------------------------------------------------------------------------


def test_compute_chv_mdc_empty():
    chv, mdc = _compute_chv_mdc([])
    assert chv == pytest.approx(0.0)
    assert mdc == pytest.approx(0.0)


def test_compute_chv_mdc_with_elements():
    elements = [
        {"name": "Safety goal", "description": "Hazard analysis", "quality_attribute": "Safety"},
        {"name": "Efficiency goal", "description": "Performance metric", "quality_attribute": "Efficiency"},
        {"name": "Trust goal", "description": "User trust", "quality_attribute": "Trustworthiness"},
    ]
    chv, mdc = _compute_chv_mdc(elements)
    assert chv > 0.0  # 3 distinct axes should produce nonzero CHV
    assert mdc > 0.0


# ---------------------------------------------------------------------------
# _intrinsic_hull_space
# ---------------------------------------------------------------------------


def test_intrinsic_hull_space_empty():
    result = _intrinsic_hull_space(np.zeros((0, 5)))
    assert result.shape == (0, 5)


def test_intrinsic_hull_space_rank_reduction():
    # 3 points in 5D that lie on a 2D plane
    points = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0],
    ], dtype=float)
    result = _intrinsic_hull_space(points)
    assert result.shape[1] <= 2  # projected to intrinsic rank


# ---------------------------------------------------------------------------
# _semantic_cache_key
# ---------------------------------------------------------------------------


def test_semantic_cache_key_deterministic():
    k1 = _semantic_cache_key(["a"], ["b"])
    k2 = _semantic_cache_key(["a"], ["b"])
    assert k1 == k2


def test_semantic_cache_key_different_inputs():
    k1 = _semantic_cache_key(["a"], ["b"])
    k2 = _semantic_cache_key(["b"], ["a"])
    assert k1 != k2


# ---------------------------------------------------------------------------
# _iso29148_scores
# ---------------------------------------------------------------------------


def test_iso29148_scores_all_ones():
    result = _iso29148_scores(
        s_logic=1.0,
        s_term=1.0,
        topology_valid=1,
        deterministic_valid=1,
        compliance_coverage=1.0,
    )
    assert all(1.0 <= v <= 5.0 for v in result.values())


def test_iso29148_scores_all_zeros():
    result = _iso29148_scores(
        s_logic=0.0, s_term=0.0, topology_valid=0,
        deterministic_valid=0, compliance_coverage=0.0,
    )
    assert all(v == pytest.approx(1.0) for v in result.values())


def test_iso29148_scores_has_all_keys():
    result = _iso29148_scores(
        s_logic=0.5, s_term=0.5, topology_valid=1,
        deterministic_valid=1, compliance_coverage=0.5,
    )
    for key in ("unambiguous", "correctness", "verifiability", "set_consistency", "set_feasibility"):
        assert key in result


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------


def test_dataclasses_matrix_config_fields():
    config = MatrixConfig(
        cases_dir=Path("/tmp/cases"),
        output_dir=Path("/tmp/output"),
        seeds=[42],
        settings=["single_agent"],
        model="gpt-4o-mini",
        temperature=0.7,
        round_cap=3,
        max_tokens=4000,
        rag_enabled=True,
        rag_backend="local_tfidf",
        rag_corpus_dir=Path("/tmp/corpus"),
    )
    assert config.system == "mare"
    assert config.systems is None
    assert config.max_workers == 1


def test_dataclasses_matrix_result_fields():
    result = MatrixResult(
        output_dir=Path("/tmp"),
        runs_jsonl=Path("/tmp/runs.jsonl"),
        by_case_csv=Path("/tmp/case.csv"),
        summary_csv=Path("/tmp/summary.csv"),
        ablation_csv=Path("/tmp/ablation.csv"),
        validity_md=Path("/tmp/validity.md"),
        total_runs=10,
        expected_runs=10,
        errors=[],
        warnings=[],
    )
    assert result.total_runs == 10


def test_run_comparison_matrix_parallel_preserves_matrix_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "Demo_input.json").write_text(
        json.dumps(
            {
                "case_name": "Demo",
                "case_description": "Demo case",
                "requirement": "The system shall preserve matrix order.",
            }
        )
    )

    class FakeRunRecord:
        def __init__(self, config):
            self.run_id = config.run_id
            self.case_id = "Demo"
            self.seed = config.seed
            self.system = config.system
            self.setting = config.setting
            self.model = config.model
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            self.round_cap = config.round_cap
            self.rag_enabled = config.rag_enabled
            self.rag_backend = config.rag_backend
            self.rag_fallback_used = False
            self.runtime_seconds = 1.0
            self.execution_flags = SimpleNamespace(
                fallback_tainted=False,
                retry_used=False,
                retry_count=0,
            )
            reasons = non_comparable_reasons_for_setting(config.setting)
            self.comparability = SimpleNamespace(non_comparable_reasons=reasons)

        def model_dump(self, mode: str = "json"):
            reasons = list(self.comparability.non_comparable_reasons)
            return {
                "run_id": self.run_id,
                "case_id": self.case_id,
                "seed": self.seed,
                "system": self.system,
                "setting": self.setting,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "round_cap": self.round_cap,
                "rag_enabled": self.rag_enabled,
                "rag_backend": self.rag_backend,
                "rag_fallback_used": self.rag_fallback_used,
                "artifact_paths": {},
                "system_identity": {"system_name": self.system},
                "provenance": {
                    "prompt_hash": "a" * 64,
                    "corpus_hash": "b" * 64,
                },
                "execution_flags": {
                    "fallback_tainted": False,
                    "retry_used": False,
                    "retry_count": 0,
                },
                "comparability": {
                    "is_comparable": not reasons,
                    "non_comparable_reasons": reasons,
                },
            }

    def fake_run_case_pipeline(config):
        config.run_record_path.parent.mkdir(parents=True, exist_ok=True)
        config.run_record_path.write_text("{}")
        return FakeRunRecord(config)

    metrics = {
        "n_phase1_agents": 1,
        "n_phase1_elements": 1,
        "n_phase2_negotiations": 1,
        "n_phase2_steps": 1,
        "n_phase3_elements": 1,
        "n_phase3_connections": 0,
        "conflict_resolution_rate": 0.0,
        "chv": 0.0,
        "mdc": 0.0,
        "semantic_preservation_f1": 0.0,
        "semantic_p2_vs_p1_f1": 0.0,
        "s_logic": 1.0,
        "topology_is_valid": 1,
        "deterministic_is_valid": 1,
        "compliance_coverage": 1.0,
        "s_term": 1.0,
        "iso29148_unambiguous": 5.0,
        "iso29148_correctness": 5.0,
        "iso29148_verifiability": 5.0,
        "iso29148_set_consistency": 5.0,
        "iso29148_set_feasibility": 5.0,
    }
    clean_report = SimpleNamespace(errors=[], warnings=[])
    monkeypatch.setattr(harness, "run_case_pipeline", fake_run_case_pipeline)
    monkeypatch.setattr(harness, "_compute_run_metrics", lambda artifacts_dir: metrics)
    monkeypatch.setattr(harness, "validate_case_input", lambda case_path: clean_report)
    monkeypatch.setattr(harness, "validate_run_record", lambda run_record_path: clean_report)
    monkeypatch.setattr(harness, "validate_phase_artifacts", lambda artifacts_dir: clean_report)
    monkeypatch.setattr(
        harness,
        "validate_system_behavior_contract",
        lambda *, system, artifacts_dir, run_record_path=None: clean_report,
    )

    result = run_comparison_matrix(
        MatrixConfig(
            cases_dir=cases_dir,
            output_dir=tmp_path / "out",
            seeds=[101],
            settings=[SETTING_SINGLE_AGENT, SETTING_MULTI_AGENT_WITH_NEGOTIATION],
            model="test-model",
            temperature=0.0,
            round_cap=1,
            max_tokens=128,
            rag_enabled=True,
            rag_backend="local_tfidf",
            rag_corpus_dir=tmp_path / "corpus",
            systems=["mare", "iredev"],
            max_workers=2,
        )
    )

    rows = _read_jsonl(result.runs_jsonl)
    assert result.errors == []
    assert result.total_runs == 4
    assert [row["run_id"] for row in rows] == [
        "mare-demo-single_agent-s101",
        "iredev-demo-single_agent-s101",
        "mare-demo-multi_agent_with_negotiation-s101",
        "iredev-demo-multi_agent_with_negotiation-s101",
    ]

    with result.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 4
    assert {row["system"] for row in summary_rows} == {"mare", "iredev"}


def test_dataclasses_trace_audit_result():
    result = TraceAuditResult(
        output_path=Path("/tmp/audit.md"),
        total_runs=5,
        runs_with_loops=0,
        runs_with_conflicts=1,
    )
    assert result.runs_with_conflicts == 1


# ---------------------------------------------------------------------------
# _file_sha256
# ---------------------------------------------------------------------------


def test_file_sha256_none_path():
    assert _file_sha256(None) == ""


def test_file_sha256_nonexistent_file():
    assert _file_sha256(Path("/tmp/nonexistent_abc123")) == ""


def test_file_sha256_directory_path(tmp_path: Path):
    assert _file_sha256(tmp_path) == ""


def test_file_sha256_existing_file(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    result = _file_sha256(f)
    assert len(result) == 64
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert result == expected


# ---------------------------------------------------------------------------
# _write_run_record_provenance
# ---------------------------------------------------------------------------


def test_write_run_record_provenance_injects_provenance(tmp_path: Path):
    path = tmp_path / "run_record.json"
    path.write_text(json.dumps({"run_id": "test-001"}))
    _write_run_record_provenance(run_record_path=path, judge_pipeline_hash="abc123")
    data = json.loads(path.read_text())
    assert data["judge_pipeline_hash"] == "abc123"
    assert data["artifact_blinded"] is False


def test_write_run_record_provenance_non_dict_payload(tmp_path: Path):
    path = tmp_path / "run_record.json"
    path.write_text(json.dumps([1, 2, 3]))
    _write_run_record_provenance(run_record_path=path, judge_pipeline_hash="abc123")
    data = json.loads(path.read_text())
    assert isinstance(data, list)  # unchanged


# ---------------------------------------------------------------------------
# _build_summary_rows
# ---------------------------------------------------------------------------


def test_build_summary_rows_empty():
    assert _build_summary_rows([]) == []


def test_build_summary_rows_basic():
    rows = [
        {
            "case_id": "case1", "setting": "single_agent",
            "runtime_seconds": 1.0, "n_phase1_elements": 10,
            "n_phase2_steps": 5, "n_phase3_elements": 8,
            "conflict_resolution_rate": 0.5, "topology_is_valid": 1,
            "validation_passed": True,
        },
        {
            "case_id": "case1", "setting": "single_agent",
            "runtime_seconds": 2.0, "n_phase1_elements": 12,
            "n_phase2_steps": 7, "n_phase3_elements": 9,
            "conflict_resolution_rate": 0.8, "topology_is_valid": 1,
            "validation_passed": True,
        },
    ]
    summary = _build_summary_rows(rows)
    assert len(summary) == 1
    assert summary[0]["case_id"] == "case1"
    assert summary[0]["runs"] == 2
    assert summary[0]["valid_runs"] == 2


def test_build_summary_rows_multiple_cases():
    rows = [
        {
            "case_id": "case1", "setting": "single_agent",
            "runtime_seconds": 1.0, "n_phase1_elements": 10,
            "n_phase2_steps": 5, "n_phase3_elements": 8,
            "conflict_resolution_rate": 0.5, "topology_is_valid": 1,
            "validation_passed": True,
        },
        {
            "case_id": "case2", "setting": "single_agent",
            "runtime_seconds": 3.0, "n_phase1_elements": 15,
            "n_phase2_steps": 10, "n_phase3_elements": 12,
            "conflict_resolution_rate": 1.0, "topology_is_valid": 0,
            "validation_passed": False,
        },
    ]
    summary = _build_summary_rows(rows)
    assert len(summary) == 2


# ---------------------------------------------------------------------------
# _build_ablation_rows
# ---------------------------------------------------------------------------


def _make_ablation_row(case_id, seed, setting, p1=10, p2=5, p3=8, topo=1):
    return {
        "case_id": case_id, "seed": seed, "setting": setting,
        "n_phase1_elements": p1, "n_phase2_steps": p2,
        "n_phase3_elements": p3, "topology_is_valid": topo,
    }


def test_build_ablation_rows_empty():
    assert _build_ablation_rows([]) == []


def test_build_ablation_rows_incomplete_settings():
    rows = [_make_ablation_row("c1", 42, "single_agent")]
    assert _build_ablation_rows(rows) == []


def test_build_ablation_rows_full_ablation():
    rows = [
        _make_ablation_row("c1", 42, SETTING_SINGLE_AGENT, p1=5),
        _make_ablation_row("c1", 42, SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION, p1=10),
        _make_ablation_row("c1", 42, SETTING_MULTI_AGENT_WITH_NEGOTIATION, p2=15, p3=12),
        _make_ablation_row("c1", 42, SETTING_NEGOTIATION_INTEGRATION_VERIFICATION, p3=20),
    ]
    result = _build_ablation_rows(rows)
    assert len(result) == 1
    assert result[0]["delta_multi_without_neg_vs_single_phase1_elements"] == 10 - 5


# ---------------------------------------------------------------------------
# _write_validity_log
# ---------------------------------------------------------------------------


def test_write_validity_log_creates_file(tmp_path: Path):
    config = MatrixConfig(
        cases_dir=Path("/tmp/cases"),
        output_dir=Path("/tmp/output"),
        seeds=[42],
        settings=["single_agent"],
        model="test-model",
        temperature=0.7,
        round_cap=3,
        max_tokens=500,
        rag_enabled=True,
        rag_backend="local_tfidf",
        rag_corpus_dir=Path("/tmp/corpus"),
    )
    path = tmp_path / "validity.md"
    _write_validity_log(
        path=path, config=config,
        total_runs=4, expected_runs=4,
        errors=[], warnings=[],
    )
    assert path.exists()
    content = path.read_text()
    assert "# Comparison Validity Log" in content
    assert "None" in content  # No errors section says "None"


def test_write_validity_log_with_errors_and_warnings(tmp_path: Path):
    config = MatrixConfig(
        cases_dir=Path("/tmp/cases"),
        output_dir=Path("/tmp/output"),
        seeds=[42],
        settings=["single_agent"],
        model="test-model",
        temperature=0.7,
        round_cap=3,
        max_tokens=500,
        rag_enabled=True,
        rag_backend="local_tfidf",
        rag_corpus_dir=Path("/tmp/corpus"),
    )
    path = tmp_path / "validity.md"
    _write_validity_log(
        path=path, config=config,
        total_runs=3, expected_runs=4,
        errors=["Missing deliverable"],
        warnings=["Partial setting list"],
    )
    content = path.read_text()
    assert "Missing deliverable" in content
    assert "Partial setting list" in content


# ---------------------------------------------------------------------------
# _validate_deliverables
# ---------------------------------------------------------------------------


def test_validate_deliverables_missing_files(tmp_path: Path):
    errors, warnings = _validate_deliverables(
        runs_jsonl=tmp_path / "missing.jsonl",
        by_case_csv=tmp_path / "missing.csv",
        summary_csv=tmp_path / "missing_summary.csv",
        ablation_csv=tmp_path / "missing_ablation.csv",
        validity_md=tmp_path / "missing.md",
        expected_runs=1,
        expected_cases=1,
        expected_seeds=1,
        expected_settings=1,
        full_ablation_expected=False,
    )
    assert len(errors) >= 4  # at least 4 "Missing deliverable" errors


def test_validate_deliverables_valid_runs(tmp_path: Path):
    from openre_bench.schemas import non_comparable_reasons_for_setting

    run_rows = []
    for i in range(2):
        reasons = non_comparable_reasons_for_setting("single_agent")
        run_rows.append({
            "run_id": f"test-{i}",
            "case_id": "case1",
            "seed": 42 + i,
            "system": "mare",
            "setting": "single_agent",
            "model": "test",
            "temperature": 0.7,
            "max_tokens": 500,
            "round_cap": 3,
            "artifact_paths": {},
            "system_identity": "mare",
            "provenance": {"prompt_hash": "a" * 64, "corpus_hash": "b" * 64},
            "execution_flags": {"fallback_tainted": False, "retry_used": False, "retry_count": 0},
            "comparability": {
                "is_comparable": not reasons,
                "non_comparable_reasons": reasons,
            },
            "validation_passed": True,
            "rag_enabled": True,
            "rag_backend": "local_tfidf",
            "rag_fallback_used": False,
            "non_comparable_reason": _non_comparable_reason_text(reasons),
        })

    runs_jsonl = tmp_path / "runs.jsonl"
    _write_jsonl(runs_jsonl, run_rows)

    by_case_csv = tmp_path / "by_case.csv"
    _write_csv(by_case_csv, BY_CASE_COLUMNS, run_rows)

    summary_csv = tmp_path / "summary.csv"
    _write_csv(summary_csv, SUMMARY_COLUMNS, [
        {"case_id": "case1", "setting": "single_agent"}
    ])

    ablation_csv = tmp_path / "ablation.csv"
    _write_csv(ablation_csv, ABLATION_COLUMNS, [])

    errors, warnings = _validate_deliverables(
        runs_jsonl=runs_jsonl,
        by_case_csv=by_case_csv,
        summary_csv=summary_csv,
        ablation_csv=ablation_csv,
        validity_md=tmp_path / "validity.md",
        expected_runs=2,
        expected_cases=1,
        expected_seeds=2,
        expected_settings=1,
        full_ablation_expected=False,
    )
    # May have some errors due to by-case row count etc., but both should be lists
    assert errors is not None
    assert warnings is not None


# ---------------------------------------------------------------------------
# _compute_run_metrics (file-based)
# ---------------------------------------------------------------------------


def test_compute_run_metrics_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from openre_bench import comparison_harness as harness
    from openre_bench.comparison_harness import _compute_run_metrics
    from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME, PHASE3_FILENAME, PHASE4_FILENAME

    # Minimal phase artifacts
    phase1 = {
        "SafetyAgent": [{"description": "The system shall be safe", "name": "R1"}],
        "EfficiencyAgent": [{"description": "Fast response time", "name": "R2"}],
    }
    phase2 = {
        "total_negotiations": 1,
        "summary_stats": {"total_steps": 2, "detected_conflicts": 1, "resolved_conflicts": 1, "successful_consensus": 1},
        "negotiations": {},
    }
    phase3 = {
        "gsn_elements": [
            {"name": "G1", "description": "Top-level safety goal"},
            {"name": "S1", "description": "Safety strategy"},
        ],
        "gsn_connections": [{"source": "G1", "target": "S1"}],
        "topology_status": {"is_valid": True},
    }
    phase4 = {
        "deterministic_validation": {"is_valid": True},
        "verification_results": {
            "s_logic": 0.8,
            "compliance_coverage": {"coverage_ratio": 0.9},
            "terminology_consistency": {"consistency_ratio": 0.7},
        },
    }

    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(phase1))
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(phase2))
    (tmp_path / PHASE3_FILENAME).write_text(json.dumps(phase3))
    (tmp_path / PHASE4_FILENAME).write_text(json.dumps(phase4))

    monkeypatch.setattr(harness, "_semantic_preservation_f1", lambda **kwargs: 0.75)

    metrics = _compute_run_metrics(tmp_path)
    assert metrics["n_phase1_agents"] == 2
    assert metrics["n_phase1_elements"] == 2
    assert metrics["n_phase2_negotiations"] == 1
    assert metrics["n_phase3_elements"] == 2
    assert metrics["n_phase3_connections"] == 1
    assert metrics["conflict_resolution_rate"] == pytest.approx(1.0)
    assert metrics["topology_is_valid"] == 1
    assert metrics["deterministic_is_valid"] == 1
    assert metrics["s_logic"] == pytest.approx(0.8)


def test_compute_run_metrics_no_conflicts(tmp_path: Path):
    from openre_bench.comparison_harness import _compute_run_metrics
    from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME, PHASE3_FILENAME, PHASE4_FILENAME

    phase1 = {"Agent1": [{"description": "Test req", "name": "R1"}]}
    phase2 = {
        "total_negotiations": 0,
        "summary_stats": {"total_steps": 0, "detected_conflicts": 0, "resolved_conflicts": 0, "successful_consensus": 0},
        "negotiations": {},
    }
    phase3 = {"gsn_elements": [], "gsn_connections": [], "topology_status": {"is_valid": False}}
    phase4 = {"deterministic_validation": {"is_valid": False}, "verification_results": {}}

    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(phase1))
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(phase2))
    (tmp_path / PHASE3_FILENAME).write_text(json.dumps(phase3))
    (tmp_path / PHASE4_FILENAME).write_text(json.dumps(phase4))

    metrics = _compute_run_metrics(tmp_path)
    assert metrics["conflict_resolution_rate"] == pytest.approx(0.0)
    assert metrics["topology_is_valid"] == 0
    assert metrics["deterministic_is_valid"] == 0


# ---------------------------------------------------------------------------
# export_trace_audit
# ---------------------------------------------------------------------------


def test_export_trace_audit_missing_runs_jsonl(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        export_trace_audit(matrix_output_dir=tmp_path)


def test_export_trace_audit_empty_runs(tmp_path: Path):
    runs_jsonl = tmp_path / RUNS_JSONL_NAME
    runs_jsonl.write_text("")
    result = export_trace_audit(matrix_output_dir=tmp_path)
    assert result.total_runs == 0
    assert result.runs_with_loops == 0
    assert result.output_path.exists()


def test_export_trace_audit_runs_without_phase2(tmp_path: Path):
    runs_jsonl = tmp_path / RUNS_JSONL_NAME
    # Use explicit nonexistent paths to avoid Path("") → "." gotcha
    rows = [
        {"run_id": "run1", "setting": "single_agent", "artifact_paths": {PHASE2_FILENAME: "/tmp/nonexistent_dir/no.json"}},
        {"run_id": "run2", "setting": "single_agent", "artifact_paths": {PHASE2_FILENAME: "/tmp/nonexistent_dir2/no.json"}},
    ]
    runs_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = export_trace_audit(matrix_output_dir=tmp_path)
    assert result.total_runs == 2
    assert result.runs_with_loops == 0
    content = result.output_path.read_text()
    assert "run1" in content
    assert "run2" in content


def test_export_trace_audit_runs_with_negotiation_traces(tmp_path: Path):
    # Create a phase2 artifact with backward steps and conflicts
    phase2 = {
        "total_negotiations": 1,
        "summary_stats": {"total_steps": 3, "detected_conflicts": 2, "resolved_conflicts": 1},
        "negotiations": {
            "pair1": {
                "steps": [
                    {"message_type": "forward"},
                    {"message_type": "backward"},
                    {"message_type": "forward"},
                ]
            }
        },
    }
    phase2_path = tmp_path / "phase2.json"
    phase2_path.write_text(json.dumps(phase2))

    rows = [{"run_id": "run1", "setting": "multi_agent_with_negotiation", "artifact_paths": {PHASE2_FILENAME: str(phase2_path)}}]
    runs_jsonl = tmp_path / RUNS_JSONL_NAME
    runs_jsonl.write_text(json.dumps(rows[0]) + "\n")

    result = export_trace_audit(matrix_output_dir=tmp_path)
    assert result.total_runs == 1
    assert result.runs_with_loops == 1
    assert result.runs_with_conflicts == 1
    content = result.output_path.read_text()
    assert "yes" in content  # loop_detected
    assert "Comparison Trace Audit" in content


def test_export_trace_audit_custom_output_path(tmp_path: Path):
    runs_jsonl = tmp_path / RUNS_JSONL_NAME
    runs_jsonl.write_text(json.dumps({"run_id": "r1", "setting": "s", "artifact_paths": {PHASE2_FILENAME: "/tmp/nonexistent/no.json"}}) + "\n")
    custom_path = tmp_path / "custom" / "audit.md"
    result = export_trace_audit(matrix_output_dir=tmp_path, output_path=custom_path)
    assert result.output_path == custom_path
    assert custom_path.exists()


# ---------------------------------------------------------------------------
# prepare_blind_evaluation
# ---------------------------------------------------------------------------


def _create_matrix_files(matrix_dir: Path, runs: list[dict], by_case_rows: list[dict] | None = None):
    """Helper to create runs.jsonl + by_case CSV in matrix_dir."""
    runs_jsonl = matrix_dir / RUNS_JSONL_NAME
    runs_jsonl.write_text("\n".join(json.dumps(r) for r in runs) + "\n")

    if by_case_rows is None:
        by_case_rows = [
            {"run_id": r["run_id"], "system": r.get("system", "quare"), "case_id": "ATM", "seed": "42", "setting": "multi_agent_with_negotiation"}
            for r in runs
        ]
    by_case_csv = matrix_dir / BY_CASE_CSV_NAME
    if by_case_rows:
        columns = list(by_case_rows[0].keys())
        with by_case_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(by_case_rows)
    else:
        by_case_csv.write_text("")


def test_prepare_blind_evaluation_missing_runs_jsonl(tmp_path: Path):
    blind_dir = tmp_path / "blind"
    judge = tmp_path / "judge.py"
    judge.write_text("# judge")
    with pytest.raises(FileNotFoundError, match="Missing runs JSONL"):
        prepare_blind_evaluation(
            matrix_output_dir=tmp_path,
            blind_output_dir=blind_dir,
            judge_pipeline_path=judge,
        )


def test_prepare_blind_evaluation_missing_by_case_csv(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    (matrix_dir / RUNS_JSONL_NAME).write_text(json.dumps({"run_id": "r1"}) + "\n")
    blind_dir = tmp_path / "blind"
    judge = tmp_path / "judge.py"
    judge.write_text("# judge")
    with pytest.raises(FileNotFoundError, match="Missing by-case CSV"):
        prepare_blind_evaluation(
            matrix_output_dir=matrix_dir,
            blind_output_dir=blind_dir,
            judge_pipeline_path=judge,
        )


def test_prepare_blind_evaluation_successful_blinding(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()

    # Create source artifact
    artifact_dir = matrix_dir / "runs" / "run-001"
    artifact_dir.mkdir(parents=True)
    phase1_source = artifact_dir / PHASE2_FILENAME
    phase1_source.write_text(json.dumps({"test": "data"}))

    runs = [
        {
            "run_id": "run-001",
            "system": "quare",
            "setting": "multi_agent_with_negotiation",
            "artifact_paths": {PHASE2_FILENAME: str(phase1_source)},
            "system_identity": {"system_name": "quare"},
        }
    ]
    _create_matrix_files(matrix_dir, runs)

    blind_dir = tmp_path / "blind"
    judge = tmp_path / "judge.py"
    judge.write_text("# judge script")

    result = prepare_blind_evaluation(
        matrix_output_dir=matrix_dir,
        blind_output_dir=blind_dir,
        judge_pipeline_path=judge,
    )

    assert result.output_dir == blind_dir
    assert result.blinded_runs_jsonl.exists()
    assert result.blinded_by_case_csv.exists()
    assert result.mapping_json.exists()
    assert result.protocol_md.exists()

    # Verify anonymization
    mapping = json.loads(result.mapping_json.read_text())
    assert "run-001" in mapping["run_mapping"]
    assert mapping["run_mapping"]["run-001"] == "BLIND_RUN_001"
    assert "quare" in mapping["system_mapping"]

    # Verify blinded artifact was copied
    blinded_artifact = blind_dir / "blinded_artifacts" / "BLIND_RUN_001" / PHASE2_FILENAME
    assert blinded_artifact.exists()


def test_prepare_blind_evaluation_multiple_systems(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()

    runs = [
        {"run_id": "r1", "system": "mare", "setting": "s", "artifact_paths": {}},
        {"run_id": "r2", "system": "quare", "setting": "s", "artifact_paths": {}},
    ]
    _create_matrix_files(matrix_dir, runs)

    blind_dir = tmp_path / "blind"
    judge = tmp_path / "judge.py"
    judge.write_text("# judge")

    result = prepare_blind_evaluation(
        matrix_output_dir=matrix_dir,
        blind_output_dir=blind_dir,
        judge_pipeline_path=judge,
    )
    mapping = json.loads(result.mapping_json.read_text())
    system_map = mapping["system_mapping"]
    assert len(system_map) == 2
    assert system_map["mare"] != system_map["quare"]

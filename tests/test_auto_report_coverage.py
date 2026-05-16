"""Coverage tests for openre_bench.auto_report — pure utility functions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from openre_bench.auto_report import (
    AutoReportConfig,
    AutoReportResult,
    KEY_DELTA_METRICS,
    RUNS_JSONL_NAME,
    UNSPECIFIED_SYSTEM,
    _Logger,
    _agent_log_paths,
    _artifact_phase1_requirement_texts,
    _artifact_phase2_requirement_texts,
    _audit_precision_f1_contract,
    _build_manifest,
    _build_run_key,
    _canonical_phase_label,
    _canonical_setting_label,
    _canonical_system_label,
    _coerce_text,
    _collect_system_stats,
    _compute_key_deltas,
    _compute_literal_precision_f1_gains,
    _conversation_bundle_is_fresh,
    _delta,
    _derive_precision_f1_gains_from_artifacts,
    _evaluate_paper_control_profile,
    _extract_phase_metrics,
    _file_sha256,
    _fmt,
    _infer_setting_from_text,
    _infer_system_from_run_id,
    _infer_system_hint_from_path,
    _ingest_literal_metric_node,
    _literal_phase_setting_compatible,
    _llm_seed_reproducibility_incomplete,
    _load_case_requirement_units,
    _load_ground_truth_case_ids,
    _log_matrix_snapshot,
    _matrix_outputs_complete,
    _mean_metric,
    _mirror_latest_outputs,
    _normalize_case_id,
    _phase1_agents_for_logging,
    _phase2_agents_for_logging,
    _phase2_steps_for_logging,
    _read_csv_rows,
    _read_jsonl_rows,
    _relative_gain,
    _render_agent_markdown,
    _render_timeline_markdown,
    _rewrite_readme_for_report_root,
    _runtime_semantics_string_list,
    _sanitize_component,
    _semantic_precision_recall_f1,
    _split_requirement_units,
    _state_text,
    _to_float,
    _to_int,
    _token_overlap_f1,
    _write_analysis_md,
    _write_literal_phase_metrics,
    _write_report_readme,
    PAPER_MODEL,
    PAPER_SEEDS,
    PAPER_SETTINGS,
    PAPER_TEMPERATURE,
)
from openre_bench.schemas import PHASE1_FILENAME, PHASE2_FILENAME


# ---------------------------------------------------------------------------
# _to_float / _to_int
# ---------------------------------------------------------------------------


def test_auto_report_coercions_float_none():
    assert _to_float(None) is None


def test_auto_report_coercions_float_empty():
    assert _to_float("") is None


def test_auto_report_coercions_float_na():
    assert _to_float("N/A") is None


def test_auto_report_coercions_float_valid():
    assert _to_float(3.14) == pytest.approx(3.14)


def test_auto_report_coercions_float_str():
    assert _to_float("2.5") == pytest.approx(2.5)


def test_auto_report_coercions_float_bool():
    assert _to_float(True) == pytest.approx(1.0)
    assert _to_float(False) == pytest.approx(0.0)


def test_auto_report_coercions_float_junk():
    assert _to_float("abc") is None


def test_auto_report_coercions_int_none():
    assert _to_int(None, default=99) == 99


def test_auto_report_coercions_int_valid():
    assert _to_int(42, default=0) == 42
    assert _to_int("7", default=0) == 7


def test_auto_report_coercions_int_bool():
    assert _to_int(True, default=0) == 1
    assert _to_int(False, default=0) == 0


def test_auto_report_coercions_int_float():
    assert _to_int(3.7, default=0) == 3


def test_auto_report_coercions_int_junk():
    assert _to_int("abc", default=-1) == -1


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------


def test_fmt_none():
    assert _fmt(None) == "N/A"


def test_fmt_float():
    assert _fmt(1.23456789) == "1.234568"


# ---------------------------------------------------------------------------
# _delta
# ---------------------------------------------------------------------------


def test_delta_both_present():
    assert _delta(3.0, 1.0) == pytest.approx(2.0)


def test_delta_left_none():
    assert _delta(None, 1.0) is None


def test_delta_right_none():
    assert _delta(3.0, None) is None


# ---------------------------------------------------------------------------
# _mean_metric
# ---------------------------------------------------------------------------


def test_mean_metric_empty():
    assert _mean_metric([], "x") is None


def test_mean_metric_all_na():
    assert _mean_metric([{"x": "N/A"}, {"x": None}], "x") is None


def test_mean_metric_valid():
    rows: list[dict[str, Any]] = [{"x": 1.0}, {"x": 3.0}]
    assert _mean_metric(rows, "x") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _coerce_text / _state_text / _sanitize_component
# ---------------------------------------------------------------------------


def test_text_helpers_coerce_none():
    assert _coerce_text(None) == ""
    assert _coerce_text(None, fallback="?") == "?"


def test_text_helpers_coerce_blank():
    assert _coerce_text("  ") == ""


def test_text_helpers_coerce_valid():
    assert _coerce_text("  hello  ") == "hello"


def test_text_helpers_state_true():
    assert _state_text(True) == "true"


def test_text_helpers_state_false():
    assert _state_text(False) == "false"


def test_text_helpers_state_none():
    assert _state_text(None) == "N/A"


def test_text_helpers_state_str():
    assert _state_text("maybe") == "N/A"


def test_text_helpers_sanitize_basic():
    assert _sanitize_component("Hello World!") == "Hello-World"


def test_text_helpers_sanitize_empty():
    assert _sanitize_component("!!!") == "unknown"


def test_text_helpers_sanitize_safe():
    assert _sanitize_component("file_name.txt") == "file_name.txt"


# ---------------------------------------------------------------------------
# _build_run_key
# ---------------------------------------------------------------------------


def test_build_run_key_deterministic():
    controls = {"model": "gpt-4o-mini", "seeds": [42], "settings": ["single_agent"]}
    k1 = _build_run_key(controls)
    k2 = _build_run_key(controls)
    assert k1 == k2
    assert k1.startswith("auto-")
    assert len(k1) == len("auto-") + 12


def test_build_run_key_different_controls():
    k1 = _build_run_key({"a": 1})
    k2 = _build_run_key({"a": 2})
    assert k1 != k2


# ---------------------------------------------------------------------------
# _normalize_case_id
# ---------------------------------------------------------------------------


def test_normalize_case_id_basic():
    assert _normalize_case_id("ATM System") == "ATMSYSTEM"


def test_normalize_case_id_empty():
    assert _normalize_case_id("") is None


def test_normalize_case_id_none():
    assert _normalize_case_id(None) is None


def test_normalize_case_id_special_chars():
    assert _normalize_case_id("case-1 (test)") == "CASE1TEST"


def test_normalize_case_id_all_special():
    assert _normalize_case_id("---") is None


# ---------------------------------------------------------------------------
# _canonical_phase_label
# ---------------------------------------------------------------------------


def test_canonical_phase_label_phase1_variants():
    assert _canonical_phase_label("phase1") == "phase1"
    assert _canonical_phase_label("Phase 1") == "phase1"
    assert _canonical_phase_label("p1") == "phase1"
    assert _canonical_phase_label("1") == "phase1"
    assert _canonical_phase_label("initial") == "phase1"
    assert _canonical_phase_label("generation") == "phase1"


def test_canonical_phase_label_phase2_variants():
    assert _canonical_phase_label("phase2") == "phase2"
    assert _canonical_phase_label("Phase 2") == "phase2"
    assert _canonical_phase_label("p2") == "phase2"
    assert _canonical_phase_label("2") == "phase2"
    assert _canonical_phase_label("negotiation") == "phase2"
    assert _canonical_phase_label("dialectic") == "phase2"


def test_canonical_phase_label_unknown():
    assert _canonical_phase_label("phase3") is None
    assert _canonical_phase_label("") is None
    assert _canonical_phase_label(None) is None


# ---------------------------------------------------------------------------
# _canonical_system_label
# ---------------------------------------------------------------------------


def test_canonical_system_label_mare():
    assert _canonical_system_label("mare") == "mare"


def test_canonical_system_label_quare():
    assert _canonical_system_label("quare") == "quare"


def test_canonical_system_label_unknown():
    assert _canonical_system_label("xyzzy") is None
    assert _canonical_system_label(None) is None


# ---------------------------------------------------------------------------
# _canonical_setting_label
# ---------------------------------------------------------------------------


def test_canonical_setting_label_single():
    assert _canonical_setting_label("SingleAgent") == "single_agent"


def test_canonical_setting_label_multi_without():
    result = _canonical_setting_label("MultiAgentWithoutNegotiation")
    assert result == "multi_agent_without_negotiation"


def test_canonical_setting_label_unknown():
    assert _canonical_setting_label("random_str") is None


# ---------------------------------------------------------------------------
# _infer_system_from_run_id
# ---------------------------------------------------------------------------


def test_infer_system_from_run_id_mare_prefix():
    assert _infer_system_from_run_id("mare-atm-s042") == "mare"


def test_infer_system_from_run_id_quare_prefix():
    assert _infer_system_from_run_id("quare-atm-s042") == "quare"


def test_infer_system_from_run_id_embedded():
    assert _infer_system_from_run_id("run-mare-case1") == "mare"


def test_infer_system_from_run_id_unknown():
    assert _infer_system_from_run_id("unknown-run-id") is None


def test_infer_system_from_run_id_empty():
    assert _infer_system_from_run_id("") is None
    assert _infer_system_from_run_id(None) is None


# ---------------------------------------------------------------------------
# _infer_setting_from_text
# ---------------------------------------------------------------------------


def test_infer_setting_from_text_direct():
    assert _infer_setting_from_text("single_agent") == "single_agent"


def test_infer_setting_from_text_embedded():
    assert _infer_setting_from_text("mare-single_agent-s042") == "single_agent"


def test_infer_setting_from_text_canonical_fallback():
    assert _infer_setting_from_text("SingleAgent") == "single_agent"


def test_infer_setting_from_text_unknown():
    assert _infer_setting_from_text("random") is None


def test_infer_setting_from_text_empty():
    assert _infer_setting_from_text("") is None


# ---------------------------------------------------------------------------
# _infer_system_hint_from_path
# ---------------------------------------------------------------------------


def test_infer_system_hint_from_path_from_parts():
    assert _infer_system_hint_from_path(Path("/tmp/report/mare/runs/file.json")) == "mare"


def test_infer_system_hint_from_path_from_filename():
    assert _infer_system_hint_from_path(Path("/tmp/report/quare-atm-s042.json")) == "quare"


# ---------------------------------------------------------------------------
# _split_requirement_units
# ---------------------------------------------------------------------------


def test_split_requirement_units_basic():
    text = "The system shall be safe. The system shall be efficient!"
    result = _split_requirement_units(text)
    assert len(result) == 2


def test_split_requirement_units_short_chunks_filtered():
    result = _split_requirement_units("Short. Also short.")
    assert len(result) == 0


def test_split_requirement_units_empty():
    assert _split_requirement_units("") == []


# ---------------------------------------------------------------------------
# _token_overlap_f1
# ---------------------------------------------------------------------------


def test_token_overlap_f1_identical():
    assert _token_overlap_f1(left="hello world", right="hello world") == pytest.approx(1.0)


def test_token_overlap_f1_partial():
    result = _token_overlap_f1(left="hello world", right="hello there")
    assert 0.0 < result < 1.0


def test_token_overlap_f1_no_overlap():
    assert _token_overlap_f1(left="abc", right="xyz") == pytest.approx(0.0)


def test_token_overlap_f1_empty():
    assert _token_overlap_f1(left="", right="test") == pytest.approx(0.0)
    assert _token_overlap_f1(left="test", right="") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _relative_gain
# ---------------------------------------------------------------------------


def test_relative_gain_basic():
    assert _relative_gain(1.5, 1.0) == pytest.approx(0.5)


def test_relative_gain_none_current():
    assert _relative_gain(None, 1.0) is None


def test_relative_gain_none_baseline():
    assert _relative_gain(1.0, None) is None


def test_relative_gain_zero_baseline():
    assert _relative_gain(1.0, 0.0) is None


# ---------------------------------------------------------------------------
# _semantic_precision_recall_f1
# ---------------------------------------------------------------------------


def test_semantic_precision_recall_f1_empty_candidates():
    assert _semantic_precision_recall_f1(candidates=[], references=["a"]) is None


def test_semantic_precision_recall_f1_empty_references():
    assert _semantic_precision_recall_f1(candidates=["a"], references=[]) is None


def test_semantic_precision_recall_f1_identical():
    result = _semantic_precision_recall_f1(
        candidates=["hello world test foo"],
        references=["hello world test foo"],
    )
    assert result is not None
    precision, recall, f1 = result
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _extract_phase_metrics
# ---------------------------------------------------------------------------


def test_extract_phase_metrics_direct_precision():
    result = _extract_phase_metrics({"precision": 0.85, "f1": 0.9})
    assert result["precision"] == pytest.approx(0.85)
    assert result["f1"] == pytest.approx(0.9)


def test_extract_phase_metrics_nested_metrics():
    result = _extract_phase_metrics({"metrics": {"precision": 0.7, "f1": 0.75}})
    assert result["precision"] == pytest.approx(0.7)
    assert result["f1"] == pytest.approx(0.75)


def test_extract_phase_metrics_tp_fp_fn():
    result = _extract_phase_metrics({"tp": 8, "fp": 2, "fn": 1})
    assert result["precision"] == pytest.approx(0.8)
    assert "f1" in result


def test_extract_phase_metrics_empty():
    assert _extract_phase_metrics({}) == {}


def test_extract_phase_metrics_alternative_keys():
    result = _extract_phase_metrics({"prec": 0.6, "f1_score": 0.7})
    assert result["precision"] == pytest.approx(0.6)
    assert result["f1"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _write_literal_phase_metrics
# ---------------------------------------------------------------------------


def test_write_literal_phase_metrics_write_new():
    output: dict = {}
    _write_literal_phase_metrics(
        output=output, system="mare", case_id="case1", seed=42,
        phase="phase1", metrics={"precision": 0.8, "f1": 0.9},
    )
    assert output[("mare", "case1", 42)]["phase1"]["precision"] == pytest.approx(0.8)


def test_write_literal_phase_metrics_negative_seed_skipped():
    output: dict = {}
    _write_literal_phase_metrics(
        output=output, system="mare", case_id="case1", seed=-1,
        phase="phase1", metrics={"precision": 0.8},
    )
    assert len(output) == 0


def test_write_literal_phase_metrics_invalid_phase_skipped():
    output: dict = {}
    _write_literal_phase_metrics(
        output=output, system="mare", case_id="case1", seed=42,
        phase="phase3", metrics={"precision": 0.8},
    )
    assert len(output) == 0


def test_write_literal_phase_metrics_keeps_lower_value():
    output: dict = {}
    _write_literal_phase_metrics(
        output=output, system="mare", case_id="case1", seed=42,
        phase="phase1", metrics={"precision": 0.8},
    )
    _write_literal_phase_metrics(
        output=output, system="mare", case_id="case1", seed=42,
        phase="phase1", metrics={"precision": 0.6},
    )
    assert output[("mare", "case1", 42)]["phase1"]["precision"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _literal_phase_setting_compatible
# ---------------------------------------------------------------------------


def test_literal_phase_setting_compatible_none_setting():
    assert _literal_phase_setting_compatible(phase="phase1", setting=None) is True


def test_literal_phase_setting_compatible_phase1_correct():
    assert _literal_phase_setting_compatible(
        phase="phase1", setting="multi_agent_without_negotiation"
    ) is True


def test_literal_phase_setting_compatible_phase1_wrong():
    assert _literal_phase_setting_compatible(
        phase="phase1", setting="single_agent"
    ) is False


def test_literal_phase_setting_compatible_phase2_correct():
    assert _literal_phase_setting_compatible(
        phase="phase2", setting="multi_agent_with_negotiation"
    ) is True


def test_literal_phase_setting_compatible_unknown_phase_none_setting():
    # None setting always returns True regardless of phase
    assert _literal_phase_setting_compatible(phase="phase3", setting=None) is True


def test_literal_phase_setting_compatible_unknown_phase_with_setting():
    assert _literal_phase_setting_compatible(phase="phase3", setting="single_agent") is False


# ---------------------------------------------------------------------------
# _llm_seed_reproducibility_incomplete
# ---------------------------------------------------------------------------


def test_llm_seed_reproducibility_incomplete_not_dict():
    assert _llm_seed_reproducibility_incomplete(None) is False
    assert _llm_seed_reproducibility_incomplete("str") is False


def test_llm_seed_reproducibility_incomplete_not_enabled():
    assert _llm_seed_reproducibility_incomplete({"enabled": False}) is False


def test_llm_seed_reproducibility_incomplete_no_turns():
    assert _llm_seed_reproducibility_incomplete({"enabled": True, "turns": 0}) is False


def test_llm_seed_reproducibility_incomplete_complete():
    assert _llm_seed_reproducibility_incomplete({
        "enabled": True, "turns": 5, "seed_applied_turns": 5
    }) is False


def test_llm_seed_reproducibility_incomplete_incomplete():
    assert _llm_seed_reproducibility_incomplete({
        "enabled": True, "turns": 5, "seed_applied_turns": 3
    }) is True


def test_llm_seed_reproducibility_incomplete_no_seed_applied():
    assert _llm_seed_reproducibility_incomplete({
        "enabled": True, "turns": 5
    }) is True


# ---------------------------------------------------------------------------
# _evaluate_paper_control_profile
# ---------------------------------------------------------------------------


def test_evaluate_paper_control_profile_matched():
    result = _evaluate_paper_control_profile({
        "model": PAPER_MODEL,
        "temperature": PAPER_TEMPERATURE,
        "seeds": list(PAPER_SEEDS),
        "settings": list(PAPER_SETTINGS),
    })
    assert result["is_paper_matched"] is True
    assert result["mismatches"] == []


def test_evaluate_paper_control_profile_wrong_model():
    result = _evaluate_paper_control_profile({
        "model": "wrong-model",
        "temperature": PAPER_TEMPERATURE,
        "seeds": list(PAPER_SEEDS),
        "settings": list(PAPER_SETTINGS),
    })
    assert result["is_paper_matched"] is False
    assert len(result["mismatches"]) >= 1


def test_evaluate_paper_control_profile_wrong_temperature():
    result = _evaluate_paper_control_profile({
        "model": PAPER_MODEL,
        "temperature": 0.0,
        "seeds": list(PAPER_SEEDS),
        "settings": list(PAPER_SETTINGS),
    })
    assert result["is_paper_matched"] is False


def test_evaluate_paper_control_profile_wrong_seeds():
    result = _evaluate_paper_control_profile({
        "model": PAPER_MODEL,
        "temperature": PAPER_TEMPERATURE,
        "seeds": [1, 2],
        "settings": list(PAPER_SETTINGS),
    })
    assert result["is_paper_matched"] is False


# ---------------------------------------------------------------------------
# _runtime_semantics_string_list
# ---------------------------------------------------------------------------


def test_runtime_semantics_string_list_valid():
    assert _runtime_semantics_string_list(["a", "b"]) == ["a", "b"]


def test_runtime_semantics_string_list_not_list():
    assert _runtime_semantics_string_list("abc") is None
    assert _runtime_semantics_string_list(None) is None


def test_runtime_semantics_string_list_mixed_types():
    assert _runtime_semantics_string_list([1, None, "x"]) == ["1", "None", "x"]


# ---------------------------------------------------------------------------
# _Logger
# ---------------------------------------------------------------------------


def test_logger_info(tmp_path: Path):
    path = tmp_path / "test.log"
    logger = _Logger(path)
    logger.info("hello")
    content = path.read_text()
    assert "[INFO]" in content
    assert "hello" in content


def test_logger_warning(tmp_path: Path):
    path = tmp_path / "test.log"
    logger = _Logger(path)
    logger.warning("warn!")
    content = path.read_text()
    assert "[WARN]" in content


def test_logger_error(tmp_path: Path):
    path = tmp_path / "test.log"
    logger = _Logger(path)
    logger.error("fail!")
    content = path.read_text()
    assert "[ERROR]" in content


def test_logger_multiple_writes(tmp_path: Path):
    path = tmp_path / "test.log"
    logger = _Logger(path)
    logger.info("one")
    logger.info("two")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# _read_jsonl_rows / _read_csv_rows
# ---------------------------------------------------------------------------


def test_io_functions_jsonl_nonexistent(tmp_path: Path):
    assert _read_jsonl_rows(tmp_path / "missing.jsonl") == []


def test_io_functions_jsonl_basic(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n')
    rows = _read_jsonl_rows(path)
    assert len(rows) == 2


def test_io_functions_jsonl_blank_lines(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n')
    rows = _read_jsonl_rows(path)
    assert len(rows) == 2


def test_io_functions_csv_nonexistent(tmp_path: Path):
    assert _read_csv_rows(tmp_path / "missing.csv") == []


def test_io_functions_csv_basic(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,2\n3,4\n")
    rows = _read_csv_rows(path)
    assert len(rows) == 2
    assert rows[0]["x"] == "1"


# ---------------------------------------------------------------------------
# _file_sha256
# ---------------------------------------------------------------------------


def test_auto_report_file_sha256_basic(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert _file_sha256(f) == expected


# ---------------------------------------------------------------------------
# _collect_system_stats
# ---------------------------------------------------------------------------


def test_collect_system_stats_empty():
    stats = _collect_system_stats("mare", [])
    assert stats["total_runs"] == 0
    assert stats["validation_passed_runs"] == 0


def test_collect_system_stats_basic_valid_run():
    row = {
        "validation_passed": True,
        "system_identity": {"name": "mare"},
        "provenance": {"hash": "abc"},
        "execution_flags": {
            "fallback_tainted": False,
            "retry_used": False,
            "rag_fallback_used": False,
            "llm_fallback_used": False,
        },
        "comparability": {"is_comparable": True},
        "notes": {},
        "setting": "single_agent",
        "artifacts_dir": "/tmp/nonexistent",
    }
    stats = _collect_system_stats("mare", [row])
    assert stats["total_runs"] == 1
    assert stats["validation_passed_runs"] == 1
    assert stats["fallback_tainted_runs"] == 0


def test_collect_system_stats_tainted_run():
    row = {
        "validation_passed": False,
        "system_identity": {"name": "mare"},
        "provenance": {"hash": "abc"},
        "execution_flags": {
            "fallback_tainted": True,
            "retry_used": True,
            "rag_fallback_used": True,
            "llm_fallback_used": True,
        },
        "comparability": {"is_comparable": True},
        "notes": {},
        "setting": "single_agent",
        "artifacts_dir": "/tmp/nonexistent",
    }
    stats = _collect_system_stats("mare", [row])
    assert stats["validation_passed_runs"] == 0
    assert stats["fallback_tainted_runs"] == 1
    assert stats["retry_used_runs"] == 1


def test_collect_system_stats_metadata_incomplete():
    row = {
        "validation_passed": True,
        "system_identity": "not_a_dict",  # malformed
        "provenance": "not_a_dict",  # malformed
        "execution_flags": "not_a_dict",  # malformed
        "comparability": "not_a_dict",  # malformed
        "notes": {},
        "setting": "single_agent",
        "artifacts_dir": "/tmp/nonexistent",
    }
    stats = _collect_system_stats("mare", [row])
    assert stats["metadata_incomplete_runs"] == 1


# ---------------------------------------------------------------------------
# _matrix_outputs_complete
# ---------------------------------------------------------------------------


def test_matrix_outputs_complete_missing_dir(tmp_path: Path):
    ok, reason = _matrix_outputs_complete(
        output_dir=tmp_path / "nonexistent",
        expected_runs=1,
        system="mare",
    )
    assert ok is False
    assert "missing output directory" in reason


def test_matrix_outputs_complete_missing_files(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ok, reason = _matrix_outputs_complete(
        output_dir=output_dir,
        expected_runs=1,
        system="mare",
    )
    assert ok is False
    assert "missing" in reason


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_auto_report_dataclasses_config():
    config = AutoReportConfig(
        report_dir=Path("/tmp/report"),
        cases_dir=Path("/tmp/cases"),
        seeds=[42],
        settings=["single_agent"],
        model="gpt-4o-mini",
        temperature=0.7,
        round_cap=3,
        max_tokens=4000,
        rag_enabled=True,
        rag_backend="local_tfidf",
        rag_corpus_dir=Path("/tmp/corpus"),
        judge_pipeline_path=Path("/tmp/judge"),
    )
    assert config.model == "gpt-4o-mini"


def test_auto_report_dataclasses_result():
    result = AutoReportResult(
        run_key="auto-abc123",
        run_dir=Path("/tmp/run"),
        logs_dir=Path("/tmp/logs"),
        mare_dir=Path("/tmp/mare"),
        quare_dir=Path("/tmp/quare"),
        report_readme=Path("/tmp/README.md"),
        report_analysis=Path("/tmp/analysis.md"),
        proofs_dir=Path("/tmp/proofs"),
        verdict_path=Path("/tmp/verdict.json"),
        hard_failures=[],
        warnings=[],
    )
    assert result.run_key == "auto-abc123"


# ---------------------------------------------------------------------------
# _load_case_requirement_units
# ---------------------------------------------------------------------------


def test_load_case_requirement_units_no_controls(tmp_path: Path):
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert result == {}


def test_load_case_requirement_units_invalid_json(tmp_path: Path):
    (tmp_path / "controls.json").write_text("{bad json")
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert result == {}


def test_load_case_requirement_units_not_dict(tmp_path: Path):
    (tmp_path / "controls.json").write_text('"just a string"')
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert result == {}


def test_load_case_requirement_units_no_cases_dir(tmp_path: Path):
    (tmp_path / "controls.json").write_text(json.dumps({"cases_dir": "/nowhere"}))
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert result == {}


def test_load_case_requirement_units_cases_dir_not_string(tmp_path: Path):
    (tmp_path / "controls.json").write_text(json.dumps({"cases_dir": 42}))
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert result == {}


def test_load_case_requirement_units_auto_discover_case_files(tmp_path: Path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case = {
        "case_name": "ATM",
        "requirement": "The system shall handle deposits safely and securely at all times.",
    }
    (cases_dir / "atm_input.json").write_text(json.dumps(case))
    (tmp_path / "controls.json").write_text(json.dumps({"cases_dir": str(cases_dir)}))
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert "ATM" in result
    assert len(result["ATM"]) > 0


def test_load_case_requirement_units_explicit_case_files(tmp_path: Path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case = {
        "case_name": "ATM",
        "requirement": "The system shall handle deposits safely and securely at all times.",
    }
    (cases_dir / "my_case.json").write_text(json.dumps(case))
    (tmp_path / "controls.json").write_text(
        json.dumps({"cases_dir": str(cases_dir), "case_files": ["my_case.json"]})
    )
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert "ATM" in result


def test_load_case_requirement_units_missing_case_name(tmp_path: Path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "bad_input.json").write_text(json.dumps({"requirement": "something long enough to pass threshold requirement"}))
    (tmp_path / "controls.json").write_text(json.dumps({"cases_dir": str(cases_dir)}))
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert result == {}


def test_load_case_requirement_units_short_requirement(tmp_path: Path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "tiny_input.json").write_text(json.dumps({"case_name": "X", "requirement": "short"}))
    (tmp_path / "controls.json").write_text(json.dumps({"cases_dir": str(cases_dir)}))
    result = _load_case_requirement_units(run_dir=tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# _artifact_phase1_requirement_texts
# ---------------------------------------------------------------------------


def test_artifact_phase1_requirement_texts_missing_file(tmp_path: Path):
    assert _artifact_phase1_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase1_requirement_texts_invalid_json(tmp_path: Path):
    (tmp_path / PHASE1_FILENAME).write_text("{bad")
    assert _artifact_phase1_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase1_requirement_texts_non_dict(tmp_path: Path):
    (tmp_path / PHASE1_FILENAME).write_text('"string"')
    assert _artifact_phase1_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase1_requirement_texts_valid(tmp_path: Path):
    payload = {
        "requirements": [
            {"description": "Must handle deposits"},
            {"description": "Must handle withdrawals"},
        ],
        "constraints": [
            {"description": "Max 1000 concurrent users"},
        ],
    }
    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(payload))
    texts = _artifact_phase1_requirement_texts(artifacts_dir=tmp_path)
    assert len(texts) == 3


def test_artifact_phase1_requirement_texts_non_list_values(tmp_path: Path):
    payload = {"info": "not a list"}
    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(payload))
    assert _artifact_phase1_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase1_requirement_texts_non_dict_elements(tmp_path: Path):
    payload = {"items": ["a string", 42]}
    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(payload))
    assert _artifact_phase1_requirement_texts(artifacts_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# _artifact_phase2_requirement_texts
# ---------------------------------------------------------------------------


def test_artifact_phase2_requirement_texts_missing_file(tmp_path: Path):
    assert _artifact_phase2_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase2_requirement_texts_invalid_json(tmp_path: Path):
    (tmp_path / PHASE2_FILENAME).write_text("{bad")
    assert _artifact_phase2_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase2_requirement_texts_non_dict(tmp_path: Path):
    (tmp_path / PHASE2_FILENAME).write_text('"string"')
    assert _artifact_phase2_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase2_requirement_texts_negotiations_not_dict(tmp_path: Path):
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps({"negotiations": "nope"}))
    assert _artifact_phase2_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase2_requirement_texts_valid_backward_step(tmp_path: Path):
    payload = {
        "negotiations": {
            "pair1": {
                "steps": [
                    {
                        "message_type": "forward",
                        "kaos_elements": [{"description": "old requirement"}],
                    },
                    {
                        "message_type": "backward",
                        "kaos_elements": [{"description": "refined requirement"}],
                    },
                ]
            }
        }
    }
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(payload))
    texts = _artifact_phase2_requirement_texts(artifacts_dir=tmp_path)
    assert "refined requirement" in texts
    assert "old requirement" not in texts


def test_artifact_phase2_requirement_texts_fallback_to_last_step(tmp_path: Path):
    payload = {
        "negotiations": {
            "pair1": {
                "steps": [
                    {
                        "message_type": "forward",
                        "kaos_elements": [{"description": "only forward"}],
                    },
                ]
            }
        }
    }
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(payload))
    texts = _artifact_phase2_requirement_texts(artifacts_dir=tmp_path)
    assert "only forward" in texts


def test_artifact_phase2_requirement_texts_uses_name_fallback(tmp_path: Path):
    payload = {
        "negotiations": {
            "pair1": {
                "steps": [
                    {
                        "message_type": "backward",
                        "kaos_elements": [{"name": "deposit_goal"}],
                    }
                ]
            }
        }
    }
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(payload))
    texts = _artifact_phase2_requirement_texts(artifacts_dir=tmp_path)
    assert "deposit_goal" in texts


def test_artifact_phase2_requirement_texts_deduplication(tmp_path: Path):
    payload = {
        "negotiations": {
            "pair1": {
                "steps": [{"message_type": "backward", "kaos_elements": [{"description": "req A"}]}]
            },
            "pair2": {
                "steps": [{"message_type": "backward", "kaos_elements": [{"description": "req A"}]}]
            },
        }
    }
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(payload))
    texts = _artifact_phase2_requirement_texts(artifacts_dir=tmp_path)
    # req A appears twice in source, but set removes dups -> list
    assert len(texts) == 1
    assert texts[0] == "req A"


def test_artifact_phase2_requirement_texts_empty_steps(tmp_path: Path):
    payload = {"negotiations": {"pair1": {"steps": []}}}
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(payload))
    assert _artifact_phase2_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase2_requirement_texts_non_list_steps(tmp_path: Path):
    payload = {"negotiations": {"pair1": {"steps": "not a list"}}}
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(payload))
    assert _artifact_phase2_requirement_texts(artifacts_dir=tmp_path) == []


def test_artifact_phase2_requirement_texts_non_dict_negotiation(tmp_path: Path):
    payload = {"negotiations": {"pair1": "not a dict"}}
    (tmp_path / PHASE2_FILENAME).write_text(json.dumps(payload))
    assert _artifact_phase2_requirement_texts(artifacts_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# _derive_precision_f1_gains_from_artifacts
# ---------------------------------------------------------------------------


def test_derive_precision_f1_gains_from_artifacts_no_requirements(tmp_path: Path):
    result = _derive_precision_f1_gains_from_artifacts(run_dir=tmp_path, rows=[])
    assert result["method"] == "token_overlap_requirement_alignment"
    assert result["pair_count"] == 0


def test_derive_precision_f1_gains_from_artifacts_no_baseline_negotiated_pair(tmp_path: Path):
    # Set up controls with requirements
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case = {
        "case_name": "ATM",
        "requirement": "The system shall handle deposits safely and securely at all times throughout operation.",
    }
    (cases_dir / "atm_input.json").write_text(json.dumps(case))
    (tmp_path / "controls.json").write_text(json.dumps({"cases_dir": str(cases_dir)}))

    # Only single_agent rows, no baseline/negotiated pair
    rows = [{"case_id": "ATM", "setting": "single_agent", "run_id": "run1", "seed": 42}]
    result = _derive_precision_f1_gains_from_artifacts(run_dir=tmp_path, rows=rows)
    assert result["pair_count"] == 0


# ---------------------------------------------------------------------------
# _phase2_steps_for_logging
# ---------------------------------------------------------------------------


def test_phase2_steps_for_logging_empty():
    assert _phase2_steps_for_logging({}) == []


def test_phase2_steps_for_logging_negotiations_not_dict():
    assert _phase2_steps_for_logging({"negotiations": "nope"}) == []


def test_phase2_steps_for_logging_basic():
    payload = {
        "negotiations": {
            "pair1": {
                "focus_agent": "alice",
                "reviewer_agents": ["bob"],
                "steps": [
                    {
                        "step_id": 1,
                        "round_number": 1,
                        "message_type": "forward",
                        "focus_agent": "alice",
                        "reviewer_agent": "bob",
                        "analysis_text": "initial analysis",
                    }
                ],
            }
        }
    }
    steps = _phase2_steps_for_logging(payload)
    assert len(steps) == 1
    assert steps[0]["focus_agent"] == "alice"
    assert steps[0]["message_type"] == "forward"


def test_phase2_steps_for_logging_non_dict_step():
    payload = {
        "negotiations": {
            "pair1": {
                "steps": ["not_a_dict"],
            }
        }
    }
    steps = _phase2_steps_for_logging(payload)
    assert steps == []


def test_phase2_steps_for_logging_non_dict_negotiation():
    payload = {"negotiations": {"pair1": "not_a_dict"}}
    steps = _phase2_steps_for_logging(payload)
    assert steps == []


# ---------------------------------------------------------------------------
# _phase2_agents_for_logging
# ---------------------------------------------------------------------------


def test_phase2_agents_for_logging_empty():
    agents = _phase2_agents_for_logging(phase2_payload={}, steps=[])
    assert agents == []


def test_phase2_agents_for_logging_from_steps():
    steps = [{"focus_agent": "alice", "reviewer_agent": "bob"}]
    agents = _phase2_agents_for_logging(phase2_payload={}, steps=steps)
    assert "alice" in agents
    assert "bob" in agents


def test_phase2_agents_for_logging_from_payload():
    payload = {
        "negotiations": {
            "pair1": {
                "focus_agent": "charlie",
                "reviewer_agents": ["dave"],
            }
        }
    }
    agents = _phase2_agents_for_logging(phase2_payload=payload, steps=[])
    assert "charlie" in agents
    assert "dave" in agents


def test_phase2_agents_for_logging_non_dict_negotiation():
    payload = {"negotiations": {"pair1": "not_a_dict"}}
    agents = _phase2_agents_for_logging(phase2_payload=payload, steps=[])
    assert agents == []


# ---------------------------------------------------------------------------
# _phase1_agents_for_logging
# ---------------------------------------------------------------------------


def test_phase1_agents_for_logging_missing_file(tmp_path: Path):
    assert _phase1_agents_for_logging(artifacts_dir=tmp_path) == []


def test_phase1_agents_for_logging_invalid_json(tmp_path: Path):
    (tmp_path / PHASE1_FILENAME).write_text("{bad")
    assert _phase1_agents_for_logging(artifacts_dir=tmp_path) == []


def test_phase1_agents_for_logging_non_dict(tmp_path: Path):
    (tmp_path / PHASE1_FILENAME).write_text('"string"')
    assert _phase1_agents_for_logging(artifacts_dir=tmp_path) == []


def test_phase1_agents_for_logging_valid(tmp_path: Path):
    payload = {
        "requirements_engineer": [],
        "safety_analyst": [],
    }
    (tmp_path / PHASE1_FILENAME).write_text(json.dumps(payload))
    agents = _phase1_agents_for_logging(artifacts_dir=tmp_path)
    assert "requirements_engineer" in agents
    assert "safety_analyst" in agents


# ---------------------------------------------------------------------------
# _agent_log_paths
# ---------------------------------------------------------------------------


def test_agent_log_paths_basic(tmp_path: Path):
    agents_dir = tmp_path / "agents"
    mapping = _agent_log_paths(agents_dir=agents_dir, expected_agents=["alice", "bob"])
    assert "alice" in mapping
    assert "bob" in mapping
    assert mapping["alice"].suffix == ".md"


def test_agent_log_paths_deduplication(tmp_path: Path):
    """Two agents that sanitize to the same base produce distinct file names."""
    agents_dir = tmp_path / "agents"
    # Both sanitize to "alice"
    mapping = _agent_log_paths(agents_dir=agents_dir, expected_agents=["alice", "Alice"])
    paths = list(mapping.values())
    assert len(paths) == 2
    assert paths[0] != paths[1]


# ---------------------------------------------------------------------------
# _conversation_bundle_is_fresh
# ---------------------------------------------------------------------------


def test_conversation_bundle_is_fresh_no_meta(tmp_path: Path):
    timeline = tmp_path / "timeline.md"
    timeline.write_text("# Timeline")
    assert _conversation_bundle_is_fresh(
        meta_path=tmp_path / "meta.json",
        source_hash="abc",
        timeline_path=timeline,
        agent_paths=[],
    ) is False


def test_conversation_bundle_is_fresh_no_timeline(tmp_path: Path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"source_hash": "abc"}))
    assert _conversation_bundle_is_fresh(
        meta_path=meta,
        source_hash="abc",
        timeline_path=tmp_path / "timeline.md",
        agent_paths=[],
    ) is False


def test_conversation_bundle_is_fresh_missing_agent_path(tmp_path: Path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"source_hash": "abc"}))
    timeline = tmp_path / "timeline.md"
    timeline.write_text("# Timeline")
    assert _conversation_bundle_is_fresh(
        meta_path=meta,
        source_hash="abc",
        timeline_path=timeline,
        agent_paths=[tmp_path / "agent.md"],
    ) is False


def test_conversation_bundle_is_fresh_hash_mismatch(tmp_path: Path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"source_hash": "old"}))
    timeline = tmp_path / "timeline.md"
    timeline.write_text("# Timeline")
    assert _conversation_bundle_is_fresh(
        meta_path=meta,
        source_hash="new",
        timeline_path=timeline,
        agent_paths=[],
    ) is False


def test_conversation_bundle_is_fresh_fresh(tmp_path: Path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"source_hash": "abc123"}))
    timeline = tmp_path / "timeline.md"
    timeline.write_text("# Timeline")
    agent = tmp_path / "agent.md"
    agent.write_text("# Agent")
    assert _conversation_bundle_is_fresh(
        meta_path=meta,
        source_hash="abc123",
        timeline_path=timeline,
        agent_paths=[agent],
    ) is True


def test_conversation_bundle_is_fresh_invalid_meta_json(tmp_path: Path):
    meta = tmp_path / "meta.json"
    meta.write_text("{bad")
    timeline = tmp_path / "timeline.md"
    timeline.write_text("# Timeline")
    assert _conversation_bundle_is_fresh(
        meta_path=meta,
        source_hash="abc",
        timeline_path=timeline,
        agent_paths=[],
    ) is False


# ---------------------------------------------------------------------------
# _render_timeline_markdown / _render_agent_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_timeline_empty_steps():
    result = _render_timeline_markdown(
        run_id="run1", system="mare", case_id="ATM",
        setting="single_agent", seed=42, source_hash="abc",
        steps=[],
    )
    assert "# Conversation Timeline" in result
    assert "run1" in result


def test_render_markdown_timeline_with_steps():
    steps = [
        {
            "negotiation_pair": "p1",
            "step_id": 1,
            "round_number": 1,
            "message_type": "forward",
            "focus_agent": "alice",
            "reviewer_agent": "bob",
            "analysis_text": "Initial proposal",
            "feedback": "Looks good",
            "conflict_detected": True,
            "resolution_state": "resolved",
            "requires_refinement": False,
        }
    ]
    result = _render_timeline_markdown(
        run_id="run1", system="mare", case_id="ATM",
        setting="single_agent", seed=42, source_hash="abc",
        steps=steps,
    )
    assert "alice" in result


def test_render_markdown_agent_empty_steps():
    result = _render_agent_markdown(
        run_id="run1", system="mare", case_id="ATM",
        setting="single_agent", seed=42, agent="alice",
        source_hash="abc", steps=[],
    )
    assert "alice" in result


def test_render_markdown_agent_with_steps():
    steps = [
        {
            "focus_agent": "alice",
            "reviewer_agent": "bob",
            "step_id": 1,
            "round_number": 1,
            "message_type": "forward",
            "analysis_text": "My analysis",
            "feedback": "OK",
            "conflict_detected": False,
            "resolution_state": "none",
            "requires_refinement": True,
        }
    ]
    result = _render_agent_markdown(
        run_id="run1", system="mare", case_id="ATM",
        setting="single_agent", seed=42, agent="alice",
        source_hash="abc", steps=steps,
    )
    assert "My analysis" in result


# ---------------------------------------------------------------------------
# _ingest_literal_metric_node
# ---------------------------------------------------------------------------


def test_ingest_literal_metric_node_non_dict_node():
    output: dict = {}
    _ingest_literal_metric_node(node="not a dict", output=output)
    assert output == {}


def test_ingest_literal_metric_node_list_node_delegates():
    output: dict = {}
    nodes = [
        {
            "case_id": "ATM",
            "system": "quare",
            "phase": "phase1",
            "seed": 42,
            "setting": "multi_agent_without_negotiation",
            "precision": 0.8,
            "f1": 0.85,
        }
    ]
    _ingest_literal_metric_node(node=nodes, output=output)
    key = ("quare", "ATM", 42)
    assert key in output
    assert output[key]["phase1"]["precision"] == pytest.approx(0.8)


def test_ingest_literal_metric_node_direct_node():
    output: dict = {}
    node = {
        "case_id": "ATM",
        "system": "quare",
        "phase": "phase1",
        "seed": 42,
        "setting": "multi_agent_without_negotiation",
        "precision": 0.7,
        "f1": 0.8,
    }
    _ingest_literal_metric_node(node=node, output=output)
    key = ("quare", "ATM", 42)
    assert key in output


def test_ingest_literal_metric_node_nested_phase_block():
    """Test node with phase1/phase2 sub-dicts."""
    output: dict = {}
    node = {
        "case_id": "ATM",
        "system": "quare",
        "seed": 42,
        "setting": "multi_agent_without_negotiation",
        "phase1": {"precision": 0.6, "f1": 0.7},
    }
    _ingest_literal_metric_node(node=node, output=output)
    key = ("quare", "ATM", 42)
    assert key in output
    assert "phase1" in output[key]


def test_ingest_literal_metric_node_nested_case_results():
    """Test node with nested case_results list."""
    output: dict = {}
    node = {
        "case_results": [
            {
                "case_id": "ATM",
                "system": "quare",
                "phase": "phase1",
                "seed": 42,
                "setting": "multi_agent_without_negotiation",
                "precision": 0.9,
                "f1": 0.95,
            }
        ]
    }
    _ingest_literal_metric_node(node=node, output=output)
    key = ("quare", "ATM", 42)
    assert key in output


def test_ingest_literal_metric_node_infer_system_from_run_id():
    output: dict = {}
    node = {
        "case_id": "ATM",
        "run_id": "quare-atm-s042",
        "phase": "phase1",
        "seed": 42,
        "setting": "multi_agent_without_negotiation",
        "precision": 0.5,
    }
    _ingest_literal_metric_node(node=node, output=output)
    key = ("quare", "ATM", 42)
    assert key in output


def test_ingest_literal_metric_node_unspecified_system():
    output: dict = {}
    node = {
        "case_id": "ATM",
        "phase": "phase1",
        "seed": 42,
        "precision": 0.5,
    }
    _ingest_literal_metric_node(node=node, output=output)
    key = (UNSPECIFIED_SYSTEM, "ATM", 42)
    assert key in output


def test_ingest_literal_metric_node_setting_based_blocks():
    """Test node with setting-named sub-dicts mapped to phases."""
    output: dict = {}
    node = {
        "case_id": "ATM",
        "system": "quare",
        "seed": 42,
        "multi_agent_without_negotiation": {"precision": 0.6, "f1": 0.7},
    }
    _ingest_literal_metric_node(node=node, output=output)
    key = ("quare", "ATM", 42)
    assert key in output


# ---------------------------------------------------------------------------
# _load_ground_truth_case_ids
# ---------------------------------------------------------------------------


def test_load_ground_truth_case_ids_empty_contract():
    assert _load_ground_truth_case_ids({}) == set()


def test_load_ground_truth_case_ids_non_dict_contract():
    assert _load_ground_truth_case_ids("not a dict") == set()


def test_load_ground_truth_case_ids_no_found_key():
    assert _load_ground_truth_case_ids({"something": "else"}) == set()


def test_load_ground_truth_case_ids_found_not_dict():
    assert _load_ground_truth_case_ids({"found": "not a dict"}) == set()


def test_load_ground_truth_case_ids_from_file_stem(tmp_path: Path):
    """Extract case ID from file stem when file doesn't exist."""
    missing = tmp_path / "ATM_gt.json"
    result = _load_ground_truth_case_ids({
        "found": {"local_ground_truth_files": [str(missing)]}
    })
    # file stem is ATM_gt → normalized to ATMGT
    assert "ATMGT" in result


def test_load_ground_truth_case_ids_from_file_content(tmp_path: Path):
    """Extract case ID from JSON content of ground truth file."""
    gt_file = tmp_path / "ground_truth.json"
    gt_file.write_text(json.dumps({"case_name": "ATM System"}))
    result = _load_ground_truth_case_ids({
        "found": {"local_ground_truth_files": [str(gt_file)]}
    })
    assert "ATMSYSTEM" in result


def test_load_ground_truth_case_ids_external_ground_truth_files(tmp_path: Path):
    """External ground truth files also contribute case IDs."""
    gt_file = tmp_path / "external_gt.json"
    gt_file.write_text(json.dumps({"case_id": "hospital", "other": "data"}))
    result = _load_ground_truth_case_ids({
        "found": {"external_ground_truth_files": [str(gt_file)]}
    })
    assert "HOSPITAL" in result


def test_load_ground_truth_case_ids_invalid_json_file(tmp_path: Path):
    """Malformed JSON is silently skipped."""
    gt_file = tmp_path / "bad_gt.json"
    gt_file.write_text("{bad")
    result = _load_ground_truth_case_ids({
        "found": {"local_ground_truth_files": [str(gt_file)]}
    })
    # Still gets case ID from stem
    assert "BAD_GT" in result or "BADGT" in result


# ---------------------------------------------------------------------------
# _audit_precision_f1_contract
# ---------------------------------------------------------------------------


def test_audit_precision_f1_contract_no_controls(tmp_path: Path):
    result = _audit_precision_f1_contract(run_dir=tmp_path)
    assert result["contract_available"] is False


def test_audit_precision_f1_contract_no_judge_path(tmp_path: Path):
    controls = {"judge_pipeline_path": "/nonexistent/judge.py"}
    (tmp_path / "controls.json").write_text(json.dumps(controls))
    result = _audit_precision_f1_contract(run_dir=tmp_path)
    assert result["contract_available"] is False


def test_audit_precision_f1_contract_judge_exists_no_labels(tmp_path: Path):
    judge = tmp_path / "judge.py"
    judge.write_text("# judge script")
    controls = {"judge_pipeline_path": str(judge)}
    (tmp_path / "controls.json").write_text(json.dumps(controls))
    result = _audit_precision_f1_contract(run_dir=tmp_path)
    assert result["contract_available"] is False


# ---------------------------------------------------------------------------
# _compute_literal_precision_f1_gains
# ---------------------------------------------------------------------------


def test_compute_literal_precision_f1_gains_contract_not_available(tmp_path: Path):
    result = _compute_literal_precision_f1_gains(
        run_dir=tmp_path,
        contract={"contract_available": False},
    )
    assert result is None


def test_compute_literal_precision_f1_gains_no_label_files(tmp_path: Path):
    result = _compute_literal_precision_f1_gains(
        run_dir=tmp_path,
        contract={"contract_available": True, "found": {"label_files": []}},
    )
    assert result is None


def test_compute_literal_precision_f1_gains_label_files_missing(tmp_path: Path):
    result = _compute_literal_precision_f1_gains(
        run_dir=tmp_path,
        contract={"contract_available": True, "found": {"label_files": ["/nonexistent"]}},
    )
    assert result is None


def test_compute_literal_precision_f1_gains_valid_label_file(tmp_path: Path):
    label_file = tmp_path / "quare-atm-labels.json"
    labels = {
        "case_id": "ATM",
        "system": "quare",
        "seed": 42,
        "phase": "phase1",
        "setting": "multi_agent_without_negotiation",
        "precision": 0.8,
        "f1": 0.85,
    }
    label_file.write_text(json.dumps(labels))
    result = _compute_literal_precision_f1_gains(
        run_dir=tmp_path,
        contract={
            "contract_available": True,
            "found": {"label_files": [str(label_file)]},
        },
    )
    # Single label file with only phase1 won't produce gains (need phase1+phase2)
    assert result is not None
    assert result["method"] == "literal_manual_label_ground_truth_contract"


# ---------------------------------------------------------------------------
# _compute_key_deltas
# ---------------------------------------------------------------------------


def _write_csv_for_test(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    import csv as _csv
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_key_deltas_empty_csvs(tmp_path: Path):
    mare_csv = tmp_path / "mare.csv"
    quare_csv = tmp_path / "quare.csv"
    _write_csv_for_test(mare_csv, [])
    _write_csv_for_test(quare_csv, [])
    result = _compute_key_deltas(mare_csv=mare_csv, quare_csv=quare_csv)
    assert result["by_setting"] == {}
    assert "overall" in result


def test_compute_key_deltas_with_rows(tmp_path: Path):
    mare_csv = tmp_path / "mare.csv"
    quare_csv = tmp_path / "quare.csv"
    mare_rows = [
        {"setting": "multi_agent_with_negotiation", "semantic_preservation_f1": "0.8", "conflict_resolution_rate": "0.5"},
    ]
    quare_rows = [
        {"setting": "multi_agent_with_negotiation", "semantic_preservation_f1": "0.9", "conflict_resolution_rate": "0.7"},
    ]
    _write_csv_for_test(mare_csv, mare_rows)
    _write_csv_for_test(quare_csv, quare_rows)
    result = _compute_key_deltas(mare_csv=mare_csv, quare_csv=quare_csv)
    assert "multi_agent_with_negotiation" in result["by_setting"]
    assert result["overall"] is not None
    assert len(result["metrics"]) == len(KEY_DELTA_METRICS)


# ---------------------------------------------------------------------------
# _write_analysis_md
# ---------------------------------------------------------------------------


def test_write_analysis_md_basic(tmp_path: Path):
    path = tmp_path / "analysis.md"
    _write_analysis_md(
        path=path,
        verdict={"final_completion_verdict": "GO", "check_results": {"F1>0.7": True}},
        deltas={"by_setting": {"multi_agent_with_negotiation": {"semantic_preservation_f1": {"mare_mean": 0.8, "quare_mean": 0.9, "quare_minus_mare": 0.1}}}},
        replay={"systems": {"mare": {"total_runs": 10, "error_items": 0, "warning_items": 1}, "quare": {"total_runs": 10, "error_items": 0, "warning_items": 0}}},
        paper_claims={"claim_summary": {"pass_count": 5, "hard_fail_count": 0, "blocked_hard_fail_count": 0, "non_comparable_count": 1}, "claims": [{"claim_id": "C1", "status": "PASS", "paper_target": ">0.7", "observed": 0.85}]},
        conversation_summary={"expected_runs": 10, "complete_runs": 10, "coverage_ratio": 1.0, "is_complete": True, "missing_run_ids": [], "runs_with_missing_agent_logs": 0},
        warnings=["Test warning"],
    )
    assert path.exists()
    content = path.read_text()
    assert "MARE/QUARE Strict Analysis" in content
    assert "GO" in content
    assert "Test warning" in content
    assert "C1" in content


def test_write_analysis_md_no_warnings(tmp_path: Path):
    path = tmp_path / "analysis.md"
    _write_analysis_md(
        path=path,
        verdict={"final_completion_verdict": "NO-GO", "check_results": {}},
        deltas={"by_setting": {}},
        replay={"systems": {}},
        paper_claims={"claim_summary": {}, "claims": []},
        conversation_summary={},
        warnings=[],
    )
    content = path.read_text()
    assert "Warnings" not in content


def test_write_analysis_md_observed_dict_claim(tmp_path: Path):
    path = tmp_path / "analysis.md"
    _write_analysis_md(
        path=path,
        verdict={"final_completion_verdict": "GO", "check_results": {}},
        deltas={"by_setting": {}},
        replay={"systems": {}},
        paper_claims={"claim_summary": {}, "claims": [{"claim_id": "C2", "status": "PASS", "paper_target": "T", "observed": {"precision": 0.9, "recall": 0.8}}]},
        conversation_summary={},
        warnings=[],
    )
    content = path.read_text()
    assert "precision" in content


# ---------------------------------------------------------------------------
# _build_manifest
# ---------------------------------------------------------------------------


def test_build_manifest_empty(tmp_path: Path):
    result = _build_manifest(key_paths=[], root_dir=tmp_path)
    assert result["files"] == {}


def test_build_manifest_with_files(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    result = _build_manifest(key_paths=[f], root_dir=tmp_path)
    assert "test.txt" in result["files"]
    assert len(result["files"]["test.txt"]) == 64


def test_build_manifest_missing_file_skipped(tmp_path: Path):
    f = tmp_path / "nonexistent.txt"
    result = _build_manifest(key_paths=[f], root_dir=tmp_path)
    assert result["files"] == {}


# ---------------------------------------------------------------------------
# _log_matrix_snapshot
# ---------------------------------------------------------------------------


def test_log_matrix_snapshot_basic(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runs_jsonl = output_dir / RUNS_JSONL_NAME
    runs_jsonl.write_text(json.dumps({"run_id": "r1", "setting": "s", "validation_passed": True}) + "\n")
    log_path = tmp_path / "snapshot.log"
    _log_matrix_snapshot(output_dir=output_dir, system="quare", log_path=log_path)
    content = log_path.read_text()
    assert "system=quare" in content
    assert "run_id=r1" in content


# ---------------------------------------------------------------------------
# _rewrite_readme_for_report_root
# ---------------------------------------------------------------------------


def test_rewrite_readme_for_report_root_replaces_relative_prefix():
    readme = "See [log](../../logs/run-key-001/agent.md)."
    result = _rewrite_readme_for_report_root(readme=readme, run_key="run-key-001")
    assert "logs/run-key-001/agent.md" in result
    assert "../../" not in result


def test_rewrite_readme_for_report_root_replaces_legacy_prefix():
    readme = "See [log](../logs/run-key-001/timeline.md)."
    result = _rewrite_readme_for_report_root(readme=readme, run_key="run-key-001")
    assert "logs/run-key-001/timeline.md" in result
    assert "../" not in result


def test_rewrite_readme_for_report_root_no_match():
    readme = "No relative links here."
    result = _rewrite_readme_for_report_root(readme=readme, run_key="run-key-001")
    assert result == readme


# ---------------------------------------------------------------------------
# _mirror_latest_outputs
# ---------------------------------------------------------------------------


def test_mirror_latest_outputs_basic(tmp_path: Path):
    report_dir = tmp_path / "report"
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)

    readme = tmp_path / "README.md"
    readme.write_text("# Test README")
    analysis = tmp_path / "analysis.md"
    analysis.write_text("# Analysis")
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "proof1.json").write_text(json.dumps({"test": True}))

    _mirror_latest_outputs(
        report_dir=report_dir,
        run_key="run-001",
        run_dir=run_dir,
        report_readme=readme,
        report_analysis=analysis,
        proofs_dir=proofs,
    )

    assert (report_dir / "README.md").exists()
    assert (report_dir / "analysis.md").exists()
    assert (report_dir / "proofs" / "proof1.json").exists()
    assert (report_dir / "latest_run_pointer.json").exists()


# ---------------------------------------------------------------------------
# _write_report_readme
# ---------------------------------------------------------------------------


def test_write_report_readme_basic(tmp_path: Path):
    path = tmp_path / "README.md"
    _write_report_readme(
        path=path,
        run_dir=tmp_path,
        verdict={"final_completion_verdict": "GO"},
        validation_evidence={
            "systems": {
                "mare": {"total_runs": 10, "validation_passed_runs": 8, "llm_turns_sum": 120},
                "quare": {"total_runs": 10, "validation_passed_runs": 10, "llm_turns_sum": 80},
            }
        },
        conversation_summary={"expected_runs": 20, "complete_runs": 20, "coverage_ratio": 1.0},
        paper_claims={"claim_summary": {"pass_count": 5, "hard_fail_count": 0, "blocked_hard_fail_count": 0, "non_comparable_count": 1}},
    )
    assert path.exists()
    content = path.read_text()
    assert "Auto Report" in content
    assert "GO" in content

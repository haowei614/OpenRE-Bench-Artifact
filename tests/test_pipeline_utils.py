"""Tests for openre_bench.pipeline._utils — text helpers, fragments, topology, validation, quality."""

from __future__ import annotations


import pytest

from openre_bench.pipeline._utils import (
    _agent_fragment_start_index,
    _agent_fragment_window,
    _build_deterministic_validation,
    _compliance_coverage,
    _compute_topology_status,
    _extract_external_rules,
    _extract_requirement_fragments,
    _fact_checking,
    _hierarchy_view,
    _logical_consistency,
    _negotiation_enabled,
    _parent_child_mappings,
    _prompt_contract_hash,
    _quality_axis_for_agent,
    _quality_lens_phrase,
    _rotate_fragments,
    _runtime_semantics_mode,
    _sha256_payload,
    _summarize_text,
    _terminology_consistency,
    _text_overlap_score,
    _to_float,
    _to_int,
    _tokens,
    _verification_executed,
    _agent_quality_mapping_for_setting,
    default_run_id,
)

# Also import types to ensure _types.py coverage
from openre_bench.pipeline._types import (
    IREDEV_ROLE_QUALITY_ATTRIBUTES,
    IREDEV_RUNTIME_SEMANTICS_MODE,
    IREDEV_RUNTIME_TRACE_VERSION,
    MARE_ROLE_QUALITY_ATTRIBUTES,
    MARE_RUNTIME_SEMANTICS_MODE,
    MARE_RUNTIME_TRACE_VERSION,
    RuntimeExecutionMeta,
    PHASE2_LLM_RETRY_LIMIT,
    Phase2ExecutionMeta,
    PipelineConfig,
    QUALITY_LENS_CUES,
)


# ---------------------------------------------------------------------------
# _tokens
# ---------------------------------------------------------------------------


def test_tokens_basic():
    assert _tokens("Hello World 123") == ["ello", "orld", "123"]


def test_tokens_empty():
    assert _tokens("") == []


def test_tokens_special_chars_only():
    assert _tokens("!!!@@@") == []


def test_tokens_underscores():
    assert _tokens("hello_world") == ["hello_world"]


# ---------------------------------------------------------------------------
# _summarize_text
# ---------------------------------------------------------------------------


def test_summarize_text_within_limit():
    assert _summarize_text("short", 100) == "short"


def test_summarize_text_exceeds_limit():
    result = _summarize_text("a" * 200, 50)
    assert len(result) <= 50
    assert result.endswith("...")


def test_summarize_text_whitespace_normalization():
    assert _summarize_text("hello   world\n\ttab", 100) == "hello world tab"


def test_summarize_text_exact_limit():
    text = "a" * 50
    assert _summarize_text(text, 50) == text


# ---------------------------------------------------------------------------
# _text_overlap_score
# ---------------------------------------------------------------------------


def test_text_overlap_score_identical():
    assert _text_overlap_score("hello world", "hello world") == pytest.approx(1.0)


def test_text_overlap_score_no_overlap():
    assert _text_overlap_score("abc", "xyz") == pytest.approx(0.0)


def test_text_overlap_score_partial_overlap():
    score = _text_overlap_score("hello world foo", "hello world bar")
    # tokens: {hello, world, foo} vs {hello, world, bar} → 2/3 overlap
    assert score == pytest.approx(2.0 / 3.0)


def test_text_overlap_score_empty_left():
    assert _text_overlap_score("", "hello") == pytest.approx(0.0)


def test_text_overlap_score_empty_right():
    assert _text_overlap_score("hello", "") == pytest.approx(0.0)


def test_text_overlap_score_both_empty():
    assert _text_overlap_score("", "") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _to_int / _to_float
# ---------------------------------------------------------------------------


def test_coercions_to_int_valid():
    assert _to_int(42) == 42
    assert _to_int("7") == 7


def test_coercions_to_int_invalid():
    assert _to_int("abc", 99) == 99
    assert _to_int(None, -1) == -1


def test_coercions_to_float_valid():
    assert _to_float(3.14) == pytest.approx(3.14)
    assert _to_float("2.5") == pytest.approx(2.5)


def test_coercions_to_float_invalid():
    assert _to_float("abc", 0.0) == pytest.approx(0.0)
    assert _to_float(None, -1.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# _sha256_payload
# ---------------------------------------------------------------------------


def test_sha256_payload_deterministic():
    payload = {"key": "value", "num": 42}
    h1 = _sha256_payload(payload)
    h2 = _sha256_payload(payload)
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_payload_key_order_independent():
    assert _sha256_payload({"a": 1, "b": 2}) == _sha256_payload({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# _extract_requirement_fragments
# ---------------------------------------------------------------------------


def test_extract_requirement_fragments_sentence_split():
    text = "The system shall authenticate users. The system shall log events."
    frags = _extract_requirement_fragments(text)
    assert len(frags) == 2


def test_extract_requirement_fragments_short_fragments_filtered():
    text = "Short. This is a long enough fragment sentence for test."
    frags = _extract_requirement_fragments(text)
    assert all(len(f) >= 20 for f in frags)


def test_extract_requirement_fragments_empty():
    assert _extract_requirement_fragments("") == []


# ---------------------------------------------------------------------------
# _rotate_fragments
# ---------------------------------------------------------------------------


def test_rotate_fragments_basic():
    frags = ["a", "b", "c"]
    assert _rotate_fragments(frags, 1) == ["b", "c", "a"]


def test_rotate_fragments_empty():
    assert _rotate_fragments([], 5) == []


def test_rotate_fragments_zero_seed():
    frags = ["x", "y"]
    assert _rotate_fragments(frags, 0) == frags


def test_rotate_fragments_seed_wraps():
    frags = ["a", "b", "c"]
    assert _rotate_fragments(frags, 3) == frags  # offset = 3 % 3 = 0


# ---------------------------------------------------------------------------
# _agent_fragment_window / _agent_fragment_start_index
# ---------------------------------------------------------------------------


def test_agent_fragment_window_basic():
    frags = ["a", "b", "c", "d"]
    result = _agent_fragment_window(rotated=frags, agent_index=1, total_agents=2, leaf_count=2)
    assert len(result) == 2


def test_agent_fragment_window_empty():
    assert _agent_fragment_window(rotated=[], agent_index=1, total_agents=1, leaf_count=3) == []


def test_agent_fragment_window_start_index_zero_fragments():
    assert _agent_fragment_start_index(fragment_count=0, agent_index=1, total_agents=2) == 0


def test_agent_fragment_window_start_index_wrapping():
    idx = _agent_fragment_start_index(fragment_count=4, agent_index=2, total_agents=2)
    # agent_slot = max(0, 2-1) = 1, (1 * 4) // 2 = 2, 2 % 4 = 2
    assert idx == 2


# ---------------------------------------------------------------------------
# _quality_lens_phrase
# ---------------------------------------------------------------------------


def test_quality_lens_phrase_known_axis():
    phrase = _quality_lens_phrase(quality_attribute="Safety", leaf_index=1)
    # leaf_index=1 → index = max(0, 1-1) % 3 = 0 → first Safety cue
    assert phrase == "hazard prevention"


def test_quality_lens_phrase_known_axis_second_cue():
    phrase = _quality_lens_phrase(quality_attribute="Safety", leaf_index=2)
    # leaf_index=2 → index = max(0, 2-1) % 3 = 1 → second Safety cue
    assert phrase == "fault tolerance"


def test_quality_lens_phrase_unknown_axis_falls_back():
    phrase = _quality_lens_phrase(quality_attribute="UnknownAxis", leaf_index=1)
    # Unknown axis falls back to Integrated cues, index 0
    assert phrase == "cross-quality balance"


# ---------------------------------------------------------------------------
# Phase-setting queries
# ---------------------------------------------------------------------------


def test_setting_queries_negotiation_enabled():
    from openre_bench.schemas import SETTING_MULTI_AGENT_WITH_NEGOTIATION
    assert _negotiation_enabled(SETTING_MULTI_AGENT_WITH_NEGOTIATION) is True
    assert _negotiation_enabled("single_agent") is False


def test_setting_queries_verification_executed():
    from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
    assert _verification_executed(SETTING_NEGOTIATION_INTEGRATION_VERIFICATION) is True
    assert _verification_executed("multi_agent_with_negotiation") is False


def test_setting_queries_agent_quality_mapping_single():
    mapping = _agent_quality_mapping_for_setting("single_agent")
    assert "SingleAgent" in mapping


def test_setting_queries_agent_quality_mapping_multi():
    mapping = _agent_quality_mapping_for_setting("multi_agent_with_negotiation")
    assert "SingleAgent" not in mapping


# ---------------------------------------------------------------------------
# _runtime_semantics_mode
# ---------------------------------------------------------------------------


def test_runtime_semantics_mode_mare_multi():
    mode = _runtime_semantics_mode(system="mare", setting="multi_agent_with_negotiation")
    assert mode == "mare_paper_workflow_v1"


def test_runtime_semantics_mode_mare_single():
    mode = _runtime_semantics_mode(system="mare", setting="single_agent")
    assert mode == "mare_single_agent_baseline"


def test_runtime_semantics_mode_iredev_multi():
    mode = _runtime_semantics_mode(system="iredev", setting="multi_agent_with_negotiation")
    assert mode == "iredev_knowledge_driven_v1"


def test_runtime_semantics_mode_iredev_single():
    mode = _runtime_semantics_mode(system="iredev", setting="single_agent")
    assert mode == "iredev_single_agent_baseline"


def test_runtime_semantics_mode_quare():
    mode = _runtime_semantics_mode(system="quare", setting="multi_agent_with_negotiation")
    assert mode == "quare_dialectic_scaffold_v1"


def test_runtime_semantics_mode_quare_single():
    mode = _runtime_semantics_mode(system="quare", setting="single_agent")
    assert mode == "quare_dialectic_scaffold_v1"


# ---------------------------------------------------------------------------
# _prompt_contract_hash
# ---------------------------------------------------------------------------


def test_prompt_contract_hash_deterministic():
    h1 = _prompt_contract_hash(system="quare", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    h2 = _prompt_contract_hash(system="quare", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    assert h1 == h2
    assert len(h1) == 64


def test_prompt_contract_hash_mare_system():
    h = _prompt_contract_hash(system="mare", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    assert len(h) == 64


def test_prompt_contract_hash_iredev_system():
    h = _prompt_contract_hash(system="iredev", setting="multi_agent_with_negotiation", round_cap=3, max_tokens=500)
    assert len(h) == 64


# ---------------------------------------------------------------------------
# default_run_id
# ---------------------------------------------------------------------------


def test_default_run_id_format():
    run_id = default_run_id("Test Case!", 42)
    assert "test-case" in run_id
    assert "s042" in run_id


def test_default_run_id_empty_name():
    run_id = default_run_id("", 0)
    assert "s000" in run_id


# ---------------------------------------------------------------------------
# _hierarchy_view / _parent_child_mappings
# ---------------------------------------------------------------------------


def test_hierarchy_view():
    item = {"id": "G1", "name": "Goal", "quality_attribute": "Safety", "parent_goal_id": None}
    view = _hierarchy_view(item)
    assert view["id"] == "G1"
    assert view["parent_goal_id"] is None


def test_parent_child_mappings():
    elements = [
        {"id": "G1", "parent_goal_id": None},
        {"id": "T1", "parent_goal_id": "G1"},
        {"id": "T2", "parent_goal_id": "G1"},
    ]
    mappings = _parent_child_mappings(elements)
    assert "G1" in mappings
    assert len(mappings["G1"]) == 2


# ---------------------------------------------------------------------------
# _compute_topology_status
# ---------------------------------------------------------------------------


def test_topology_status_valid_tree():
    elements = [
        {"id": "G1", "gsn_type": "Goal", "parent_goal_id": None},
        {"id": "T1", "gsn_type": "Task", "parent_goal_id": "G1"},
    ]
    status = _compute_topology_status(elements)
    assert status["is_valid"] is True
    assert status["is_dag"] is True
    assert status["orphan_count"] == 0


def test_topology_status_orphan_detection():
    elements = [
        {"id": "G1", "gsn_type": "Goal", "parent_goal_id": None},
        {"id": "T1", "gsn_type": "Task", "parent_goal_id": "MISSING"},
    ]
    status = _compute_topology_status(elements)
    assert status["orphan_count"] == 1
    assert status["orphan_elements"] == ["T1"]
    assert status["is_valid"] is False


def test_topology_status_invalid_leaf_goal():
    elements = [
        {"id": "G1", "gsn_type": "Goal", "parent_goal_id": None},
        {"id": "G2", "gsn_type": "Goal", "parent_goal_id": "G1"},
    ]
    status = _compute_topology_status(elements)
    assert status["invalid_leaf_count"] == 1
    assert status["invalid_leaves"] == ["G2"]


def test_topology_status_orphan_non_goal_root():
    elements = [
        {"id": "T1", "gsn_type": "Task", "parent_goal_id": None},
    ]
    status = _compute_topology_status(elements)
    assert status["orphan_count"] == 1
    assert status["orphan_elements"] == ["T1"]


def test_topology_status_cycle_detection():
    """Test cycle detection when elements form a circular parent chain."""
    elements_cycle = [
        {"id": "G1", "gsn_type": "Goal", "parent_goal_id": "G3"},
        {"id": "G2", "gsn_type": "Goal", "parent_goal_id": "G1"},
        {"id": "G3", "gsn_type": "Goal", "parent_goal_id": "G2"},
    ]
    status = _compute_topology_status(elements_cycle)
    assert status["is_dag"] is False
    assert status["cycle_count"] >= 1
    assert status["is_valid"] is False


# ---------------------------------------------------------------------------
# _logical_consistency
# ---------------------------------------------------------------------------


def test_logical_consistency_no_contradictions():
    elements = [{"id": "R1", "description": "The system shall respond quickly."}]
    result = _logical_consistency(elements)
    assert result["score"] == 5
    assert len(result["contradictions"]) == 0


def test_logical_consistency_numerical_contradiction_leq_geq():
    elements = [
        {"id": "R1", "description": "latency ≤ 5"},
        {"id": "R2", "description": "latency ≥ 10"},
    ]
    result = _logical_consistency(elements)
    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["type"] == "numerical_contradiction"
    assert result["contradictions"][0]["variable"] == "latency"
    # 1 contradiction / 2 elements = 0.5 → score tier 1 (>0.20)
    assert result["score"] == 1


def test_logical_consistency_equal_contradiction():
    elements = [
        {"id": "R1", "description": "timeout = 5"},
        {"id": "R2", "description": "timeout = 10"},
    ]
    result = _logical_consistency(elements)
    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["type"] == "numerical_contradiction"


def test_logical_consistency_forbidden_conflict():
    elements = [
        {"id": "R1", "description": "The system must not timeout"},
        {"id": "R2", "description": "timeout ≥ 5"},
    ]
    result = _logical_consistency(elements)
    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["type"] == "forbidden_conflict"


def test_logical_consistency_scoring_tier_5_no_contradictions():
    """Score 5: contradiction_rate == 0."""
    elements = [{"id": "R1", "description": "response_time ≤ 100"}]
    result = _logical_consistency(elements)
    assert result["score"] == 5
    assert result["contradiction_rate"] == pytest.approx(0.0)


def test_logical_consistency_scoring_tier_4():
    """Score 4: 0 < contradiction_rate <= 0.05 → 1/21 = 0.0476."""
    elements = [{"id": f"R{i}", "description": f"latency ≤ {i}"} for i in range(1, 22)]
    elements.append({"id": "RCONFLICT", "description": "latency ≥ 100"})
    result = _logical_consistency(elements)
    # All 21 latency≤i constraints conflict with latency≥100 where i < 100
    # contradiction_rate = contradictions / 22 elements
    assert result["score"] <= 4


def test_logical_consistency_scoring_tier_1_many_contradictions():
    """Score 1: contradiction_rate > 0.20."""
    elements = [
        {"id": "R1", "description": "latency ≤ 5"},
        {"id": "R2", "description": "latency ≥ 10"},
    ]
    result = _logical_consistency(elements)
    # 1 contradiction / 2 elements = 0.5 → score 1
    assert result["score"] == 1
    assert result["contradiction_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _terminology_consistency
# ---------------------------------------------------------------------------


def test_terminology_consistency_consistent():
    elements = [
        {"id": "R1", "description": "The user shall login."},
        {"id": "R2", "description": "The user shall logout."},
    ]
    result = _terminology_consistency(elements)
    assert result["score"] == 5
    assert result["consistency_ratio"] == pytest.approx(1.0)
    assert len(result["conflicts"]) == 0


def test_terminology_consistency_synonyms_detected():
    elements = [
        {"id": "R1", "description": "The user shall login."},
        {"id": "R2", "description": "The client shall register."},
    ]
    result = _terminology_consistency(elements)
    # "user" and "client" are synonyms in the 'user' group
    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["concept"] == "user"
    assert set(result["conflicts"][0]["terms_used"]) == {"user", "client"}


def test_terminology_consistency_no_concepts():
    elements = [{"id": "R1", "description": "xyz abc 12345"}]
    result = _terminology_consistency(elements)
    assert result["score"] == 5
    assert result["consistency_ratio"] == pytest.approx(1.0)
    assert result["total_concepts"] == 0


# ---------------------------------------------------------------------------
# _compliance_coverage
# ---------------------------------------------------------------------------


def test_compliance_coverage_full_coverage():
    requirement = "The system shall authenticate users and log all access events."
    elements = [
        {"name": "Auth", "description": "The system shall authenticate users"},
        {"name": "Log", "description": "The system shall log all access events"},
    ]
    result = _compliance_coverage(elements, requirement)
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["satisfied_applicable_clauses"] == result["total_applicable_clauses"]


def test_compliance_coverage_no_match():
    """Test when requirement text has no overlap with elements."""
    requirement = "The system shall fly to the moon using quantum teleportation."
    elements = [
        {"name": "Auth", "description": "The system shall authenticate users"},
    ]
    result = _compliance_coverage(elements, requirement)
    assert result["coverage_ratio"] == pytest.approx(0.0)
    assert result["satisfied_applicable_clauses"] == 0


def test_compliance_coverage_no_elements():
    result = _compliance_coverage([], "The system shall perform well.")
    assert result["coverage_ratio"] == pytest.approx(0.0)


def test_compliance_coverage_empty_requirement():
    result = _compliance_coverage([], "")
    assert result["total_applicable_clauses"] == 0


# ---------------------------------------------------------------------------
# _fact_checking
# ---------------------------------------------------------------------------


def test_fact_checking_with_source():
    elements = [{"id": "R1", "properties": {"source": "doc1.pdf"}}]
    result = _fact_checking(elements)
    assert len(result["hallucination_reports"]) == 0


def test_fact_checking_missing_source():
    elements = [{"id": "R1", "properties": {}}]
    result = _fact_checking(elements)
    assert len(result["hallucination_reports"]) == 1
    assert result["flagged_elements"] == ["R1"]


def test_fact_checking_no_properties():
    elements = [{"id": "R1"}]
    result = _fact_checking(elements)
    assert len(result["hallucination_reports"]) == 1


# ---------------------------------------------------------------------------
# _build_deterministic_validation
# ---------------------------------------------------------------------------


def test_build_deterministic_validation_not_executed():
    result = _build_deterministic_validation(
        topology_status={"is_valid": True},
        logical={"contradictions": []},
        verification_executed=False,
    )
    assert result["status"] == "not_executed"
    assert result["is_valid"] is False


def test_build_deterministic_validation_valid():
    result = _build_deterministic_validation(
        topology_status={"is_valid": True, "cycle_count": 0},
        logical={"contradictions": []},
        verification_executed=True,
    )
    assert result["is_valid"] is True
    assert "logical_consistency" in result["passed_rules"]


def test_build_deterministic_validation_with_contradictions():
    result = _build_deterministic_validation(
        topology_status={"is_valid": True, "cycle_count": 0},
        logical={"contradictions": [{"type": "numerical_contradiction", "left": {"element_id": "R1"}}]},
        verification_executed=True,
    )
    assert result["is_valid"] is False
    assert len(result["violations"]) == 1


def test_build_deterministic_validation_with_cycles():
    result = _build_deterministic_validation(
        topology_status={"is_valid": False, "cycle_count": 1},
        logical={"contradictions": []},
        verification_executed=True,
    )
    assert any(v["rule_id"] == "topology_cycle" for v in result["violations"])


# ---------------------------------------------------------------------------
# _quality_axis_for_agent
# ---------------------------------------------------------------------------


def test_quality_axis_for_agent_quare_agents():
    assert _quality_axis_for_agent("SafetyAgent") == "Safety"
    assert _quality_axis_for_agent("EfficiencyAgent") == "Efficiency"


def test_quality_axis_for_agent_mare_agents():
    assert _quality_axis_for_agent("Stakeholders") == "Responsibility"
    assert _quality_axis_for_agent("Checker") == "Safety"


def test_quality_axis_for_agent_unknown_agent():
    assert _quality_axis_for_agent("UnknownAgent") == "UnknownAgent"


def test_quality_axis_for_agent_whitespace():
    assert _quality_axis_for_agent("  SafetyAgent  ") == "Safety"


# ---------------------------------------------------------------------------
# _extract_external_rules
# ---------------------------------------------------------------------------


def test_extract_external_rules_basic():
    rules = _extract_external_rules(
        requirement="The system shall authenticate users.",
        fragments=["The system shall authenticate users"],
    )
    assert len(rules) == 1
    assert rules[0]["rule_id"] == "R001"


def test_extract_external_rules_trigger_detection():
    rules = _extract_external_rules(
        requirement="If the user logs in, then grant access.",
        fragments=["If the user logs in, then grant access to the system"],
    )
    assert any(r["rule_type"] == "trigger" for r in rules)


def test_extract_external_rules_obligation():
    rules = _extract_external_rules(
        requirement="The system must always log events to the audit trail.",
        fragments=["The system must always log events to the audit trail"],
    )
    assert any(r["rule_type"] == "obligation" for r in rules)


def test_extract_external_rules_prohibition():
    rules = _extract_external_rules(
        requirement="The restricted areas are forbidden for regular users of the platform.",
        fragments=["The restricted areas are forbidden for regular users of the platform"],
    )
    assert any(r["rule_type"] == "prohibition" for r in rules)


def test_extract_external_rules_empty_fragments_uses_requirement():
    rules = _extract_external_rules(
        requirement="The system shall perform validation and ensure data integrity.",
        fragments=[],
    )
    assert len(rules) == 1
    assert rules[0]["rule_id"] == "R001"


def test_extract_external_rules_caps_at_12():
    rules = _extract_external_rules(
        requirement="test",
        fragments=[f"Fragment {i} is long enough for a rule" for i in range(20)],
    )
    assert len(rules) <= 12


# ---------------------------------------------------------------------------
# _types.py coverage (import-level)
# ---------------------------------------------------------------------------


def test_types_pipeline_config_defaults():
    from pathlib import Path
    cfg = PipelineConfig(
        case_input=Path("/tmp/case.json"),
        artifacts_dir=Path("/tmp/artifacts"),
        run_record_path=Path("/tmp/run.json"),
        run_id="test-001",
        setting="single_agent",
        seed=42,
        model="gpt-4o-mini",
        temperature=0.7,
        round_cap=3,
        max_tokens=500,
    )
    assert cfg.system == "mare"
    assert cfg.rag_enabled is True
    assert cfg.llm_client is None


def test_types_runtime_meta_defaults():
    meta = RuntimeExecutionMeta()
    assert meta.llm_enabled is False
    assert meta.execution_mode == "deterministic_emulation"


def test_types_phase2_meta_defaults():
    meta = Phase2ExecutionMeta()
    assert meta.llm_source == "disabled"


def test_types_constants():
    assert PHASE2_LLM_RETRY_LIMIT == 1
    assert len(MARE_ROLE_QUALITY_ATTRIBUTES) == 5
    assert len(IREDEV_ROLE_QUALITY_ATTRIBUTES) == 6
    assert len(QUALITY_LENS_CUES) == 6  # Safety, Efficiency, Sustainability, Trustworthiness, Responsibility, Integrated
    assert set(QUALITY_LENS_CUES.keys()) == {
        "Safety", "Efficiency", "Sustainability",
        "Trustworthiness", "Responsibility", "Integrated",
    }
    assert MARE_RUNTIME_SEMANTICS_MODE == "mare_paper_workflow_v1"
    assert IREDEV_RUNTIME_SEMANTICS_MODE == "iredev_knowledge_driven_v1"
    assert MARE_RUNTIME_TRACE_VERSION == "1"
    assert IREDEV_RUNTIME_TRACE_VERSION == "1"

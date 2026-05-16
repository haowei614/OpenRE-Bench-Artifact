"""Regression tests for CHV/MDC quality-space metric behavior."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import openre_bench.comparison_harness as harness
from openre_bench.schemas import PHASE1_FILENAME
from openre_bench.schemas import PHASE2_FILENAME
from openre_bench.schemas import PHASE3_FILENAME
from openre_bench.schemas import PHASE4_FILENAME
from openre_bench.schemas import write_json_file


def test_compute_chv_mdc_rewards_multi_agent_quality_diversity() -> None:
    single_agent = [
        {
            "name": f"Integrated Requirement {index}",
            "description": (
                "Integrated requirement covering system behavior and consistency checks."
            ),
            "quality_attribute": "Integrated",
            "measurable_criteria": "Integrated acceptance criteria",
        }
        for index in range(6)
    ]
    multi_agent = [
        {
            "name": "Safety Goal",
            "description": "Mitigate hazard and fault scenarios with secure shutdown.",
            "quality_attribute": "Safety",
        },
        {
            "name": "Efficiency Goal",
            "description": "Reduce latency and optimize throughput under peak runtime load.",
            "quality_attribute": "Efficiency",
        },
        {
            "name": "Sustainability Goal",
            "description": "Lower energy and carbon overhead with green lifecycle controls.",
            "quality_attribute": "Sustainability",
        },
        {
            "name": "Trustworthiness Goal",
            "description": "Improve audit trace and verification evidence consistency.",
            "quality_attribute": "Trustworthiness",
        },
        {
            "name": "Responsibility Goal",
            "description": "Enforce regulatory compliance and accountability reporting.",
            "quality_attribute": "Responsibility",
        },
        {
            "name": "Safety Follow-up",
            "description": "Contain incidents and preserve safety integrity.",
            "quality_attribute": "Safety, Efficiency",
        },
    ]

    single_chv, single_mdc = harness._compute_chv_mdc(single_agent)
    multi_chv, multi_mdc = harness._compute_chv_mdc(multi_agent)

    assert multi_chv > single_chv
    assert multi_mdc > single_mdc


def test_compute_chv_mdc_returns_zero_for_identical_points() -> None:
    repeated = [
        {
            "name": "Same",
            "description": "same same same",
            "quality_attribute": "Integrated",
        }
        for _ in range(4)
    ]

    chv, mdc = harness._compute_chv_mdc(repeated)
    assert chv == 0.0
    assert mdc == 0.0


def test_compute_chv_mdc_is_deterministic_for_sparse_text() -> None:
    sparse = [
        {"name": "Token Alpha", "description": "x y z", "quality_attribute": ""},
        {"name": "Token Beta", "description": "q r s", "quality_attribute": ""},
        {"name": "Token Gamma", "description": "k l m", "quality_attribute": ""},
        {"name": "Token Delta", "description": "u v w", "quality_attribute": ""},
    ]

    first = harness._compute_chv_mdc(sparse)
    second = harness._compute_chv_mdc(sparse)

    assert first == second
    assert all(math.isfinite(value) for value in first)


def test_compute_chv_mdc_does_not_inflate_from_keyword_stuffing() -> None:
    baseline = [
        {
            "name": "Req A",
            "description": "basic requirement text",
            "quality_attribute": "Safety",
        },
        {
            "name": "Req B",
            "description": "basic requirement text",
            "quality_attribute": "Efficiency",
        },
        {
            "name": "Req C",
            "description": "basic requirement text",
            "quality_attribute": "Sustainability",
        },
        {
            "name": "Req D",
            "description": "basic requirement text",
            "quality_attribute": "Trustworthiness",
        },
        {
            "name": "Req E",
            "description": "basic requirement text",
            "quality_attribute": "Responsibility",
        },
        {
            "name": "Req F",
            "description": "basic requirement text",
            "quality_attribute": "Safety",
        },
    ]
    stuffed = [
        {
            "name": item["name"],
            "description": (
                "safe hazard risk secure verify compliance policy trust audit "
                "latency throughput energy carbon green ")
            * 4,
            "quality_attribute": item["quality_attribute"],
        }
        for item in baseline
    ]

    baseline_scores = harness._compute_chv_mdc(baseline)
    stuffed_scores = harness._compute_chv_mdc(stuffed)

    baseline_chv, baseline_mdc = baseline_scores
    stuffed_chv, stuffed_mdc = stuffed_scores

    assert math.isfinite(stuffed_chv)
    assert math.isfinite(stuffed_mdc)
    assert stuffed_chv <= (baseline_chv * 1.15)
    assert stuffed_mdc <= (baseline_mdc * 1.15)


def test_compute_chv_mdc_infers_axes_from_mare_role_metadata() -> None:
    elements = [
        {"stakeholder": "Stakeholders", "description": "role-tagged output"},
        {"stakeholder": "Collector", "description": "role-tagged output"},
        {"stakeholder": "Modeler", "description": "role-tagged output"},
        {"stakeholder": "Checker", "description": "role-tagged output"},
        {"stakeholder": "Documenter", "description": "role-tagged output"},
    ]

    chv, mdc = harness._compute_chv_mdc(elements)

    assert chv > 0.0
    assert mdc > 0.0


def test_compute_chv_mdc_uses_intrinsic_space_for_rank_deficient_points() -> None:
    elements = [
        {"quality_attribute": "Safety", "description": "same text"},
        {"quality_attribute": "Efficiency", "description": "same text"},
        {"quality_attribute": "Sustainability", "description": "same text"},
        {"quality_attribute": "Trustworthiness", "description": "same text"},
        {"quality_attribute": "Responsibility", "description": "same text"},
        {"quality_attribute": "Safety, Efficiency", "description": "same text"},
    ]

    chv, mdc = harness._compute_chv_mdc(elements)

    assert chv > 0.0
    assert mdc > 0.0


def test_compute_chv_mdc_integrated_metadata_does_not_force_uniform_vector() -> None:
    uniform_integrated = [
        {
            "quality_attribute": "Integrated",
            "description": "integrated requirement text",
        }
        for _ in range(4)
    ]

    integrated_elements = [
        {
            "quality_attribute": "Integrated",
            "description": "hazard mitigation and fail-safe operation",
        },
        {
            "quality_attribute": "Integrated",
            "description": "throughput optimization and latency reduction",
        },
        {
            "quality_attribute": "Integrated",
            "description": "audit evidence and traceability controls",
        },
        {
            "quality_attribute": "Integrated",
            "description": "regulatory compliance and accountability reporting",
        },
    ]

    uniform_chv, uniform_mdc = harness._compute_chv_mdc(uniform_integrated)
    varied_chv, varied_mdc = harness._compute_chv_mdc(integrated_elements)

    assert varied_mdc > uniform_mdc
    assert varied_chv >= uniform_chv


def test_quality_vector_preserves_absolute_semantic_axis_scale() -> None:
    semantic_scores = np.array([0.08, 0.16, 0.0, 0.04, 0.0], dtype=float)

    vector = harness._quality_vector_from_element(
        {"description": "metadata-light requirement"},
        semantic_scores=semantic_scores,
    )

    assert vector[0] == pytest.approx(0.08)
    assert vector[1] == pytest.approx(0.16)
    assert float(np.max(vector)) == pytest.approx(0.16)


def test_conflict_resolution_rate_stays_zero_when_no_conflicts_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        harness,
        "_semantic_preservation_f1",
        lambda *, candidates, references: 0.9,
    )

    phase1_payload = {
        "SafetyAgent": [
            {
                "id": "S-L1-001",
                "name": "Safety Root",
                "description": "Maintain safe operation.",
                "element_type": "Goal",
                "quality_attribute": "Safety",
                "hierarchy_level": 1,
            }
        ]
    }
    phase2_payload = {
        "total_negotiations": 1,
        "negotiations": {},
        "summary_stats": {
            "total_steps": 2,
            "successful_consensus": 1,
            "detected_conflicts": 0,
            "resolved_conflicts": 0,
        },
    }
    phase3_payload = {
        "gsn_elements": [
            {
                "id": "S-L1-001",
                "name": "Safety Root",
                "description": "Maintain safe operation.",
                "gsn_type": "Goal",
                "quality_attribute": "Safety",
                "parent_goal_id": None,
                "properties": {"source": "rag"},
            }
        ],
        "gsn_connections": [],
        "topology_status": {"is_valid": True},
    }
    phase4_payload = {
        "verification_results": {
            "s_logic": 1.0,
            "terminology_consistency": {"consistency_ratio": 1.0},
            "compliance_coverage": {"coverage_ratio": 1.0},
        },
        "deterministic_validation": {"is_valid": True},
    }

    write_json_file(tmp_path / PHASE1_FILENAME, phase1_payload)
    write_json_file(tmp_path / PHASE2_FILENAME, phase2_payload)
    write_json_file(tmp_path / PHASE3_FILENAME, phase3_payload)
    write_json_file(tmp_path / PHASE4_FILENAME, phase4_payload)

    metrics = harness._compute_run_metrics(tmp_path)
    assert metrics["conflict_resolution_rate"] == 0.0

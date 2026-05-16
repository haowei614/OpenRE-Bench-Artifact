"""Focused MVP tests for dual-runtime routing and strict metadata emission."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openre_bench.comparison_validator import validate_run_record
from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline._core import _latest_backward_elements
from openre_bench.pipeline import run_case_pipeline
from openre_bench.schemas import DEFAULT_AGENT_QUALITY_ATTRIBUTES
from openre_bench.schemas import PHASE1_FILENAME
from openre_bench.schemas import PHASE2_FILENAME
from openre_bench.schemas import PHASE3_FILENAME
from openre_bench.schemas import PHASE4_FILENAME
from openre_bench.schemas import SETTING_MULTI_AGENT_WITH_NEGOTIATION
from openre_bench.schemas import SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SETTING_SINGLE_AGENT
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import SYSTEM_QUARE
from openre_bench.schemas import load_json_file
from openre_bench.schemas import write_json_file
from tests.fake_mare_llm import ScriptedMareLLMClient


class StableQuareLLMClient:
    """Deterministic fake client that always returns resolved JSON."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        return (
            '{"analysis_text":"Resolved","feedback":"Accepted",'
            '"conflict_detected":false,"conflict_resolved":true,'
            '"requires_refinement":false,"element_updates":[]}'
        )


def _write_case(path: Path, *, requirement: str | None = None) -> None:
    case_requirement = requirement or (
        "The ATM shall authenticate users securely, detect fraudulent patterns, "
        "and complete approved withdrawals with low latency."
    )
    write_json_file(
        path,
        {
            "case_name": "ATM",
            "case_description": "ATM requirements with safety and efficiency concerns.",
            "requirement": case_requirement,
        },
    )


def _write_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "guidance.md").write_text(
        (
            "ATM domain guidance requires strong user authentication, transaction integrity, "
            "fraud detection controls, and performance limits for customer interactions."
        ),
        encoding="utf-8",
    )


def _run(
    tmp_path: Path,
    *,
    system: str,
    setting: str,
    llm_client: Any | None = None,
    requirement: str | None = None,
) -> tuple[Path, object]:
    case_path = tmp_path / f"{system}-{setting}-case.json"
    corpus_dir = tmp_path / f"{system}-{setting}-corpus"
    artifacts_dir = tmp_path / f"{system}-{setting}-artifacts"
    run_record_path = artifacts_dir / "run_record.json"
    _write_case(case_path, requirement=requirement)
    _write_corpus(corpus_dir)

    effective_llm_client = llm_client
    if effective_llm_client is None and system == SYSTEM_MARE and setting != SETTING_SINGLE_AGENT:
        effective_llm_client = ScriptedMareLLMClient()

    config = PipelineConfig(
        case_input=case_path,
        artifacts_dir=artifacts_dir,
        run_record_path=run_record_path,
        run_id=f"{system}-{setting}-s101",
        setting=setting,
        seed=101,
        model="gpt-4o-mini",
        temperature=0.7,
        round_cap=3,
        max_tokens=4000,
        system=system,
        rag_enabled=True,
        rag_backend="local_tfidf",
        rag_corpus_dir=corpus_dir,
        llm_client=effective_llm_client,
    )
    run_record = run_case_pipeline(config)
    return run_record_path, run_record


def test_dual_runtime_system_identity_and_artifacts(tmp_path: Path) -> None:
    mare_record_path, mare_record = _run(
        tmp_path,
        system=SYSTEM_MARE,
        setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    )
    quare_record_path, quare_record = _run(
        tmp_path,
        system=SYSTEM_QUARE,
        setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
        llm_client=StableQuareLLMClient(),
    )

    assert mare_record.system == SYSTEM_MARE
    assert quare_record.system == SYSTEM_QUARE
    assert mare_record.system_identity.system_name == SYSTEM_MARE
    assert quare_record.system_identity.system_name == SYSTEM_QUARE

    for run_record_path in (mare_record_path, quare_record_path):
        payload = load_json_file(run_record_path)
        artifact_paths = payload["artifact_paths"]
        for filename in (PHASE1_FILENAME, PHASE2_FILENAME, PHASE3_FILENAME, PHASE4_FILENAME):
            assert Path(artifact_paths[filename]).exists()

        assert len(payload["provenance"]["prompt_hash"]) == 64
        assert len(payload["provenance"]["corpus_hash"]) == 64
        report = validate_run_record(run_record_path)
        assert report.errors == []


def test_single_agent_emits_explicit_non_comparable_reason(tmp_path: Path) -> None:
    run_record_path, run_record = _run(
        tmp_path,
        system=SYSTEM_MARE,
        setting=SETTING_SINGLE_AGENT,
    )

    assert run_record.comparability.is_comparable is False
    assert run_record.comparability.non_comparable_reasons == [
        "single_agent_baseline_partial_phase_equivalence"
    ]

    report = validate_run_record(run_record_path)
    assert report.errors == []


def test_quare_multi_agent_phase1_emits_fixed_per_agent_leaf_budget(tmp_path: Path) -> None:
    run_record_path, _ = _run(
        tmp_path,
        system=SYSTEM_QUARE,
        setting=SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
    )
    payload = load_json_file(run_record_path)
    phase1 = load_json_file(Path(payload["artifact_paths"][PHASE1_FILENAME]))

    assert sorted(phase1.keys()) == sorted(
        [
            "SafetyAgent",
            "EfficiencyAgent",
            "GreenAgent",
            "TrustworthinessAgent",
            "ResponsibilityAgent",
        ]
    )
    assert all(len(elements) == 7 for elements in phase1.values())


def test_quare_multi_agent_phase1_window_avoids_pathological_fragment_repeats(tmp_path: Path) -> None:
    requirement = " ".join(
        (
            f"Clause {index:02d} requires deterministic handling of scenario {index:02d} "
            "for independent verification."
        )
        for index in range(1, 11)
    )
    run_record_path, _ = _run(
        tmp_path,
        system=SYSTEM_QUARE,
        setting=SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
        requirement=requirement,
    )
    payload = load_json_file(run_record_path)
    phase1 = load_json_file(Path(payload["artifact_paths"][PHASE1_FILENAME]))

    for elements in phase1.values():
        leaves = [item for item in elements if int(item.get("hierarchy_level", 1)) == 2]
        leaf_descriptions = [str(item.get("description", "")) for item in leaves]
        assert len(leaf_descriptions) == 6
        assert len(set(leaf_descriptions)) >= 5


def test_quare_multi_agent_phase1_window_covers_all_fragments_when_budget_allows(tmp_path: Path) -> None:
    requirement = " ".join(
        (
            f"Clause {index:02d} states that the service shall preserve scenario {index:02d} "
            "traceability for independent requirement validation."
        )
        for index in range(1, 25)
    )
    run_record_path, _ = _run(
        tmp_path,
        system=SYSTEM_QUARE,
        setting=SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
        requirement=requirement,
    )
    payload = load_json_file(run_record_path)
    phase1 = load_json_file(Path(payload["artifact_paths"][PHASE1_FILENAME]))

    matched_clause_ids: set[str] = set()
    clause_pattern = re.compile(r"Clause\s+(\d{2})")
    for elements in phase1.values():
        for element in elements:
            if int(element.get("hierarchy_level", 1)) != 2:
                continue
            description = str(element.get("description", ""))
            match = clause_pattern.search(description)
            if match:
                matched_clause_ids.add(match.group(1))

    expected_clause_ids = {f"{index:02d}" for index in range(1, 25)}
    assert expected_clause_ids.issubset(matched_clause_ids)


def test_quare_multi_agent_with_negotiation_compresses_phase3_elements(tmp_path: Path) -> None:
    run_record_path, _ = _run(
        tmp_path,
        system=SYSTEM_QUARE,
        setting=SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        llm_client=StableQuareLLMClient(),
    )
    payload = load_json_file(run_record_path)
    phase1 = load_json_file(Path(payload["artifact_paths"][PHASE1_FILENAME]))
    phase3 = load_json_file(Path(payload["artifact_paths"][PHASE3_FILENAME]))

    phase1_count = sum(len(elements) for elements in phase1.values())
    phase3_elements = phase3.get("gsn_elements", [])
    phase3_count = len(phase3_elements)
    phase3_leaf_qualities = {
        str(element.get("quality_attribute", "Integrated"))
        for element in phase3_elements
        if int(element.get("hierarchy_level", 1)) >= 2
    }
    assert phase1_count == 35
    assert phase3_count < phase1_count
    assert phase3_count <= 11
    assert set(DEFAULT_AGENT_QUALITY_ATTRIBUTES.values()).issubset(phase3_leaf_qualities)


def test_latest_backward_elements_prefers_latest_step_and_stable_order() -> None:
    phase2_payload = {
        "negotiations": {
            "pair_b": {
                "steps": [
                    {
                        "step_id": 1,
                        "message_type": "forward",
                        "kaos_elements": [{"id": "B-001", "description": "forward b"}],
                    },
                    {
                        "step_id": 2,
                        "message_type": "backward",
                        "kaos_elements": [
                            {"id": "A-001", "description": "older backward duplicate"},
                            {"id": "B-001", "description": "backward b"},
                        ],
                    },
                ]
            },
            "pair_a": {
                "steps": [
                    {
                        "step_id": 7,
                        "message_type": "backward",
                        "kaos_elements": [
                            {"id": "A-001", "description": "latest backward duplicate"}
                        ],
                    }
                ]
            },
            "pair_c": {
                "steps": [
                    {
                        "step_id": 3,
                        "message_type": "forward",
                        "kaos_elements": [{"id": "C-001", "description": "forward-only c"}],
                    }
                ]
            },
        }
    }

    elements = _latest_backward_elements(phase2_payload)
    assert [str(item["id"]) for item in elements] == ["A-001", "B-001", "C-001"]

    by_id = {str(item["id"]): str(item.get("description", "")) for item in elements}
    assert by_id["A-001"] == "latest backward duplicate"
    assert by_id["B-001"] == "backward b"
    assert by_id["C-001"] == "forward-only c"

"""Tests for QUARE-only protocol artifacts beyond canonical phase1-4 outputs."""

from __future__ import annotations

from pathlib import Path

from openre_bench.comparison_validator import validate_system_behavior_contract
from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline import run_case_pipeline
from openre_bench.schemas import PHASE0_FILENAME
from openre_bench.schemas import PHASE25_FILENAME
from openre_bench.schemas import PHASE5_FILENAME
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import SYSTEM_QUARE
from openre_bench.schemas import load_json_file
from openre_bench.schemas import write_json_file
from tests.fake_mare_llm import ScriptedMareLLMClient


class StableQuareLLMClient:
    """Deterministic fake client that returns valid QUARE review JSON."""

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


def _write_case(path: Path) -> None:
    write_json_file(
        path,
        {
            "case_name": "ATM",
            "case_description": "ATM requirement set with explicit tradeoff conflict.",
            "requirement": (
                "The ATM shall enforce strong anti-fraud controls while keeping low latency. "
                "If tradeoff conflicts appear, reviewers must negotiate and resolve them."
            ),
        },
    )


def _write_no_conflict_case(path: Path) -> None:
    write_json_file(
        path,
        {
            "case_name": "Library",
            "case_description": "Library requirement set with no explicit tradeoff trigger terms.",
            "requirement": (
                "The library system shall support secure member login, accurate catalog lookup, "
                "and audit logging for borrowing workflows."
            ),
        },
    )


def _write_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "guidance.md").write_text(
        (
            "ATM guidance prioritizes safety, efficiency, trustworthiness, and responsibility. "
            "Conflicts should be explicitly traced and resolved with auditable rationale."
        ),
        encoding="utf-8",
    )


def _run(tmp_path: Path, *, system: str) -> Path:
    case_path = tmp_path / f"{system}-case.json"
    corpus_dir = tmp_path / f"{system}-corpus"
    artifacts_dir = tmp_path / f"{system}-artifacts"
    run_record_path = artifacts_dir / "run_record.json"

    _write_case(case_path)
    _write_corpus(corpus_dir)

    llm_client = StableQuareLLMClient() if system == SYSTEM_QUARE else ScriptedMareLLMClient()
    run_case_pipeline(
        PipelineConfig(
            case_input=case_path,
            artifacts_dir=artifacts_dir,
            run_record_path=run_record_path,
            run_id=f"{system}-atm-s101",
            setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
            seed=101,
            model="gpt-4o-mini",
            temperature=0.7,
            round_cap=3,
            max_tokens=4000,
            system=system,
            rag_enabled=True,
            rag_backend="local_tfidf",
            rag_corpus_dir=corpus_dir,
            llm_client=llm_client,
        )
    )
    return artifacts_dir


def _run_no_conflict_case(tmp_path: Path, *, system: str) -> Path:
    case_path = tmp_path / f"{system}-no-conflict-case.json"
    corpus_dir = tmp_path / f"{system}-no-conflict-corpus"
    artifacts_dir = tmp_path / f"{system}-no-conflict-artifacts"
    run_record_path = artifacts_dir / "run_record.json"

    _write_no_conflict_case(case_path)
    _write_corpus(corpus_dir)

    llm_client = StableQuareLLMClient() if system == SYSTEM_QUARE else ScriptedMareLLMClient()
    run_case_pipeline(
        PipelineConfig(
            case_input=case_path,
            artifacts_dir=artifacts_dir,
            run_record_path=run_record_path,
            run_id=f"{system}-library-s101",
            setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
            seed=101,
            model="gpt-4o-mini",
            temperature=0.7,
            round_cap=3,
            max_tokens=4000,
            system=system,
            rag_enabled=True,
            rag_backend="local_tfidf",
            rag_corpus_dir=corpus_dir,
            llm_client=llm_client,
        )
    )
    return artifacts_dir


def test_quare_emits_optional_parity_artifacts(tmp_path: Path) -> None:
    artifacts_dir = _run(tmp_path, system=SYSTEM_QUARE)

    phase0 = artifacts_dir / PHASE0_FILENAME
    phase25 = artifacts_dir / PHASE25_FILENAME
    phase5 = artifacts_dir / PHASE5_FILENAME
    assert phase0.exists()
    assert phase25.exists()
    assert phase5.exists()

    phase0_payload = load_json_file(phase0)
    phase25_payload = load_json_file(phase25)
    phase5_payload = load_json_file(phase5)

    assert phase0_payload["phase"] == "0_external_spec_processing"
    assert isinstance(phase0_payload["extracted_rules"], list)

    assert phase25_payload["phase"] == "2.5_conflict_resolution"
    assert isinstance(phase25_payload["conflict_map"], dict)
    assert isinstance(phase25_payload["summary"], dict)

    assert phase5_payload["phase"] == "5_software_materials_generation"
    assert isinstance(phase5_payload["materials"], dict)
    assert isinstance(phase5_payload["quality_signals"], dict)

    behavior_report = validate_system_behavior_contract(
        system=SYSTEM_QUARE,
        artifacts_dir=artifacts_dir,
    )
    assert behavior_report.errors == []


def test_mare_does_not_emit_quare_only_artifacts(tmp_path: Path) -> None:
    artifacts_dir = _run(tmp_path, system=SYSTEM_MARE)
    assert not (artifacts_dir / PHASE0_FILENAME).exists()
    assert not (artifacts_dir / PHASE25_FILENAME).exists()
    assert not (artifacts_dir / PHASE5_FILENAME).exists()


def test_quare_behavior_contract_fails_when_optional_artifact_missing(tmp_path: Path) -> None:
    artifacts_dir = _run(tmp_path, system=SYSTEM_QUARE)
    (artifacts_dir / PHASE25_FILENAME).unlink()

    behavior_report = validate_system_behavior_contract(
        system=SYSTEM_QUARE,
        artifacts_dir=artifacts_dir,
    )
    assert any(PHASE25_FILENAME in error for error in behavior_report.errors)


def test_phase25_does_not_mark_no_conflict_pairs_as_resolved(tmp_path: Path) -> None:
    artifacts_dir = _run_no_conflict_case(tmp_path, system=SYSTEM_QUARE)
    phase25_payload = load_json_file(artifacts_dir / PHASE25_FILENAME)
    conflict_map = phase25_payload["conflict_map"]

    assert conflict_map
    for pair_payload in conflict_map.values():
        assert bool(pair_payload["detected_conflict"]) is False
        assert bool(pair_payload["resolved_conflict"]) is False

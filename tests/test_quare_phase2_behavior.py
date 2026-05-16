"""Behavioral separation tests for QUARE vs MARE phase-2 negotiation semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openre_bench.comparison_validator import validate_run_record
from openre_bench.comparison_validator import validate_system_behavior_contract
from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline import run_case_pipeline
from openre_bench.schemas import SETTING_MULTI_AGENT_WITH_NEGOTIATION
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import SYSTEM_QUARE
from openre_bench.schemas import load_json_file
from openre_bench.schemas import write_json_file
from tests.fake_mare_llm import ScriptedMareLLMClient


class AlternatingQuareLLMClient:
    """Deterministic fake client that forces one refinement then one resolution."""

    def __init__(self) -> None:
        self._call_count = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        self._call_count += 1
        if self._call_count % 2 == 1:
            payload = {
                "analysis_text": "Reviewer detects unresolved tradeoff and asks for refinement.",
                "feedback": "Conflict unresolved. Continue dialectic refinement.",
                "conflict_detected": True,
                "conflict_resolved": False,
                "requires_refinement": True,
                "element_updates": [],
            }
        else:
            payload = {
                "analysis_text": "Reviewer accepts refined proposal and records resolution.",
                "feedback": "Conflict resolved after dialectic refinement.",
                "conflict_detected": True,
                "conflict_resolved": True,
                "requires_refinement": False,
                "element_updates": [],
            }
        return json.dumps(payload)


class FlakyRecoveryLLMClient:
    """Fake client that triggers one retry and one parse recovery, then succeeds."""

    def __init__(self) -> None:
        self._call_count = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("transient failure")
        payload = {
            "analysis_text": "Recovered response",
            "feedback": "Proceed",
            "conflict_detected": False,
            "conflict_resolved": True,
            "requires_refinement": False,
            "element_updates": [],
        }
        return f"```json\n{json.dumps(payload)}\n```"


class AlwaysFailLLMClient:
    """Fake client that forces deterministic fallback on every turn."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        raise RuntimeError("llm unavailable")


class MalformedQuareLLMClient:
    """Fake client that returns recoverable QUARE JSON with raw string newlines."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        return """
        {
          "analysis_text": "Reviewer detects a recoverable tradeoff
          and requests dialectic refinement.",
          "feedback": "Continue refinement.",
          "conflict_detected": true,
          "conflict_resolved": false,
          "requires_refinement": true,
          "element_updates": []
        }
        """


class InconsistentTupleLLMClient:
    """Fake client that reports unresolved conflict without refinement flag."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        payload = {
            "analysis_text": "Conflict remains.",
            "feedback": "Unresolved.",
            "conflict_detected": True,
            "conflict_resolved": False,
            "requires_refinement": False,
            "element_updates": [],
        }
        return json.dumps(payload)


class PrematureResolutionLLMClient:
    """Fake client that resolves first-round conflicts without refinement."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        payload = {
            "analysis_text": "Conflict detected and resolved immediately.",
            "feedback": "Resolved.",
            "conflict_detected": True,
            "conflict_resolved": True,
            "requires_refinement": False,
            "element_updates": [],
        }
        return json.dumps(payload)


class ConflictThenAcceptLLMClient:
    """Fake client that reports conflict first, then clean acceptance."""

    def __init__(self) -> None:
        self._call_count = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        self._call_count += 1
        if self._call_count == 1:
            payload = {
                "analysis_text": "Conflict detected.",
                "feedback": "Refine once.",
                "conflict_detected": True,
                "conflict_resolved": False,
                "requires_refinement": True,
                "element_updates": [],
            }
        else:
            payload = {
                "analysis_text": "Refined model accepted.",
                "feedback": "Accepted.",
                "conflict_detected": False,
                "conflict_resolved": False,
                "requires_refinement": False,
                "element_updates": [],
            }
        return json.dumps(payload)


class AlwaysAcceptLLMClient:
    """Fake client that always accepts with no conflict and tracks calls."""

    def __init__(self) -> None:
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        self.call_count += 1
        payload = {
            "analysis_text": "No additional conflicts found.",
            "feedback": "Accepted as-is.",
            "conflict_detected": False,
            "conflict_resolved": False,
            "requires_refinement": False,
            "element_updates": [],
        }
        return json.dumps(payload)


class NoConflictResolvedTupleLLMClient:
    """Fake client that returns inconsistent no-conflict/resolved tuple."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        payload = {
            "analysis_text": "No conflict detected.",
            "feedback": "Accepted.",
            "conflict_detected": False,
            "conflict_resolved": True,
            "requires_refinement": False,
            "element_updates": [],
        }
        return json.dumps(payload)


class DescriptionRewriteLLMClient:
    """Fake client that proposes explicit description rewrites."""

    marker = "LLM rewrite marker"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        payload = json.loads(messages[-1]["content"])
        focus_elements = payload.get("focus_elements", [])
        target_id = None
        for item in focus_elements:
            if int(item.get("hierarchy_level", 1)) == 2:
                target_id = item.get("id")
                break
        updates = []
        if target_id:
            updates.append(
                {
                    "id": target_id,
                    "description": f"{self.marker}: rewritten by reviewer.",
                    "validation_status": "resolved",
                    "conflict_resolved_by": "EfficiencyAgent",
                }
            )

        response = {
            "analysis_text": "Reviewer provides structured rewrite.",
            "feedback": "Accepted.",
            "conflict_detected": False,
            "conflict_resolved": False,
            "requires_refinement": False,
            "element_updates": updates,
        }
        return json.dumps(response)


class AlwaysRefineLLMClient:
    """Fake client that keeps conflicts unresolved across all rounds."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        payload = {
            "analysis_text": "Conflict remains unresolved.",
            "feedback": "Continue refinement.",
            "conflict_detected": True,
            "conflict_resolved": False,
            "requires_refinement": True,
            "element_updates": [],
        }
        return json.dumps(payload)


def _write_case(path: Path) -> None:
    write_json_file(
        path,
        {
            "case_name": "ATM",
            "case_description": "ATM requirements with explicit safety-performance conflicts.",
            "requirement": (
                "The ATM shall enforce strong fraud detection and strict authentication while "
                "maintaining low latency; if a tradeoff conflict appears, the system shall "
                "negotiate and resolve it with explicit reviewer feedback."
            ),
        },
    )


def _write_non_conflict_case(path: Path) -> None:
    write_json_file(
        path,
        {
            "case_name": "Library",
            "case_description": "Library system requirement with no explicit tradeoff words.",
            "requirement": (
                "The library system shall support secure member login and accurate catalog lookup "
                "with clear audit logging for borrowing workflows."
            ),
        },
    )


def _write_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "guidance.md").write_text(
        (
            "ATM guidance prioritizes safety, efficiency, trustworthiness, and responsibility. "
            "Tradeoff conflicts between fraud controls and response-time goals must be handled "
            "through auditable negotiation and clear conflict-resolution rationale."
        ),
        encoding="utf-8",
    )


def _run_case(
    tmp_path: Path,
    *,
    system: str,
    llm_client: Any | None = None,
) -> tuple[Path, dict[str, Any]]:
    return _run_case_for_setting(
        tmp_path,
        system=system,
        setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
        llm_client=llm_client,
    )


def _run_case_for_setting(
    tmp_path: Path,
    *,
    system: str,
    setting: str,
    llm_client: Any | None = None,
    round_cap: int = 3,
) -> tuple[Path, dict[str, Any]]:
    case_path = tmp_path / f"{system}-case.json"
    corpus_dir = tmp_path / f"{system}-corpus"
    artifacts_dir = tmp_path / f"{system}-artifacts"
    run_record_path = artifacts_dir / "run_record.json"

    _write_case(case_path)
    _write_corpus(corpus_dir)

    effective_llm_client = llm_client
    if effective_llm_client is None and system == SYSTEM_MARE:
        effective_llm_client = ScriptedMareLLMClient()

    run_record = run_case_pipeline(
        PipelineConfig(
            case_input=case_path,
            artifacts_dir=artifacts_dir,
            run_record_path=run_record_path,
            run_id=f"{system}-atm-s101",
            setting=setting,
            seed=101,
            model="gpt-4o-mini",
            temperature=0.7,
            round_cap=round_cap,
            max_tokens=4000,
            system=system,
            rag_enabled=True,
            rag_backend="local_tfidf",
            rag_corpus_dir=corpus_dir,
            llm_client=effective_llm_client,
        )
    )
    phase2_path = Path(run_record.artifact_paths["phase2_negotiation_trace.json"])
    return phase2_path, load_json_file(phase2_path)


def _run_case_no_conflict_terms(
    tmp_path: Path,
    *,
    system: str,
    llm_client: Any | None = None,
) -> tuple[Path, dict[str, Any]]:
    case_path = tmp_path / f"{system}-no-conflict-case.json"
    corpus_dir = tmp_path / f"{system}-no-conflict-corpus"
    artifacts_dir = tmp_path / f"{system}-no-conflict-artifacts"
    run_record_path = artifacts_dir / "run_record.json"

    _write_non_conflict_case(case_path)
    _write_corpus(corpus_dir)

    effective_llm_client = llm_client
    if effective_llm_client is None and system == SYSTEM_MARE:
        effective_llm_client = ScriptedMareLLMClient()

    run_record = run_case_pipeline(
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
            llm_client=effective_llm_client,
        )
    )
    phase2_path = Path(run_record.artifact_paths["phase2_negotiation_trace.json"])
    return phase2_path, load_json_file(phase2_path)


def test_quare_phase2_multi_turn_llm_path_separates_from_mare(tmp_path: Path) -> None:
    mare_phase2_path, mare_phase2 = _run_case(tmp_path, system=SYSTEM_MARE)
    quare_phase2_path, quare_phase2 = _run_case(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=AlternatingQuareLLMClient(),
    )

    mare_rounds = [
        int(negotiation["total_rounds"])
        for negotiation in mare_phase2["negotiations"].values()
    ]
    assert mare_rounds
    assert all(rounds == 1 for rounds in mare_rounds)

    mare_steps = [
        step
        for negotiation in mare_phase2["negotiations"].values()
        for step in negotiation["steps"]
    ]
    assert all(step.get("negotiation_mode") in {None, ""} for step in mare_steps)

    quare_rounds = [
        int(negotiation["total_rounds"])
        for negotiation in quare_phase2["negotiations"].values()
    ]
    assert quare_rounds
    assert any(rounds > 1 for rounds in quare_rounds)

    quare_steps = [
        step
        for negotiation in quare_phase2["negotiations"].values()
        for step in negotiation["steps"]
    ]
    assert any(step.get("negotiation_mode") == "quare_dialectic" for step in quare_steps)

    quare_summary = quare_phase2["summary_stats"]
    assert int(quare_summary["llm_turns"]) > 0
    assert int(quare_summary["llm_fallback_turns"]) == 0
    assert int(quare_summary["llm_retry_count"]) == 0
    assert int(quare_summary["llm_parse_recoveries"]) == 0

    assert int(quare_phase2["summary_stats"]["total_steps"]) > int(
        mare_phase2["summary_stats"]["total_steps"]
    )

    mare_run_report = validate_run_record(mare_phase2_path.parent / "run_record.json")
    quare_run_report = validate_run_record(quare_phase2_path.parent / "run_record.json")
    assert mare_run_report.errors == []
    assert quare_run_report.errors == []

    mare_behavior_report = validate_system_behavior_contract(
        system=SYSTEM_MARE,
        artifacts_dir=mare_phase2_path.parent,
    )
    quare_behavior_report = validate_system_behavior_contract(
        system=SYSTEM_QUARE,
        artifacts_dir=quare_phase2_path.parent,
    )
    assert mare_behavior_report.errors == []
    assert quare_behavior_report.errors == []


def test_quare_fallback_sets_taint(tmp_path: Path) -> None:
    quare_phase2_path, quare_phase2 = _run_case(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=AlwaysFailLLMClient(),
    )

    summary = quare_phase2["summary_stats"]
    assert int(summary["llm_turns"]) == 0
    assert int(summary["llm_fallback_turns"]) > 0

    report = validate_run_record(quare_phase2_path.parent / "run_record.json")
    assert any("Fallback-tainted metadata detected" in error for error in report.errors)


def test_quare_transport_retry_does_not_mark_retry_taint(tmp_path: Path) -> None:
    quare_phase2_path, quare_phase2 = _run_case(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=FlakyRecoveryLLMClient(),
    )

    summary = quare_phase2["summary_stats"]
    assert int(summary["llm_retry_count"]) == 0
    assert int(summary["llm_parse_recoveries"]) > 0

    report = validate_run_record(quare_phase2_path.parent / "run_record.json")
    assert all("Retry-tainted metadata detected" not in error for error in report.errors)


def test_quare_recovers_malformed_llm_payload_without_retry_taint(tmp_path: Path) -> None:
    quare_phase2_path, quare_phase2 = _run_case(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=MalformedQuareLLMClient(),
    )

    summary = quare_phase2["summary_stats"]
    assert int(summary["llm_turns"]) > 0
    assert int(summary["llm_fallback_turns"]) == 0
    assert int(summary["llm_retry_count"]) == 0
    assert int(summary["llm_parse_recoveries"]) > 0

    report = validate_run_record(quare_phase2_path.parent / "run_record.json")
    assert all("Fallback-tainted metadata detected" not in error for error in report.errors)
    assert all("Retry-tainted metadata detected" not in error for error in report.errors)


def test_mare_guardrail_rejects_quare_trace_markers(tmp_path: Path) -> None:
    mare_phase2_path, mare_phase2 = _run_case(tmp_path, system=SYSTEM_MARE)

    first_negotiation = next(iter(mare_phase2["negotiations"].values()))
    first_step = first_negotiation["steps"][0]
    first_step["negotiation_mode"] = "quare_dialectic"
    write_json_file(mare_phase2_path, mare_phase2)

    behavior_report = validate_system_behavior_contract(
        system=SYSTEM_MARE,
        artifacts_dir=mare_phase2_path.parent,
    )
    assert any("MARE baseline guardrail failed" in error for error in behavior_report.errors)


def test_quare_coerces_inconsistent_unresolved_tuple_into_refinement(tmp_path: Path) -> None:
    _, quare_phase2 = _run_case(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=InconsistentTupleLLMClient(),
    )

    rounds = [int(negotiation["total_rounds"]) for negotiation in quare_phase2["negotiations"].values()]
    assert rounds
    assert any(round_count > 1 for round_count in rounds)

    backward_steps = [
        step
        for negotiation in quare_phase2["negotiations"].values()
        for step in negotiation["steps"]
        if step.get("message_type") == "backward"
    ]
    assert backward_steps
    assert any(bool(step.get("requires_refinement")) for step in backward_steps)


def test_quare_forces_refinement_for_first_round_conflict_signal(tmp_path: Path) -> None:
    _, quare_phase2 = _run_case_no_conflict_terms(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=PrematureResolutionLLMClient(),
    )

    assert int(quare_phase2["summary_stats"]["detected_conflicts"]) > 0
    rounds = [int(negotiation["total_rounds"]) for negotiation in quare_phase2["negotiations"].values()]
    assert rounds
    assert any(round_count > 1 for round_count in rounds)


def test_quare_accept_followup_marks_conflicts_resolved(tmp_path: Path) -> None:
    _, quare_phase2 = _run_case_no_conflict_terms(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=ConflictThenAcceptLLMClient(),
    )

    summary = quare_phase2["summary_stats"]
    assert int(summary["detected_conflicts"]) > 0
    assert int(summary["resolved_conflicts"]) == int(summary["detected_conflicts"])

    for negotiation in quare_phase2["negotiations"].values():
        assert bool(negotiation["final_consensus"]) is True


def test_quare_skips_first_round_llm_call_for_deterministic_conflict(tmp_path: Path) -> None:
    client = AlwaysAcceptLLMClient()
    _, quare_phase2 = _run_case(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=client,
    )

    summary = quare_phase2["summary_stats"]
    total_negotiations = int(quare_phase2["total_negotiations"])
    assert int(summary["llm_turns"]) == client.call_count
    assert int(summary["llm_turns"]) == total_negotiations

    backward_steps = [
        step
        for negotiation in quare_phase2["negotiations"].values()
        for step in negotiation["steps"]
        if step.get("message_type") == "backward"
    ]
    assert any(
        "pre-LLM refinement guardrail" in str(step.get("analysis_text", ""))
        for step in backward_steps
    )


def test_quare_niv_blocks_description_rewrites_but_multi_agent_allows_them(tmp_path: Path) -> None:
    client = DescriptionRewriteLLMClient()

    _, niv_phase2 = _run_case_for_setting(
        tmp_path / "niv",
        system=SYSTEM_QUARE,
        setting=SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
        llm_client=client,
    )
    _, man_phase2 = _run_case_for_setting(
        tmp_path / "man",
        system=SYSTEM_QUARE,
        setting=SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        llm_client=client,
    )

    niv_descriptions = [
        str(element.get("description", ""))
        for negotiation in niv_phase2["negotiations"].values()
        for step in negotiation["steps"]
        if step.get("message_type") == "backward"
        for element in step.get("kaos_elements", [])
    ]
    man_descriptions = [
        str(element.get("description", ""))
        for negotiation in man_phase2["negotiations"].values()
        for step in negotiation["steps"]
        if step.get("message_type") == "backward"
        for element in step.get("kaos_elements", [])
    ]

    assert all(DescriptionRewriteLLMClient.marker not in text for text in niv_descriptions)
    assert any(DescriptionRewriteLLMClient.marker in text for text in man_descriptions)


def test_quare_no_conflict_tuple_is_not_coerced_into_conflict(tmp_path: Path) -> None:
    _, quare_phase2 = _run_case_no_conflict_terms(
        tmp_path,
        system=SYSTEM_QUARE,
        llm_client=NoConflictResolvedTupleLLMClient(),
    )

    summary = quare_phase2["summary_stats"]
    assert int(summary["detected_conflicts"]) == 0
    assert int(summary["resolved_conflicts"]) == 0

    backward_steps = [
        step
        for negotiation in quare_phase2["negotiations"].values()
        for step in negotiation["steps"]
        if step.get("message_type") == "backward"
    ]
    assert backward_steps
    assert all(step.get("requires_refinement") is False for step in backward_steps)


def test_quare_round_cap_preserves_unresolved_conflicts(tmp_path: Path) -> None:
    phase2_path, phase2 = _run_case_for_setting(
        tmp_path,
        system=SYSTEM_QUARE,
        setting=SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        llm_client=AlwaysRefineLLMClient(),
        round_cap=1,
    )

    summary = phase2["summary_stats"]
    assert int(summary["detected_conflicts"]) > 0
    assert int(summary["resolved_conflicts"]) == 0
    assert int(summary["round_cap_hits"]) > 0

    backward_steps = [
        step
        for negotiation in phase2["negotiations"].values()
        for step in negotiation["steps"]
        if step.get("message_type") == "backward"
    ]
    assert backward_steps
    assert any(step.get("resolution_state") == "unresolved" for step in backward_steps)
    assert any("round cap reached" in str(step.get("analysis_text", "")).lower() for step in backward_steps)

    phase25 = load_json_file(phase2_path.parent / "phase2_conflict_map.json")
    phase25_summary = phase25["summary"]
    assert int(phase25_summary["detected_conflict_pairs"]) > 0
    assert int(phase25_summary["resolved_conflict_pairs"]) == 0
    assert int(phase25_summary["unresolved_conflict_pairs"]) == int(
        phase25_summary["detected_conflict_pairs"]
    )

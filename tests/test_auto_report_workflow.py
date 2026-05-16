"""Focused tests for /auto report helper determinism and resume checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openre_bench.auto_report import _build_verdict
from openre_bench.auto_report import _build_paper_claim_validation
from openre_bench.auto_report import _collect_system_stats
from openre_bench.auto_report import _build_run_key
from openre_bench.auto_report import _audit_precision_f1_contract
from openre_bench.auto_report import _evaluate_paper_control_profile
from openre_bench.auto_report import _generate_conversation_logs
from openre_bench.auto_report import _matrix_outputs_complete
from openre_bench.auto_report import _mirror_latest_outputs
from openre_bench.auto_report import _write_report_readme
from openre_bench.schemas import PHASE1_FILENAME
from openre_bench.schemas import PHASE2_FILENAME
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import SYSTEM_QUARE
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION


def _paper_controls_pass() -> dict[str, object]:
    return {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "seeds": [101, 202, 303],
        "settings": [
            "single_agent",
            "multi_agent_without_negotiation",
            "multi_agent_with_negotiation",
            "negotiation_integration_verification",
        ],
    }


def _paper_control_check_pass() -> dict[str, object]:
    return {"is_paper_matched": True, "mismatches": []}


def _paper_claims_pass() -> dict[str, object]:
    return {"claim_summary": {"hard_fail_count": 0, "blocked_hard_fail_count": 0}}


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_build_run_key_is_deterministic_for_same_controls() -> None:
    controls_a = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "seeds": [101, 202],
        "settings": ["single_agent", "negotiation_integration_verification"],
    }
    controls_b = {
        "settings": ["single_agent", "negotiation_integration_verification"],
        "seeds": [101, 202],
        "temperature": 0.7,
        "model": "gpt-4o-mini",
    }

    assert _build_run_key(controls_a) == _build_run_key(controls_b)


def test_matrix_outputs_complete_accepts_strict_complete_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "mare"
    _touch(output_dir / "comparison_metrics_by_case.csv")
    _touch(output_dir / "comparison_metrics_summary.csv")
    _touch(output_dir / "comparison_ablation_table.csv")
    _touch(output_dir / "comparison_validity_log.md")
    (output_dir / "comparison_runs.jsonl").write_text(
        "\n".join(
            [
                '{"run_id":"mare-atm-single_agent-s101","system":"mare","validation_passed":true}',
                '{"run_id":"mare-atm-single_agent-s202","system":"mare","validation_passed":true}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    complete, reason = _matrix_outputs_complete(
        output_dir=output_dir,
        expected_runs=2,
        system="mare",
    )
    assert complete is True
    assert reason == "all strict checks passed"


def test_matrix_outputs_complete_rejects_failed_validation_row(tmp_path: Path) -> None:
    output_dir = tmp_path / "quare"
    _touch(output_dir / "comparison_metrics_by_case.csv")
    _touch(output_dir / "comparison_metrics_summary.csv")
    _touch(output_dir / "comparison_ablation_table.csv")
    _touch(output_dir / "comparison_validity_log.md")
    (output_dir / "comparison_runs.jsonl").write_text(
        '{"run_id":"quare-atm-single_agent-s101","system":"quare","validation_passed":false}\n',
        encoding="utf-8",
    )

    complete, reason = _matrix_outputs_complete(
        output_dir=output_dir,
        expected_runs=1,
        system="quare",
    )
    assert complete is False
    assert "validation failed" in reason


def _sample_phase2_payload() -> dict[str, object]:
    return {
        "total_negotiations": 1,
        "negotiations": {
            "SafetyAgent_EfficiencyAgent": {
                "negotiation_id": "n1",
                "focus_agent": "SafetyAgent",
                "reviewer_agents": ["EfficiencyAgent"],
                "start_timestamp": "2026-02-15T00:00:00Z",
                "end_timestamp": "2026-02-15T00:00:01Z",
                "final_consensus": True,
                "total_rounds": 1,
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-02-15T00:00:00Z",
                        "focus_agent": "SafetyAgent",
                        "reviewer_agent": "EfficiencyAgent",
                        "round_number": 1,
                        "message_type": "forward",
                        "analysis_text": "Forward analysis",
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-02-15T00:00:01Z",
                        "focus_agent": "SafetyAgent",
                        "reviewer_agent": "EfficiencyAgent",
                        "round_number": 1,
                        "message_type": "backward",
                        "analysis_text": "Backward analysis",
                        "feedback": "Resolved",
                        "negotiation_mode": "quare_dialectic",
                        "conflict_detected": True,
                        "resolution_state": "resolved",
                        "requires_refinement": False,
                    },
                ],
            }
        },
        "summary_stats": {
            "detected_conflicts": 1,
            "resolved_conflicts": 1,
            "llm_turns": 1,
        },
    }


def _sample_phase2_without_negotiation_payload() -> dict[str, object]:
    return {
        "total_negotiations": 0,
        "negotiations": {},
        "summary_stats": {
            "total_steps": 0,
            "successful_consensus": 0,
            "average_rounds": 0.0,
            "detected_conflicts": 0,
            "resolved_conflicts": 0,
            "llm_enabled": False,
            "llm_turns": 0,
            "llm_fallback_turns": 0,
            "llm_retry_count": 0,
            "llm_parse_recoveries": 0,
            "llm_source": "disabled",
        },
    }


def _sample_phase1_payload() -> dict[str, object]:
    return {
        "SafetyAgent": [{"id": "SAFE-L1-001"}],
        "EfficiencyAgent": [{"id": "EFF-L1-001"}],
    }


def _make_row(
    tmp_path: Path,
    *,
    system: str,
    run_id: str,
    phase2_payload: dict[str, object] | None = None,
    phase1_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    artifacts_dir = tmp_path / "artifacts" / system / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "phase2_negotiation_trace.json").write_text(
        json.dumps(phase2_payload if phase2_payload is not None else _sample_phase2_payload()),
        encoding="utf-8",
    )
    if phase1_payload is not None:
        (artifacts_dir / PHASE1_FILENAME).write_text(json.dumps(phase1_payload), encoding="utf-8")

    return {
        "run_id": run_id,
        "system": system,
        "case_id": "atm",
        "setting": SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
        "seed": 101,
        "artifacts_dir": str(artifacts_dir),
    }


def test_generate_conversation_logs_creates_index_and_agent_markdown(tmp_path: Path) -> None:
    logs_dir = tmp_path / "report" / "logs" / "auto-test"
    logs_dir.mkdir(parents=True, exist_ok=True)
    mare_row = _make_row(tmp_path, system="mare", run_id="mare-atm-niv-s101")

    summary = _generate_conversation_logs(
        run_key="auto-test",
        logs_dir=logs_dir,
        mare_rows=[mare_row],
        quare_rows=[],
    )

    assert summary["is_complete"] is True
    assert summary["expected_runs"] == 1
    assert summary["complete_runs"] == 1

    index_path = logs_dir / "conversation_index.jsonl"
    assert index_path.exists()
    rows = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1

    conversations_root = logs_dir / "conversations"
    timeline_path = next(conversations_root.glob("**/timeline.md"))
    agent_logs = list(conversations_root.glob("**/agents/*.md"))
    assert timeline_path.exists()
    assert len(agent_logs) == 2


def test_generate_conversation_logs_reuses_fresh_bundle_without_regeneration(tmp_path: Path) -> None:
    logs_dir = tmp_path / "report" / "logs" / "auto-test"
    logs_dir.mkdir(parents=True, exist_ok=True)
    quare_row = _make_row(tmp_path, system="quare", run_id="quare-atm-niv-s101")

    first = _generate_conversation_logs(
        run_key="auto-test",
        logs_dir=logs_dir,
        mare_rows=[],
        quare_rows=[quare_row],
    )
    second = _generate_conversation_logs(
        run_key="auto-test",
        logs_dir=logs_dir,
        mare_rows=[],
        quare_rows=[quare_row],
    )

    assert first["regenerated_runs"] == 1
    assert second["regenerated_runs"] == 0
    assert second["reused_runs"] == 1


def test_generate_conversation_logs_uses_phase1_agents_when_phase2_has_no_negotiations(
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "report" / "logs" / "auto-test"
    logs_dir.mkdir(parents=True, exist_ok=True)
    mare_row = _make_row(
        tmp_path,
        system="mare",
        run_id="mare-atm-single-s101",
        phase2_payload=_sample_phase2_without_negotiation_payload(),
        phase1_payload=_sample_phase1_payload(),
    )

    summary = _generate_conversation_logs(
        run_key="auto-test",
        logs_dir=logs_dir,
        mare_rows=[mare_row],
        quare_rows=[],
    )

    assert summary["is_complete"] is True
    assert summary["runs_with_missing_agent_logs"] == 0

    agent_logs = list((logs_dir / "conversations").glob("**/agents/*.md"))
    assert len(agent_logs) == 2


def test_generate_conversation_logs_requires_agents_to_mark_bundle_complete(tmp_path: Path) -> None:
    logs_dir = tmp_path / "report" / "logs" / "auto-test"
    logs_dir.mkdir(parents=True, exist_ok=True)
    mare_row = _make_row(
        tmp_path,
        system="mare",
        run_id="mare-atm-single-s101",
        phase2_payload=_sample_phase2_without_negotiation_payload(),
    )

    summary = _generate_conversation_logs(
        run_key="auto-test",
        logs_dir=logs_dir,
        mare_rows=[mare_row],
        quare_rows=[],
    )

    assert summary["is_complete"] is False
    assert summary["complete_runs"] == 0
    assert summary["runs_with_missing_agent_logs"] == 1

    index_rows = [
        json.loads(line)
        for line in (logs_dir / "conversation_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(index_rows) == 1
    assert index_rows[0]["log_complete"] is False
    assert index_rows[0]["missing_expected_agents"] is True


def test_verdict_fails_when_conversation_logs_are_incomplete() -> None:
    mare_stats = {
        "total_runs": 1,
        "validation_passed_runs": 1,
        "validation_error_items": 0,
        "validation_warning_items": 0,
        "fallback_tainted_runs": 0,
        "retry_used_runs": 0,
        "rag_fallback_used_runs": 0,
        "llm_fallback_used_runs": 0,
        "quare_mode_markers": 0,
        "llm_turns_sum": 0,
        "metadata_incomplete_runs": 0,
    }
    quare_stats = {
        "total_runs": 1,
        "validation_passed_runs": 1,
        "validation_error_items": 0,
        "validation_warning_items": 0,
        "fallback_tainted_runs": 0,
        "retry_used_runs": 0,
        "rag_fallback_used_runs": 0,
        "llm_fallback_used_runs": 0,
        "quare_mode_markers": 1,
        "llm_turns_sum": 1,
        "metadata_incomplete_runs": 0,
    }
    replay = {
        "systems": {
            "mare": {"error_items": 0, "warning_items": 0},
            "quare": {"error_items": 0, "warning_items": 0},
        }
    }
    deltas = {
        "by_setting": {
            SETTING_NEGOTIATION_INTEGRATION_VERIFICATION: {
                "semantic_preservation_f1": {"quare_minus_mare": 0.0}
            }
        }
    }
    conversation_summary = {
        "is_complete": False,
        "coverage_ratio": 0.5,
        "complete_runs": 1,
        "expected_runs": 2,
    }

    verdict = _build_verdict(
        expected_runs=1,
        settings=[SETTING_NEGOTIATION_INTEGRATION_VERIFICATION],
        controls=_paper_controls_pass(),
        mare_stats=mare_stats,
        quare_stats=quare_stats,
        replay=replay,
        deltas=deltas,
        paper_control_check=_paper_control_check_pass(),
        paper_claims=_paper_claims_pass(),
        conversation_summary=conversation_summary,
    )

    assert verdict["check_results"]["conversation_logs_complete"] is False
    assert verdict["final_completion_verdict"] == "NO-GO"


def test_verdict_fails_when_mare_runtime_semantics_are_incomplete() -> None:
    mare_stats = {
        "total_runs": 1,
        "validation_passed_runs": 1,
        "validation_error_items": 0,
        "validation_warning_items": 0,
        "fallback_tainted_runs": 0,
        "retry_used_runs": 0,
        "rag_fallback_used_runs": 0,
        "llm_fallback_used_runs": 0,
        "quare_mode_markers": 0,
        "llm_turns_sum": 0,
        "metadata_incomplete_runs": 0,
        "mare_semantics_incomplete_runs": 1,
    }
    quare_stats = {
        "total_runs": 1,
        "validation_passed_runs": 1,
        "validation_error_items": 0,
        "validation_warning_items": 0,
        "fallback_tainted_runs": 0,
        "retry_used_runs": 0,
        "rag_fallback_used_runs": 0,
        "llm_fallback_used_runs": 0,
        "quare_mode_markers": 1,
        "llm_turns_sum": 1,
        "metadata_incomplete_runs": 0,
    }
    replay = {
        "systems": {
            "mare": {"error_items": 0, "warning_items": 0},
            "quare": {"error_items": 0, "warning_items": 0},
        }
    }
    deltas = {
        "by_setting": {
            SETTING_NEGOTIATION_INTEGRATION_VERIFICATION: {
                "semantic_preservation_f1": {"quare_minus_mare": 0.0}
            }
        }
    }
    conversation_summary = {
        "is_complete": True,
        "coverage_ratio": 1.0,
        "complete_runs": 2,
        "expected_runs": 2,
    }

    verdict = _build_verdict(
        expected_runs=1,
        settings=[SETTING_NEGOTIATION_INTEGRATION_VERIFICATION],
        controls=_paper_controls_pass(),
        mare_stats=mare_stats,
        quare_stats=quare_stats,
        replay=replay,
        deltas=deltas,
        paper_control_check=_paper_control_check_pass(),
        paper_claims=_paper_claims_pass(),
        conversation_summary=conversation_summary,
    )

    assert verdict["check_results"]["runtime_semantics_complete"] is False
    assert verdict["final_completion_verdict"] == "NO-GO"


def test_verdict_fails_when_mare_runtime_is_emulated() -> None:
    mare_stats = {
        "total_runs": 1,
        "validation_passed_runs": 1,
        "validation_error_items": 0,
        "validation_warning_items": 0,
        "fallback_tainted_runs": 0,
        "retry_used_runs": 0,
        "rag_fallback_used_runs": 0,
        "llm_fallback_used_runs": 0,
        "quare_mode_markers": 0,
        "llm_turns_sum": 9,
        "metadata_incomplete_runs": 0,
        "mare_semantics_incomplete_runs": 0,
        "mare_llm_emulation_runs": 1,
    }
    quare_stats = {
        "total_runs": 1,
        "validation_passed_runs": 1,
        "validation_error_items": 0,
        "validation_warning_items": 0,
        "fallback_tainted_runs": 0,
        "retry_used_runs": 0,
        "rag_fallback_used_runs": 0,
        "llm_fallback_used_runs": 0,
        "quare_mode_markers": 1,
        "llm_turns_sum": 1,
        "metadata_incomplete_runs": 0,
    }
    replay = {
        "systems": {
            "mare": {"error_items": 0, "warning_items": 0},
            "quare": {"error_items": 0, "warning_items": 0},
        }
    }
    deltas = {
        "by_setting": {
            SETTING_NEGOTIATION_INTEGRATION_VERIFICATION: {
                "semantic_preservation_f1": {"quare_minus_mare": 0.0}
            }
        }
    }
    conversation_summary = {
        "is_complete": True,
        "coverage_ratio": 1.0,
        "complete_runs": 2,
        "expected_runs": 2,
    }

    verdict = _build_verdict(
        expected_runs=1,
        settings=[SETTING_NEGOTIATION_INTEGRATION_VERIFICATION],
        controls=_paper_controls_pass(),
        mare_stats=mare_stats,
        quare_stats=quare_stats,
        replay=replay,
        deltas=deltas,
        paper_control_check=_paper_control_check_pass(),
        paper_claims=_paper_claims_pass(),
        conversation_summary=conversation_summary,
    )

    assert verdict["check_results"]["runtime_semantics_complete"] is False
    assert verdict["final_completion_verdict"] == "NO-GO"


def test_collect_system_stats_handles_malformed_runtime_semantics_lists() -> None:
    rows = [
        {
            "run_id": "mare-atm-niv-s101",
            "setting": SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
            "validation_passed": True,
            "validation_errors": [],
            "validation_warnings": [],
            "system_identity": {},
            "provenance": {},
            "execution_flags": {
                "fallback_tainted": False,
                "retry_used": False,
                "rag_fallback_used": False,
                "llm_fallback_used": False,
            },
            "comparability": {},
            "notes": {
                "runtime_semantics": {
                    "roles_executed": None,
                    "actions_executed": None,
                    "workspace_digest": "",
                    "execution_mode": "llm_driven",
                    "llm_turns": len([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                    "llm_fallback_turns": 0,
                    "llm_actions": None,
                }
            },
            "artifacts_dir": "",
        }
    ]

    stats = _collect_system_stats(SYSTEM_MARE, rows)
    assert stats["mare_semantics_incomplete_runs"] == 1
    assert stats["mare_llm_emulation_runs"] == 1


def test_generate_conversation_logs_marks_missing_phase2_as_incomplete(tmp_path: Path) -> None:
    logs_dir = tmp_path / "report" / "logs" / "auto-test"
    logs_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = tmp_path / "artifacts" / "mare" / "mare-atm-niv-s101"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": "mare-atm-niv-s101",
        "system": "mare",
        "case_id": "atm",
        "setting": SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
        "seed": 101,
        "artifacts_dir": str(artifacts_dir),
    }

    summary = _generate_conversation_logs(
        run_key="auto-test",
        logs_dir=logs_dir,
        mare_rows=[row],
        quare_rows=[],
    )

    assert summary["is_complete"] is False
    assert summary["missing_run_ids"] == ["mare-atm-niv-s101"]


def test_report_readme_uses_run_relative_conversation_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "report" / "runs" / "auto-abc123"
    run_dir.mkdir(parents=True, exist_ok=True)
    readme_path = run_dir / "README.md"

    _write_report_readme(
        path=readme_path,
        run_dir=run_dir,
        verdict={"final_completion_verdict": "GO"},
        validation_evidence={"systems": {"mare": {}, "quare": {}}},
        conversation_summary={},
        paper_claims={"claim_summary": {}},
    )

    readme_text = readme_path.read_text(encoding="utf-8")
    assert f"../../logs/{run_dir.name}/conversation_index.jsonl" in readme_text
    assert f"../../logs/{run_dir.name}/conversation_coverage.json" in readme_text


def test_mirror_latest_outputs_rewrites_conversation_links_for_report_root(tmp_path: Path) -> None:
    run_key = "auto-abc123"
    report_dir = tmp_path / "report"
    run_dir = report_dir / "runs" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    report_readme = run_dir / "README.md"
    report_analysis = run_dir / "analysis.md"
    proofs_dir = run_dir / "proofs"
    proofs_dir.mkdir(parents=True, exist_ok=True)

    report_readme.write_text(
        f"- `../../logs/{run_key}/conversation_index.jsonl`\n"
        f"- `../../logs/{run_key}/conversation_coverage.json`\n",
        encoding="utf-8",
    )
    report_analysis.write_text("analysis\n", encoding="utf-8")
    (proofs_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    _mirror_latest_outputs(
        report_dir=report_dir,
        run_key=run_key,
        run_dir=run_dir,
        report_readme=report_readme,
        report_analysis=report_analysis,
        proofs_dir=proofs_dir,
    )

    top_readme = (report_dir / "README.md").read_text(encoding="utf-8")
    assert f"logs/{run_key}/conversation_index.jsonl" in top_readme
    assert f"logs/{run_key}/conversation_coverage.json" in top_readme
    assert f"../../logs/{run_key}/" not in top_readme


def test_evaluate_paper_control_profile_flags_mismatch() -> None:
    result = _evaluate_paper_control_profile(
        {
            "model": "gpt-4o",
            "temperature": 0.0,
            "seeds": [101],
            "settings": ["single_agent"],
        }
    )
    assert result["is_paper_matched"] is False
    assert len(result["mismatches"]) >= 1


def test_build_paper_claim_validation_reports_blocked_precision_when_contract_missing(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_id,case_id,seed,system,setting,chv,mdc,n_phase1_elements,n_phase3_elements,conflict_resolution_rate,semantic_preservation_f1,topology_is_valid",
                "quare-ad-single_agent-s101,AD,101,quare,single_agent,0.04,0.70,8,8,0.0,0.90,1",
                "quare-ad-multi_agent_without_negotiation-s101,AD,101,quare,multi_agent_without_negotiation,0.06,0.84,20,20,0.0,0.92,1",
                "quare-ad-multi_agent_with_negotiation-s101,AD,101,quare,multi_agent_with_negotiation,0.05,0.85,20,20,1.0,0.94,1",
                "quare-ad-negotiation_integration_verification-s101,AD,101,quare,negotiation_integration_verification,0.05,0.85,20,20,1.0,0.95,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    claims = _build_paper_claim_validation(
        quare_csv=csv_path,
        controls=_paper_controls_pass(),
        control_check=_paper_control_check_pass(),
    )
    status_by_id = {
        str(item.get("claim_id")): str(item.get("status"))
        for item in claims.get("claims", [])
        if isinstance(item, dict)
    }
    assert status_by_id["RQ1_precision_gain"] == "BLOCKED"
    assert status_by_id["RQ1_f1_gain"] == "BLOCKED"
    assert int(claims["claim_summary"]["blocked_hard_fail_count"]) >= 1


def test_collect_system_stats_counts_missing_seed_reproducibility() -> None:
    rows = [
        {
            "run_id": "quare-ad-multi_agent_with_negotiation-s101",
            "setting": "multi_agent_with_negotiation",
            "validation_passed": True,
            "validation_errors": [],
            "validation_warnings": [],
            "system_identity": {},
            "provenance": {},
            "execution_flags": {
                "fallback_tainted": False,
                "retry_used": False,
                "rag_fallback_used": False,
                "llm_fallback_used": False,
            },
            "comparability": {},
            "notes": {
                "phase2_llm": {
                    "enabled": True,
                    "turns": 2,
                    "seed_applied_turns": 1,
                }
            },
            "artifacts_dir": "",
        }
    ]

    stats = _collect_system_stats("quare", rows)
    assert stats["seed_reproducibility_incomplete_runs"] == 1


def test_audit_precision_f1_contract_reports_missing_requirements(tmp_path: Path) -> None:
    run_dir = tmp_path / "report" / "runs" / "auto-test"
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = _audit_precision_f1_contract(run_dir=run_dir)
    assert audit["contract_available"] is False
    assert len(audit["missing_requirements"]) >= 1


def test_audit_precision_f1_contract_uses_stable_roots_and_script_alternatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "OpenRE-Bench"
    run_dir = repo_root / "report" / "runs" / "auto-test"
    run_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'openre_bench-test'\n", encoding="utf-8")
    (repo_root / "src" / "openre_bench").mkdir(parents=True, exist_ok=True)

    # Label artifacts exist under the run directory.
    labels = run_dir / "proofs" / "precision_evaluation_progress.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text("{}\n", encoding="utf-8")

    # Ground truth and only one of the script alternatives exist.
    external_root = repo_root.parent / "OpenRE-Bench"
    (external_root / "data" / "ground_truth").mkdir(parents=True, exist_ok=True)
    (external_root / "data" / "ground_truth" / "ad.json").write_text("{}\n", encoding="utf-8")
    (external_root / "scripts").mkdir(parents=True, exist_ok=True)
    (external_root / "scripts" / "evaluate_precision_recall_f1.py").write_text(
        "# placeholder\n", encoding="utf-8"
    )

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(outside_dir)
    audit = _audit_precision_f1_contract(run_dir=run_dir)

    assert audit["contract_available"] is True
    scripts = audit["found"]["scripts"]
    assert any(bool(value) for value in scripts.values())
    assert str(repo_root) in audit["checked_paths"]


def test_build_paper_claim_validation_precision_gain_is_seed_stable(tmp_path: Path) -> None:
    repo_root = tmp_path / "OpenRE-Bench"
    run_dir = repo_root / "report" / "runs" / "auto-seed-stable"
    csv_path = run_dir / "quare" / "comparison_metrics_by_case.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'openre_bench-test'\n", encoding="utf-8")
    (repo_root / "src" / "openre_bench").mkdir(parents=True, exist_ok=True)

    labels = run_dir / "proofs" / "precision_evaluation_progress.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(
        json.dumps(
            [
                {
                    "case_id": "AD",
                    "seed": 202,
                    "phase": "phase2",
                    "precision": 0.82,
                    "f1": 0.41,
                },
                {
                    "case_id": "AD",
                    "seed": 101,
                    "phase": "phase1",
                    "precision": 0.10,
                    "f1": 0.20,
                },
                {
                    "case_id": "AD",
                    "seed": 101,
                    "phase": "phase2",
                    "precision": 0.20,
                    "f1": 0.30,
                },
                {
                    "case_id": "AD",
                    "seed": 202,
                    "phase": "phase1",
                    "precision": 0.80,
                    "f1": 0.40,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    external_root = repo_root.parent / "OpenRE-Bench"
    (external_root / "data" / "ground_truth").mkdir(parents=True, exist_ok=True)
    (external_root / "data" / "ground_truth" / "ad.json").write_text("{}\n", encoding="utf-8")
    (external_root / "scripts").mkdir(parents=True, exist_ok=True)
    (external_root / "scripts" / "evaluate_precision_recall_f1.py").write_text(
        "# placeholder\n", encoding="utf-8"
    )

    csv_path.write_text(
        "\n".join(
            [
                "run_id,case_id,seed,system,setting,chv,mdc,n_phase1_elements,n_phase3_elements,conflict_resolution_rate,semantic_preservation_f1,topology_is_valid,precision,f1",
                # Intentionally shuffled rows to catch order-sensitive gain logic.
                "quare-ad-multi_agent_with_negotiation-s101,AD,101,quare,multi_agent_with_negotiation,0.05,0.85,20,20,1.0,0.94,1,0.20,0.30",
                "quare-ad-multi_agent_without_negotiation-s202,AD,202,quare,multi_agent_without_negotiation,0.06,0.84,20,20,0.0,0.92,1,0.80,0.40",
                "quare-ad-multi_agent_with_negotiation-s202,AD,202,quare,multi_agent_with_negotiation,0.05,0.85,20,20,1.0,0.94,1,0.82,0.41",
                "quare-ad-single_agent-s101,AD,101,quare,single_agent,0.04,0.70,8,8,0.0,0.90,1,0.10,0.20",
                "quare-ad-single_agent-s202,AD,202,quare,single_agent,0.04,0.70,8,8,0.0,0.90,1,0.10,0.20",
                "quare-ad-multi_agent_without_negotiation-s101,AD,101,quare,multi_agent_without_negotiation,0.06,0.84,20,20,0.0,0.92,1,0.10,0.20",
                "quare-ad-negotiation_integration_verification-s101,AD,101,quare,negotiation_integration_verification,0.05,0.85,20,20,1.0,0.95,1,0.20,0.30",
                "quare-ad-negotiation_integration_verification-s202,AD,202,quare,negotiation_integration_verification,0.05,0.85,20,20,1.0,0.95,1,0.82,0.41",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    claims = _build_paper_claim_validation(
        quare_csv=csv_path,
        controls=_paper_controls_pass(),
        control_check=_paper_control_check_pass(),
    )
    by_id = {
        str(item.get("claim_id")): item
        for item in claims.get("claims", [])
        if isinstance(item, dict)
    }
    assert by_id["RQ1_precision_gain"]["status"] == "PASS"
    assert by_id["RQ1_f1_gain"]["status"] == "PASS"
    assert by_id["RQ1_precision_gain"]["blocker"] is None
    assert by_id["RQ1_f1_gain"]["blocker"] is None
    assert str(by_id["RQ1_precision_gain"]["observed"]["source_column"]).startswith("literal:")
    assert str(by_id["RQ1_f1_gain"]["observed"]["source_column"]).startswith("literal:")
    assert float(by_id["RQ1_precision_gain"]["observed"]["gain"]) == 1.0
    assert float(by_id["RQ1_f1_gain"]["observed"]["gain"]) == 0.5


def test_build_paper_claim_validation_literal_labels_are_isolated_by_system_and_setting(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "OpenRE-Bench"
    run_dir = repo_root / "report" / "runs" / "auto-literal-isolation"
    csv_path = run_dir / "quare" / "comparison_metrics_by_case.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'openre_bench-test'\n", encoding="utf-8")
    (repo_root / "src" / "openre_bench").mkdir(parents=True, exist_ok=True)

    labels = run_dir / "proofs" / "precision_evaluation_progress.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(
        json.dumps(
            [
                {
                    "system": SYSTEM_MARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_without_negotiation",
                    "precision": 0.10,
                    "f1": 0.20,
                },
                {
                    "system": SYSTEM_MARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_with_negotiation",
                    "precision": 0.90,
                    "f1": 0.90,
                },
                {
                    "system": SYSTEM_QUARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_without_negotiation",
                    "precision": 0.40,
                    "f1": 0.50,
                },
                {
                    "system": SYSTEM_QUARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_with_negotiation",
                    "precision": 0.50,
                    "f1": 0.60,
                },
                {
                    "system": SYSTEM_QUARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "negotiation_integration_verification",
                    "precision": 0.99,
                    "f1": 0.99,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    external_root = repo_root.parent / "OpenRE-Bench"
    (external_root / "data" / "ground_truth").mkdir(parents=True, exist_ok=True)
    (external_root / "data" / "ground_truth" / "ad.json").write_text("{}\n", encoding="utf-8")
    (external_root / "scripts").mkdir(parents=True, exist_ok=True)
    (external_root / "scripts" / "evaluate_precision_recall_f1.py").write_text(
        "# placeholder\n", encoding="utf-8"
    )

    csv_path.write_text(
        "\n".join(
            [
                "run_id,case_id,seed,system,setting,chv,mdc,n_phase1_elements,n_phase3_elements,conflict_resolution_rate,semantic_preservation_f1,topology_is_valid",
                "quare-ad-multi_agent_without_negotiation-s101,AD,101,quare,multi_agent_without_negotiation,0.06,0.84,20,20,0.0,0.92,1",
                "quare-ad-multi_agent_with_negotiation-s101,AD,101,quare,multi_agent_with_negotiation,0.05,0.85,20,20,1.0,0.94,1",
                "quare-ad-negotiation_integration_verification-s101,AD,101,quare,negotiation_integration_verification,0.05,0.85,20,20,1.0,0.95,1",
                "quare-ad-single_agent-s101,AD,101,quare,single_agent,0.04,0.70,8,8,0.0,0.90,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    claims = _build_paper_claim_validation(
        quare_csv=csv_path,
        controls=_paper_controls_pass(),
        control_check=_paper_control_check_pass(),
    )
    by_id = {
        str(item.get("claim_id")): item
        for item in claims.get("claims", [])
        if isinstance(item, dict)
    }
    assert float(by_id["RQ1_precision_gain"]["observed"]["gain"]) == pytest.approx(0.25)
    assert float(by_id["RQ1_f1_gain"]["observed"]["gain"]) == pytest.approx(0.2)
    assert str(by_id["RQ1_precision_gain"]["observed"]["source_column"]).startswith("literal:")
    assert str(by_id["RQ1_f1_gain"]["observed"]["source_column"]).startswith("literal:")


def test_build_paper_claim_validation_literal_labels_merge_partial_updates(tmp_path: Path) -> None:
    repo_root = tmp_path / "OpenRE-Bench"
    run_dir = repo_root / "report" / "runs" / "auto-literal-partial-merge"
    csv_path = run_dir / "quare" / "comparison_metrics_by_case.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'openre_bench-test'\n", encoding="utf-8")
    (repo_root / "src" / "openre_bench").mkdir(parents=True, exist_ok=True)

    labels = run_dir / "proofs" / "precision_evaluation_progress.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(
        json.dumps(
            [
                {
                    "system": SYSTEM_QUARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_with_negotiation",
                    "precision": 0.80,
                },
                {
                    "system": SYSTEM_QUARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_without_negotiation",
                    "f1": 0.50,
                },
                {
                    "system": SYSTEM_QUARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_with_negotiation",
                    "f1": 0.75,
                },
                {
                    "system": SYSTEM_QUARE,
                    "case_id": "AD",
                    "seed": 101,
                    "setting": "multi_agent_without_negotiation",
                    "precision": 0.40,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    external_root = repo_root.parent / "OpenRE-Bench"
    (external_root / "data" / "ground_truth").mkdir(parents=True, exist_ok=True)
    (external_root / "data" / "ground_truth" / "ad.json").write_text("{}\n", encoding="utf-8")
    (external_root / "scripts").mkdir(parents=True, exist_ok=True)
    (external_root / "scripts" / "evaluate_precision_recall_f1.py").write_text(
        "# placeholder\n", encoding="utf-8"
    )

    csv_path.write_text(
        "\n".join(
            [
                "run_id,case_id,seed,system,setting,chv,mdc,n_phase1_elements,n_phase3_elements,conflict_resolution_rate,semantic_preservation_f1,topology_is_valid",
                "quare-ad-single_agent-s101,AD,101,quare,single_agent,0.04,0.70,8,8,0.0,0.90,1",
                "quare-ad-multi_agent_without_negotiation-s101,AD,101,quare,multi_agent_without_negotiation,0.06,0.84,20,20,0.0,0.92,1",
                "quare-ad-multi_agent_with_negotiation-s101,AD,101,quare,multi_agent_with_negotiation,0.05,0.85,20,20,1.0,0.94,1",
                "quare-ad-negotiation_integration_verification-s101,AD,101,quare,negotiation_integration_verification,0.05,0.85,20,20,1.0,0.95,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    claims = _build_paper_claim_validation(
        quare_csv=csv_path,
        controls=_paper_controls_pass(),
        control_check=_paper_control_check_pass(),
    )
    by_id = {
        str(item.get("claim_id")): item
        for item in claims.get("claims", [])
        if isinstance(item, dict)
    }
    assert float(by_id["RQ1_precision_gain"]["observed"]["gain"]) == pytest.approx(1.0)
    assert float(by_id["RQ1_f1_gain"]["observed"]["gain"]) == pytest.approx(0.5)
    assert by_id["RQ1_precision_gain"]["status"] == "PASS"
    assert by_id["RQ1_f1_gain"]["status"] == "PASS"
    assert str(by_id["RQ1_precision_gain"]["observed"]["source_column"]).startswith("literal:")
    assert str(by_id["RQ1_f1_gain"]["observed"]["source_column"]).startswith("literal:")


def test_build_paper_claim_validation_derives_precision_f1_from_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "OpenRE-Bench"
    run_dir = repo_root / "report" / "runs" / "auto-derived"
    csv_path = run_dir / "quare" / "comparison_metrics_by_case.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "openre_bench").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'openre_bench-test'\n", encoding="utf-8")

    cases_dir = repo_root / "data" / "case_studies"
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_path = cases_dir / "AD_input.json"
    case_path.write_text(
        json.dumps(
            {
                "case_name": "AD",
                "case_description": "Autonomous driving safety requirements",
                "requirement": (
                    "System shall encrypt customer data at rest using AES256 and verify "
                    "key rotation every 90 days. "
                    "System shall record security audit events with actor identity and "
                    "immutable timestamps for each access."
                ),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "controls.json").write_text(
        json.dumps(
            {
                "cases_dir": str(cases_dir),
                "case_files": ["AD_input.json"],
            }
        ),
        encoding="utf-8",
    )

    baseline_run_id = "quare-ad-multi_agent_without_negotiation-s101"
    negotiated_run_id = "quare-ad-multi_agent_with_negotiation-s101"
    baseline_artifacts = run_dir / "quare" / "runs" / baseline_run_id
    baseline_artifacts.mkdir(parents=True, exist_ok=True)
    (baseline_artifacts / PHASE1_FILENAME).write_text(
        json.dumps(
            {
                "SafetyAgent": [
                    {"id": "s1", "description": "System shall encrypt data at rest."},
                    {"id": "s2", "description": "System shall record audit events."},
                ]
            }
        ),
        encoding="utf-8",
    )

    negotiated_artifacts = run_dir / "quare" / "runs" / negotiated_run_id
    negotiated_artifacts.mkdir(parents=True, exist_ok=True)
    (negotiated_artifacts / PHASE2_FILENAME).write_text(
        json.dumps(
            {
                "negotiations": {
                    "SafetyAgent_EfficiencyAgent": {
                        "steps": [
                            {
                                "message_type": "backward",
                                "kaos_elements": [
                                    {
                                        "description": (
                                            "System shall encrypt customer data at rest using "
                                            "AES256 and verify key rotation every 90 days"
                                        )
                                    },
                                    {
                                        "description": (
                                            "System shall record security audit events with "
                                            "actor identity and immutable timestamps for each "
                                            "access"
                                        )
                                    },
                                ],
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    csv_path.write_text(
        "\n".join(
            [
                "run_id,case_id,seed,system,setting,chv,mdc,n_phase1_elements,n_phase3_elements,conflict_resolution_rate,semantic_preservation_f1,topology_is_valid",
                "quare-ad-single_agent-s101,AD,101,quare,single_agent,0.04,0.70,8,8,0.0,0.90,1",
                f"{baseline_run_id},AD,101,quare,multi_agent_without_negotiation,0.06,0.84,20,20,0.0,0.92,1",
                f"{negotiated_run_id},AD,101,quare,multi_agent_with_negotiation,0.05,0.85,20,20,1.0,0.94,1",
                "quare-ad-negotiation_integration_verification-s101,AD,101,quare,negotiation_integration_verification,0.05,0.85,20,20,1.0,0.95,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    controls = _paper_controls_pass()
    controls.update({"cases_dir": str(cases_dir), "case_files": ["AD_input.json"]})
    claims = _build_paper_claim_validation(
        quare_csv=csv_path,
        controls=controls,
        control_check=_paper_control_check_pass(),
    )
    by_id = {
        str(item.get("claim_id")): item
        for item in claims.get("claims", [])
        if isinstance(item, dict)
    }
    assert by_id["RQ1_precision_gain"]["status"] == "BLOCKED"
    assert by_id["RQ1_f1_gain"]["status"] == "BLOCKED"
    assert by_id["RQ1_precision_gain"]["blocker"] is not None
    assert by_id["RQ1_f1_gain"]["blocker"] is not None
    assert str(by_id["RQ1_precision_gain"]["observed"]["source_column"]).startswith("derived:")
    assert str(by_id["RQ1_f1_gain"]["observed"]["source_column"]).startswith("derived:")

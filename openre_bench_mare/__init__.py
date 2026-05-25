"""MARE 5-agent / 9-action workflow (ASE 2024 paper)."""

from __future__ import annotations

import json
import re
from typing import Any

from openre_bench.llm import LLMContract
from openre_bench.schemas import KAOSElement
from openre_bench.schemas import MARE_ACTIONS
from openre_bench.schemas import MARE_AGENT_ROLES
from openre_bench.schemas import MARE_ROLE_ACTIONS
from openre_bench.schemas import NegotiationHistory
from openre_bench.schemas import NegotiationStep
from openre_bench.schemas import SETTING_SINGLE_AGENT
from openre_bench.schemas import utc_timestamp

from openre_bench.runtime_support import (
    ActionRunResult,
    MARE_ROLE_QUALITY_ATTRIBUTES,
    MARE_RUNTIME_SEMANTICS_MODE,
    MARE_RUNTIME_TRACE_VERSION,
    NegotiationBuildResult as _NegotiationBuildResult,
    RuntimeExecutionMeta,
    apply_negotiation_adjustments as _apply_negotiation_adjustments,
    backward_analysis_text as _backward_analysis_text,
    backward_feedback as _backward_feedback,
    build_base_phase1 as _build_phase1,
    coerce_action_items as _coerce_action_items,
    detect_conflict as _detect_conflict,
    extract_requirement_fragments as _extract_requirement_fragments,
    rag_payload as _rag_payload,
    rotate_fragments as _rotate_fragments,
    run_llm_action_with_fallback as _run_llm_action_with_fallback,
    sha256_payload as _sha256_payload,
    summarize_text as _summarize_text,
    tokens as _tokens,
)

__all__ = [
    "build_negotiation_history",
    "build_optional_artifacts",
    "build_phase1",
    "build_phase1_mare_semantics",
    "run_mare_action_workflow",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_phase1_mare_semantics(
    *,
    case: Any,
    seed: int,
    setting: str,
    rag_context: dict[str, Any],
    llm_client: LLMContract | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], RuntimeExecutionMeta]:
    """Build phase-1 payload via paper-faithful MARE 5-agent/9-action workflow."""

    if setting == SETTING_SINGLE_AGENT:
        phase1 = _build_phase1(case, seed, setting, rag_context)
        return phase1, {}, RuntimeExecutionMeta(llm_source=llm_source)

    fragments = _extract_requirement_fragments(case.requirement)
    if not fragments:
        fragments = [case.requirement.strip() or "Requirement text unavailable"]
    rotated = _rotate_fragments(fragments, seed)

    workspace, llm_meta = run_mare_action_workflow(
        case_name=case.case_name,
        fragments=rotated,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )

    phase1: dict[str, list[dict[str, Any]]] = {}
    for role_index, role in enumerate(MARE_AGENT_ROLES, start=1):
        quality_attribute = MARE_ROLE_QUALITY_ATTRIBUTES.get(role, "Integrated")
        role_prefix = re.sub(r"[^A-Z]", "", role.upper()) or f"R{role_index}"
        leaf_texts = _mare_role_leaf_texts(role=role, workspace=workspace, rotated=rotated)

        root_id = f"{role_prefix[:3]}-L1-{role_index:03d}"
        root_query = leaf_texts[0] if leaf_texts else " ".join(rotated)
        root_rag_payload = _rag_payload(query=root_query, rag_context=rag_context)
        root = KAOSElement(
            id=root_id,
            name=f"{role} Goal",
            description=(
                f"{role} objective for {case.case_name}: "
                f"{_summarize_text(root_query, 220)}"
            ),
            element_type="Goal",
            quality_attribute=quality_attribute,
            hierarchy_level=1,
            stakeholder=role,
            measurable_criteria=f"{role} deliverables remain traceable in shared workspace",
            source=root_rag_payload["source"],
            source_chunk_id=root_rag_payload["source_chunk_id"],
            source_document=root_rag_payload["source_document"],
            retrieved_chunks=root_rag_payload["retrieved_chunks"],
            validation_status="pending",
        )

        role_elements: list[dict[str, Any]] = [root.model_dump(mode="json")]
        for leaf_index, leaf_text in enumerate(leaf_texts[:4], start=1):
            rag_payload = _rag_payload(query=leaf_text, rag_context=rag_context)
            leaf = KAOSElement(
                id=f"{role_prefix[:3]}-L2-{role_index:03d}-{leaf_index:02d}",
                name=f"{role} Output {leaf_index}",
                description=f"The system shall {_summarize_text(leaf_text, 220)}",
                element_type="Task",
                quality_attribute=quality_attribute,
                hierarchy_level=2,
                parent_goal_id=root_id,
                stakeholder=role,
                measurable_criteria=f"{role} action artifact {leaf_index}",
                source=rag_payload["source"],
                source_chunk_id=rag_payload["source_chunk_id"],
                source_document=rag_payload["source_document"],
                retrieved_chunks=rag_payload["retrieved_chunks"],
                validation_status="candidate_resolution" if role == "Checker" else "pending",
            )
            role_elements.append(leaf.model_dump(mode="json"))

        phase1[role] = role_elements

    runtime_semantics = {
        "workflow_version": MARE_RUNTIME_TRACE_VERSION,
        "paper_workflow_enabled": True,
        "setting": setting,
        "llm_required": True,
        "llm_source": llm_meta.llm_source,
        "llm_enabled": llm_meta.llm_enabled,
        "llm_seed": llm_seed,
        "llm_seed_applied_turns": llm_meta.llm_seed_applied_turns,
        "llm_turns": llm_meta.llm_turns,
        "llm_fallback_turns": llm_meta.llm_fallback_turns,
        "llm_retry_count": llm_meta.llm_retry_count,
        "llm_parse_recoveries": llm_meta.llm_parse_recoveries,
        "execution_mode": llm_meta.execution_mode,
        "llm_actions": workspace["llm_actions"],
        "fallback_actions": workspace["fallback_actions"],
        "roles_executed": list(MARE_AGENT_ROLES),
        "actions_executed": list(MARE_ACTIONS),
        "role_action_map": {role: list(actions) for role, actions in MARE_ROLE_ACTIONS.items()},
        "action_trace": workspace["action_trace"],
        "workspace_summary": workspace["workspace_summary"],
        "workspace_digest": _sha256_payload(workspace),
    }
    return phase1, runtime_semantics, llm_meta


def build_phase1(
    *,
    case: Any,
    seed: int,
    setting: str,
    rag_context: dict[str, Any],
    llm_client: LLMContract | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], RuntimeExecutionMeta]:
    """Build MARE phase 1 for the selected benchmark setting."""

    return build_phase1_mare_semantics(
        case=case,
        seed=seed,
        setting=setting,
        rag_context=rag_context,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )


def build_negotiation_history(
    *,
    run_id: str,
    pair_key: str,
    focus_agent: str,
    reviewer_agent: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    requirement: str,
    step_counter: int,
    **_: Any,
) -> _NegotiationBuildResult:
    """Build MARE single-round peer negotiation history."""

    conflict_detected = _detect_conflict(
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=focus_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
    )
    negotiated_elements = _apply_negotiation_adjustments(
        elements=focus_elements,
        reviewer_agent=reviewer_agent,
        conflict_detected=conflict_detected,
    )
    conflict_resolved = conflict_detected

    forward_text = f"{focus_agent} proposes initial model for peer review."
    forward_step = NegotiationStep(
        step_id=step_counter,
        timestamp=utc_timestamp(),
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        round_number=1,
        message_type="forward",
        kaos_elements=focus_elements,
        analysis_text=forward_text,
        analysis=forward_text,
        conflict_detected=conflict_detected,
    )
    step_counter += 1

    backward_text = _backward_analysis_text(reviewer_agent, conflict_detected)
    backward_step = NegotiationStep(
        step_id=step_counter,
        timestamp=utc_timestamp(),
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        round_number=1,
        message_type="backward",
        kaos_elements=negotiated_elements,
        analysis_text=backward_text,
        analysis=backward_text,
        feedback=_backward_feedback(conflict_detected),
        conflict_detected=conflict_detected,
    )
    step_counter += 1

    history = NegotiationHistory(
        negotiation_id=f"neg_{pair_key}_{run_id}",
        focus_agent=focus_agent,
        reviewer_agents=[reviewer_agent],
        start_timestamp=utc_timestamp(),
        end_timestamp=utc_timestamp(),
        steps=[forward_step, backward_step],
        final_consensus=conflict_resolved or not conflict_detected,
        total_rounds=1,
    )

    return _NegotiationBuildResult(
        history=history,
        next_step_id=step_counter,
        conflict_detected=conflict_detected,
        conflict_resolved=conflict_resolved,
    )


def build_optional_artifacts(**_: Any) -> dict[str, dict[str, Any]]:
    """MARE has no extra phase artifacts beyond the benchmark phase contract."""

    return {}


def run_mare_action_workflow(
    *,
    case_name: str,
    fragments: list[str],
    llm_client: LLMContract | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, Any], RuntimeExecutionMeta]:
    """Execute MARE 5-agent/9-action workflow with LLM-first action execution."""

    action_trace: list[dict[str, Any]] = []
    llm_actions: list[str] = []
    fallback_actions: list[str] = []
    llm_meta = RuntimeExecutionMeta(
        llm_enabled=llm_client is not None,
        llm_source=llm_source,
    )

    workspace: dict[str, Any] = {
        "case_name": case_name,
        "fragments": list(fragments),
        "user_stories": [],
        "questions": [],
        "answers": [],
        "requirement_draft": [],
        "entities": [],
        "relations": [],
        "findings": [],
        "srs_sections": [],
        "check_report": [],
    }

    def _run_action(
        *,
        role: str,
        action: str,
        instruction: str,
        fallback_items: list[str],
        max_items: int,
    ) -> tuple[list[str], str]:
        result: ActionRunResult = _run_llm_action_with_fallback(
            llm_client=llm_client,
            messages=_build_mare_llm_messages(
                role=role,
                action=action,
                case_name=case_name,
                fragments=fragments,
                workspace=workspace,
                instruction=instruction,
            ),
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_source=llm_source,
            llm_meta=llm_meta,
            coerce_items_fn=_coerce_action_items,
            max_items=max_items,
            fallback_items=fallback_items,
        )

        if result.llm_generated:
            llm_actions.append(action)
        else:
            fallback_actions.append(action)

        action_trace.append(
            {
                "role": role,
                "action": action,
                "execution_mode": result.execution_mode,
                "llm_generated": result.llm_generated,
                "llm_source": llm_source,
                "fallback_reason": result.fallback_reason,
                "output_count": len(result.items),
                "summary": result.summary,
            }
        )
        return result.items, result.summary

    fallback_user_stories = [
        f"As a stakeholder of {case_name}, I need {_summarize_text(fragment, 180)}"
        for fragment in fragments[:6]
    ]
    if not fallback_user_stories:
        fallback_user_stories = [f"As a stakeholder, I need reliable behavior for {case_name}."]
    user_stories, _ = _run_action(
        role="Stakeholders",
        action="SpeakUserStories",
        instruction=(
            "Write concise user stories aligned to stakeholder needs for the case requirement."
        ),
        fallback_items=fallback_user_stories,
        max_items=6,
    )
    workspace["user_stories"] = user_stories

    fallback_questions = [
        f"What acceptance condition should constrain story {index}?"
        for index in range(1, min(3, len(user_stories)) + 1)
    ]
    if not fallback_questions:
        fallback_questions = ["What acceptance condition defines successful delivery?"]
    questions, _ = _run_action(
        role="Collector",
        action="ProposeQuestion",
        instruction=(
            "Ask clarification questions that resolve ambiguity and make requirements verifiable."
        ),
        fallback_items=fallback_questions,
        max_items=4,
    )
    workspace["questions"] = questions

    fallback_answers = [
        f"Answer {index}: The system shall {_summarize_text(fragments[(index - 1) % len(fragments)], 130)}"
        for index in range(1, len(questions) + 1)
    ]
    answers, _ = _run_action(
        role="Stakeholders",
        action="AnswerQuestion",
        instruction=(
            "Answer each collector question with concrete constraints and acceptance-oriented detail."
        ),
        fallback_items=fallback_answers,
        max_items=max(1, len(fallback_answers)),
    )
    workspace["answers"] = answers

    fallback_requirement_draft = [
        f"Draft requirement {index}: The system shall {_summarize_text(story, 150)}"
        for index, story in enumerate(user_stories[:4], start=1)
    ]
    fallback_requirement_draft.extend(answers[:2])
    requirement_draft, _ = _run_action(
        role="Collector",
        action="WriteReqDraft",
        instruction=(
            "Write requirement draft statements with explicit 'shall' language and measurable intent."
        ),
        fallback_items=fallback_requirement_draft,
        max_items=6,
    )
    workspace["requirement_draft"] = requirement_draft

    entity_tokens = _tokens(" ".join(requirement_draft).lower())
    fallback_entities = [token for token in entity_tokens if len(token) >= 5][:8]
    if not fallback_entities:
        fallback_entities = ["requirement", "verification", "stakeholder"]
    entities, _ = _run_action(
        role="Modeler",
        action="ExtractEntity",
        instruction=(
            "Extract candidate requirement entities (nouns/concepts) from the draft requirements."
        ),
        fallback_items=fallback_entities,
        max_items=8,
    )
    workspace["entities"] = entities

    fallback_relation_items: list[str] = []
    for index in range(len(entities) - 1):
        fallback_relation_items.append(f"{entities[index]}|refines|{entities[index + 1]}")
    if not fallback_relation_items and entities:
        fallback_relation_items = [f"{entities[0]}|supports|{entities[0]}"]
    relation_items, _ = _run_action(
        role="Modeler",
        action="ExtractRelation",
        instruction=(
            "Extract entity relations using 'source|relation|target' format for each output item."
        ),
        fallback_items=fallback_relation_items,
        max_items=8,
    )
    relations = _coerce_mare_relations(relation_items)
    workspace["relations"] = relations

    fallback_findings: list[str] = []
    requirement_text = " ".join(fragments).lower()
    if any(token in requirement_text for token in ("conflict", "tradeoff", "trade-off")):
        fallback_findings.append(
            "Potential trade-off detected: responses shall not relax core safety and security constraints."
        )
    fallback_findings.append("All requirements should remain verifiable with explicit measurable criteria.")
    findings, _ = _run_action(
        role="Checker",
        action="CheckRequirement",
        instruction=(
            "List requirement quality findings covering ambiguity, conflict risk, and verifiability."
        ),
        fallback_items=fallback_findings,
        max_items=6,
    )
    workspace["findings"] = findings

    fallback_srs_sections = [
        "SRS-1 Scope and Stakeholders",
        "SRS-2 Functional and Quality Requirements",
        "SRS-3 Verification and Acceptance Criteria",
    ]
    srs_sections, _ = _run_action(
        role="Documenter",
        action="WriteSRS",
        instruction=(
            "Draft compact SRS section headings and intents aligned with the checked requirements."
        ),
        fallback_items=fallback_srs_sections,
        max_items=4,
    )
    workspace["srs_sections"] = srs_sections

    fallback_check_report = [
        f"Check report item {index}: {finding}"
        for index, finding in enumerate(findings, start=1)
    ]
    check_report, _ = _run_action(
        role="Documenter",
        action="WriteCheckReport",
        instruction=(
            "Write numbered check report items that tie findings to verification expectations."
        ),
        fallback_items=fallback_check_report,
        max_items=8,
    )
    workspace["check_report"] = check_report

    if llm_meta.llm_turns == len(MARE_ACTIONS) and llm_meta.llm_fallback_turns == 0:
        llm_meta.execution_mode = "llm_driven"
    elif llm_meta.llm_turns > 0:
        llm_meta.execution_mode = "llm_hybrid_fallback"
    elif llm_meta.llm_enabled:
        llm_meta.execution_mode = "deterministic_fallback"
    else:
        llm_meta.execution_mode = "deterministic_emulation"

    workspace_summary = {
        "user_stories": len(user_stories),
        "questions": len(questions),
        "answers": len(answers),
        "requirement_draft": len(requirement_draft),
        "entities": len(entities),
        "relations": len(relations),
        "findings": len(findings),
        "srs_sections": len(srs_sections),
        "check_report": len(check_report),
        "llm_turns": llm_meta.llm_turns,
        "llm_fallback_turns": llm_meta.llm_fallback_turns,
    }
    return {
        "workflow": MARE_RUNTIME_SEMANTICS_MODE,
        "trace_version": MARE_RUNTIME_TRACE_VERSION,
        "user_stories": user_stories,
        "questions": questions,
        "answers": answers,
        "requirement_draft": requirement_draft,
        "entities": entities,
        "relations": relations,
        "findings": findings,
        "srs_sections": srs_sections,
        "check_report": check_report,
        "workspace_summary": workspace_summary,
        "action_trace": action_trace,
        "llm_actions": llm_actions,
        "fallback_actions": fallback_actions,
    }, llm_meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_mare_llm_messages(
    *,
    role: str,
    action: str,
    case_name: str,
    fragments: list[str],
    workspace: dict[str, Any],
    instruction: str,
) -> list[dict[str, str]]:
    """Build one action-scoped MARE prompt contract for LLM execution."""

    payload = {
        "task": "MARE paper workflow action execution",
        "role": role,
        "action": action,
        "instruction": instruction,
        "case_name": case_name,
        "requirement_fragments": fragments[:6],
        "workspace_snapshot": _mare_workspace_snapshot(workspace),
        "output_schema": {
            "items": ["non-empty string"],
            "summary": "short summary string",
        },
        "constraints": [
            "Return exactly one JSON object.",
            "Do not emit markdown fences.",
            "Keep outputs concise and requirement-grounded.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are executing one role/action in the MARE requirements workflow. "
                "Return exactly one JSON object with fields: items, summary."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _mare_workspace_snapshot(workspace: dict[str, Any]) -> dict[str, Any]:
    """Compact shared-workspace snapshot passed into MARE action prompts."""

    snapshot: dict[str, Any] = {}
    for key in (
        "user_stories",
        "questions",
        "answers",
        "requirement_draft",
        "entities",
        "relations",
        "findings",
        "srs_sections",
        "check_report",
    ):
        value = workspace.get(key)
        if isinstance(value, list):
            if key == "relations":
                snapshot[key] = [
                    f"{item.get('source')}|{item.get('relation')}|{item.get('target')}"
                    for item in value[:6]
                    if isinstance(item, dict)
                ]
            else:
                snapshot[key] = [str(item).strip() for item in value[:6] if str(item).strip()]
    return snapshot


def _coerce_mare_relations(items: list[str]) -> list[dict[str, str]]:
    """Parse relation triples from action items into structured relation records."""

    relations: list[dict[str, str]] = []
    for item in items:
        parts = [part.strip() for part in item.split("|")]
        if len(parts) == 3 and all(parts):
            source, relation, target = parts
        else:
            words = _tokens(item)
            if len(words) < 2:
                continue
            source = words[0]
            target = words[-1]
            relation = "related_to"
        relations.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
            }
        )
    return relations


def _mare_role_leaf_texts(*, role: str, workspace: dict[str, Any], rotated: list[str]) -> list[str]:
    """Select role-specific leaf material from shared workspace."""

    if role == "Stakeholders":
        values = workspace.get("user_stories", [])
    elif role == "Collector":
        values = workspace.get("requirement_draft", [])
    elif role == "Modeler":
        entities = [f"Entity: {item}" for item in workspace.get("entities", [])]
        relations = [
            f"Relation: {item.get('source')} {item.get('relation')} {item.get('target')}"
            for item in workspace.get("relations", [])
            if isinstance(item, dict)
        ]
        values = entities + relations
    elif role == "Checker":
        values = workspace.get("check_report", [])
    else:
        values = workspace.get("srs_sections", [])

    leaf_texts = [str(item).strip() for item in values if str(item).strip()]
    if leaf_texts:
        return leaf_texts
    return rotated[:3]

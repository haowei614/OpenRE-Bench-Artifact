"""iReDev 6-agent / 17-action knowledge-driven workflow (Jin et al., TOSEM 2025)."""

from __future__ import annotations

import json
import re
from typing import Any

from openre_bench.llm import LLMContract
from openre_bench.schemas import IREDEV_ACTIONS
from openre_bench.schemas import IREDEV_AGENT_ROLES
from openre_bench.schemas import IREDEV_ROLE_ACTIONS
from openre_bench.schemas import KAOSElement
from openre_bench.schemas import NegotiationHistory
from openre_bench.schemas import NegotiationStep
from openre_bench.schemas import SETTING_SINGLE_AGENT
from openre_bench.schemas import utc_timestamp

from openre_bench.runtime_support import (
    ActionRunResult,
    IREDEV_ROLE_QUALITY_ATTRIBUTES,
    IREDEV_RUNTIME_SEMANTICS_MODE,
    IREDEV_RUNTIME_TRACE_VERSION,
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
    "build_phase1_iredev_semantics",
    "run_iredev_action_workflow",
]


# Maximum dialogue rounds per multi-round stakeholder interaction (§4.3).
_IREDEV_DIALOGUE_ROUNDS = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_phase1_iredev_semantics(
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
    """Build phase-1 payload via paper-faithful iReDev 6-agent/17-action workflow."""

    if setting == SETTING_SINGLE_AGENT:
        phase1 = _build_phase1(case, seed, setting, rag_context)
        return phase1, {}, RuntimeExecutionMeta(llm_source=llm_source)

    fragments = _extract_requirement_fragments(case.requirement)
    if not fragments:
        fragments = [case.requirement.strip() or "Requirement text unavailable"]
    rotated = _rotate_fragments(fragments, seed)

    workspace, llm_meta = run_iredev_action_workflow(
        case_name=case.case_name,
        fragments=rotated,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )

    phase1: dict[str, list[dict[str, Any]]] = {}
    for role_index, role in enumerate(IREDEV_AGENT_ROLES, start=1):
        quality_attribute = IREDEV_ROLE_QUALITY_ATTRIBUTES.get(role, "Integrated")
        role_prefix = re.sub(r"[^A-Z]", "", role.upper()) or f"R{role_index}"
        leaf_texts = _iredev_role_leaf_texts(role=role, workspace=workspace, rotated=rotated)

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
            measurable_criteria=f"{role} deliverables remain traceable in artifacts pool",
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
                validation_status="candidate_resolution" if role == "Reviewer" else "pending",
            )
            role_elements.append(leaf.model_dump(mode="json"))

        phase1[role] = role_elements

    runtime_semantics = {
        "workflow_version": IREDEV_RUNTIME_TRACE_VERSION,
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
        "roles_executed": list(IREDEV_AGENT_ROLES),
        "actions_executed": list(IREDEV_ACTIONS),
        "role_action_map": {role: list(actions) for role, actions in IREDEV_ROLE_ACTIONS.items()},
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
    """Build iReDev phase 1 for the selected benchmark setting."""

    return build_phase1_iredev_semantics(
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
    """Build iReDev single-round artifact-pool review history."""

    conflict_detected = _detect_conflict(
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=focus_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
    )
    reviewed_elements = _apply_negotiation_adjustments(
        elements=focus_elements,
        reviewer_agent=reviewer_agent,
        conflict_detected=conflict_detected,
    )
    conflict_resolved = conflict_detected

    forward_text = f"{focus_agent} submits artifact-pool output for review."
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
        kaos_elements=reviewed_elements,
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
    """iReDev has no extra phase artifacts beyond the benchmark phase contract."""

    return {}


def run_iredev_action_workflow(
    *,
    case_name: str,
    fragments: list[str],
    llm_client: LLMContract | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, Any], RuntimeExecutionMeta]:
    """Execute iReDev 6-agent/17-action knowledge-driven workflow with LLM-first action execution."""

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
        "interview_records": [],
        "user_requirements": [],
        "operating_environment": [],
        "enduser_responses": [],
        "enduser_questions": [],
        "enduser_confirmations": [],
        "deployer_responses": [],
        "deployer_questions": [],
        "deployer_confirmations": [],
        "system_requirements": [],
        "selected_model": [],
        "requirement_model": [],
        "srs_sections": [],
        "review_findings": [],
        "closure_confirmations": [],
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
            messages=_build_iredev_llm_messages(
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

    # Each action declares its *preconditions* — the artifact pool keys that
    # must be non-empty before the action may fire.  A monitor loop scans
    # all pending actions, fires the first one whose preconditions are met,
    # updates the pool, and repeats until all 17 distinct actions are done.
    #
    # Multi-round dialogue (paper Fig. 3: "Conduct multi-round dialogue")
    # is modelled by allowing dialogue-class actions to fire up to
    # _IREDEV_DIALOGUE_ROUNDS times.  The distinct-action count stays 17.

    # ── Action descriptors ───────────────────────────────────────────────
    # Each entry: (role, action, instruction, fallback_builder, max_items, preconditions, target_key)
    # preconditions: tuple of workspace keys that must be non-empty
    # target_key: workspace key to append results into

    def _fb_enduser_dialogue() -> list[str]:
        fb = [
            f"As a user of {case_name}, I need {_summarize_text(fragment, 180)}"
            for fragment in fragments[:6]
        ]
        return fb or [f"As a user, I need reliable behavior for {case_name}."]

    def _fb_enduser_respond() -> list[str]:
        return [
            f"Response: {_summarize_text(fragments[i % len(fragments)], 130)}"
            for i in range(min(4, max(1, len(workspace.get('enduser_responses', [])) + 1)))
        ]

    def _fb_enduser_raise() -> list[str]:
        return ["Could you clarify the priority between performance and accuracy?"]

    def _fb_enduser_confirm() -> list[str]:
        resps = workspace.get("enduser_responses", [])
        return [f"Confirmed: {_summarize_text(resp, 100)}" for resp in resps[:3]] or ["Confirmed: requirements are consistent."]

    def _fb_interview_records() -> list[str]:
        resps = workspace.get("enduser_responses", [])
        return [
            f"Interview record {i + 1}: {_summarize_text(item, 150)}"
            for i, item in enumerate(resps[:4])
        ] or ["Interview record 1: Initial stakeholder interview completed."]

    def _fb_user_requirements() -> list[str]:
        return [
            f"UR-{i + 1}: The system shall {_summarize_text(fragment, 150)}"
            for i, fragment in enumerate(fragments[:6])
        ]

    def _fb_deployer_dialogue() -> list[str]:
        fb = [
            f"What infrastructure constraints apply to {_summarize_text(fragment, 100)}?"
            for fragment in fragments[:3]
        ]
        return fb or [f"What deployment environment does {case_name} target?"]

    def _fb_deployer_respond() -> list[str]:
        return [
            f"Deploy constraint: {_summarize_text(fragments[i % len(fragments)], 130)}"
            for i in range(min(3, max(1, len(workspace.get('deployer_responses', [])) + 1)))
        ]

    def _fb_deployer_raise() -> list[str]:
        return ["What are the expected peak concurrent connections?"]

    def _fb_deployer_confirm() -> list[str]:
        resps = workspace.get("deployer_responses", [])
        return [f"Confirmed: {_summarize_text(resp, 100)}" for resp in resps[:3]] or ["Confirmed: deployment constraints are consistent."]

    def _fb_operating_environment() -> list[str]:
        return [
            "ENV-1: Standard web server with HTTPS termination",
            "ENV-2: Relational database with backup and recovery procedures",
            "ENV-3: Authentication and access control framework",
        ]

    def _fb_system_requirements() -> list[str]:
        ur = workspace.get("user_requirements", [])
        env = workspace.get("operating_environment", [])
        result = [f"SR-{i + 1}: The system shall {_summarize_text(req, 150)}" for i, req in enumerate(ur[:6])]
        result.extend([f"SR-deploy-{i + 1}: The system shall {_summarize_text(e, 150)}" for i, e in enumerate(env[:3])])
        return result

    def _fb_select_model() -> list[str]:
        return ["Selected model: Use Case Diagram (UML) — suitable for stakeholder interactions"]

    def _fb_build_model() -> list[str]:
        sr = workspace.get("system_requirements", [])
        entity_tokens = _tokens(" ".join(sr).lower())
        entities = [t for t in entity_tokens if len(t) >= 5][:8]
        if not entities:
            entities = ["system", "user", "requirement"]
        return [f"Model element: {e}" for e in entities[:6]]

    def _fb_srs() -> list[str]:
        return [
            "SRS-1 Purpose and Scope",
            "SRS-2 Functional Requirements",
            "SRS-3 Non-Functional Requirements and Quality Attributes",
            "SRS-4 External Interface Requirements",
        ]

    def _fb_evaluate() -> list[str]:
        requirement_text = " ".join(fragments).lower()
        findings: list[str] = []
        if any(token in requirement_text for token in ("conflict", "tradeoff", "trade-off")):
            findings.append("Finding: Potential trade-off detected between conflicting quality attributes.")
        findings.append("Finding: All requirements should include verifiable acceptance criteria per ISO/IEC/IEEE 29148.")
        findings.append("Finding: Ensure traceability from user requirements to system requirements.")
        return findings

    def _fb_closure() -> list[str]:
        rf = workspace.get("review_findings", [])
        return [f"Closure: {_summarize_text(f, 120)} — resolved" for f in rf[:4]] or ["Closure: review complete."]

    # Action descriptors: (role, action, instruction, fallback_fn, max_items, preconditions, target_key, max_fires)
    # max_fires > 1 for dialogue actions (multi-round)
    _ACTION_DESCRIPTORS: list[tuple[str, str, str, Any, int, tuple[str, ...], str, int]] = [
        # ── Elicitation: Interviewer ↔ EndUser ──
        ("Interviewer", "DialogueWithEndUser",
         "Generate interview questions that elicit user goals, pain points, and acceptance criteria using 5W1H and Socratic questioning techniques.",
         _fb_enduser_dialogue, 6, (), "enduser_dialogue", _IREDEV_DIALOGUE_ROUNDS),
        ("EndUser", "RespondEndUser",
         "Respond to interview questions with specific pain points, expectations, and usage scenarios from a business perspective.",
         _fb_enduser_respond, 4, ("enduser_dialogue",), "enduser_responses", _IREDEV_DIALOGUE_ROUNDS),
        ("EndUser", "RaiseQuestionEndUser",
         "Raise clarification questions about ambiguous or conflicting requirements.",
         _fb_enduser_raise, 3, ("enduser_responses",), "enduser_questions", _IREDEV_DIALOGUE_ROUNDS),
        ("EndUser", "ConfirmOrRefineEndUser",
         "Validate earlier requirements or refine them based on new information.",
         _fb_enduser_confirm, 3, ("enduser_questions",), "enduser_confirmations", _IREDEV_DIALOGUE_ROUNDS),
        # ── Consolidation artifacts (depend on dialogue completion) ──
        ("Interviewer", "WriteInterviewRecords",
         "Consolidate the EndUser dialogue into structured elicitation records following ISO/IEC/IEEE 29148 documentation standards.",
         _fb_interview_records, 6, ("enduser_confirmations",), "interview_records", 1),
        ("Interviewer", "WriteUserRequirementsList",
         "Synthesize a hierarchical, prioritized list of user requirements with MoSCoW priority and traceability to interview statements.",
         _fb_user_requirements, 8, ("interview_records",), "user_requirements", 1),
        # ── Elicitation: Interviewer ↔ Deployer ──
        ("Interviewer", "DialogueWithDeployer",
         "Formulate targeted questions for the Deployer that probe infrastructure constraints, security mandates, and scalability expectations.",
         _fb_deployer_dialogue, 4, ("user_requirements",), "deployer_dialogue", _IREDEV_DIALOGUE_ROUNDS),
        ("Deployer", "RespondDeployer",
         "Provide infrastructure constraints, security mandates, scalability targets, and operational procedures that answer the interviewer's questions.",
         _fb_deployer_respond, 3, ("deployer_dialogue",), "deployer_responses", _IREDEV_DIALOGUE_ROUNDS),
        ("Deployer", "RaiseQuestionDeployer",
         "Request clarification on queries lacking sufficient operational context.",
         _fb_deployer_raise, 3, ("deployer_responses",), "deployer_questions", _IREDEV_DIALOGUE_ROUNDS),
        ("Deployer", "ConfirmOrRefineDeployer",
         "Validate earlier deployment requirements or adjust in light of new information.",
         _fb_deployer_confirm, 3, ("deployer_questions",), "deployer_confirmations", _IREDEV_DIALOGUE_ROUNDS),
        # ── Consolidation artifacts (depend on deployer dialogue) ──
        ("Interviewer", "WriteOperatingEnvironmentList",
         "Compile a comprehensive environment specification capturing hardware, network, and compliance prerequisites from the deployer dialogue.",
         _fb_operating_environment, 6, ("deployer_confirmations",), "operating_environment", 1),
        # ── Analysis ──
        ("Analyst", "WriteSystemRequirementsList",
         "Transform user-level and environment-level requirements into a consolidated system requirements list adhering to IEEE 830 format.",
         _fb_system_requirements, 10, ("user_requirements", "operating_environment"), "system_requirements", 1),
        ("Analyst", "SelectRequirementModel",
         "Evaluate software system context to choose an appropriate modeling methodology (e.g., use case diagram, SysML diagram) with justification.",
         _fb_select_model, 2, ("system_requirements",), "selected_model", 1),
        ("Analyst", "BuildRequirementModel",
         "Construct requirements model with textual notation (e.g., PlantUML), highlighting conflicts or gaps for subsequent validation.",
         _fb_build_model, 8, ("selected_model",), "requirement_model", 1),
        # ── Specification ──
        ("Archivist", "WriteSRS",
         "Consolidate the approved system requirements list and requirement model into a standard-structured SRS following IEEE 830 guidelines.",
         _fb_srs, 6, ("system_requirements", "requirement_model"), "srs_sections", 1),
        # ── Validation ──
        ("Reviewer", "Evaluate",
         "Apply ISO/IEC/IEEE 29148 quality attributes (clarity, feasibility, verifiability, traceability, consistency) to evaluate the SRS. Record findings citing specific sections and violated attributes.",
         _fb_evaluate, 6, ("srs_sections",), "review_findings", 1),
        ("Reviewer", "ConfirmClosure",
         "Examine the SRS after revisions to verify that all findings are resolved. Confirm closure for each finding.",
         _fb_closure, 6, ("review_findings",), "closure_confirmations", 1),
    ]

    # Add dialogue accumulator keys to workspace
    workspace["enduser_dialogue"] = []
    workspace["deployer_dialogue"] = []

    # ── Monitor-trigger loop (artifact pool event-driven pattern) ────────
    # Track how many times each action has fired
    fire_counts: dict[str, int] = {desc[1]: 0 for desc in _ACTION_DESCRIPTORS}
    distinct_actions_fired: set[str] = set()
    max_iterations = sum(desc[7] for desc in _ACTION_DESCRIPTORS) + 5  # safety cap

    for _ in range(max_iterations):
        fired_this_iteration = False
        for role, action, instruction, fallback_fn, max_items, preconditions, target_key, max_fires in _ACTION_DESCRIPTORS:
            if fire_counts[action] >= max_fires:
                continue
            # Check preconditions: all required pool keys must be non-empty
            if all(len(workspace.get(pre, [])) > 0 for pre in preconditions):
                items, _ = _run_action(
                    role=role,
                    action=action,
                    instruction=instruction,
                    fallback_items=fallback_fn(),
                    max_items=max_items,
                )
                workspace[target_key] = workspace.get(target_key, []) + items
                fire_counts[action] += 1
                distinct_actions_fired.add(action)
                fired_this_iteration = True
                break  # Re-scan from top after each state change
        if not fired_this_iteration:
            break  # All actions completed or no preconditions satisfiable

    # Assign final local names for workspace summary
    interview_records = workspace["interview_records"]
    user_requirements = workspace["user_requirements"]
    operating_environment = workspace["operating_environment"]
    enduser_responses = workspace["enduser_responses"]
    enduser_questions = workspace["enduser_questions"]
    enduser_confirmations = workspace["enduser_confirmations"]
    deployer_responses = workspace["deployer_responses"]
    deployer_questions = workspace["deployer_questions"]
    deployer_confirmations = workspace["deployer_confirmations"]
    system_requirements = workspace["system_requirements"]
    selected_model = workspace["selected_model"]
    requirement_model = workspace["requirement_model"]
    srs_sections = workspace["srs_sections"]
    review_findings = workspace["review_findings"]
    closure_confirmations = workspace["closure_confirmations"]

    # ── Finalize execution metadata ─────────────────────────────────────

    total_fires = sum(fire_counts.values())
    if llm_meta.llm_turns == total_fires and llm_meta.llm_fallback_turns == 0:
        llm_meta.execution_mode = "llm_driven"
    elif llm_meta.llm_turns > 0:
        llm_meta.execution_mode = "llm_hybrid_fallback"
    elif llm_meta.llm_enabled:
        llm_meta.execution_mode = "deterministic_fallback"
    else:
        llm_meta.execution_mode = "deterministic_emulation"

    workspace_summary = {
        "interview_records": len(interview_records),
        "user_requirements": len(user_requirements),
        "operating_environment": len(operating_environment),
        "enduser_responses": len(enduser_responses),
        "enduser_questions": len(enduser_questions),
        "enduser_confirmations": len(enduser_confirmations),
        "deployer_responses": len(deployer_responses),
        "deployer_questions": len(deployer_questions),
        "deployer_confirmations": len(deployer_confirmations),
        "system_requirements": len(system_requirements),
        "selected_model": len(selected_model),
        "requirement_model": len(requirement_model),
        "srs_sections": len(srs_sections),
        "review_findings": len(review_findings),
        "closure_confirmations": len(closure_confirmations),
        "dialogue_rounds": _IREDEV_DIALOGUE_ROUNDS,
        "total_action_fires": total_fires,
        "distinct_actions_fired": len(distinct_actions_fired),
        "llm_turns": llm_meta.llm_turns,
        "llm_fallback_turns": llm_meta.llm_fallback_turns,
    }
    return {
        "workflow": IREDEV_RUNTIME_SEMANTICS_MODE,
        "trace_version": IREDEV_RUNTIME_TRACE_VERSION,
        "interview_records": interview_records,
        "user_requirements": user_requirements,
        "operating_environment": operating_environment,
        "enduser_responses": enduser_responses,
        "enduser_questions": enduser_questions,
        "enduser_confirmations": enduser_confirmations,
        "deployer_responses": deployer_responses,
        "deployer_questions": deployer_questions,
        "deployer_confirmations": deployer_confirmations,
        "system_requirements": system_requirements,
        "selected_model": selected_model,
        "requirement_model": requirement_model,
        "srs_sections": srs_sections,
        "review_findings": review_findings,
        "closure_confirmations": closure_confirmations,
        "workspace_summary": workspace_summary,
        "action_trace": action_trace,
        "llm_actions": llm_actions,
        "fallback_actions": fallback_actions,
    }, llm_meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_iredev_llm_messages(
    *,
    role: str,
    action: str,
    case_name: str,
    fragments: list[str],
    workspace: dict[str, Any],
    instruction: str,
) -> list[dict[str, str]]:
    """Build one action-scoped iReDev prompt contract for LLM execution."""

    payload = {
        "task": "iReDev knowledge-driven agent action execution",
        "role": role,
        "action": action,
        "instruction": instruction,
        "case_name": case_name,
        "requirement_fragments": fragments[:6],
        "workspace_snapshot": _iredev_workspace_snapshot(workspace),
        "knowledge_injection": _iredev_knowledge_context(role),
        "output_schema": {
            "items": ["non-empty string"],
            "summary": "short summary string",
        },
        "constraints": [
            "Return exactly one JSON object.",
            "Do not emit markdown fences.",
            "Keep outputs concise and requirement-grounded.",
            "Apply Chain-of-Thought reasoning to decompose complex requirements.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are executing one role/action in the iReDev knowledge-driven requirements "
                "development workflow. Use Chain-of-Thought prompting to reason through "
                "complex requirements. Return exactly one JSON object with fields: items, summary."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _iredev_workspace_snapshot(workspace: dict[str, Any]) -> dict[str, Any]:
    """Compact artifacts pool snapshot passed into iReDev action prompts."""

    snapshot: dict[str, Any] = {}
    for key in (
        "interview_records",
        "user_requirements",
        "operating_environment",
        "enduser_responses",
        "enduser_questions",
        "enduser_confirmations",
        "deployer_responses",
        "deployer_questions",
        "deployer_confirmations",
        "system_requirements",
        "selected_model",
        "requirement_model",
        "srs_sections",
        "review_findings",
        "closure_confirmations",
    ):
        value = workspace.get(key)
        if isinstance(value, list):
            snapshot[key] = [str(item).strip() for item in value[:6] if str(item).strip()]
    return snapshot


def _iredev_knowledge_context(role: str) -> dict[str, Any]:
    """Return role-specific knowledge injection context for iReDev agents."""

    knowledge: dict[str, dict[str, Any]] = {
        "Interviewer": {
            "standards": ["ISO/IEC/IEEE 29148", "BABOK v3"],
            "techniques": ["5W1H", "Socratic questioning", "iterative paraphrasing"],
            "strategies": ["MoSCoW prioritization", "life-cycle trade-off reasoning"],
        },
        "EndUser": {
            "perspective": "business scenario",
            "focus": ["pain points", "expectations", "usage feedback"],
        },
        "Deployer": {
            "standards": ["ISO/IEC/IEEE 29148 deploy environment checklist"],
            "focus": ["infrastructure constraints", "security", "scalability", "cost"],
            "strategies": ["availability-cost-performance trade-off"],
        },
        "Analyst": {
            "standards": ["IEEE 830", "ISO/IEC/IEEE 29148"],
            "models": ["UML Use Case Diagram", "SysML"],
            "focus": ["consistency", "quality attributes", "gap analysis"],
        },
        "Archivist": {
            "standards": ["IEEE 830 SRS template", "ISO/IEC/IEEE 29148"],
            "focus": ["accuracy", "completeness", "auditability"],
        },
        "Reviewer": {
            "standards": ["ISO/IEC/IEEE 29148"],
            "quality_attributes": ["clarity", "feasibility", "verifiability", "traceability", "consistency"],
            "defect_catalogue": ["ambiguity", "conflict", "redundancy"],
        },
    }
    return knowledge.get(role, {})


def _iredev_role_leaf_texts(*, role: str, workspace: dict[str, Any], rotated: list[str]) -> list[str]:
    """Select role-specific leaf material from iReDev artifacts pool."""

    if role == "Interviewer":
        values = workspace.get("interview_records", []) + workspace.get("user_requirements", [])
    elif role == "EndUser":
        values = workspace.get("enduser_responses", []) + workspace.get("enduser_confirmations", [])
    elif role == "Deployer":
        values = workspace.get("deployer_responses", []) + workspace.get("deployer_confirmations", [])
    elif role == "Analyst":
        values = workspace.get("system_requirements", []) + workspace.get("requirement_model", [])
    elif role == "Archivist":
        values = workspace.get("srs_sections", [])
    elif role == "Reviewer":
        values = workspace.get("review_findings", []) + workspace.get("closure_confirmations", [])
    else:
        values = []

    leaf_texts = [str(item).strip() for item in values if str(item).strip()]
    if leaf_texts:
        return leaf_texts
    return rotated[:3]

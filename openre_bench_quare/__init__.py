"""QUARE multi-turn dialectic negotiation and optional artifacts.

Extracted from the monolithic ``_core.py`` for single-system cohesion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openre_bench.llm import LLMClientError
from openre_bench.llm import LLMContract
from openre_bench.llm import chat_with_seed as _chat_with_seed
from openre_bench.schemas import NegotiationHistory
from openre_bench.schemas import NegotiationStep
from openre_bench.schemas import PHASE0_FILENAME
from openre_bench.schemas import PHASE25_FILENAME
from openre_bench.schemas import PHASE5_FILENAME
from openre_bench.schemas import utc_timestamp

from openre_bench.runtime_support import (
    PHASE2_LLM_RETRY_LIMIT,
    NegotiationBuildResult as _NegotiationBuildResult,
    RuntimeExecutionMeta,
    SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    apply_negotiation_adjustments as _apply_negotiation_adjustments,
    backward_analysis_text as _backward_analysis_text,
    backward_feedback as _backward_feedback,
    build_base_phase1 as _build_phase1,
    coerce_non_empty_text as _coerce_non_empty_text,
    detect_conflict as _detect_conflict,
    extract_requirement_fragments as _extract_requirement_fragments,
    parse_quare_llm_payload as _parse_quare_llm_payload,
    summarize_text as _summarize_text,
    to_float as _to_float,
    to_int as _to_int,
)

__all__ = [
    "build_negotiation_history",
    "build_optional_artifacts",
    "build_phase1",
    "build_quare_negotiation_history",
    "build_quare_optional_artifacts",
]


# ---------------------------------------------------------------------------
# QUARE multi-turn dialectic negotiation
# ---------------------------------------------------------------------------


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
    """Build QUARE phase 1 with its quality-lens agent model."""

    del llm_client, llm_temperature, llm_max_tokens, llm_seed
    phase1 = _build_phase1(case, seed, setting, rag_context)
    return phase1, {}, RuntimeExecutionMeta(llm_source=llm_source)


def build_negotiation_history(
    *,
    run_id: str,
    pair_key: str,
    focus_agent: str,
    reviewer_agent: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    requirement: str,
    setting: str,
    round_cap: int,
    step_counter: int,
    llm_client: LLMContract | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> _NegotiationBuildResult:
    """Build QUARE phase 2 dialectic negotiation history."""

    return build_quare_negotiation_history(
        run_id=run_id,
        pair_key=pair_key,
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=focus_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
        setting=setting,
        round_cap=round_cap,
        step_counter=step_counter,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )


def build_optional_artifacts(
    *,
    case: Any,
    phase2: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    setting: str,
    round_cap: int,
    rag_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build QUARE phase 0, phase 2.5, and phase 5 artifacts."""

    return build_quare_optional_artifacts(
        case=case,
        phase2=phase2,
        phase3=phase3,
        phase4=phase4,
        setting=setting,
        round_cap=round_cap,
        rag_context=rag_context,
    )


def build_quare_negotiation_history(
    *,
    run_id: str,
    pair_key: str,
    focus_agent: str,
    reviewer_agent: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    requirement: str,
    setting: str,
    round_cap: int,
    step_counter: int,
    llm_client: LLMContract | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> _NegotiationBuildResult:
    """Build QUARE-specific multi-turn dialectic negotiation history."""

    steps: list[NegotiationStep] = []
    current_elements = [dict(item) for item in focus_elements]
    max_rounds = max(1, round_cap)
    total_rounds = 0
    round_cap_hits = 0
    llm_turns = 0
    llm_fallback_turns = 0
    llm_retry_count = 0
    llm_parse_recoveries = 0
    llm_seed_applied_turns = 0

    conflict_detected = False
    conflict_resolved = False
    initial_conflict = _detect_conflict(
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=current_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
    )

    for round_number in range(1, max_rounds + 1):
        total_rounds = round_number
        forward_conflict = _detect_conflict(
            focus_agent=focus_agent,
            reviewer_agent=reviewer_agent,
            focus_elements=current_elements,
            reviewer_elements=reviewer_elements,
            requirement=requirement,
        )
        if initial_conflict and round_number == 1:
            forward_conflict = True

        conflict_detected = conflict_detected or forward_conflict

        forward_text = (
            f"Round {round_number}: {focus_agent} submits model for dialectic review by "
            f"{reviewer_agent}."
        )
        steps.append(
            NegotiationStep(
                step_id=step_counter,
                timestamp=utc_timestamp(),
                focus_agent=focus_agent,
                reviewer_agent=reviewer_agent,
                round_number=round_number,
                message_type="forward",
                kaos_elements=current_elements,
                analysis_text=forward_text,
                analysis=forward_text,
                conflict_detected=forward_conflict,
                negotiation_mode="quare_dialectic",
            )
        )
        step_counter += 1

        # Avoid a redundant first-round LLM call when deterministic heuristics already
        # require refinement and additional rounds are available.
        if round_number == 1 and max_rounds > 1 and initial_conflict:
            candidate_elements = _apply_quare_round_refinement(
                elements=current_elements,
                reviewer_agent=reviewer_agent,
                round_number=round_number,
            )
            steps.append(
                NegotiationStep(
                    step_id=step_counter,
                    timestamp=utc_timestamp(),
                    focus_agent=focus_agent,
                    reviewer_agent=reviewer_agent,
                    round_number=round_number,
                    message_type="backward",
                    kaos_elements=candidate_elements,
                    analysis_text=(
                        f"Round {round_number}: deterministic pre-LLM refinement guardrail "
                        f"activated by {reviewer_agent}."
                    ),
                    analysis=(
                        f"Round {round_number}: deterministic pre-LLM refinement guardrail "
                        f"activated by {reviewer_agent}."
                    ),
                    feedback="Additional dialectic refinement required before LLM acceptance.",
                    conflict_detected=True,
                    negotiation_mode="quare_dialectic",
                    resolution_state="rejected",
                    requires_refinement=True,
                )
            )
            step_counter += 1
            current_elements = candidate_elements
            continue

        llm_turn_result = _run_quare_llm_turn(
            focus_agent=focus_agent,
            reviewer_agent=reviewer_agent,
            requirement=requirement,
            focus_elements=current_elements,
            reviewer_elements=reviewer_elements,
            allow_description_updates=(
                setting != SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
            ),
            llm_client=llm_client,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
        )
        llm_retry_count += llm_turn_result.retry_count
        llm_parse_recoveries += llm_turn_result.parse_recoveries
        if llm_turn_result.llm_used:
            llm_turns += 1
        if llm_turn_result.llm_seed_applied:
            llm_seed_applied_turns += 1
        if llm_turn_result.used_fallback:
            llm_fallback_turns += 1

        round_conflict = llm_turn_result.conflict_detected
        conflict_detected = conflict_detected or round_conflict

        candidate_elements = [dict(item) for item in llm_turn_result.negotiated_elements]
        resolution_state = llm_turn_result.resolution_state
        requires_refinement = llm_turn_result.requires_refinement
        analysis_text = llm_turn_result.analysis_text
        feedback = llm_turn_result.feedback

        # QUARE dialectics require at least one reject/refine turn when any first-round
        # conflict signal is present and additional rounds are available.
        if round_number == 1 and max_rounds > 1 and (initial_conflict or round_conflict):
            round_conflict = True
            conflict_detected = True
            if not requires_refinement:
                requires_refinement = True
                resolution_state = "rejected"
                feedback = (
                    f"{feedback} Additional dialectic refinement required before acceptance."
                )

        # Preserve QUARE dialectic semantics: refinement always dominates a resolved claim.
        effective_conflict_resolved = llm_turn_result.conflict_resolved and not requires_refinement

        if round_conflict and not effective_conflict_resolved:
            if round_number < max_rounds and requires_refinement:
                candidate_elements = _apply_quare_round_refinement(
                    elements=candidate_elements,
                    reviewer_agent=reviewer_agent,
                    round_number=round_number,
                )
                resolution_state = "rejected"
                requires_refinement = True
                if llm_turn_result.used_fallback:
                    analysis_text = (
                        f"Round {round_number}: fallback reviewer critique requests "
                        "refinement due to unresolved conflict."
                    )
                steps.append(
                    NegotiationStep(
                        step_id=step_counter,
                        timestamp=utc_timestamp(),
                        focus_agent=focus_agent,
                        reviewer_agent=reviewer_agent,
                        round_number=round_number,
                        message_type="backward",
                        kaos_elements=candidate_elements,
                        analysis_text=analysis_text,
                        analysis=analysis_text,
                        feedback=feedback,
                        conflict_detected=True,
                        negotiation_mode="quare_dialectic",
                        resolution_state=resolution_state,
                        requires_refinement=requires_refinement,
                    )
                )
                step_counter += 1
                current_elements = candidate_elements
                continue

            if round_number >= max_rounds:
                round_cap_hits += 1
                # Reaching the negotiation cap terminates the dialectic loop,
                # but it must not be counted as a resolved conflict.
                candidate_elements = _finalize_quare_acceptance(
                    elements=candidate_elements,
                    reviewer_agent=reviewer_agent,
                )
                resolution_state = "unresolved"
                requires_refinement = False
                analysis_text = (
                    f"Round {round_number}: round cap reached; {reviewer_agent} "
                    "records unresolved state for downstream prioritization."
                )
                feedback = (
                    "Round cap reached; conflict remains unresolved and requires integration-phase "
                    "prioritization or human review."
                )
                conflict_resolved = False
        elif round_conflict:
            candidate_elements = _apply_quare_resolution_adjustments(
                elements=candidate_elements,
                reviewer_agent=reviewer_agent,
                conflict_detected=True,
            )
            conflict_resolved = True
            resolution_state = "resolved"
            requires_refinement = False
        else:
            candidate_elements = _finalize_quare_acceptance(
                elements=candidate_elements,
                reviewer_agent=reviewer_agent,
            )
            # If any earlier round detected conflict and we now accept, mark it resolved.
            if conflict_detected:
                conflict_resolved = True
            resolution_state = "accepted"
            requires_refinement = False

        steps.append(
            NegotiationStep(
                step_id=step_counter,
                timestamp=utc_timestamp(),
                focus_agent=focus_agent,
                reviewer_agent=reviewer_agent,
                round_number=round_number,
                message_type="backward",
                kaos_elements=candidate_elements,
                analysis_text=analysis_text,
                analysis=analysis_text,
                feedback=feedback,
                conflict_detected=round_conflict,
                negotiation_mode="quare_dialectic",
                resolution_state=resolution_state,
                requires_refinement=requires_refinement,
            )
        )
        step_counter += 1
        break

    history = NegotiationHistory(
        negotiation_id=f"neg_{pair_key}_{run_id}",
        focus_agent=focus_agent,
        reviewer_agents=[reviewer_agent],
        start_timestamp=utc_timestamp(),
        end_timestamp=utc_timestamp(),
        steps=steps,
        final_consensus=conflict_resolved or not conflict_detected,
        total_rounds=total_rounds,
    )

    return _NegotiationBuildResult(
        history=history,
        next_step_id=step_counter,
        conflict_detected=conflict_detected,
        conflict_resolved=conflict_resolved,
        llm_turns=llm_turns,
        llm_fallback_turns=llm_fallback_turns,
        llm_retry_count=llm_retry_count,
        llm_parse_recoveries=llm_parse_recoveries,
        llm_seed_applied_turns=llm_seed_applied_turns,
        round_cap_hits=round_cap_hits,
    )


# ---------------------------------------------------------------------------
# QUARE LLM turn
# ---------------------------------------------------------------------------


@dataclass
class _QuareLLMTurnResult:
    """One QUARE reviewer turn result after optional LLM parsing/recovery."""

    conflict_detected: bool
    conflict_resolved: bool
    requires_refinement: bool
    resolution_state: str
    analysis_text: str
    feedback: str
    negotiated_elements: list[dict[str, Any]]
    llm_used: bool
    llm_seed_applied: bool
    used_fallback: bool
    retry_count: int
    parse_recoveries: int
    error_reason: str


def _run_quare_llm_turn(
    *,
    focus_agent: str,
    reviewer_agent: str,
    requirement: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    allow_description_updates: bool,
    llm_client: LLMContract | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> _QuareLLMTurnResult:
    """Run one QUARE dialectic reviewer turn with deterministic fallback."""

    baseline_conflict = _detect_conflict(
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=focus_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
    )
    baseline_elements = _apply_negotiation_adjustments(
        elements=focus_elements,
        reviewer_agent=reviewer_agent,
        conflict_detected=baseline_conflict,
    )
    baseline_refinement = baseline_conflict
    baseline_resolution_state = "rejected" if baseline_refinement else "accepted"

    if llm_client is None:
        return _QuareLLMTurnResult(
            conflict_detected=baseline_conflict,
            conflict_resolved=False if baseline_conflict else True,
            requires_refinement=baseline_refinement,
            resolution_state=baseline_resolution_state,
            analysis_text=_backward_analysis_text(reviewer_agent, baseline_conflict),
            feedback=_backward_feedback(baseline_conflict),
            negotiated_elements=baseline_elements,
            llm_used=False,
            llm_seed_applied=False,
            used_fallback=True,
            retry_count=0,
            parse_recoveries=0,
            error_reason="llm_unavailable",
        )

    retry_count = 0
    parse_recoveries = 0
    last_error = ""
    max_attempts = max(1, PHASE2_LLM_RETRY_LIMIT + 1)

    for attempt in range(max_attempts):
        try:
            raw_response, seed_applied = _chat_with_seed(
                llm_client=llm_client,
                messages=_build_quare_llm_messages(
                    focus_agent=focus_agent,
                    reviewer_agent=reviewer_agent,
                    requirement=requirement,
                    focus_elements=focus_elements,
                    reviewer_elements=reviewer_elements,
                    llm_model=llm_model,
                ),
                temperature=max(0.0, min(float(llm_temperature), 1.0)),
                max_tokens=max(128, int(llm_max_tokens)),
                seed=llm_seed,
            )
        except (LLMClientError, RuntimeError, ValueError) as exc:
            last_error = f"request_failed: {exc}"
            continue

        try:
            payload, recovered = _parse_quare_llm_payload(raw_response)
        except ValueError as exc:
            payload = _recover_quare_payload_shape(raw_response)
            if payload:
                recovered = True
            else:
                last_error = f"parse_failed: {exc}"
                if attempt < max_attempts - 1:
                    retry_count += 1
                continue

        if recovered:
            parse_recoveries += 1

        conflict_detected = _coerce_optional_bool(payload.get("conflict_detected"))
        if conflict_detected is None:
            conflict_detected = baseline_conflict

        conflict_resolved = _coerce_optional_bool(payload.get("conflict_resolved"))
        if conflict_resolved is None:
            conflict_resolved = not conflict_detected

        requires_refinement = _coerce_optional_bool(payload.get("requires_refinement"))
        if requires_refinement is None:
            requires_refinement = conflict_detected and not conflict_resolved

        # Coerce inconsistent tuples into guardrail-safe semantics.
        if requires_refinement:
            conflict_detected = True
            conflict_resolved = False
        elif conflict_detected:
            requires_refinement = not conflict_resolved
        else:
            conflict_detected = False
            conflict_resolved = False
            requires_refinement = False

        analysis_text = _coerce_non_empty_text(
            payload.get("analysis_text") or payload.get("analysis"),
            fallback=_backward_analysis_text(reviewer_agent, conflict_detected),
        )
        feedback = _coerce_non_empty_text(
            payload.get("feedback"),
            fallback=_backward_feedback(conflict_detected),
        )

        negotiated_elements = _merge_quare_llm_element_updates(
            element_updates=payload.get("element_updates") or payload.get("kaos_elements"),
            base_elements=focus_elements,
            reviewer_agent=reviewer_agent,
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            allow_description_updates=allow_description_updates,
        )

        if requires_refinement:
            resolution_state = "rejected"
        elif conflict_detected and conflict_resolved:
            resolution_state = "resolved"
        elif conflict_detected:
            resolution_state = "unresolved"
        else:
            resolution_state = "accepted"

        return _QuareLLMTurnResult(
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            requires_refinement=requires_refinement,
            resolution_state=resolution_state,
            analysis_text=analysis_text,
            feedback=feedback,
            negotiated_elements=negotiated_elements,
            llm_used=True,
            llm_seed_applied=seed_applied,
            used_fallback=False,
            retry_count=retry_count,
            parse_recoveries=parse_recoveries,
            error_reason="",
        )

    return _QuareLLMTurnResult(
        conflict_detected=baseline_conflict,
        conflict_resolved=False if baseline_conflict else True,
        requires_refinement=baseline_refinement,
        resolution_state=baseline_resolution_state,
        analysis_text=_backward_analysis_text(reviewer_agent, baseline_conflict),
        feedback=_backward_feedback(baseline_conflict),
        negotiated_elements=baseline_elements,
        llm_used=False,
        llm_seed_applied=False,
        used_fallback=True,
        retry_count=retry_count,
        parse_recoveries=parse_recoveries,
        error_reason=last_error or "llm_invalid_response",
    )


# ---------------------------------------------------------------------------
# QUARE LLM prompt building & parsing helpers
# ---------------------------------------------------------------------------


def _build_quare_llm_messages(
    *,
    focus_agent: str,
    reviewer_agent: str,
    requirement: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    llm_model: str,
) -> list[dict[str, str]]:
    """Build one compact structured prompt for QUARE review turns."""

    focus_payload = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "description": item.get("description"),
            "hierarchy_level": item.get("hierarchy_level"),
            "validation_status": item.get("validation_status"),
        }
        for item in focus_elements
    ]
    reviewer_payload = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "description": item.get("description"),
            "hierarchy_level": item.get("hierarchy_level"),
            "validation_status": item.get("validation_status"),
        }
        for item in reviewer_elements
    ]

    instruction_payload = {
        "task": "QUARE Phase-2 dialectic negotiation",
        "model": llm_model,
        "focus_agent": focus_agent,
        "reviewer_agent": reviewer_agent,
        "requirement": _summarize_text(requirement, 420),
        "focus_elements": focus_payload,
        "reviewer_elements": reviewer_payload,
        "output_schema": {
            "analysis_text": "string",
            "feedback": "string",
            "conflict_detected": "boolean",
            "conflict_resolved": "boolean",
            "requires_refinement": "boolean",
            "element_updates": [
                {
                    "id": "existing focus element id",
                    "description": "optional updated description",
                    "validation_status": "pending|candidate_resolution|resolved",
                    "conflict_resolved_by": "reviewer agent when resolved"
                }
            ]
        }
    }
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one valid JSON object. Use JSON booleans true/false. "
                "Keep string values single-line. Do not emit markdown fences or prose."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(instruction_payload, ensure_ascii=False),
        },
    ]


def _recover_quare_payload_shape(raw_response: str) -> dict[str, Any]:
    """Recover QUARE fields from malformed JSON without changing turn semantics."""

    text = raw_response.strip()
    if not text:
        return {}

    text = re.sub(r"```[a-zA-Z0-9_-]*", "", text).replace("```", "")
    payload: dict[str, Any] = {}

    for field_name in (
        "conflict_detected",
        "conflict_resolved",
        "requires_refinement",
    ):
        value = _extract_relaxed_bool_field(text, field_name)
        if value is not None:
            payload[field_name] = value

    for field_name in ("analysis_text", "analysis", "feedback"):
        value = _extract_relaxed_text_field(text, field_name)
        if value:
            payload[field_name] = value

    if "element_updates" in text:
        payload["element_updates"] = []

    return payload


def _extract_relaxed_bool_field(text: str, field_name: str) -> bool | None:
    """Extract one JSON-like boolean field from an otherwise malformed response."""

    match = re.search(
        rf"""["']?{re.escape(field_name)}["']?\s*:\s*["']?(true|false|yes|no|1|0)["']?""",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _coerce_optional_bool(match.group(1))


def _extract_relaxed_text_field(text: str, field_name: str) -> str:
    """Extract one JSON-like text field while tolerating raw newlines in strings."""

    match = re.search(rf"""["']?{re.escape(field_name)}["']?\s*:""", text)
    if not match:
        return ""

    tail = text[match.end() :].lstrip()
    if not tail:
        return ""

    if tail[0] in {'"', "'"}:
        raw_value = _read_relaxed_quoted_value(tail[1:], tail[0])
    else:
        raw_value = re.split(r""",\s*["']?[A-Za-z_][A-Za-z0-9_]*["']?\s*:|[}\n]""", tail, maxsplit=1)[0]

    return _summarize_text(
        raw_value.replace("\\n", " ").replace('\\"', '"').strip().strip(","),
        320,
    )


def _read_relaxed_quoted_value(text: str, quote: str) -> str:
    """Read a quoted JSON-like value until the next top-level key boundary."""

    boundary = re.search(
        rf"""{re.escape(quote)}?\s*,\s*(?=\n?\s*["']?[A-Za-z_][A-Za-z0-9_]*["']?\s*:)""",
        text,
    )
    if boundary:
        return text[: boundary.start()].strip()

    end_index = text.rfind(quote)
    if end_index >= 0:
        return text[:end_index].strip()
    return text.strip()


def _merge_quare_llm_element_updates(
    *,
    element_updates: Any,
    base_elements: list[dict[str, Any]],
    reviewer_agent: str,
    conflict_detected: bool,
    conflict_resolved: bool,
    allow_description_updates: bool,
) -> list[dict[str, Any]]:
    """Merge optional LLM element updates over base focus elements."""

    updates: dict[str, dict[str, Any]] = {}
    if isinstance(element_updates, list):
        for item in element_updates:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            if item_id:
                updates[item_id] = item

    merged: list[dict[str, Any]] = []
    for base in base_elements:
        element = dict(base)
        update = updates.get(str(element.get("id", "")))
        if update:
            description = update.get("description")
            if (
                allow_description_updates
                and isinstance(description, str)
                and description.strip()
            ):
                element["description"] = description.strip()
            validation_status = update.get("validation_status")
            if isinstance(validation_status, str) and validation_status.strip():
                element["validation_status"] = validation_status.strip()
            conflict_resolved_by = update.get("conflict_resolved_by")
            if isinstance(conflict_resolved_by, str) and conflict_resolved_by.strip():
                element["conflict_resolved_by"] = conflict_resolved_by.strip()

        merged.append(element)

    if conflict_detected and conflict_resolved:
        return _apply_quare_resolution_adjustments(
            elements=merged,
            reviewer_agent=reviewer_agent,
            conflict_detected=True,
        )
    return merged


def _coerce_optional_bool(value: Any) -> bool | None:
    """Return bool for permissive boolean-like values; None when unknown."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


# ---------------------------------------------------------------------------
# QUARE refinement / resolution / acceptance helpers
# ---------------------------------------------------------------------------


def _apply_quare_round_refinement(
    *,
    elements: list[dict[str, Any]],
    reviewer_agent: str,
    round_number: int,
) -> list[dict[str, Any]]:
    """Apply one QUARE refinement turn before final acceptance."""

    updated = [dict(item) for item in elements]
    for element in updated:
        if int(element.get("hierarchy_level", 1)) != 2:
            continue
        # Keep source descriptions stable to reduce semantic drift during QUARE rounds.
        element["validation_status"] = "candidate_resolution"
        element["conflict_resolved_by"] = None
    return updated


def _apply_quare_resolution_adjustments(
    *,
    elements: list[dict[str, Any]],
    reviewer_agent: str,
    conflict_detected: bool,
) -> list[dict[str, Any]]:
    """Finalize QUARE conflict resolution without mutating requirement wording."""

    updated = [dict(item) for item in elements]
    if not conflict_detected:
        return updated

    for element in updated:
        if int(element.get("hierarchy_level", 1)) != 2:
            continue
        element["conflict_resolved_by"] = reviewer_agent
        element["validation_status"] = "resolved"
    return updated


def _finalize_quare_acceptance(
    *,
    elements: list[dict[str, Any]],
    reviewer_agent: str,
) -> list[dict[str, Any]]:
    """Finalize QUARE run elements when reviewer accepts without further rejection."""

    updated = [dict(item) for item in elements]
    for element in updated:
        if int(element.get("hierarchy_level", 1)) != 2:
            continue
        status = str(element.get("validation_status", "")).lower()
        if status == "candidate_resolution":
            element["validation_status"] = "resolved"
            element["conflict_resolved_by"] = reviewer_agent
        elif not status:
            element["validation_status"] = "accepted"
    return updated


# ---------------------------------------------------------------------------
# QUARE optional artifacts (phase 0, 2.5, 5)
# ---------------------------------------------------------------------------


def build_quare_optional_artifacts(
    *,
    case: Any,
    phase2: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    setting: str,
    round_cap: int,
    rag_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build QUARE-only protocol artifacts beyond canonical phase1-4 outputs."""

    phase0_payload = _build_phase0_external_spec_rules(
        case=case,
        setting=setting,
        rag_context=rag_context,
    )
    phase25_payload = _build_phase25_conflict_map(
        case=case,
        phase2=phase2,
        setting=setting,
        round_cap=round_cap,
    )
    phase5_payload = _build_phase5_software_materials(
        case=case,
        phase3=phase3,
        phase4=phase4,
        phase25=phase25_payload,
        setting=setting,
    )
    return {
        PHASE0_FILENAME: phase0_payload,
        PHASE25_FILENAME: phase25_payload,
        PHASE5_FILENAME: phase5_payload,
    }


def _build_phase0_external_spec_rules(
    *,
    case: Any,
    setting: str,
    rag_context: dict[str, Any],
) -> dict[str, Any]:
    """Extract deterministic external-spec style control rules from requirement text."""

    requirement = case.requirement.strip()
    fragments = _extract_requirement_fragments(requirement)
    rules = _extract_external_rules(requirement=requirement, fragments=fragments)
    return {
        "phase": "0_external_spec_processing",
        "case_id": case.case_name,
        "setting": setting,
        "generated_at": utc_timestamp(),
        "extracted_rules": rules,
        "extraction_metadata": {
            "rule_count": len(rules),
            "rag_enabled": bool(rag_context.get("rag_enabled", False)),
            "rag_corpus_hash": str(rag_context.get("corpus_hash", "")),
        },
    }


def _build_phase25_conflict_map(
    *,
    case: Any,
    phase2: dict[str, Any],
    setting: str,
    round_cap: int,
) -> dict[str, Any]:
    """Summarize per-pair conflict lifecycle for QUARE phase-2.5 parity output."""

    negotiations = phase2.get("negotiations", {})
    conflict_map: dict[str, dict[str, Any]] = {}
    detected_pairs = 0
    resolved_pairs = 0
    max_round_observed = 0

    if isinstance(negotiations, dict):
        for pair_key, negotiation in negotiations.items():
            if not isinstance(negotiation, dict):
                continue
            steps = negotiation.get("steps", [])
            total_rounds = _to_int(negotiation.get("total_rounds"), default=0)
            max_round_observed = max(max_round_observed, total_rounds)
            detected = False
            refinement_count = 0
            latest_resolution = ""
            for step in steps if isinstance(steps, list) else []:
                if not isinstance(step, dict):
                    continue
                if bool(step.get("conflict_detected", False)):
                    detected = True
                if bool(step.get("requires_refinement", False)):
                    refinement_count += 1
                resolution = str(step.get("resolution_state", "")).strip()
                if resolution:
                    latest_resolution = resolution

            final_consensus = bool(negotiation.get("final_consensus", False))
            resolved = detected and (
                final_consensus or latest_resolution in {"resolved", "accepted"}
            )
            if detected:
                detected_pairs += 1
            if detected and resolved:
                resolved_pairs += 1

            conflict_map[str(pair_key)] = {
                "detected_conflict": detected,
                "resolved_conflict": resolved,
                "total_rounds": total_rounds,
                "max_round_observed": total_rounds,
                "requires_refinement_count": refinement_count,
                "latest_resolution_state": latest_resolution,
                "final_consensus": final_consensus,
            }

    unresolved_pairs = max(0, detected_pairs - resolved_pairs)
    phase2_summary = phase2.get("summary_stats", {})
    phase2_detected = _to_int(
        phase2_summary.get("detected_conflicts", 0) if isinstance(phase2_summary, dict) else 0,
        default=0,
    )
    phase2_resolved = _to_int(
        phase2_summary.get("resolved_conflicts", 0) if isinstance(phase2_summary, dict) else 0,
        default=0,
    )
    return {
        "phase": "2.5_conflict_resolution",
        "case_id": case.case_name,
        "setting": setting,
        "generated_at": utc_timestamp(),
        "round_cap": max(1, int(round_cap)),
        "conflict_map": conflict_map,
        "summary": {
            "total_pairs": len(conflict_map),
            "detected_conflict_pairs": detected_pairs,
            "resolved_conflict_pairs": resolved_pairs,
            "unresolved_conflict_pairs": unresolved_pairs,
            "max_round_observed": max_round_observed,
            "round_cap_reached": bool(max_round_observed >= max(1, int(round_cap))),
            "phase2_detected_conflicts": phase2_detected,
            "phase2_resolved_conflicts": phase2_resolved,
            "phase2_alignment_ok": detected_pairs == phase2_detected
            and resolved_pairs == phase2_resolved,
        },
    }


def _build_phase5_software_materials(
    *,
    case: Any,
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase25: dict[str, Any],
    setting: str,
) -> dict[str, Any]:
    """Generate QUARE-style downstream software materials from integrated outputs."""

    gsn_elements = phase3.get("gsn_elements", [])
    traceability: list[dict[str, Any]] = []
    for element in gsn_elements if isinstance(gsn_elements, list) else []:
        if not isinstance(element, dict):
            continue
        traceability.append(
            {
                "element_id": str(element.get("id", "")),
                "quality_attribute": str(element.get("quality_attribute", "")),
                "statement": _summarize_text(str(element.get("description", "")), 180),
                "verification_anchor": str(
                    element.get("properties", {}).get("validation_status", "pending")
                ),
            }
        )

    verification_results = phase4.get("verification_results", {})
    compliance = _to_float(
        verification_results.get("compliance_coverage", {}).get("coverage_ratio")
        if isinstance(verification_results, dict)
        else 0.0,
        default=0.0,
    )
    s_logic = _to_float(
        verification_results.get("s_logic") if isinstance(verification_results, dict) else 0.0,
        default=0.0,
    )
    topology_valid = bool(phase3.get("topology_status", {}).get("is_valid", False))
    deterministic_valid = bool(phase4.get("deterministic_validation", {}).get("is_valid", False))

    conflict_summary = phase25.get("summary", {}) if isinstance(phase25, dict) else {}
    return {
        "phase": "5_software_materials_generation",
        "case_id": case.case_name,
        "setting": setting,
        "generated_at": utc_timestamp(),
        "materials": {
            "srs_outline": [
                "System Scope and Stakeholders",
                "Quality-Attribute Requirements (Safety/Efficiency/Green/Trustworthiness/Responsibility)",
                "Negotiation and Conflict Resolution Decisions",
                "Verification and Compliance Evidence",
            ],
            "implementation_checklist": [
                "Preserve canonical phase artifact compatibility",
                "Enforce strict provenance and taint controls",
                "Trace each integrated requirement to a verification signal",
            ],
            "traceability_matrix": traceability,
        },
        "quality_signals": {
            "topology_is_valid": topology_valid,
            "deterministic_is_valid": deterministic_valid,
            "compliance_coverage": round(compliance, 6),
            "s_logic": round(s_logic, 6),
            "detected_conflict_pairs": _to_int(conflict_summary.get("detected_conflict_pairs"), 0),
            "resolved_conflict_pairs": _to_int(conflict_summary.get("resolved_conflict_pairs"), 0),
        },
    }


def _extract_external_rules(*, requirement: str, fragments: list[str]) -> list[dict[str, Any]]:
    """Derive normalized rule stubs from requirement language."""

    rules: list[dict[str, Any]] = []
    normalized_fragments = fragments or [requirement]
    for index, fragment in enumerate(normalized_fragments[:12], start=1):
        text = _summarize_text(fragment, 220)
        lowered = text.lower()
        rule_type = "constraint"
        if "if " in lowered or "when " in lowered:
            rule_type = "trigger"
        elif any(token in lowered for token in ("must", "shall", "should")):
            rule_type = "obligation"
        elif any(token in lowered for token in ("not", "never", "forbid")):
            rule_type = "prohibition"
        rules.append(
            {
                "rule_id": f"R{index:03d}",
                "rule_type": rule_type,
                "statement": text,
                "source": "requirement_text",
            }
        )
    return rules

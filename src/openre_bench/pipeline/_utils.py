"""Shared utility functions for the pipeline package.

Contains text helpers, hashing, fragment rotation, topology analysis,
validation primitives, and phase-setting query helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from openre_bench.schemas import (
    DEFAULT_AGENT_QUALITY_ATTRIBUTES,
    IREDEV_ACTIONS,
    IREDEV_AGENT_ROLES,
    MARE_ACTIONS,
    MARE_AGENT_ROLES,
    SETTING_MULTI_AGENT_WITH_NEGOTIATION,
    SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    SETTING_SINGLE_AGENT,
    SYSTEM_IREDEV,
    SYSTEM_MARE,
    utc_timestamp,
)
from openre_bench.pipeline._types import (
    IREDEV_ROLE_QUALITY_ATTRIBUTES,
    IREDEV_RUNTIME_SEMANTICS_MODE,
    MARE_ROLE_QUALITY_ATTRIBUTES,
    MARE_RUNTIME_SEMANTICS_MODE,
    QUALITY_LENS_CUES,
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    """Tokenize to lowercase alphanumeric words (input must be pre-lowered)."""

    return re.findall(r"[a-z0-9_]+", text)


def _summarize_text(text: str, limit: int) -> str:
    """Trim text to a bounded length preserving readability."""

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _text_overlap_score(left: str, right: str) -> float:
    """Compute token overlap F1 for two text strings."""

    left_tokens = set(_tokens(left.lower()))
    right_tokens = set(_tokens(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _to_int(value: Any, default: int = 0) -> int:
    """Parse integer-like values safely for optional artifact accounting."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Parse float-like values safely for optional artifact accounting."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sha256_payload(payload: dict[str, Any]) -> str:
    """Return deterministic SHA256 digest for JSON-like payload."""

    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fragment & agent helpers
# ---------------------------------------------------------------------------


def _extract_requirement_fragments(requirement: str) -> list[str]:
    """Split requirement text into sentence-like fragments."""

    normalized = requirement.replace("\r", "\n")
    chunks = re.split(r"[\n\.\?\!]+", normalized)
    fragments = [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks]
    return [item for item in fragments if len(item) >= 20]


def _rotate_fragments(fragments: list[str], seed: int) -> list[str]:
    """Apply deterministic rotation without changing fragment content."""

    if not fragments:
        return []
    offset = seed % len(fragments)
    return fragments[offset:] + fragments[:offset]


def _agent_fragment_window(
    *,
    rotated: list[str],
    agent_index: int,
    total_agents: int,
    leaf_count: int,
) -> list[str]:
    """Return deterministic per-agent fragment windows with fixed leaf count."""

    if not rotated:
        return []
    fragment_count = len(rotated)
    window_size = max(1, leaf_count)
    start = _agent_fragment_start_index(
        fragment_count=fragment_count,
        agent_index=agent_index,
        total_agents=total_agents,
    )
    selected: list[str] = []
    for offset in range(window_size):
        cursor = (start + offset) % fragment_count
        selected.append(rotated[cursor])
    return selected


def _agent_fragment_start_index(*, fragment_count: int, agent_index: int, total_agents: int) -> int:
    """Compute an evenly distributed start index for one agent fragment window."""

    if fragment_count <= 0:
        return 0
    agent_slot = max(0, agent_index - 1)
    safe_total_agents = max(1, total_agents)
    return ((agent_slot * fragment_count) // safe_total_agents) % fragment_count


def _quality_lens_phrase(*, quality_attribute: str, leaf_index: int) -> str:
    """Return deterministic quality-lens phrasing to diversify multi-agent outputs."""

    cues = QUALITY_LENS_CUES.get(quality_attribute, QUALITY_LENS_CUES["Integrated"])
    index = max(0, leaf_index - 1) % len(cues)
    return cues[index]


# ---------------------------------------------------------------------------
# Phase-setting queries
# ---------------------------------------------------------------------------


def _negotiation_enabled(setting: str) -> bool:
    """Whether phase 2 negotiation should run for the setting."""

    return setting in {
        SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    }


def _verification_executed(setting: str) -> bool:
    """Whether phase 4 verification should run for the setting."""

    return setting == SETTING_NEGOTIATION_INTEGRATION_VERIFICATION


def _agent_quality_mapping_for_setting(setting: str) -> dict[str, str]:
    """Return agent-quality mapping for a given experimental setting."""

    if setting == SETTING_SINGLE_AGENT:
        return {"SingleAgent": "Integrated"}
    return DEFAULT_AGENT_QUALITY_ATTRIBUTES


def _runtime_semantics_mode(*, system: str, setting: str) -> str:
    """Return machine-checkable runtime semantics mode label."""

    if system == SYSTEM_MARE and setting != SETTING_SINGLE_AGENT:
        return MARE_RUNTIME_SEMANTICS_MODE
    if system == SYSTEM_MARE:
        return "mare_single_agent_baseline"
    if system == SYSTEM_IREDEV and setting != SETTING_SINGLE_AGENT:
        return IREDEV_RUNTIME_SEMANTICS_MODE
    if system == SYSTEM_IREDEV:
        return "iredev_single_agent_baseline"
    return "quare_dialectic_scaffold_v1"


def _prompt_contract_hash(*, system: str, setting: str, round_cap: int, max_tokens: int) -> str:
    """Hash deterministic generation contract to detect configuration drift."""

    runtime_mode = _runtime_semantics_mode(system=system, setting=setting)
    contract: dict[str, Any] = {
        "generator": "openre_bench-deterministic-parity-pipeline",
        "system": system,
        "setting": setting,
        "round_cap": round_cap,
        "max_tokens": max_tokens,
        "runtime_semantics_mode": runtime_mode,
        "agent_quality_mapping": _agent_quality_mapping_for_setting(setting),
    }
    if runtime_mode == MARE_RUNTIME_SEMANTICS_MODE:
        contract["agent_quality_mapping"] = {
            role: MARE_ROLE_QUALITY_ATTRIBUTES.get(role, "Integrated")
            for role in MARE_AGENT_ROLES
        }
        contract["mare_roles"] = list(MARE_AGENT_ROLES)
        contract["mare_actions"] = list(MARE_ACTIONS)
    elif runtime_mode == IREDEV_RUNTIME_SEMANTICS_MODE:
        contract["agent_quality_mapping"] = {
            role: IREDEV_ROLE_QUALITY_ATTRIBUTES.get(role, "Integrated")
            for role in IREDEV_AGENT_ROLES
        }
        contract["iredev_roles"] = list(IREDEV_AGENT_ROLES)
        contract["iredev_actions"] = list(IREDEV_ACTIONS)
    serialized = json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Run ID helper
# ---------------------------------------------------------------------------


def default_run_id(case_name: str, seed: int) -> str:
    """Build a readable run id for one-case scaffold runs."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", case_name.strip().lower()).strip("-")
    return f"{normalized}-{utc_timestamp().replace(':', '').replace('-', '')}-s{seed:03d}"


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------


def _hierarchy_view(item: dict[str, Any]) -> dict[str, Any]:
    """Return hierarchy view shape used by phase 3 export."""

    return {
        "id": item["id"],
        "name": item["name"],
        "quality_attribute": item.get("quality_attribute"),
        "parent_goal_id": item.get("parent_goal_id"),
    }


def _parent_child_mappings(elements: list[dict[str, Any]]) -> dict[str, list[str]]:
    mappings: dict[str, list[str]] = {}
    for element in elements:
        parent_id = element.get("parent_goal_id")
        if not parent_id:
            continue
        mappings.setdefault(parent_id, []).append(element["id"])
    return mappings


# ---------------------------------------------------------------------------
# Topology & verification analysis
# ---------------------------------------------------------------------------


def _compute_topology_status(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute DAG topology validity from GSN elements."""

    ids = {item["id"] for item in elements}
    children: dict[str, list[str]] = {item["id"]: [] for item in elements}
    orphan_elements: list[str] = []

    for item in elements:
        parent_id = item.get("parent_goal_id")
        if not parent_id:
            if item.get("gsn_type") != "Goal":
                orphan_elements.append(item["id"])
            continue
        if parent_id not in ids:
            orphan_elements.append(item["id"])
            continue
        children[parent_id].append(item["id"])

    cycles: list[list[str]] = []
    visited: set[str] = set()
    active: set[str] = set()

    def walk(node: str, path: list[str]) -> None:
        if node in active:
            start = path.index(node) if node in path else 0
            cycles.append(path[start:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        active.add(node)
        for child in children.get(node, []):
            walk(child, path + [child])
        active.remove(node)

    for node_id in children:
        walk(node_id, [node_id])

    invalid_leaves = [
        item["id"]
        for item in elements
        if not children.get(item["id"], []) and item.get("gsn_type") == "Goal"
    ]

    is_valid = not orphan_elements and not cycles and not invalid_leaves
    return {
        "status": "Valid" if is_valid else "Invalid",
        "is_valid": is_valid,
        "is_dag": not cycles,
        "orphan_count": len(orphan_elements),
        "cycle_count": len(cycles),
        "invalid_leaf_count": len(invalid_leaves),
        "orphan_elements": orphan_elements,
        "cycles": cycles,
        "invalid_leaves": invalid_leaves,
    }


def _logical_consistency(gsn_elements: list[dict[str, Any]]) -> dict[str, Any]:
    """ISO29148-style contradiction checks for logical consistency."""

    constraints: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []

    for element in gsn_elements:
        description = str(element.get("description", ""))
        element_id = str(element.get("id", ""))
        for var, op, value in re.findall(r"([a-zA-Z_]+)\s*(≤|≥|<|>|=)\s*(\d+\.?\d*)", description):
            constraints.append(
                {
                    "element_id": element_id,
                    "variable": var,
                    "operator": op,
                    "value": float(value),
                }
            )

        forbidden = re.search(r"\b(must|shall|should)\s+not\s+(\w+)", description, re.IGNORECASE)
        if forbidden:
            constraints.append(
                {
                    "element_id": element_id,
                    "variable": forbidden.group(2).lower(),
                    "operator": "forbidden",
                    "value": None,
                }
            )

    for index, left in enumerate(constraints):
        for right in constraints[index + 1 :]:
            if left["variable"].lower() != right["variable"].lower():
                continue
            if left["operator"] == "forbidden" and right["operator"] != "forbidden":
                contradictions.append(
                    {
                        "type": "forbidden_conflict",
                        "variable": left["variable"],
                        "left": left,
                        "right": right,
                    }
                )
                continue
            if right["operator"] == "forbidden" and left["operator"] != "forbidden":
                contradictions.append(
                    {
                        "type": "forbidden_conflict",
                        "variable": left["variable"],
                        "left": left,
                        "right": right,
                    }
                )
                continue

            if left["value"] is None or right["value"] is None:
                continue
            if left["operator"] in {"≤", "<", "="} and right["operator"] in {"≥", ">", "="}:
                if left["operator"] == "≤" and right["operator"] == "≥" and left["value"] < right["value"]:
                    contradictions.append(
                        {
                            "type": "numerical_contradiction",
                            "variable": left["variable"],
                            "left": left,
                            "right": right,
                        }
                    )
                if left["operator"] == "=" and right["operator"] == "=" and left["value"] != right["value"]:
                    contradictions.append(
                        {
                            "type": "numerical_contradiction",
                            "variable": left["variable"],
                            "left": left,
                            "right": right,
                        }
                    )

    total_requirements = max(1, len(gsn_elements))
    contradiction_rate = len(contradictions) / total_requirements
    if contradiction_rate == 0:
        score = 5
    elif contradiction_rate <= 0.05:
        score = 4
    elif contradiction_rate <= 0.10:
        score = 3
    elif contradiction_rate <= 0.20:
        score = 2
    else:
        score = 1

    return {
        "score": score,
        "normalized_score": round(score / 5.0, 6),
        "contradiction_rate": round(contradiction_rate, 6),
        "contradictions": contradictions,
        "recommendations": [
            "Resolve contradictory numeric bounds" if contradictions else "No contradiction detected"
        ],
    }


def _terminology_consistency(gsn_elements: list[dict[str, Any]]) -> dict[str, Any]:
    """ISO29148-style terminology consistency checks."""

    synonym_groups = {
        "user": ["user", "client", "customer", "member"],
        "system": ["system", "application", "platform", "service"],
        "account": ["account", "checking-account", "saving-account"],
        "transaction": ["transaction", "operation", "action"],
        "data": ["data", "information", "record"],
        "access": ["access", "gain access", "obtain access"],
        "verify": ["verify", "validate", "check", "authenticate"],
    }

    terminology_map: dict[str, list[tuple[str, str]]] = {}
    for element in gsn_elements:
        description = str(element.get("description", "")).lower()
        element_id = str(element.get("id", ""))
        for concept, synonyms in synonym_groups.items():
            for term in synonyms:
                if term in description:
                    terminology_map.setdefault(concept, []).append((term, element_id))

    conflicts: list[dict[str, Any]] = []
    for concept, usage in terminology_map.items():
        terms = {term for term, _ in usage}
        if len(terms) > 1:
            conflicts.append(
                {
                    "concept": concept,
                    "terms_used": sorted(terms),
                    "requirements": sorted({element_id for _, element_id in usage}),
                }
            )

    total_concepts = len(terminology_map)
    consistent = total_concepts - len(conflicts)
    consistency_ratio = consistent / total_concepts if total_concepts else 1.0
    if consistency_ratio >= 0.95:
        score = 5
    elif consistency_ratio >= 0.85:
        score = 4
    elif consistency_ratio >= 0.70:
        score = 3
    elif consistency_ratio >= 0.50:
        score = 2
    else:
        score = 1

    return {
        "score": score,
        "consistency_ratio": round(consistency_ratio, 6),
        "conflicts": conflicts,
        "total_concepts": total_concepts,
        "consistent_concepts": consistent,
    }


def _compliance_coverage(gsn_elements: list[dict[str, Any]], requirement: str) -> dict[str, Any]:
    """Compute compliance coverage as satisfied clauses / applicable clauses."""

    clauses = _extract_requirement_fragments(requirement)
    if not clauses:
        clauses = [requirement.strip()] if requirement.strip() else []

    requirement_texts = [
        f"{item.get('name', '')}. {item.get('description', '')}".strip()
        for item in gsn_elements
    ]

    satisfied = 0
    clause_results: list[dict[str, Any]] = []
    for clause in clauses:
        best = 0.0
        for generated in requirement_texts:
            score = _text_overlap_score(clause, generated)
            if score > best:
                best = score
        matched = best >= 0.45
        if matched:
            satisfied += 1
        clause_results.append(
            {
                "clause": _summarize_text(clause, 120),
                "best_overlap": round(best, 6),
                "matched": matched,
            }
        )

    total = len(clauses)
    ratio = satisfied / total if total else 0.0
    return {
        "satisfied_applicable_clauses": satisfied,
        "total_applicable_clauses": total,
        "coverage_ratio": round(ratio, 6),
        "clauses": clause_results,
    }


def _fact_checking(gsn_elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Basic source-evidence checks for generated elements."""

    flagged: list[dict[str, Any]] = []
    for item in gsn_elements:
        properties = item.get("properties", {})
        if properties.get("source"):
            continue
        flagged.append(
            {
                "element_id": item.get("id"),
                "issue": "missing_source",
            }
        )
    return {
        "hallucination_reports": flagged,
        "flagged_elements": [entry["element_id"] for entry in flagged],
        "total_checked": len(gsn_elements),
    }


def _build_deterministic_validation(
    *,
    topology_status: dict[str, Any],
    logical: dict[str, Any],
    verification_executed: bool,
) -> dict[str, Any]:
    """Construct deterministic validation payload used by phase4 artifact."""

    violations: list[dict[str, Any]] = []
    for contradiction in logical["contradictions"]:
        violations.append(
            {
                "rule_id": "logical_consistency",
                "rule_name": contradiction.get("type", "logical_consistency"),
                "element_id": contradiction.get("left", {}).get("element_id", "N/A"),
                "element_name": "N/A",
                "severity": "medium",
                "message": "Detected contradiction between requirement constraints",
            }
        )

    if topology_status.get("cycle_count", 0) > 0:
        violations.append(
            {
                "rule_id": "topology_cycle",
                "rule_name": "cycle_detection",
                "element_id": "N/A",
                "element_name": "N/A",
                "severity": "high",
                "message": "Detected cycle in integrated topology",
            }
        )

    passed_rules = ["topology_validity", "traceability"]
    if not violations:
        passed_rules.append("logical_consistency")

    if not verification_executed:
        return {
            "is_valid": False,
            "violations": [],
            "passed_rules": [],
            "status": "not_executed",
        }

    return {
        "is_valid": bool(topology_status.get("is_valid", False) and not violations),
        "violations": violations,
        "passed_rules": passed_rules,
    }


# ---------------------------------------------------------------------------
# Quality axis helpers (negotiation support)
# ---------------------------------------------------------------------------


def _quality_axis_for_agent(agent_name: str) -> str:
    """Normalize QUARE and MARE runtime role names onto the paper quality axes."""

    normalized = str(agent_name).strip()
    aliases = {
        "SafetyAgent": "Safety",
        "EfficiencyAgent": "Efficiency",
        "GreenAgent": "Sustainability",
        "TrustworthinessAgent": "Trustworthiness",
        "ResponsibilityAgent": "Responsibility",
        "Stakeholders": "Responsibility",
        "Collector": "Efficiency",
        "Modeler": "Trustworthiness",
        "Checker": "Safety",
        "Documenter": "Sustainability",
    }
    return aliases.get(normalized, normalized)


# ---------------------------------------------------------------------------
# External rule extraction
# ---------------------------------------------------------------------------


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

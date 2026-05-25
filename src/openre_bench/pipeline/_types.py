"""Shared types, constants, and quality mappings for the pipeline package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openre_bench.llm import LLMContract
from openre_bench.schemas import (
    SYSTEM_MARE,
)


@dataclass
class PipelineConfig:
    """Run configuration used to generate parity artifacts."""

    case_input: Path
    artifacts_dir: Path
    run_record_path: Path
    run_id: str
    setting: str
    seed: int
    model: str
    temperature: float
    round_cap: int
    max_tokens: int
    system: str = SYSTEM_MARE
    rag_enabled: bool = True
    rag_backend: str = "local_tfidf"
    rag_corpus_dir: Path | None = None
    llm_client: LLMContract | None = None


@dataclass
class Phase2ExecutionMeta:
    """Execution metadata from phase 2 used for strict run-record tainting."""

    llm_enabled: bool = False
    llm_turns: int = 0
    llm_fallback_turns: int = 0
    llm_retry_count: int = 0
    llm_parse_recoveries: int = 0
    llm_seed_applied_turns: int = 0
    llm_source: str = "disabled"


@dataclass
class RuntimeExecutionMeta:
    """Execution metadata for one system-specific runtime workflow."""

    llm_enabled: bool = False
    llm_turns: int = 0
    llm_fallback_turns: int = 0
    llm_retry_count: int = 0
    llm_parse_recoveries: int = 0
    llm_seed_applied_turns: int = 0
    llm_source: str = "disabled"
    execution_mode: str = "deterministic_emulation"


@dataclass
class ActionRunResult:
    """Result from one LLM-first action with deterministic fallback."""

    items: list[str]
    summary: str
    execution_mode: str
    llm_generated: bool
    fallback_reason: str


# Module-level caches
_RAG_CHUNK_CACHE: dict[str, list[dict[str, Any]]] = {}
_CORPUS_HASH_CACHE: dict[str, str] = {}

PHASE2_LLM_RETRY_LIMIT = 1
MARE_RUNTIME_SEMANTICS_MODE = "mare_paper_workflow_v1"
MARE_RUNTIME_TRACE_VERSION = "1"
IREDEV_RUNTIME_SEMANTICS_MODE = "iredev_knowledge_driven_v1"
IREDEV_RUNTIME_TRACE_VERSION = "1"

# NOTE: The MARE paper (ASE 2024) does not define quality attributes
# per role. These OpenRE-Bench benchmark-axis labels keep target adapters
# comparable without treating any one target as the metric owner.
MARE_ROLE_QUALITY_ATTRIBUTES: dict[str, str] = {
    "Stakeholders": "Responsibility",
    "Collector": "Efficiency",
    "Modeler": "Trustworthiness",
    "Checker": "Safety",
    "Documenter": "Sustainability",
}

# NOTE: The iReDev paper (TOSEM 2025) does not define quality
# attributes per role. Same rationale as MARE_ROLE_QUALITY_ATTRIBUTES above.
IREDEV_ROLE_QUALITY_ATTRIBUTES: dict[str, str] = {
    "Interviewer": "Responsibility",
    "EndUser": "Efficiency",
    "Deployer": "Safety",
    "Analyst": "Trustworthiness",
    "Archivist": "Sustainability",
    "Reviewer": "Integrated",
}

QUALITY_LENS_CUES: dict[str, tuple[str, ...]] = {
    "Safety": (
        "hazard prevention",
        "fault tolerance",
        "risk mitigation",
    ),
    "Efficiency": (
        "latency optimization",
        "throughput stability",
        "resource utilization",
    ),
    "Sustainability": (
        "energy footprint reduction",
        "resource lifecycle control",
        "environmental impact awareness",
    ),
    "Trustworthiness": (
        "security assurance",
        "auditability",
        "integrity guarantees",
    ),
    "Responsibility": (
        "regulatory accountability",
        "stakeholder transparency",
        "ethical compliance",
    ),
    "Integrated": (
        "cross-quality balance",
        "holistic requirement coherence",
        "end-to-end requirement traceability",
    ),
}

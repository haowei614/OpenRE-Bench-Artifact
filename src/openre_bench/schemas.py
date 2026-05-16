"""Canonical schema models for comparison parity artifacts."""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import RootModel

PHASE1_FILENAME = "phase1_initial_models.json"
PHASE2_FILENAME = "phase2_negotiation_trace.json"
PHASE3_FILENAME = "phase3_integrated_kaos_model.json"
PHASE4_FILENAME = "phase4_verification_report.json"
PHASE0_FILENAME = "phase0_external_spec_rules.json"
PHASE25_FILENAME = "phase2_conflict_map.json"
PHASE5_FILENAME = "phase5_software_materials.json"

PHASE_FILENAMES = (
    PHASE1_FILENAME,
    PHASE2_FILENAME,
    PHASE3_FILENAME,
    PHASE4_FILENAME,
)

QUARE_OPTIONAL_PHASE_FILENAMES = (
    PHASE0_FILENAME,
    PHASE25_FILENAME,
    PHASE5_FILENAME,
)

SYSTEM_MARE = "mare"
SYSTEM_QUARE = "quare"
SYSTEM_IREDEV = "iredev"
SUPPORTED_SYSTEMS = (SYSTEM_MARE, SYSTEM_IREDEV, SYSTEM_QUARE)

SETTING_SINGLE_AGENT = "single_agent"
SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION = "multi_agent_without_negotiation"
SETTING_MULTI_AGENT_WITH_NEGOTIATION = "multi_agent_with_negotiation"
SETTING_NEGOTIATION_INTEGRATION_VERIFICATION = "negotiation_integration_verification"

DEFAULT_MATRIX_SETTINGS = (
    SETTING_SINGLE_AGENT,
    SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION,
    SETTING_MULTI_AGENT_WITH_NEGOTIATION,
    SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
)

DEFAULT_AGENT_QUALITY_ATTRIBUTES = {
    "SafetyAgent": "Safety",
    "EfficiencyAgent": "Efficiency",
    "GreenAgent": "Sustainability",
    "TrustworthinessAgent": "Trustworthiness",
    "ResponsibilityAgent": "Responsibility",
}

MARE_AGENT_ROLES = (
    "Stakeholders",
    "Collector",
    "Modeler",
    "Checker",
    "Documenter",
)

MARE_ACTIONS = (
    "SpeakUserStories",
    "ProposeQuestion",
    "AnswerQuestion",
    "WriteReqDraft",
    "ExtractEntity",
    "ExtractRelation",
    "CheckRequirement",
    "WriteSRS",
    "WriteCheckReport",
)

MARE_ROLE_ACTIONS: dict[str, tuple[str, ...]] = {
    "Stakeholders": (
        "SpeakUserStories",
        "AnswerQuestion",
    ),
    "Collector": (
        "ProposeQuestion",
        "WriteReqDraft",
    ),
    "Modeler": (
        "ExtractEntity",
        "ExtractRelation",
    ),
    "Checker": ("CheckRequirement",),
    "Documenter": ("WriteSRS", "WriteCheckReport"),
}

IREDEV_AGENT_ROLES = (
    "Interviewer",
    "EndUser",
    "Deployer",
    "Analyst",
    "Archivist",
    "Reviewer",
)

IREDEV_ACTIONS = (
    "DialogueWithEndUser",
    "WriteInterviewRecords",
    "WriteUserRequirementsList",
    "DialogueWithDeployer",
    "WriteOperatingEnvironmentList",
    "RespondEndUser",
    "RaiseQuestionEndUser",
    "ConfirmOrRefineEndUser",
    "RespondDeployer",
    "RaiseQuestionDeployer",
    "ConfirmOrRefineDeployer",
    "WriteSystemRequirementsList",
    "SelectRequirementModel",
    "BuildRequirementModel",
    "WriteSRS",
    "Evaluate",
    "ConfirmClosure",
)

IREDEV_ROLE_ACTIONS: dict[str, tuple[str, ...]] = {
    "Interviewer": (
        "DialogueWithEndUser",
        "WriteInterviewRecords",
        "WriteUserRequirementsList",
        "DialogueWithDeployer",
        "WriteOperatingEnvironmentList",
    ),
    "EndUser": (
        "RespondEndUser",
        "RaiseQuestionEndUser",
        "ConfirmOrRefineEndUser",
    ),
    "Deployer": (
        "RespondDeployer",
        "RaiseQuestionDeployer",
        "ConfirmOrRefineDeployer",
    ),
    "Analyst": (
        "WriteSystemRequirementsList",
        "SelectRequirementModel",
        "BuildRequirementModel",
    ),
    "Archivist": ("WriteSRS",),
    "Reviewer": ("Evaluate", "ConfirmClosure"),
}

NON_COMPARABLE_REASON_BY_SETTING: dict[str, list[str]] = {
    SETTING_SINGLE_AGENT: ["single_agent_baseline_partial_phase_equivalence"],
    SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION: [
        "negotiation_and_verification_not_executed_for_setting"
    ],
    SETTING_MULTI_AGENT_WITH_NEGOTIATION: ["verification_not_executed_for_setting"],
    SETTING_NEGOTIATION_INTEGRATION_VERIFICATION: [],
}


class CaseInput(BaseModel):
    """Canonical case input compatible with OpenRE-Bench case studies."""

    model_config = ConfigDict(extra="ignore")

    case_name: str
    case_description: str
    requirement: str


class RunSystemIdentity(BaseModel):
    """Machine-checkable runtime identity used for reproducibility audits."""

    model_config = ConfigDict(extra="forbid")

    system_name: str
    implementation: str
    implementation_version: str
    python_version: str
    platform: str
    machine: str


class RunProvenance(BaseModel):
    """Controlled knobs and content hashes required for strict comparability."""

    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float
    seed: int
    prompt_hash: str
    corpus_hash: str
    corpus_path: str


class RunExecutionFlags(BaseModel):
    """Execution taint flags that can invalidate strict comparison runs."""

    model_config = ConfigDict(extra="forbid")

    rag_fallback_used: bool = False
    llm_fallback_used: bool = False
    fallback_tainted: bool = False
    retry_used: bool = False
    retry_count: int = 0


class RunComparability(BaseModel):
    """Explicit comparability state and reasons for partial equivalence."""

    model_config = ConfigDict(extra="forbid")

    is_comparable: bool
    non_comparable_reasons: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """Canonical run metadata used for protocol-compliant reproducibility."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    case_id: str
    system: str
    setting: str
    seed: int
    model: str
    temperature: float
    round_cap: int
    system_identity: RunSystemIdentity
    provenance: RunProvenance
    execution_flags: RunExecutionFlags
    comparability: RunComparability

    max_tokens: int = 4000
    rag_enabled: bool = False
    rag_backend: str = "none"
    rag_fallback_used: bool = False

    start_timestamp: str
    end_timestamp: str
    runtime_seconds: float = 0.0

    artifacts_dir: str
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    artifact_blinded: bool = False
    blinding_scheme_version: str = ""
    blind_eval_run_id: str = ""
    judge_pipeline_hash: str = ""
    trace_audit_path: str = ""
    notes: dict[str, Any] = Field(default_factory=dict)


class KAOSElement(BaseModel):
    """KAOS element contract aligned to OpenRE-Bench structure."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str
    element_type: str
    quality_attribute: str

    priority: int = 1
    stakeholder: str | None = None
    measurable_criteria: str | None = None
    hierarchy_level: int = 1
    parent_goal_id: str | None = None

    source: str | None = None
    source_chunk_id: str | None = None
    source_document: str | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    conflict_resolved_by: str | None = None
    validation_status: str | None = None
    validation_timestamp: str | None = None
    citation_required: bool = True


class NegotiationStep(BaseModel):
    """One negotiation step in phase 2 trace."""

    model_config = ConfigDict(extra="allow")

    step_id: int
    timestamp: str
    focus_agent: str
    reviewer_agent: str
    round_number: int
    message_type: str
    kaos_elements: list[dict[str, Any]] = Field(default_factory=list)
    analysis_text: str
    analysis: str | None = None
    feedback: str | None = None
    conflict_detected: bool | None = None
    negotiation_mode: str | None = None
    resolution_state: str | None = None
    requires_refinement: bool | None = None


class NegotiationHistory(BaseModel):
    """Negotiation history per focus/reviewer pair."""

    model_config = ConfigDict(extra="allow")

    negotiation_id: str
    focus_agent: str
    reviewer_agents: list[str]
    start_timestamp: str
    end_timestamp: str | None = None
    steps: list[NegotiationStep]
    final_consensus: bool
    total_rounds: int


class Phase2Artifact(BaseModel):
    """Canonical phase 2 artifact contract."""

    model_config = ConfigDict(extra="allow")

    total_negotiations: int
    negotiations: dict[str, NegotiationHistory]
    summary_stats: dict[str, Any]


class Phase1Artifact(RootModel[dict[str, list[KAOSElement]]]):
    """Canonical phase 1 artifact contract."""


class GSNElement(BaseModel):
    """GSN element contract for phase 3 artifact."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str
    gsn_type: str
    quality_attribute: str
    priority: int = 1
    stakeholder: str | None = None
    measurable_criteria: str | None = None
    hierarchy_level: int = 1
    parent_goal_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GSNConnection(BaseModel):
    """GSN connection contract for phase 3 artifact."""

    model_config = ConfigDict(extra="allow")

    id: str
    source_id: str
    target_id: str
    connection_type: str
    description: str
    strength: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class Phase3Artifact(BaseModel):
    """Canonical phase 3 artifact contract."""

    model_config = ConfigDict(extra="allow")

    gsn_elements: list[GSNElement]
    gsn_connections: list[GSNConnection]
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    hierarchy_structure: dict[str, Any] = Field(default_factory=dict)
    topology_status: dict[str, Any] = Field(default_factory=dict)

    model_id: str
    created_timestamp: str
    total_elements: int
    total_relations: int


class Phase4Artifact(BaseModel):
    """Canonical phase 4 artifact contract."""

    model_config = ConfigDict(extra="allow")

    verification_results: dict[str, Any]
    fact_checking: dict[str, Any]
    deterministic_validation: dict[str, Any]
    topology_status: dict[str, Any]
    consistency_verification: dict[str, Any] = Field(default_factory=dict)
    universal_verification: dict[str, Any] = Field(default_factory=dict)
    correction_summary: dict[str, Any] | None = None


def non_comparable_reasons_for_setting(setting: str) -> list[str]:
    """Return protocol reasons for settings with partial phase equivalence."""

    return list(NON_COMPARABLE_REASON_BY_SETTING.get(setting, ["unknown_setting_semantics"]))


def load_json_file(path: Path) -> Any:
    """Load JSON from disk."""

    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def write_json_file(path: Path, payload: Any) -> None:
    """Write deterministic, pretty JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def utc_timestamp() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

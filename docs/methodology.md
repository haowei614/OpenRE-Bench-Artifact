# Benchmark Methodology

OpenRE-Bench evaluates requirements-engineering target adapters under a shared benchmark harness. The current adapters are MARE, iReDev, and QUARE. They run from the same case input shape, emit the same required phase artifacts, and record provenance and comparability metadata in a common run record.

Implementation details for commands and generated files are documented separately in [reference/cli.md](reference/cli.md) and [reference/artifacts.md](reference/artifacts.md).

## Scope

OpenRE-Bench treats each target adapter as an executable representation of an external requirements-engineering method. The benchmark harness provides the shared runtime boundary:

- `src/openre_bench/` contains schemas, validation, pipeline routing, matrix execution, metrics, and reporting.
- `openre_bench_mare/`, `openre_bench_iredev/`, and `openre_bench_quare/` contain target-specific runtime behavior.
- The CLI selects a target with `--system mare`, `--system iredev`, or `--system quare`.
- Matrix runs can evaluate multiple targets through `--systems`.

The benchmark records target identity and runtime semantics rather than treating different internal workflows as identical.

## Target Adapters

| Target | Runtime id | Roles or agents | Runtime focus |
|---|---|---|---|
| MARE | `mare` | Stakeholders, Collector, Modeler, Checker, Documenter | Task-specialized requirements elicitation, modeling, checking, and documentation. |
| iReDev | `iredev` | Interviewer, EndUser, Deployer, Analyst, Archivist, Reviewer | Knowledge-driven requirements development with stakeholder and environment roles. |
| QUARE | `quare` | SafetyAgent, EfficiencyAgent, GreenAgent, TrustworthinessAgent, ResponsibilityAgent | Quality-axis KAOS generation and negotiation coordinated by the QUARE orchestration role. |

Target adapters may use different internal roles, action counts, and negotiation policies. The comparison boundary is the shared input, artifact, run-record, and metric contract.

## Shared Phases

| Phase | Purpose | Notes |
|---|---|---|
| Phase 1 | Initial model generation | The selected target produces per-role or per-agent KAOS elements. |
| Phase 2 | Negotiation or debate trace | The target records its review, feedback, or negotiation behavior under the shared trace schema. |
| Phase 3 | Integration | The harness writes the integrated KAOS/GSN-style model used for most aggregate metrics. |
| Phase 4 | Verification | The run records structural checks, compliance-oriented fields, and comparability state. |

QUARE also writes target-specific intermediate artifacts for external specification parsing, conflict maps, and software materials. They supplement the shared phases and are validated as part of the current QUARE behavior contract. Those files are documented in [reference/artifacts.md](reference/artifacts.md).

## KAOS Conventions

OpenRE-Bench uses KAOS-style elements and relations as a common representation for target outputs. The runtime schema is defined in `src/openre_bench/schemas.py`.

| Element type | Meaning |
|---|---|
| Goal | Desired objective. |
| Softgoal | Non-functional or quality-related objective. |
| Task | Operational action or behavior. |
| Resource | Required system, data, or environmental resource. |
| Obstacle | Risk, failure mode, or impediment. |
| Agent | Human, organization, or system component responsible for behavior. |

| Relation type | Meaning |
|---|---|
| AND-refinement | All child goals are needed for the parent goal. |
| OR-refinement | One or more alternatives can satisfy the parent goal. |
| Contribution | A positive or negative influence on a softgoal. |
| Dependency | One element depends on another element or resource. |
| Conflict | Two elements express incompatible objectives or constraints. |
| Operationalization | A task realizes a goal. |

The model commonly uses three hierarchy levels: strategic goals, tactical requirements, and operational tasks or checks.

## Comparability Principles

OpenRE-Bench uses explicit metadata to make comparisons auditable:

- The same case input fields are used for each target: `case_name`, `case_description`, and `requirement`.
- Each run records `system`, `setting`, `seed`, model parameters, RAG state, artifact paths, and validation status.
- Fallback and retry behavior is recorded in execution flags.
- Runs can be marked non-comparable when a setting intentionally omits negotiation or verification.
- Aggregate outputs keep target identity as a first-class field.

These controls do not guarantee identical LLM outputs across runs. They document the conditions under which each output was produced.

## Evaluation Concepts

The matrix harness reports coverage, diversity, negotiation, structural, and compliance-oriented metrics. Common fields include requirement counts, quality-axis spread, semantic preservation, conflict resolution rate, topology validity, and ISO 29148-style ratings.

Manual Precision, Recall, and F1 labeling is covered in [guides/precision-f1.md](guides/precision-f1.md). INCOSE-style statement formatting is covered in [guides/incose-format.md](guides/incose-format.md).

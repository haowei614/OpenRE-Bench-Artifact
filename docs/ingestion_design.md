# Case Ingestion Design for OpenRE-Bench

## Document Status
**Design confirmed; module implemented and verified offline.** The
`ingestion/` module (steps 1–7 + smoke test) is complete and was validated
end-to-end on a synthetic snapshot. Remaining: run the fetch steps with a real
`GITHUB_TOKEN` to produce the 3–5 mined cases, then the smoke test.

Last Updated: 2026-09-29

---

## 1. Existing Case Structure Analysis

### 1.1 Case Data Format

All existing cases follow a unified JSON schema defined by `CaseInput` in `src/openre_bench/schemas.py`:

```python
class CaseInput(BaseModel):
    case_name: str           # Stable case identifier (e.g., "ATM", "AD")
    case_description: str    # Short description of the domain
    requirement: str         # Source requirement text (benchmark input)
```

**Fields**:
- `case_name`: Used in run IDs, file paths, and reports. Must be filesystem-safe and concise.
- `case_description`: Human-readable domain context (1-2 sentences).
- `requirement`: Natural language requirement text. Can be single or multi-paragraph, ranging from ~50 words (Bookkeeping) to ~400 words (AD - Autonomous Driving).

### 1.2 Existing Cases

The repository contains 5 manually constructed cases located in `data/case_studies/`:

| Case Name | File | Domain | Requirement Length | Complexity |
|-----------|------|--------|-------------------|------------|
| ATM | `ATM_input.json` | Banking/ATM system | ~170 words | Medium - Multiple account types, transaction tracking, balance constraints |
| AD | `AD_input.json` | Autonomous Driving | ~400 words | High - Multi-layer safety architecture, ODD, MRM, path-speed decoupling |
| Library | `Library_input.json` | Library management | ~65 words | Low - CRUD operations, search, borrowing limits |
| RollCall | `RollCall_input.json` | Attendance tracking | ~55 words | Low - Attendance marking, reporting, history |
| Bookkeeping | `Bookkeeping_input.json` | Accounting system | ~55 words | Low - Transaction recording, financial reports, double-entry |

**Key Observations**:
1. **Length range**: 50-400 words (ATM/AD are more detailed, others are concise)
2. **Domain diversity**: Banking, automotive, education, library, accounting
3. **All are human-constructed**: Not derived from real GitHub issues
4. **Requirement style**: Mix of functional requirements, constraints, and domain context
5. **No explicit feature request clustering**: Each case is self-contained

### 1.3 Adapter Interface

Each framework adapter (MARE, iReDev, QUARE) exposes three entry points in its `__init__.py`:

```python
def build_phase1(
    *,
    case: Any,                    # CaseInput instance
    seed: int,
    setting: str,
    rag_context: dict[str, Any],
    llm_client: LLMContract | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], RuntimeExecutionMeta]:
    """Generate initial KAOS goal models by agent/quality dimension."""

def build_negotiation_history(...) -> NegotiationBuildResult:
    """Generate negotiation trace between agent pairs."""

def build_optional_artifacts(...) -> dict[str, dict[str, Any]]:
    """Generate system-specific optional artifacts (e.g., QUARE phase 0/2.5/5)."""
```

Adapters are registered in `src/openre_bench/pipeline/_core.py` via:
```python
_SYSTEM_PIPELINE_REGISTRY: dict[str, dict[str, Any]] = {
    SYSTEM_MARE: {"module": "openre_bench_mare", ...},
    SYSTEM_IREDEV: {"module": "openre_bench_iredev", ...},
    SYSTEM_QUARE: {"module": "openre_bench_quare", ...},
}
```

The `case` parameter is the **only** input that varies between runs (other than seeds/settings). Adapters expect:
- `case.case_name`: string identifier
- `case.case_description`: domain context
- `case.requirement`: requirement text to analyze

### 1.4 Case Loading and Execution

**Loading Path** (`src/openre_bench/comparison_harness.py:455-456`):
```python
case_payload = load_json_file(case_path)
case_input = CaseInput.model_validate(case_payload)
```

**Validation** (`src/openre_bench/comparison_validator.py`):
- `validate_case_input()`: Checks schema compliance
- Pydantic enforces required fields and types

**Execution Path**:
1. CLI entry: `openre_bench --run-case --case-input <path> --system <name>`
2. `run_case_pipeline(config: PipelineConfig)` in `src/openre_bench/pipeline/_core.py`
3. Adapter's `build_phase1()` consumes `CaseInput`
4. Outputs:
   - `phase1_initial_models.json`: KAOS elements by agent
   - `phase2_negotiation_trace.json`: Negotiation history
   - `phase3_integrated_kaos_model.json`: GSN elements/connections
   - `phase4_verification_report.json`: Validation results
   - `run_record.json`: Provenance and execution metadata

**No preprocessing required**: The `requirement` field is passed as-is to adapters. No tokenization, chunking, or formatting is applied by the harness.

### 1.5 Implications for Ingestion

To be **directly consumable** by the existing harness, mined cases must:
1. Conform to the 3-field `CaseInput` schema (no extra fields)
2. Have a unique, filesystem-safe `case_name` (e.g., `django_123`)
3. Provide a concise `case_description` (~1-2 sentences)
4. Supply a `requirement` field with:
   - Natural language text (not just issue titles)
   - Sufficient context for RE analysis (not pure bug reports)
   - Comparable scale to existing cases (50-400 words recommended)
   - **Aggregated from one or more GitHub issues** (key difference from existing cases)

**Critical gap**: Existing cases are self-contained. Real issues often require aggregation (e.g., multiple feature requests for one subsystem) to match the input scale.

---

## 2. Ingestion Design (Step 2 — Awaiting User Confirmation)

### 2.1 Project Selection Criteria

**Goal**: Mine 3-5 cases from real GitHub repositories that are:
- Relevant to MSR (mining software repositories)
- Comparable to existing cases in complexity and scale
- Diverse in domain and language

**Criteria**:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Stars** | ≥ 5,000 | Ensures project maturity and community engagement (MSR relevance) |
| **Primary Language** | Python, Java, JavaScript, or TypeScript | Aligns with common RE/SE research targets; ensures language diversity |
| **Open Issues** | ≥ 100 | Sufficient pool for filtering feature requests |
| **Closed Issues** | ≥ 500 | Evidence of active maintenance and issue resolution |
| **Activity** | ≥ 1 commit in last 6 months | Confirms ongoing development |
| **License** | OSI-approved | Ensures ethical reuse of issue text |
| **Domain Diversity** | No more than 1 project per domain category | Prevents over-representation (e.g., avoid 3 web frameworks) |

**Domain Categories for Diversity**:
- Web Framework/Backend
- Data Processing/ML
- CLI Tool/DevTool
- Database/Storage
- System/Infrastructure
- Frontend/UI

**Exclusion**:
- Toy/tutorial projects (< 100 stars)
- Archived or inactive repos (last commit > 2 years ago)
- Non-English issue discussions (complicates text extraction)

### 2.2 Issue Filtering Criteria

**Goal**: Extract feature requests (not bugs, not discussions) that contain sufficient requirement detail.

**Label-based filtering** (at least one must match, case-insensitive):
- `enhancement`
- `feature request`
- `feature`
- `new feature`
- `proposal`

**State filtering**:
- **Prefer closed issues**: Indicates community/maintainer validation
- **Accept open issues** if they have ≥ 5 reactions (👍, ❤️) indicating user demand

**Text-based filtering**:
1. **Minimum body length**: ≥ 200 characters (excludes "Add feature X" one-liners)
2. **Exclude bug indicators**: Issues whose title or body contains:
   - `bug`, `error`, `crash`, `fail`, `broken`, `not working`, `wrong`, `incorrect`
   - **Exception**: Feature requests for error handling/validation are allowed
3. **Exclude pure discussion**: Issues with ≥ 20 comments but no code references (e.g., design debates)
4. **Require requirement signals**: Body must contain ≥ 1 of:
   - "should", "must", "need", "require", "allow", "support", "enable"
5. **Deduplication**: Remove issues with ≥ 80% cosine similarity in TF-IDF embedding (e.g., duplicates or near-duplicates)

**Data retention**:
- Snapshot raw API responses with `fetched_at` timestamp (ISO 8601)
- Log all filtering decisions (which criterion rejected each issue)

### 2.3 Aggregation Rules

**Problem**: Single GitHub issues are often too narrow (e.g., "Add --verbose flag") to match the scale of existing cases (50-400 words of structured requirements).

**Solution**: Aggregate multiple related feature requests into one case.

**Strategy A: Time-based aggregation (recommended for initial implementation)**:
- Group issues by **3-month sliding window** (e.g., Q1 2023, Q2 2023)
- Select the window with the **most feature requests** after filtering
- Take **top 3-5 issues by reaction count** within that window
- Concatenate into a single `requirement` field with clear delineation:

```
Requirement 1 (Issue #123): <body text>

Requirement 2 (Issue #456): <body text>

Requirement 3 (Issue #789): <body text>
```

**Strategy B: Topic clustering (future enhancement)**:
- Use LDA or K-means on TF-IDF to cluster issues by theme
- Select largest cluster per project
- Concatenate top 3-5 issues within cluster

**Strategy C: Milestone-based (if available)**:
- Group issues by GitHub milestone (e.g., "v2.0 features")
- Use milestone description as `case_description`
- Aggregate all feature requests in milestone

**Constraints**:
- **Final aggregated length**: 50-500 words (to match existing case range)
- **Maximum issues per case**: 5 (to prevent overwhelming requirement text)
- **Minimum issues per case**: 1 (allow single substantial feature requests)
- **Metadata tracking**: Store original issue URLs in a sidecar JSON (not in the `requirement` field itself)

**Output**:
- One case per project (for 3-5 projects = 3-5 cases)
- Original issue URLs stored in `data/mined_cases_metadata.json`:
  ```json
  {
    "django_admin": {
      "source_issues": [
        {"url": "https://github.com/django/django/issues/123", "title": "...", "created_at": "..."},
        {"url": "https://github.com/django/django/issues/456", "title": "...", "created_at": "..."}
      ],
      "aggregation_method": "time_window_q1_2023",
      "mined_at": "2026-09-29T12:00:00Z"
    }
  }
  ```

### 2.4 Candidate Projects

Ten candidates spanning 3 languages and 6 domain categories. Stats marked
**[verified 2026-09-29]** were queried live from the GitHub REST API;
others are approximate and will be re-verified at fetch time (the pipeline
re-checks every threshold against the live API before selecting a project,
so no case is admitted on unverified numbers).

None of these domains overlaps with the existing cases (banking/ATM,
autonomous driving, library, attendance, accounting), which preserves the
novelty of the mined set.

| # | Project | Lang | Stars | License | Domain | Rationale |
|---|---------|------|-------|---------|--------|-----------|
| 1 | `django/django` | Python | 91.2k **[verified]** | BSD-3 | Web Framework | Mature, disciplined issue triage with a dedicated Trac-style workflow; rich `enhancement`-labeled feature requests. |
| 2 | `huggingface/transformers` | Python | 166.8k **[verified]** | Apache-2.0 | ML / NLP | Very high feature-request volume (2.4k open); strong "Feature request" issue template producing structured bodies. |
| 3 | `pandas-dev/pandas` | Python | 49.9k **[verified]** | BSD-3 | Data Processing | 2.5k open issues, extensive `Enhancement` label usage; concise, well-scoped requests. |
| 4 | `microsoft/vscode` | TypeScript | ~163k | MIT | Editor / DevTool | `feature-request` label is heavily curated; issues follow a clear "problem → proposed feature" format. |
| 5 | `elastic/elasticsearch` | Java | ~70k | Elastic-2.0/SSPL | Database / Search | Adds Java + a data-store domain; `enhancement` label well maintained. |
| 6 | `spring-projects/spring-boot` | Java | ~75k | Apache-2.0 | Web Framework (Java) | Alternative Java option; `type: enhancement` label, detailed proposals. |
| 7 | `facebook/react` | JavaScript | ~228k | MIT | Frontend | Large JS project; RFC-style feature discussions. |
| 8 | `angular/angular` | TypeScript | ~95k | MIT | Frontend Framework | `feature` label; structured feature requests. |
| 9 | `scikit-learn/scikit-learn` | Python | ~60k | BSD-3 | ML Library | `New Feature` label; academically framed requests (good RE fit). |
| 10 | `ansible/ansible` | Python | ~63k | GPL-3.0 | Infrastructure / DevOps | `feature` label; system/automation domain adds diversity. |

**Recommended set (proceeding with these 4-5, per user delegation):**

To maximize language and domain diversity while keeping the pool small and
verifiable, the pipeline is configured with the following projects. Target is
3-5 final cases; 5 projects are configured so that if one yields too few
qualifying feature requests, others provide fallback.

1. **`django/django`** — Python, Web Framework
2. **`huggingface/transformers`** — Python, ML/NLP
3. **`microsoft/vscode`** — TypeScript, Editor/DevTool
4. **`elastic/elasticsearch`** — Java, Database/Search
5. **`pandas-dev/pandas`** — Python, Data Processing (fallback / 5th case)

This spans **3 languages** (Python, TypeScript, Java) and **5 domains**
(web, ML, editor tooling, search/database, data processing). The projects
are configurable in `ingestion/config.yaml`; swapping a project requires no
code change.

### 2.5 Authentication Requirement (Important)

The mining pipeline **requires** a GitHub personal access token exported as
`GITHUB_TOKEN`:

- **Unauthenticated**: 60 requests/hour (verified exhausted after ~4 repo
  lookups on a shared IP during design). Insufficient for mining.
- **Authenticated**: 5,000 requests/hour. Sufficient for the full pipeline.

The token is read **only** from the environment variable (or a gitignored
`ingestion/.env.local`) — never written to code, config, snapshots, or
commits. A `scrub_secret()` helper redacts the token if it ever appears in
text bound for logs. `ingestion/.env.local` is gitignored. A read-only
`public_repo` scope is sufficient.

**Snapshots are committed on purpose.** The raw API snapshots under
`ingestion/snapshots/` record `fetched_at` and contain only public issue data
(no token). Committing them freezes the exact mining input so reviewers can
re-run the filter/aggregate/convert steps offline (`run_all.py --skip-fetch`)
and reproduce the cases without any network access or token.

---

## 3. Implementation Plan (Step 3 — Awaiting Confirmation)

### 3.1 Directory Structure

```
ingestion/
├── README.md                    # Usage instructions
├── config.yaml                  # Configurable parameters (thresholds, seeds, API)
├── fetch_projects.py            # Step 1: Discover candidate projects
├── fetch_issues.py              # Step 2: Download raw issue data
├── filter_issues.py             # Step 3: Apply filtering criteria
├── aggregate_cases.py           # Step 4: Cluster/aggregate into cases
├── convert_to_schema.py         # Step 5: Transform to CaseInput schema
├── validate_cases.py            # Step 6: Run schema validation
├── requirements.txt             # Python dependencies (requests, PyYAML, scikit-learn)
├── snapshots/                   # Raw API responses (timestamped JSON)
│   ├── projects_<timestamp>.json
│   └── issues_<project>_<timestamp>.json
├── filtered/                    # Filtered issues (JSON)
│   └── <project>_filtered.json
├── aggregated/                  # Aggregated case groups (JSON)
│   └── <project>_aggregated.json
└── output/                      # Final CaseInput JSON files
    ├── <case_name>_input.json
    └── metadata.json            # Traceability to source issues
```

### 3.2 Configuration File

`ingestion/config.yaml`:
```yaml
github:
  api_base: "https://api.github.com"
  token_env: "GITHUB_TOKEN"  # Read from environment variable
  rate_limit_delay: 1.0      # Seconds between requests
  retry_limit: 3

project_selection:
  min_stars: 5000
  min_open_issues: 100
  min_closed_issues: 500
  languages: ["Python", "Java", "JavaScript", "TypeScript"]
  max_projects: 10

issue_filtering:
  labels: ["enhancement", "feature request", "feature", "new feature", "proposal"]
  min_body_length: 200
  min_reactions: 5
  bug_exclusion_keywords: ["bug", "error", "crash", "fail", "broken", "not working"]
  requirement_keywords: ["should", "must", "need", "require", "allow", "support", "enable"]
  deduplication_threshold: 0.80  # Cosine similarity

aggregation:
  method: "time_window"          # Options: time_window, topic_cluster, milestone
  time_window_months: 3
  max_issues_per_case: 5
  min_issues_per_case: 1
  target_word_count_min: 50
  target_word_count_max: 500

output:
  cases_dir: "../data/case_studies"
  metadata_file: "../data/mined_cases_metadata.json"
  stats_csv: "stats/mining_statistics.csv"
  stats_md: "stats/mining_statistics.md"

reproducibility:
  random_seed: 42                # For clustering and sampling
```

### 3.3 Workflow Steps

**Step 1: Fetch Projects**
```bash
python ingestion/fetch_projects.py
# Outputs: ingestion/snapshots/projects_<timestamp>.json
```
- Uses GitHub Search API: `/search/repositories`
- Filters by stars, language, activity
- Saves full metadata (stars, issues, commits, license)

**Step 2: Fetch Issues**
```bash
python ingestion/fetch_issues.py --projects snapshots/projects_*.json
# Outputs: ingestion/snapshots/issues_<project>_<timestamp>.json
```
- For each project, fetch all issues with target labels
- Uses `/repos/{owner}/{repo}/issues` with pagination
- Caches responses to avoid re-fetching

**Step 3: Filter Issues**
```bash
python ingestion/filter_issues.py
# Outputs: ingestion/filtered/<project>_filtered.json
```
- Applies text-based and metadata-based filters
- Logs statistics: total → after labels → after length → after deduplication
- Saves rejected issues with reasons (for paper appendix)

**Step 4: Aggregate**
```bash
python ingestion/aggregate_cases.py
# Outputs: ingestion/aggregated/<project>_aggregated.json
```
- Groups issues by time window / topic / milestone
- Selects top group per project
- Concatenates issue bodies

**Step 5: Convert to Schema**
```bash
python ingestion/convert_to_schema.py
# Outputs: data/case_studies/<case_name>_input.json
#          data/mined_cases_metadata.json
```
- Generates `case_name` (e.g., `django_admin`)
- Extracts `case_description` from project description + dominant theme
- Formats `requirement` field

**Step 6: Validate**
```bash
python ingestion/validate_cases.py
# Runs: CaseInput.model_validate() on each output
```
- Ensures schema compliance before Step 4 (smoke test)

### 3.4 Dependencies

`ingestion/requirements.txt`:
```
requests>=2.31.0
PyYAML>=6.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

Install with: `pip install -r ingestion/requirements.txt`

---

## 4. Verification Plan (Step 4 — Awaiting Confirmation)

### 4.1 Smoke Test

**Goal**: Confirm one mined case runs end-to-end through all three adapters.

**Test Setup**:
- Select 1 mined case (e.g., smallest by word count)
- Use cheapest model: `gpt-4o-mini-2024-07-18` (or local Ollama if no API key)
- Run single setting: `single_agent` (fastest, no negotiation)
- Seed: 42 (deterministic)

**Commands**:
```bash
# MARE
uv run openre_bench --run-case \
  --case-input data/case_studies/<mined_case>_input.json \
  --artifacts-dir artifacts/smoke/<mined_case>-mare \
  --run-record artifacts/smoke/<mined_case>-mare/run_record.json \
  --system mare \
  --setting single_agent \
  --seed 42

# iReDev
uv run openre_bench --run-case \
  --case-input data/case_studies/<mined_case>_input.json \
  --artifacts-dir artifacts/smoke/<mined_case>-iredev \
  --run-record artifacts/smoke/<mined_case>-iredev/run_record.json \
  --system iredev \
  --setting single_agent \
  --seed 42

# QUARE
uv run openre_bench --run-case \
  --case-input data/case_studies/<mined_case>_input.json \
  --artifacts-dir artifacts/smoke/<mined_case>-quare \
  --run-record artifacts/smoke/<mined_case>-quare/run_record.json \
  --system quare \
  --setting single_agent \
  --seed 42
```

**Success Criteria**:
1. All three runs complete without error
2. All four phase JSON files are generated
3. `run_record.json` has `validation_passed: true`
4. Schema validation passes:
   ```bash
   uv run openre_bench --validate-comparison \
     --run-record artifacts/smoke/<mined_case>-mare/run_record.json
   ```

**Failure Handling**:
- If adapter crashes: Check `requirement` field for unexpected characters or structure
- If validation fails: Compare against existing cases' phase outputs
- If LLM errors: Retry with longer timeout or local model fallback

### 4.2 Token Usage & Cost Tracking

**Instrumentation**:
- Extract token counts from `run_record.json` (if available)
- Use OpenAI tokenizer estimate if actual counts unavailable:
  ```python
  import tiktoken
  enc = tiktoken.encoding_for_model("gpt-4o-mini")
  tokens = len(enc.encode(text))
  ```

**Cost Formula** (gpt-4o-mini-2024-07-18 pricing as of Sep 2026):
```
cost_usd = (input_tokens / 1_000_000) * 0.15 
         + (output_tokens / 1_000_000) * 0.60
```

**Tracking Table** (saved to `artifacts/smoke/cost_report.md`):
```markdown
| Case | System | Setting | Input Tokens | Output Tokens | Cost (USD) | Runtime (s) |
|------|--------|---------|--------------|---------------|------------|-------------|
| X    | MARE   | single  | 1,234        | 567           | $0.0005    | 12.3        |
| X    | iReDev | single  | 1,456        | 789           | $0.0007    | 15.6        |
| X    | QUARE  | single  | 1,678        | 890           | $0.0008    | 18.2        |
```

**Budget Check**:
- Single smoke test should be < $0.01 USD
- Full 3-case × 3-system × 1-setting matrix: < $0.10 USD

---

## 5. Deliverables Summary

Upon completion of Steps 2-4:

1. **`docs/ingestion_design.md`** (this document)
   - Directly citable in paper Methods section
   - Reproducible mining methodology

2. **`ingestion/` module**
   - 6 Python scripts + config + README
   - Fully automated pipeline (one-command execution)

3. **3-5 mined cases** in `data/case_studies/`
   - Format: `<project>_<theme>_input.json`
   - Schema-compliant with existing cases

4. **`data/mined_cases_metadata.json`**
   - Traceability to source GitHub issues
   - Aggregation method documentation
   - Timestamp of mining

5. **Mining statistics** in `ingestion/stats/`
   - `mining_statistics.csv`: Numerical summary
   - `mining_statistics.md`: Human-readable report
   - Columns: project, total_issues, after_labels, after_length, after_dedup, final_count

6. **Smoke test results** in `artifacts/smoke/`
   - 3 system runs per mined case
   - `cost_report.md`: Token usage and cost
   - Validation reports

---

## 6. Next Steps

**Current Status**: Step 1 Complete — Design documented, awaiting user confirmation.

**User Action Required**:
1. Review Section 2 (Ingestion Design) — confirm or modify criteria
2. Approve candidate projects (to be populated after criteria approval)
3. Confirm aggregation strategy (time window vs. topic clustering)
4. Set GitHub API token: `export GITHUB_TOKEN=<your_token>`

**Once approved, proceed to**:
- Implement Step 3 (ingestion module)
- Execute Step 4 (smoke test)
- Finalize deliverables for paper submission

---

## Appendix: Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **GitHub API rate limits** | Mining fails mid-process | Cache responses, use authenticated token (5000 req/hr vs 60) |
| **Mined cases too short** | Adapters produce trivial outputs | Set minimum aggregated word count (50 words) |
| **Mined cases too long** | LLM context overflow | Set maximum aggregated word count (500 words) |
| **Domain overlap with existing cases** | Reduces perceived novelty | Exclude banking/library/accounting domains in project selection |
| **Low-quality issue text** | Incoherent requirements | Manual review of 1 case before full mining; add post-filter if needed |
| **License restrictions** | Cannot publish issue text | Verify OSI-approved licenses; add attribution in metadata |


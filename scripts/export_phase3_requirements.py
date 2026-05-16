#!/usr/bin/env python3
"""Export all requirement-like Phase 3 outputs for human-evaluation auditing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_human_eval_workbook import format_raw_requirement_text
from build_human_eval_workbook import format_requirement_text
from build_human_eval_workbook import is_requirement_like
from build_human_eval_workbook import normalize_text


PHASE3_FILENAME = "phase3_integrated_kaos_model.json"
RUN_RECORD_FILENAME = "run_record.json"
DEFAULT_RUNS_DIR = "experiment_outputs/mare-iredev-quare/runs"
DEFAULT_OUTPUT_CSV = "human_eval/phase3_requirements_all.csv"
DEFAULT_OUTPUT_JSON = "human_eval/phase3_requirements_all.json"
DEFAULT_SUMMARY_JSON = "human_eval/phase3_requirements_summary.json"
CASE_ORDER = ("AD", "ATM", "Bookkeeping", "Library", "RollCall")
SETTING_ORDER = (
    "single_agent",
    "multi_agent_without_negotiation",
    "multi_agent_with_negotiation",
    "negotiation_integration_verification",
)
FRAMEWORK_ORDER = ("quare", "mare", "iredev")
EXPORT_COLUMNS = (
    "Sample_ID",
    "source_file",
    "run_id",
    "framework",
    "case_study",
    "setting",
    "seed",
    "source_requirement_id",
    "raw_requirement_text",
    "cleaned_requirement_text",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract requirement-like texts from all Phase 3 integrated models."
    )
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    records = collect_phase3_requirements(runs_dir)
    assign_sample_ids(records)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    summary_json = Path(args.summary_json)
    for path in (output_csv, output_json, summary_json):
        path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(output_csv, records)
    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = build_summary(records, runs_dir)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Extracted requirement-like Phase 3 records: {len(records)}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    print(f"Wrote {summary_json}")
    return 0


def collect_phase3_requirements(runs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_record_path in sorted(runs_dir.glob(f"*/{RUN_RECORD_FILENAME}")):
        run_record = load_json(run_record_path)
        if not isinstance(run_record, dict):
            continue
        phase3_path = resolve_phase3_path(run_record, run_record_path.parent)
        if phase3_path is None or not phase3_path.exists():
            continue
        phase3 = load_json(phase3_path)
        elements = phase3.get("gsn_elements", []) if isinstance(phase3, dict) else []
        if not isinstance(elements, list):
            continue

        for element in elements:
            if not isinstance(element, dict):
                continue
            raw_text = format_raw_requirement_text(element)
            cleaned_text = format_requirement_text(element)
            if not cleaned_text or not normalize_text(cleaned_text):
                continue
            if not is_requirement_like(
                element,
                requirement_text=cleaned_text,
                include_goals=True,
            ):
                continue
            records.append(
                {
                    "Sample_ID": "",
                    "source_file": str(phase3_path),
                    "run_id": str(run_record.get("run_id", "")).strip(),
                    "framework": str(run_record.get("system", "")).strip().lower(),
                    "case_study": str(run_record.get("case_id", "")).strip(),
                    "setting": str(run_record.get("setting", "")).strip(),
                    "seed": to_int(run_record.get("seed"), default=0),
                    "source_requirement_id": str(element.get("id", "")).strip(),
                    "raw_requirement_text": raw_text,
                    "cleaned_requirement_text": cleaned_text,
                }
            )

    records.sort(key=record_sort_key)
    return records


def assign_sample_ids(records: list[dict[str, Any]]) -> None:
    width = max(4, len(str(len(records))))
    for index, record in enumerate(records, start=1):
        record["Sample_ID"] = f"REQ-{index:0{width}d}"


def resolve_phase3_path(run_record: dict[str, Any], run_dir: Path) -> Path | None:
    artifact_paths = run_record.get("artifact_paths", {})
    if isinstance(artifact_paths, dict):
        raw_path = str(artifact_paths.get(PHASE3_FILENAME, "")).strip()
        if raw_path:
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate
            relative_candidate = run_dir / candidate.name
            if relative_candidate.exists():
                return relative_candidate
    artifacts_dir = str(run_record.get("artifacts_dir", "")).strip()
    if artifacts_dir:
        candidate = Path(artifacts_dir) / PHASE3_FILENAME
        if candidate.exists():
            return candidate
    candidate = run_dir / PHASE3_FILENAME
    if candidate.exists():
        return candidate
    return None


def record_sort_key(record: dict[str, Any]) -> tuple[int, int, int, int, str, str]:
    return (
        order_index(str(record["framework"]), FRAMEWORK_ORDER),
        order_index(str(record["case_study"]), CASE_ORDER),
        order_index(str(record["setting"]), SETTING_ORDER),
        int(record["seed"]),
        str(record["run_id"]),
        str(record["source_requirement_id"]),
    )


def order_index(value: str, order: tuple[str, ...]) -> int:
    try:
        return order.index(value)
    except ValueError:
        return len(order)


def build_summary(records: list[dict[str, Any]], runs_dir: Path) -> dict[str, Any]:
    by_framework = Counter(str(record["framework"]) for record in records)
    by_case = Counter(str(record["case_study"]) for record in records)
    by_setting = Counter(str(record["setting"]) for record in records)
    by_framework_case = Counter(
        (str(record["framework"]), str(record["case_study"])) for record in records
    )
    by_framework_case_setting = Counter(
        (
            str(record["framework"]),
            str(record["case_study"]),
            str(record["setting"]),
        )
        for record in records
    )
    unique_cleaned = {
        (
            str(record["framework"]),
            str(record["case_study"]),
            str(record["setting"]),
            normalize_text(str(record["cleaned_requirement_text"])),
        )
        for record in records
    }
    return {
        "runs_dir": str(runs_dir),
        "total_records": len(records),
        "unique_cleaned_texts_by_framework_case_setting": len(unique_cleaned),
        "by_framework": dict(sorted(by_framework.items())),
        "by_case_study": dict(sorted(by_case.items())),
        "by_setting": dict(sorted(by_setting.items())),
        "by_framework_case": [
            {"framework": framework, "case_study": case_study, "n": count}
            for (framework, case_study), count in sorted(by_framework_case.items())
        ],
        "by_framework_case_setting": [
            {
                "framework": framework,
                "case_study": case_study,
                "setting": setting,
                "n": count,
            }
            for (framework, case_study, setting), count in sorted(
                by_framework_case_setting.items()
            )
        ],
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPORT_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in EXPORT_COLUMNS})


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def to_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())

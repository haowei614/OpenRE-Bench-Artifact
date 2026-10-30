"""Step 6: Validate mined case files against the harness schema.

Loads each mined case listed in the metadata sidecar and validates it with the
same `CaseInput` model the harness uses, so a case that passes here is
guaranteed to load in the comparison matrix. Also enforces the configured word
range as a soft check (warning only).

Usage:
    uv run python ingestion/validate_cases.py [--config ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import LOGGER
from _common import load_config
from _common import read_json
from _common import resolve_path

# Make the harness package importable when run as a plain script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from openre_bench.schemas import CaseInput  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate mined cases against CaseInput.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    cases_dir = resolve_path(config["output"]["cases_dir"])
    metadata_path = resolve_path(config["output"]["metadata_file"])
    word_min = int(config["aggregation"]["target_word_count_min"])
    # Body budget plus a header allowance (~15 words per aggregated issue), since
    # the requirement string prepends a "Requirement N (Issue #...)" line each.
    max_issues = int(config["aggregation"]["max_issues_per_case"])
    word_max = int(config["aggregation"]["target_word_count_max"]) + 15 * max_issues

    if not metadata_path.exists():
        raise SystemExit("No metadata file. Run convert_to_schema.py first.")

    metadata = read_json(metadata_path)
    case_names = list(metadata.get("cases", {}).keys())
    if not case_names:
        raise SystemExit("Metadata lists no mined cases.")

    failures = 0
    for case_name in case_names:
        case_path = cases_dir / f"{case_name}_input.json"
        if not case_path.exists():
            LOGGER.error("MISSING: %s", case_path)
            failures += 1
            continue
        try:
            payload = read_json(case_path)
            case = CaseInput.model_validate(payload)
        except Exception as error:  # noqa: BLE001 - report any validation failure
            LOGGER.error("INVALID %s: %s", case_path.name, error)
            failures += 1
            continue

        words = len(case.requirement.split())
        status = "OK"
        if not (word_min <= words <= word_max):
            status = f"OK (word count {words} outside [{word_min},{word_max}] soft range)"
        LOGGER.info("%s: %s (%d words)", case_path.name, status, words)

    if failures:
        LOGGER.error("Validation FAILED for %d case(s).", failures)
        raise SystemExit(1)
    LOGGER.info("All %d mined case(s) passed CaseInput validation.", len(case_names))


if __name__ == "__main__":
    main()

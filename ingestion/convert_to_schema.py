"""Step 5: Convert aggregated case groups into CaseInput JSON files.

Emits `<case_name>_input.json` files into the case-studies directory (matching
the existing 3-field schema exactly) plus a sidecar metadata file recording
provenance (source issue URLs, aggregation method, mined-at timestamp). The
case files themselves contain ONLY case_name/case_description/requirement so
they are byte-compatible with the existing loader.

Usage:
    uv run python ingestion/convert_to_schema.py [--config ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import LOGGER
from _common import load_config
from _common import read_json
from _common import resolve_path
from _common import utc_timestamp
from _common import write_json
from _textproc import anonymize_mentions

AGGREGATED_DIR = Path(__file__).resolve().parent / "aggregated"


def _case_name(template: str, slug: str, window: str) -> str:
    """Build a filesystem-safe case name from the template."""

    raw = template.format(slug=slug, window=window)
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in raw)


def _build_requirement(aggregated: dict[str, Any]) -> str:
    """Concatenate selected issue bodies into a single requirement string."""

    parts: list[str] = []
    for index, issue in enumerate(aggregated["selected_issues"], start=1):
        title = anonymize_mentions(issue["title"])
        body = anonymize_mentions(issue["body_for_case"])
        header = f"Requirement {index} (Issue #{issue['number']}: {title})"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def _build_description(aggregated: dict[str, Any]) -> str:
    """Build a concise domain description for the case."""

    return (
        f"Feature requests mined from the {aggregated['repo']} project "
        f"({aggregated['domain']}), aggregated from GitHub issues in "
        f"{aggregated['window_label'].upper()}."
    )


def convert_project(slug: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one aggregated file into a CaseInput JSON + metadata entry."""

    aggregated_path = AGGREGATED_DIR / f"{slug}_aggregated.json"
    if not aggregated_path.exists():
        LOGGER.info("%s: no aggregated file (project skipped upstream).", slug)
        return None

    aggregated = read_json(aggregated_path)
    template = config["output"]["case_name_template"]
    case_name = _case_name(template, slug, aggregated["window_label"])

    case_input = {
        "case_name": case_name,
        "case_description": _build_description(aggregated),
        "requirement": _build_requirement(aggregated),
    }

    cases_dir = resolve_path(config["output"]["cases_dir"])
    out_path = cases_dir / f"{case_name}_input.json"
    write_json(out_path, case_input)
    LOGGER.info("Wrote case %s (%d words).", out_path.name, len(case_input["requirement"].split()))

    metadata_entry = {
        "case_name": case_name,
        "repo": aggregated["repo"],
        "domain": aggregated["domain"],
        "aggregation_method": aggregated["aggregation_method"],
        "window_label": aggregated["window_label"],
        "mined_at": utc_timestamp(),
        "source_fetched_at": aggregated["fetched_at"],
        "source_issues": [
            {
                "number": issue["number"],
                "title": issue["title"],
                "url": issue["html_url"],
                "created_at": issue["created_at"],
                "state": issue["state"],
                "reactions": issue["reactions"],
            }
            for issue in aggregated["selected_issues"]
        ],
    }
    return metadata_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert aggregated groups to CaseInput.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    entries: dict[str, Any] = {}
    for repo_cfg in config["projects"]:
        entry = convert_project(repo_cfg["slug"], config)
        if entry is not None:
            entries[entry["case_name"]] = entry

    metadata_path = resolve_path(config["output"]["metadata_file"])
    metadata = {
        "generated_at": utc_timestamp(),
        "note": (
            "Provenance for mined OpenRE-Bench cases. Case JSON files hold only "
            "case_name/case_description/requirement; source issue links live here."
        ),
        "cases": entries,
    }
    write_json(metadata_path, metadata)
    LOGGER.info("Wrote %d case(s) and metadata to %s.", len(entries), metadata_path)


if __name__ == "__main__":
    main()

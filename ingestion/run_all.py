"""Orchestrator: run the full ingestion pipeline end to end.

Runs steps 1-7 in order. Steps that hit the GitHub API (fetch_projects,
fetch_issues) require GITHUB_TOKEN; the remaining steps operate on cached
snapshots and need no network access.

Usage:
    uv run python ingestion/run_all.py [--config ...] [--skip-fetch] [--force]
"""

from __future__ import annotations

import argparse

import aggregate_cases
import build_stats
import convert_to_schema
import fetch_issues
import fetch_projects
import filter_issues
import validate_cases
from _common import LOGGER


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip API steps and use cached snapshots (offline).",
    )
    parser.add_argument("--force", action="store_true", help="Force re-fetch of issues.")
    args = parser.parse_args()

    argv_base = ["--config", args.config] if args.config else []

    if not args.skip_fetch:
        LOGGER.info("=== Step 1: verify projects ===")
        _run(fetch_projects, argv_base)
        LOGGER.info("=== Step 2: fetch issues ===")
        _run(fetch_issues, argv_base + (["--force"] if args.force else []))
    else:
        LOGGER.info("Skipping API steps (--skip-fetch); using cached snapshots.")

    LOGGER.info("=== Step 3: filter issues ===")
    _run(filter_issues, argv_base)
    LOGGER.info("=== Step 4: aggregate cases ===")
    _run(aggregate_cases, argv_base)
    LOGGER.info("=== Step 5: convert to schema ===")
    _run(convert_to_schema, argv_base)
    LOGGER.info("=== Step 6: validate cases ===")
    _run(validate_cases, argv_base)
    LOGGER.info("=== Step 7: build statistics ===")
    _run(build_stats, argv_base)
    LOGGER.info("Pipeline complete.")


def _run(module: object, argv: list[str]) -> None:
    """Invoke a step module's main() with a temporary argv."""

    import sys

    saved = sys.argv
    sys.argv = [getattr(module, "__name__", "step"), *argv]
    try:
        module.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()

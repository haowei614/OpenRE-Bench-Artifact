"""Step 1: Verify candidate projects against selection thresholds.

Queries live repo metadata and issue counts, checks every threshold in
`project_selection`, and writes a timestamped snapshot. Projects that fail a
threshold are recorded (with the failing reason) but not silently dropped, so
the decision is auditable in the paper.

Usage:
    uv run python ingestion/fetch_projects.py [--config ingestion/config.yaml]
"""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from _common import LOGGER
from _common import build_client
from _common import load_config
from _common import utc_stamp_compact
from _common import utc_timestamp
from _common import write_json


def _days_since(iso_timestamp: str | None) -> float | None:
    if not iso_timestamp:
        return None
    parsed = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    return (datetime.now(UTC) - parsed).total_seconds() / 86400.0


def verify_project(client: Any, repo_cfg: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Fetch metadata and evaluate all selection thresholds for one project."""

    repo = repo_cfg["repo"]
    LOGGER.info("Verifying %s ...", repo)
    meta = client.get_repo(repo)

    closed_count = client.get_search_issue_count(repo, "type:issue state:closed")
    open_count = client.get_search_issue_count(repo, "type:issue state:open")

    stars = int(meta.get("stargazers_count", 0))
    language = meta.get("language")
    pushed_days = _days_since(meta.get("pushed_at"))
    license_info = meta.get("license") or {}
    license_id = license_info.get("spdx_id")

    failures: list[str] = []
    if stars < selection["min_stars"]:
        failures.append(f"stars {stars} < {selection['min_stars']}")
    if open_count < selection["min_open_issues"]:
        failures.append(f"open_issues {open_count} < {selection['min_open_issues']}")
    if closed_count < selection["min_closed_issues"]:
        failures.append(f"closed_issues {closed_count} < {selection['min_closed_issues']}")
    if language not in selection["languages"]:
        failures.append(f"language {language} not in {selection['languages']}")
    if pushed_days is not None and pushed_days > selection["require_active_days"]:
        failures.append(f"inactive {pushed_days:.0f}d > {selection['require_active_days']}d")

    record = {
        "repo": repo,
        "slug": repo_cfg["slug"],
        "domain": repo_cfg["domain"],
        "labels": repo_cfg.get("labels", []),
        "stars": stars,
        "language": language,
        "open_issues": open_count,
        "closed_issues": closed_count,
        "pushed_at": meta.get("pushed_at"),
        "days_since_push": round(pushed_days, 1) if pushed_days is not None else None,
        "license": license_id,
        "description": meta.get("description"),
        "passes_selection": not failures,
        "failures": failures,
    }
    verdict = "PASS" if not failures else f"FAIL ({'; '.join(failures)})"
    LOGGER.info(
        "  %s: %d stars, %s, open=%d closed=%d, license=%s -> %s",
        repo,
        stars,
        language,
        open_count,
        closed_count,
        license_id,
        verdict,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify candidate projects.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    client = build_client(config)
    selection = config["project_selection"]

    records = [verify_project(client, repo_cfg, selection) for repo_cfg in config["projects"]]

    snapshot = {
        "fetched_at": utc_timestamp(),
        "selection_thresholds": selection,
        "projects": records,
    }
    snapshots_dir = Path(__file__).resolve().parent / "snapshots"
    out_path = snapshots_dir / f"projects_{utc_stamp_compact()}.json"
    write_json(out_path, snapshot)

    # Also write/refresh a stable "latest" pointer for downstream steps.
    write_json(snapshots_dir / "projects_latest.json", snapshot)

    passed = sum(1 for record in records if record["passes_selection"])
    LOGGER.info("Wrote %s (%d/%d projects pass selection).", out_path, passed, len(records))


if __name__ == "__main__":
    main()

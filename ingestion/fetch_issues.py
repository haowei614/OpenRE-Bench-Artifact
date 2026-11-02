"""Step 2: Download raw issue data for each project.

Reads the label query from config, pages through the /issues endpoint, drops
pull requests, and writes one timestamped JSON snapshot per project. Snapshots
record `fetched_at` and contain only public API data (never the token), so the
mining is reproducible and safe to commit.

Caching: if a snapshot for a project already exists and --force is not set, the
project is skipped (the cached snapshot is reused).

Usage:
    uv run python ingestion/fetch_issues.py [--config ...] [--force] [--only django]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import LOGGER
from _common import build_client
from _common import load_config
from _common import read_json
from _common import utc_stamp_compact
from _common import utc_timestamp
from _common import write_json

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"


def _latest_snapshot_for(slug: str) -> Path | None:
    """Return the most recent issue snapshot for a project slug, if any."""

    matches = sorted(SNAPSHOTS_DIR.glob(f"issues_{slug}_*.json"))
    return matches[-1] if matches else None


def fetch_project_issues(
    client: Any,
    repo_cfg: dict[str, Any],
    github_cfg: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    """Fetch (or reuse cached) issues for one project and write a snapshot."""

    slug = repo_cfg["slug"]
    repo = repo_cfg["repo"]

    existing = _latest_snapshot_for(slug)
    if existing and not force:
        LOGGER.info("Reusing cached snapshot for %s: %s", repo, existing.name)
        return read_json(existing)

    LOGGER.info("Fetching issues for %s (labels=%s) ...", repo, repo_cfg.get("labels"))
    issues = client.iter_issues(
        repo=repo,
        labels=repo_cfg.get("labels", []),
        state="all",
        max_issues=int(github_cfg.get("max_issues_per_project", 600)),
    )

    # Retain only fields we need downstream; keeps snapshots compact and PII-light.
    trimmed = [
        {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "body": issue.get("body"),
            "state": issue.get("state"),
            "labels": [label.get("name") for label in issue.get("labels", [])],
            "comments": issue.get("comments", 0),
            "reactions_total": (issue.get("reactions") or {}).get("total_count", 0),
            "reactions_plus_one": (issue.get("reactions") or {}).get("+1", 0),
            "reactions_heart": (issue.get("reactions") or {}).get("heart", 0),
            "created_at": issue.get("created_at"),
            "closed_at": issue.get("closed_at"),
            "html_url": issue.get("html_url"),
            "milestone": (issue.get("milestone") or {}).get("title"),
        }
        for issue in issues
    ]

    snapshot = {
        "repo": repo,
        "slug": slug,
        "domain": repo_cfg["domain"],
        "query_labels": repo_cfg.get("labels", []),
        "fetched_at": utc_timestamp(),
        "issue_count": len(trimmed),
        "issues": trimmed,
    }
    out_path = SNAPSHOTS_DIR / f"issues_{slug}_{utc_stamp_compact()}.json"
    write_json(out_path, snapshot)
    LOGGER.info("Wrote %s (%d issues).", out_path, len(trimmed))
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw issues per project.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    parser.add_argument("--only", default=None, help="Fetch only this project slug.")
    args = parser.parse_args()

    config = load_config(args.config)
    client = build_client(config)
    github_cfg = config.get("github", {})

    projects = config["projects"]
    if args.only:
        projects = [proj for proj in projects if proj["slug"] == args.only]
        if not projects:
            raise SystemExit(f"No project with slug '{args.only}' in config.")

    for repo_cfg in projects:
        fetch_project_issues(client, repo_cfg, github_cfg, force=args.force)

    LOGGER.info("Done fetching issues for %d project(s).", len(projects))


if __name__ == "__main__":
    main()

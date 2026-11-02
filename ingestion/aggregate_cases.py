"""Step 4: Aggregate filtered feature requests into one RE case per project.

Default method is `time_window` (calendar-quarter buckets), which is fully
deterministic: pick the quarter with the most qualifying issues (ties broken by
earliest quarter), then take the top-N issues in that quarter by reaction count
(ties broken by issue number ascending). Bodies are truncated per-issue and the
total is capped to keep the case in the same size range as the existing cases.

Usage:
    uv run python ingestion/aggregate_cases.py [--config ...]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import LOGGER
from _common import load_config
from _common import read_json
from _common import write_json
from _textproc import truncate_words
from _textproc import word_count

FILTERED_DIR = Path(__file__).resolve().parent / "filtered"
AGGREGATED_DIR = Path(__file__).resolve().parent / "aggregated"


def _quarter_label(created_at: str) -> str:
    """Return a bucket label like '2023q1' from an ISO timestamp."""

    year = int(created_at[:4])
    month = int(created_at[5:7])
    quarter = (month - 1) // 3 + 1
    return f"{year}q{quarter}"


def _select_window(issues: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Group issues by quarter and return (label, issues) for the best window."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        created = issue.get("created_at")
        if created:
            buckets[_quarter_label(created)].append(issue)

    if not buckets:
        return "", []

    # Most issues wins; tie-break by earliest quarter label (chronological).
    def _sort_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, str]:
        label, bucket_issues = item
        return (-len(bucket_issues), label)

    best_label, best_issues = sorted(buckets.items(), key=_sort_key)[0]
    return best_label, best_issues


def _rank_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by reaction count desc, then issue number asc (deterministic)."""

    return sorted(
        issues,
        key=lambda i: (
            -(int(i.get("reactions_plus_one", 0)) + int(i.get("reactions_heart", 0))),
            int(i.get("number", 0)),
        ),
    )


def aggregate_project(slug: str, agg_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate one project's filtered issues into a single case payload."""

    filtered_path = FILTERED_DIR / f"{slug}_filtered.json"
    if not filtered_path.exists():
        raise SystemExit(f"No filtered file for '{slug}'. Run filter_issues.py first.")

    filtered = read_json(filtered_path)
    kept = filtered["kept_issues"]

    method = agg_cfg["method"]
    if method != "time_window":
        raise NotImplementedError(
            f"Aggregation method '{method}' not implemented; use 'time_window'."
        )

    window_label, window_issues = _select_window(kept)
    if not window_issues:
        LOGGER.warning("%s: no datable issues after filtering; skipping.", slug)
        return None

    ranked = _rank_issues(window_issues)
    max_issues = int(agg_cfg["max_issues_per_case"])
    min_issues = int(agg_cfg["min_issues_per_case"])
    per_issue_cap = int(agg_cfg["per_issue_word_cap"])
    total_cap = int(agg_cfg["target_word_count_max"])

    selected: list[dict[str, Any]] = []
    running_words = 0
    for issue in ranked:
        if len(selected) >= max_issues:
            break
        body = truncate_words(issue["_normalized_body"], per_issue_cap)
        candidate_words = word_count(body)
        if selected and running_words + candidate_words > total_cap:
            break
        issue["_case_body"] = body
        selected.append(issue)
        running_words += candidate_words

    if len(selected) < min_issues:
        LOGGER.warning(
            "%s: only %d issue(s) in best window (need >= %d); skipping.",
            slug,
            len(selected),
            min_issues,
        )
        return None

    result = {
        "slug": slug,
        "repo": filtered["repo"],
        "domain": filtered["domain"],
        "fetched_at": filtered["fetched_at"],
        "aggregation_method": "time_window",
        "window_label": window_label,
        "window_issue_count": len(window_issues),
        "selected_count": len(selected),
        "total_words": running_words,
        "selected_issues": [
            {
                "number": issue["number"],
                "title": issue["title"],
                "html_url": issue["html_url"],
                "created_at": issue["created_at"],
                "state": issue["state"],
                "reactions": int(issue.get("reactions_plus_one", 0))
                + int(issue.get("reactions_heart", 0)),
                "body_for_case": issue["_case_body"],
            }
            for issue in selected
        ],
    }
    out_path = AGGREGATED_DIR / f"{slug}_aggregated.json"
    write_json(out_path, result)
    LOGGER.info(
        "%s: window=%s, %d issues, %d words aggregated.",
        slug,
        window_label,
        len(selected),
        running_words,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate filtered issues into cases.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    agg_cfg = config["aggregation"]

    produced = 0
    for repo_cfg in config["projects"]:
        if aggregate_project(repo_cfg["slug"], agg_cfg) is not None:
            produced += 1

    LOGGER.info("Aggregation produced %d candidate case(s).", produced)


if __name__ == "__main__":
    main()

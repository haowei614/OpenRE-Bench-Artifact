"""Draw a seeded, stratified sample of issues for a two-rater precision audit.

The v1.0 precision census inspected the 18 issues that ended up in the mined
cases; those were selected by community reactions, so they cannot validate the
*filter* itself (selection bias). This script instead samples from the filter's
own populations:

  * 50 issues drawn uniformly from issues that survive all active filter
    stages (the "retained" population), and
  * 30 issues drawn uniformly from the issues the ``no_requirement_signal`` stage
    dropped (the "rejected" population),

so two raters can estimate retained-set precision, rejected-set recall proxy, and
inter-rater agreement (Cohen's kappa). Sampling uses a fixed seed (42) for
reproducibility. Rater columns are left blank on purpose --- this script never
assigns labels.

Usage:
    uv run python ingestion/sample_precision_audit.py \
        [--filtered-dir ingestion/filtered] \
        [--snapshots-dir ingestion/snapshots] \
        [--out docs/precision_audit_sample.csv] \
        [--seed 42] [--n-retained 50] [--n-rejected 30]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random

_DROP_STAGE = "no_requirement_signal"


def _truncate(text: str, limit: int = 300) -> str:
    """Collapse whitespace and truncate a body to keep the CSV skimmable."""

    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _load_snapshot_index(snapshots_dir: str, slug: str) -> dict[int, dict]:
    """Map issue number -> raw issue for one project's latest snapshot."""

    # Filtered files record their exact source snapshot; fall back to a glob.
    matches = sorted(glob.glob(os.path.join(snapshots_dir, f"issues_{slug}_*.json")))
    if not matches:
        return {}
    raw = json.load(open(matches[-1]))
    issues = raw.get("issues", raw) if isinstance(raw, dict) else raw
    index: dict[int, dict] = {}
    for issue in issues:
        num = issue.get("number")
        if num is not None:
            index[int(num)] = issue
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-dir", default="ingestion/filtered")
    parser.add_argument("--snapshots-dir", default="ingestion/snapshots")
    parser.add_argument("--out", default="docs/precision_audit_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-retained", type=int, default=50)
    parser.add_argument("--n-rejected", type=int, default=30)
    args = parser.parse_args()

    retained: list[dict] = []
    rejected: list[dict] = []

    for path in sorted(glob.glob(os.path.join(args.filtered_dir, "*_filtered.json"))):
        data = json.load(open(path))
        slug = data.get("slug") or os.path.basename(path).split("_filtered")[0]
        repo = data.get("repo", slug)

        for issue in data.get("kept_issues", []):
            retained.append(
                {
                    "stratum": "retained",
                    "repo": repo,
                    "number": issue.get("number"),
                    "url": issue.get("html_url", ""),
                    "title": issue.get("title", ""),
                    "body": _truncate(issue.get("body", "")),
                }
            )

        dropped_numbers = [
            r.get("number")
            for r in data.get("rejections", [])
            if r.get("reason") == _DROP_STAGE
        ]
        if dropped_numbers:
            snap = _load_snapshot_index(args.snapshots_dir, slug)
            for num in dropped_numbers:
                issue = snap.get(int(num), {}) if num is not None else {}
                rejected.append(
                    {
                        "stratum": "signal_dropped",
                        "repo": repo,
                        "number": num,
                        "url": issue.get("html_url", ""),
                        "title": issue.get("title", ""),
                        "body": _truncate(issue.get("body", "")),
                    }
                )

    rng = random.Random(args.seed)
    sample_retained = rng.sample(retained, min(args.n_retained, len(retained)))
    sample_rejected = rng.sample(rejected, min(args.n_rejected, len(rejected)))
    rows = sample_retained + sample_rejected

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = [
        "stratum",
        "repo",
        "number",
        "url",
        "title",
        "body_truncated",
        "label_rater1",
        "label_rater2",
    ]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stratum": row["stratum"],
                    "repo": row["repo"],
                    "number": row["number"],
                    "url": row["url"],
                    "title": row["title"],
                    "body_truncated": row["body"],
                    "label_rater1": "",
                    "label_rater2": "",
                }
            )

    print(
        f"Populations: retained={len(retained)}, "
        f"{_DROP_STAGE}_dropped={len(rejected)}"
    )
    print(
        f"Wrote {len(rows)} rows "
        f"({len(sample_retained)} retained + {len(sample_rejected)} dropped) "
        f"to {args.out} (seed {args.seed}). Rater columns are blank by design."
    )


if __name__ == "__main__":
    main()

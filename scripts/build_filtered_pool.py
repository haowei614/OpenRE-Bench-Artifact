"""Publish the filtered issue pool as an issue-level dataset.

Emits one JSON object per line (JSONL) for every issue that survived the active
filter stages (the released issue-level pool), with provenance fields and the
per-issue text de-identified (author @-mentions -> @USER, same transform the
mined cases use). Issues later aggregated into the five `CaseInput` cases are
flagged with `selected_into_case`.

Raw API snapshots remain the verbatim provenance under `ingestion/snapshots/`.
The requirement-signal stage is optional (default OFF in v1.0); ``passed_stages``
omits it when the gate was disabled for that filtered file.

Usage:
    uv run python scripts/build_filtered_pool.py \
        [--filtered-dir ingestion/filtered] \
        [--metadata data/mined_cases_metadata.json] \
        [--out data/filtered_issue_pool.jsonl]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Reuse the exact anonymizer the case pipeline uses.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from _textproc import anonymize_mentions  # noqa: E402

# Funnel stages in order. ``requirement_signal`` is included only when that
# gate was enabled for the filtered file being exported.
PASSED_STAGES_BASE = [
    "label_allow_list",
    "state_reactions_gate",
    "min_body_length",
    "bug_exclusion",
]
PASSED_STAGES_TAIL = [
    "mega_thread_cap",
    "tfidf_dedup",
]


def _passed_stages(signal_enabled: bool) -> list[str]:
    if signal_enabled:
        return [*PASSED_STAGES_BASE, "requirement_signal", *PASSED_STAGES_TAIL]
    return [*PASSED_STAGES_BASE, *PASSED_STAGES_TAIL]


def _selected_numbers(metadata_path: str) -> dict[int, str]:
    """Map issue number -> case_name for issues aggregated into a case."""

    if not os.path.exists(metadata_path):
        return {}
    cases = json.load(open(metadata_path)).get("cases", {})
    out: dict[int, str] = {}
    for case_name, entry in cases.items():
        for issue in entry.get("source_issues", []):
            num = issue.get("number")
            if num is not None:
                out[int(num)] = case_name
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-dir", default="ingestion/filtered")
    parser.add_argument("--metadata", default="data/mined_cases_metadata.json")
    parser.add_argument("--out", default="data/filtered_issue_pool.jsonl")
    args = parser.parse_args()

    selected = _selected_numbers(args.metadata)
    records: list[dict[str, object]] = []

    for path in sorted(glob.glob(os.path.join(args.filtered_dir, "*_filtered.json"))):
        data = json.load(open(path))
        repo = data.get("repo", "")
        slug = data.get("slug", "")
        domain = data.get("domain", "")
        signal_enabled = bool(data.get("requirement_signal_enabled", False))
        stages = _passed_stages(signal_enabled)
        for issue in data.get("kept_issues", []):
            number = issue.get("number")
            records.append(
                {
                    "repo": repo,
                    "slug": slug,
                    "domain": domain,
                    "number": number,
                    "url": issue.get("html_url", ""),
                    "title": anonymize_mentions(issue.get("title", "")),
                    "labels": issue.get("labels", []),
                    "state": issue.get("state", ""),
                    "created_at": issue.get("created_at", ""),
                    "closed_at": issue.get("closed_at"),
                    "comments": issue.get("comments"),
                    "reactions_plus_one": issue.get("reactions_plus_one"),
                    "reactions_heart": issue.get("reactions_heart"),
                    "reactions_total": issue.get("reactions_total"),
                    "body": anonymize_mentions(issue.get("body", "")),
                    "passed_stages": stages,
                    "selected_into_case": selected.get(int(number)) if number is not None else None,
                }
            )

    # Stable order: repo, then issue number.
    records.sort(key=lambda r: (r["repo"], r["number"] or 0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_selected = sum(1 for r in records if r["selected_into_case"])
    print(
        f"Wrote {len(records)} issues to {args.out} "
        f"({n_selected} flagged selected_into_case)."
    )


if __name__ == "__main__":
    main()

"""Step 7: Build the mining statistics table (CSV + Markdown).

Reads every per-project filtered funnel and the aggregation results, then emits
a table showing, per project: raw issues -> counts after each filter stage ->
issues aggregated into the final case. A "dropped by" breakdown quantifies how
many issues each individual filter removed. Directly citable in the paper.

Usage:
    uv run python ingestion/build_stats.py [--config ...]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _common import LOGGER
from _common import load_config
from _common import read_json
from _common import resolve_path

FILTERED_DIR = Path(__file__).resolve().parent / "filtered"
AGGREGATED_DIR = Path(__file__).resolve().parent / "aggregated"

# (funnel_key_before, funnel_key_after, human_label)
STAGE_STEPS = [
    ("raw", "after_labels", "label filter"),
    ("after_labels", "after_state_gate", "state/reactions gate"),
    ("after_state_gate", "after_min_length", "min body length"),
    ("after_min_length", "after_bug_exclusion", "bug exclusion"),
    ("after_bug_exclusion", "after_requirement_signal", "requirement signal"),
    ("after_requirement_signal", "after_discussion_cap", "mega-thread cap"),
    ("after_discussion_cap", "after_dedup", "dedup"),
]

CSV_COLUMNS = [
    "project",
    "domain",
    "raw",
    "after_labels",
    "after_state_gate",
    "after_min_length",
    "after_bug_exclusion",
    "after_requirement_signal",
    "after_discussion_cap",
    "after_dedup",
    "dropped_label",
    "dropped_state",
    "dropped_length",
    "dropped_bug",
    "dropped_signal",
    "dropped_discussion",
    "dropped_dedup",
    "final_case_issues",
    "final_word_count",
    "window_label",
]


def _collect_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo_cfg in config["projects"]:
        slug = repo_cfg["slug"]
        filtered_path = FILTERED_DIR / f"{slug}_filtered.json"
        if not filtered_path.exists():
            LOGGER.warning("No filtered file for %s; skipping in stats.", slug)
            continue
        filtered = read_json(filtered_path)
        funnel = filtered["funnel"]

        aggregated_path = AGGREGATED_DIR / f"{slug}_aggregated.json"
        if aggregated_path.exists():
            aggregated = read_json(aggregated_path)
            final_issues = aggregated["selected_count"]
            final_words = aggregated["total_words"]
            window = aggregated["window_label"]
        else:
            final_issues = 0
            final_words = 0
            window = "-"

        row = {
            "project": filtered["repo"],
            "domain": filtered["domain"],
            "raw": funnel["raw"],
            "after_labels": funnel["after_labels"],
            "after_state_gate": funnel["after_state_gate"],
            "after_min_length": funnel["after_min_length"],
            "after_bug_exclusion": funnel["after_bug_exclusion"],
            "after_requirement_signal": funnel["after_requirement_signal"],
            "after_discussion_cap": funnel["after_discussion_cap"],
            "after_dedup": funnel["after_dedup"],
            "dropped_label": funnel["raw"] - funnel["after_labels"],
            "dropped_state": funnel["after_labels"] - funnel["after_state_gate"],
            "dropped_length": funnel["after_state_gate"] - funnel["after_min_length"],
            "dropped_bug": funnel["after_min_length"] - funnel["after_bug_exclusion"],
            "dropped_signal": funnel["after_bug_exclusion"] - funnel["after_requirement_signal"],
            "dropped_discussion": funnel["after_requirement_signal"]
            - funnel["after_discussion_cap"],
            "dropped_dedup": funnel["after_discussion_cap"] - funnel["after_dedup"],
            "final_case_issues": final_issues,
            "final_word_count": final_words,
            "window_label": window,
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# OpenRE-Bench Mining Statistics\n")
    lines.append(
        "Per-project issue funnel from raw issues to the final aggregated case. "
        "Each column after `raw` shows survivors after that filter stage.\n"
    )
    lines.append(
        "> `Raw` is the fetched issue sample (most-recent label-matching issues, "
        "capped at `github.max_issues_per_project`), not the project's all-time "
        "issue total. The `Label` drop is typically 0 because the API query "
        "already restricts to the configured labels.\n"
    )

    # Funnel table
    header = (
        "| Project | Domain | Raw | Labels | State | Length | Bug | Signal | "
        "Discussion | Dedup | Final Issues | Words | Window |"
    )
    sep = "|" + "---|" * 13
    lines.append(header)
    lines.append(sep)
    for row in rows:
        lines.append(
            f"| {row['project']} | {row['domain']} | {row['raw']} | {row['after_labels']} | "
            f"{row['after_state_gate']} | {row['after_min_length']} | "
            f"{row['after_bug_exclusion']} | {row['after_requirement_signal']} | "
            f"{row['after_discussion_cap']} | {row['after_dedup']} | "
            f"{row['final_case_issues']} | {row['final_word_count']} | {row['window_label']} |"
        )

    # Per-filter drop breakdown
    lines.append("\n## Issues Dropped by Each Filter\n")
    drop_header = (
        "| Project | Label | State | Length | Bug | Signal | Discussion | Dedup |"
    )
    drop_sep = "|" + "---|" * 8
    lines.append(drop_header)
    lines.append(drop_sep)
    for row in rows:
        lines.append(
            f"| {row['project']} | {row['dropped_label']} | {row['dropped_state']} | "
            f"{row['dropped_length']} | {row['dropped_bug']} | {row['dropped_signal']} | "
            f"{row['dropped_discussion']} | {row['dropped_dedup']} |"
        )

    # Totals
    total_raw = sum(r["raw"] for r in rows)
    total_final = sum(r["final_case_issues"] for r in rows)
    cases = sum(1 for r in rows if r["final_case_issues"] > 0)
    lines.append(
        f"\n**Totals:** {total_raw} raw issues across {len(rows)} projects -> "
        f"{total_final} issues in {cases} final case(s)."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mining statistics table.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = _collect_rows(config)
    if not rows:
        raise SystemExit("No filtered data found. Run the pipeline first.")

    csv_path = resolve_path(config["output"]["stats_csv"])
    md_path = resolve_path(config["output"]["stats_md"])
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    LOGGER.info("Wrote stats: %s and %s", csv_path, md_path)


if __name__ == "__main__":
    main()

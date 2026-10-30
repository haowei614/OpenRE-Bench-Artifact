"""Step 3: Filter raw issues down to qualifying feature requests.

Applies the `issue_filtering` criteria in a fixed order and logs how many
issues each stage drops. Writes per-project filtered JSON plus a per-project
funnel record used later by the statistics table.

Filter order (each stage sees the survivors of the previous one):
    1. label allow-list
    2. state / reactions gate (closed preferred; open needs reactions)
    3. minimum normalized body length
    4. bug-report exclusion (title/body keywords)
    5. requirement-signal requirement (>= 1 keyword) — optional; default OFF
    6. mega-thread exclusion (comment count cap)
    7. near-duplicate removal (TF-IDF cosine)

Usage:
    uv run python ingestion/filter_issues.py [--config ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import LOGGER
from _common import load_config
from _common import read_json
from _common import write_json
from _textproc import deduplicate
from _textproc import normalize_issue_body
from _textproc import word_count

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"
FILTERED_DIR = Path(__file__).resolve().parent / "filtered"


def _latest_snapshot_for(slug: str) -> Path | None:
    matches = sorted(SNAPSHOTS_DIR.glob(f"issues_{slug}_*.json"))
    return matches[-1] if matches else None


def _has_allowed_label(issue: dict[str, Any], allowed_lower: set[str]) -> bool:
    return any((label or "").lower() in allowed_lower for label in issue.get("labels", []))


def _passes_state_gate(issue: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if issue.get("state") == "closed":
        return True
    reactions = int(issue.get("reactions_plus_one", 0)) + int(issue.get("reactions_heart", 0))
    return reactions >= int(cfg["min_reactions_if_open"])


def _looks_like_bug(issue: dict[str, Any], keywords: list[str]) -> bool:
    haystack = f"{issue.get('title', '')}\n{issue.get('body', '') or ''}".lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def _has_requirement_signal(normalized_body: str, keywords: list[str]) -> bool:
    lowered = normalized_body.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def filter_project(slug: str, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Filter one project's issue snapshot and return survivors + funnel counts."""

    snapshot_path = _latest_snapshot_for(slug)
    if not snapshot_path:
        raise SystemExit(f"No issue snapshot for slug '{slug}'. Run fetch_issues.py first.")

    snapshot = read_json(snapshot_path)
    issues = snapshot["issues"]
    allowed_lower = {label.lower() for label in cfg["labels"]}

    funnel: dict[str, int] = {"raw": len(issues)}
    rejections: list[dict[str, Any]] = []

    def _reject(issue: dict[str, Any], reason: str) -> None:
        rejections.append({"number": issue.get("number"), "reason": reason})

    # 1. label allow-list
    stage = [i for i in issues if _has_allowed_label(i, allowed_lower)]
    for i in issues:
        if not _has_allowed_label(i, allowed_lower):
            _reject(i, "label_not_allowed")
    funnel["after_labels"] = len(stage)

    # 2. state / reactions gate
    kept = []
    for i in stage:
        if _passes_state_gate(i, cfg):
            kept.append(i)
        else:
            _reject(i, "open_without_enough_reactions")
    stage = kept
    funnel["after_state_gate"] = len(stage)

    # Precompute normalized bodies once.
    for i in stage:
        i["_normalized_body"] = normalize_issue_body(i.get("body"))

    # 3. minimum body length
    kept = []
    for i in stage:
        if len(i["_normalized_body"]) >= int(cfg["min_body_length"]):
            kept.append(i)
        else:
            _reject(i, "body_too_short")
    stage = kept
    funnel["after_min_length"] = len(stage)

    # 4. bug-report exclusion
    kept = []
    for i in stage:
        if _looks_like_bug(i, cfg["bug_exclusion_keywords"]):
            _reject(i, "looks_like_bug")
        else:
            kept.append(i)
    stage = kept
    funnel["after_bug_exclusion"] = len(stage)

    # 5. requirement-signal requirement (optional; default OFF in config.yaml)
    signal_enabled = bool(cfg.get("requirement_signal", False))
    counterfactual_signal_drops = 0
    if signal_enabled:
        kept = []
        for i in stage:
            if _has_requirement_signal(i["_normalized_body"], cfg["requirement_keywords"]):
                kept.append(i)
            else:
                _reject(i, "no_requirement_signal")
                counterfactual_signal_drops += 1
        stage = kept
    else:
        # Ablation bookkeeping: how many would the gate have dropped.
        for i in stage:
            if not _has_requirement_signal(
                i["_normalized_body"], cfg["requirement_keywords"]
            ):
                counterfactual_signal_drops += 1
    funnel["after_requirement_signal"] = len(stage)
    funnel["requirement_signal_enabled"] = int(signal_enabled)
    funnel["counterfactual_signal_drops"] = counterfactual_signal_drops

    # 6. mega-thread exclusion
    kept = []
    for i in stage:
        if int(i.get("comments", 0)) <= int(cfg["max_discussion_comments"]):
            kept.append(i)
        else:
            _reject(i, "mega_thread")
    stage = kept
    funnel["after_discussion_cap"] = len(stage)

    # 7. near-duplicate removal (deterministic order: by issue number ascending)
    stage.sort(key=lambda i: int(i.get("number", 0)))
    keep_indices = deduplicate(
        [i["_normalized_body"] for i in stage],
        threshold=float(cfg["deduplication_threshold"]),
        seed=seed,
    )
    keep_set = set(keep_indices)
    deduped = [i for idx, i in enumerate(stage) if idx in keep_set]
    for idx, i in enumerate(stage):
        if idx not in keep_set:
            _reject(i, "near_duplicate")
    funnel["after_dedup"] = len(deduped)

    # Attach normalized word count for downstream aggregation.
    for i in deduped:
        i["_normalized_words"] = word_count(i["_normalized_body"])

    result = {
        "slug": slug,
        "repo": snapshot["repo"],
        "domain": snapshot["domain"],
        "source_snapshot": snapshot_path.name,
        "fetched_at": snapshot["fetched_at"],
        "requirement_signal_enabled": signal_enabled,
        "funnel": funnel,
        "kept_issues": deduped,
        "rejections": rejections,
    }
    out_path = FILTERED_DIR / f"{slug}_filtered.json"
    write_json(out_path, result)

    LOGGER.info(
        "%s funnel: raw=%d -> labels=%d -> state=%d -> len=%d -> bug=%d -> signal=%d -> "
        "discussion=%d -> dedup=%d",
        slug,
        funnel["raw"],
        funnel["after_labels"],
        funnel["after_state_gate"],
        funnel["after_min_length"],
        funnel["after_bug_exclusion"],
        funnel["after_requirement_signal"],
        funnel["after_discussion_cap"],
        funnel["after_dedup"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter issues into feature requests.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["issue_filtering"]
    seed = int(config["reproducibility"]["random_seed"])

    for repo_cfg in config["projects"]:
        filter_project(repo_cfg["slug"], cfg, seed)

    LOGGER.info("Filtering complete for %d project(s).", len(config["projects"]))


if __name__ == "__main__":
    main()

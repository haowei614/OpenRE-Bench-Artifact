"""Recompute the paper's reference-runs table from committed run outputs.

Reads the per-run metric CSVs the harness already wrote under
``experiment_outputs/`` (no LLM calls, no API key) and reduces them to the
mean +/- SD table the paper reports for the
``negotiation_integration_verification`` setting, aggregated over the five mined
cases and three seeds (n = 15 per adapter).

This lets a reviewer reproduce the reference-runs table offline from the
committed artifacts in seconds.

Usage:
    uv run python scripts/recompute_reference_table.py \
        [--setting negotiation_integration_verification]

By default this merges:
  * unchanged cases from ``mined-claude`` + ``mined-claude-s202303``
    (elasticsearch / transformers / vscode), and
  * refreshed ``sklearn_2021q4`` + ``pandas_2023q3`` rows from
    ``mined-claude-v1-sklearn-pandas`` (signal-off v1.0 case texts).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

METRICS = [
    ("n_phase3_elements", "P3 elems", "{:.1f}"),
    ("mdc", "MDC", "{:.3f}"),
    ("semantic_preservation_f1", "Sem.F1", "{:.3f}"),
    ("iso29148_verifiability", "ISO ver.", "{:.2f}"),
    ("runtime_seconds", "Time (s)", "{:.1f}"),
]
ADAPTER_ORDER = ["mare", "iredev", "quare"]
REFRESHED_CASES = {"sklearn_2021q4", "pandas_2023q3"}
STABLE_CASES = {"elasticsearch_2026q1", "transformers_2024q3", "vscode_2026q3"}


def _load_rows(dirs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for d in dirs:
        path = os.path.join(d, "comparison_metrics_by_case.csv")
        if not os.path.exists(path):
            matches = glob.glob(os.path.join(d, "**", "comparison_metrics_by_case.csv"), recursive=True)
            path = matches[0] if matches else path
        with open(path, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _default_merged_rows(setting: str) -> list[dict[str, str]]:
    """Build the paper's v1.0 reference-run rows without double-counting cases."""

    old = _load_rows(
        [
            "experiment_outputs/mined-claude",
            "experiment_outputs/mined-claude-s202303",
        ]
    )
    new = _load_rows(["experiment_outputs/mined-claude-v1-sklearn-pandas"])
    rows = [
        r
        for r in old
        if r["setting"] == setting and r["case_id"] in STABLE_CASES
    ]
    rows.extend(
        r
        for r in new
        if r["setting"] == setting and r["case_id"] in REFRESHED_CASES
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="If set, load these dirs verbatim (no v1.0 case merge).",
    )
    parser.add_argument("--setting", default="negotiation_integration_verification")
    args = parser.parse_args()

    if args.dirs is None:
        rows = _default_merged_rows(args.setting)
        source = "v1.0 merge (stable old + refreshed sklearn/pandas)"
    else:
        rows = [r for r in _load_rows(args.dirs) if r["setting"] == args.setting]
        source = str(args.dirs)
    if not rows:
        raise SystemExit(f"No rows for setting={args.setting} under {source}")

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: set[str] = set()
    cases: set[str] = set()
    for r in rows:
        seeds.add(r["seed"])
        cases.add(r["case_id"])
        for key, _, _ in METRICS:
            try:
                agg[r["system"]][key].append(float(r[key]))
            except (KeyError, ValueError):
                pass

    header = ["Adapter"] + [label for _, label, _ in METRICS]
    print(f"Setting: {args.setting}")
    print(f"Source: {source}")
    print(f"Cases: {sorted(cases)}  Seeds: {sorted(seeds)}")
    print("  ".join(f"{h:>18}" for h in header))
    for sysname in ADAPTER_ORDER:
        if sysname not in agg:
            continue
        cells = [f"{sysname:>18}"]
        for key, _, fmt in METRICS:
            vals = agg[sysname][key]
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{fmt.format(mean)} +/- {fmt.format(sd)}".rjust(18))
        print("  ".join(cells))
    n_each = len(next(iter(agg[ADAPTER_ORDER[0]].values())))
    print(f"\nn = {n_each} per adapter ({len(cases)} cases x {len(seeds)} seeds).")


if __name__ == "__main__":
    main()

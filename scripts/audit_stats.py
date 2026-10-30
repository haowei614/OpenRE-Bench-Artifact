"""Compute filter-audit statistics and emit LaTeX macros for the MSR paper.

Reads the labelling sheet (default ``docs/precision_audit_sample.csv``) with
columns ``label_rater1``, ``label_rater2``, and optional ``adjudicated``.

Reports:

  * **Consensus-only** retained precision and signal-drop false-omission rate
    (FOR), using rows where both raters agree (disagreements excluded).
  * **Primary** point estimates that prefer ``adjudicated`` when filled, else
    consensus; once adjudication covers disagreements, primary uses all labelled
    items in each stratum sample.
  * Cohen's kappa and disagreement count over doubly-labelled rows.
  * **Stratified pool precision** for the v1.0 pool (requirement-signal OFF):
    stratum 1 = issues in the released pool that match the keyword signal
    (size N1; estimated by the ``retained`` sample), stratum 2 = pool issues
    that lack the keyword signal (size N2; estimated by the ``signal_dropped``
    sample restricted to pool survivors). Weighted mean + normal-approx 95% CI.

While labels are blank the macros expand to ``\\todo{pending}``.

Label vocabulary (case-insensitive; see ``docs/audit_guideline.md``):
  requirement     -> positive: ``r``, ``req``, ``requirement``, ``yes``, ``y``, ``1``
  not a requirement -> negative: ``n``, ``no``, ``non``, ``not``,
                       ``not_requirement``, ``nonrequirement``, ``0``
Anything else (blank, ``?``, ``unsure``) is treated as unlabelled and skipped.

Usage:
    uv run python scripts/audit_stats.py \
        [--sample docs/precision_audit_sample.csv] \
        [--filtered-dir ingestion/filtered] \
        [--config ingestion/config.yaml] \
        [--out docs/msr_paper/audit_macros.tex]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

_POS = {"r", "req", "requirement", "yes", "y", "1", "true", "pos", "positive"}
_NEG = {
    "n", "no", "non", "not", "not_requirement", "not-requirement",
    "nonrequirement", "non_requirement", "0", "false", "neg", "negative",
}
_Z = 1.959963984540054  # 95% two-sided normal quantile


def _label(raw: str) -> bool | None:
    """Map a raw cell to True (requirement) / False (not) / None (unlabelled)."""

    key = (raw or "").strip().lower()
    if key in _POS:
        return True
    if key in _NEG:
        return False
    return None


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""

    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    z2 = _Z * _Z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two binary raters over paired labels."""

    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    p1_yes = sum(1 for a, _ in pairs if a) / n
    p2_yes = sum(1 for _, b in pairs if b) / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def _consensus(r1: bool | None, r2: bool | None) -> bool | None:
    """Agreed label when both raters labelled and agree; else None."""

    if r1 is None or r2 is None or r1 != r2:
        return None
    return r1


def _primary(r1: bool | None, r2: bool | None, adj: bool | None) -> bool | None:
    """Prefer adjudication; else consensus of the two raters."""

    if adj is not None:
        return adj
    return _consensus(r1, r2)


def _counts_with(
    group: list[tuple[bool | None, bool | None, bool | None]],
    *,
    mode: str,
) -> tuple[int, int]:
    """Return (positives, n) for consensus or primary labelling mode."""

    pos = n = 0
    for r1, r2, adj in group:
        lab = _consensus(r1, r2) if mode == "consensus" else _primary(r1, r2, adj)
        if lab is None:
            continue
        n += 1
        pos += 1 if lab else 0
    return pos, n


def _pool_strata(
    filtered_dir: str, config_path: str | None
) -> tuple[int, int, set[tuple[str, int]], set[tuple[str, int]]]:
    """Partition the released pool by keyword-signal presence.

    Returns (N1, N2, keys_signal, keys_no_signal).
    """

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
    from _common import load_config  # noqa: WPS433
    from _textproc import normalize_issue_body  # noqa: WPS433
    from filter_issues import _has_requirement_signal  # noqa: WPS433

    cfg = load_config(config_path)["issue_filtering"]
    kws = cfg["requirement_keywords"]
    keys_sig: set[tuple[str, int]] = set()
    keys_nosig: set[tuple[str, int]] = set()
    for path in sorted(Path(filtered_dir).glob("*_filtered.json")):
        data = json.loads(path.read_text())
        repo = data["repo"]
        for issue in data.get("kept_issues", []):
            key = (repo, int(issue["number"]))
            body = issue.get("_normalized_body") or normalize_issue_body(
                issue.get("body")
            )
            if _has_requirement_signal(body, kws):
                keys_sig.add(key)
            else:
                keys_nosig.add(key)
    return len(keys_sig), len(keys_nosig), keys_sig, keys_nosig


def _stratified_ci(
    p1: float, n1: int, N1: int, p2: float, n2: int, N2: int
) -> tuple[float, float, float]:
    """Stratified mean precision and normal-approx 95% CI.

    P = (N1/N)*p1 + (N2/N)*p2
    Var = (N1/N)^2 * p1(1-p1)/n1 + (N2/N)^2 * p2(1-p2)/n2
    """

    N = N1 + N2
    if N == 0 or n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"))
    w1, w2 = N1 / N, N2 / N
    p = w1 * p1 + w2 * p2
    var = (w1**2) * p1 * (1.0 - p1) / n1 + (w2**2) * p2 * (1.0 - p2) / n2
    half = _Z * math.sqrt(var)
    return (p, max(0.0, p - half), min(1.0, p + half))


def compute(
    sample_path: str,
    filtered_dir: str = "ingestion/filtered",
    config_path: str | None = None,
) -> dict[str, object]:
    """Load the labelled sheet and compute audit statistics."""

    with open(sample_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    retained: list[tuple[bool | None, bool | None, bool | None, dict]] = []
    rejected: list[tuple[bool | None, bool | None, bool | None, dict]] = []
    for row in rows:
        r1 = _label(row.get("label_rater1", ""))
        r2 = _label(row.get("label_rater2", ""))
        adj = _label(row.get("adjudicated", ""))
        item = (r1, r2, adj, row)
        if row.get("stratum") == "retained":
            retained.append(item)
        else:
            rejected.append(item)

    both = [
        (a, b)
        for a, b, _, _ in (retained + rejected)
        if a is not None and b is not None
    ]
    disagree = sum(1 for a, b in both if a != b)
    kappa = _cohen_kappa(both)
    n_adjudicated = sum(
        1 for *_, adj, __ in (retained + rejected) if adj is not None
    )

    ret_c_pos, ret_c_n = _counts_with(
        [(a, b, adj) for a, b, adj, _ in retained], mode="consensus"
    )
    rej_c_pos, rej_c_n = _counts_with(
        [(a, b, adj) for a, b, adj, _ in rejected], mode="consensus"
    )
    ret_p_pos, ret_p_n = _counts_with(
        [(a, b, adj) for a, b, adj, _ in retained], mode="primary"
    )
    rej_p_pos, rej_p_n = _counts_with(
        [(a, b, adj) for a, b, adj, _ in rejected], mode="primary"
    )

    N1, N2, _keys_sig, keys_nosig = _pool_strata(filtered_dir, config_path)
    # Stratum-2 sample: signal_dropped rows that survive in the new pool.
    rej_surv = [
        (a, b, adj)
        for a, b, adj, row in rejected
        if (row["repo"], int(row["number"])) in keys_nosig
    ]
    # If pool tagging failed to include a sample row (should not happen when
    # signal is OFF), fall back to all rejected rows for n2.
    if not rej_surv:
        rej_surv = [(a, b, adj) for a, b, adj, _ in rejected]

    s2_c_pos, s2_c_n = _counts_with(rej_surv, mode="consensus")
    s2_p_pos, s2_p_n = _counts_with(rej_surv, mode="primary")

    def _strat_from(pos1: int, n1: int, pos2: int, n2: int) -> dict[str, float | int]:
        p1 = pos1 / n1 if n1 else float("nan")
        p2 = pos2 / n2 if n2 else float("nan")
        p, lo, hi = _stratified_ci(p1, n1, N1, p2, n2, N2)
        return {
            "p1": p1,
            "n1": n1,
            "pos1": pos1,
            "p2": p2,
            "n2": n2,
            "pos2": pos2,
            "p": p,
            "lo": lo,
            "hi": hi,
            "N1": N1,
            "N2": N2,
            "N": N1 + N2,
        }

    labelled = ret_c_n + rej_c_n > 0 or ret_p_n + rej_p_n > 0

    return {
        "sample_retained": len(retained),
        "sample_rejected": len(rejected),
        "kappa": kappa,
        "disagree": disagree,
        "both_n": len(both),
        "n_adjudicated": n_adjudicated,
        "labelled": labelled,
        # Consensus-only (disagreements excluded).
        "ret_c_pos": ret_c_pos,
        "ret_c_n": ret_c_n,
        "rej_c_pos": rej_c_pos,
        "rej_c_n": rej_c_n,
        # Primary (adjudicated preferred).
        "ret_p_pos": ret_p_pos,
        "ret_p_n": ret_p_n,
        "rej_p_pos": rej_p_pos,
        "rej_p_n": rej_p_n,
        "strat_consensus": _strat_from(ret_c_pos, ret_c_n, s2_c_pos, s2_c_n),
        "strat_primary": _strat_from(ret_p_pos, ret_p_n, s2_p_pos, s2_p_n),
        "s2_survivors_sample": len(rej_surv),
    }


def render_macros(stats: dict[str, object]) -> str:
    """Render the \\newcommand block; \\todo{pending} until labelling is done."""

    todo = r"\todo{pending}"
    lines = [
        "% Auto-generated by scripts/audit_stats.py. Do not edit by hand.",
        "% Re-run `uv run python scripts/audit_stats.py` after annotation/adjudication.",
        rf"\newcommand{{\auditSampleRetained}}{{{stats['sample_retained']}}}",
        rf"\newcommand{{\auditSampleRejected}}{{{stats['sample_rejected']}}}",
        rf"\newcommand{{\auditPoolN}}{{{stats['strat_primary']['N']}}}",  # type: ignore[index]
        rf"\newcommand{{\auditPoolNOne}}{{{stats['strat_primary']['N1']}}}",  # type: ignore[index]
        rf"\newcommand{{\auditPoolNTwo}}{{{stats['strat_primary']['N2']}}}",  # type: ignore[index]
    ]

    if not stats["labelled"]:
        for name in (
            "auditPrec", "auditPrecCI", "auditFOR", "auditFORCI",
            "auditPrecConsensus", "auditPrecConsensusCI",
            "auditFORConsensus", "auditFORConsensusCI",
            "auditPoolPrec", "auditPoolPrecCI",
            "auditKappa", "auditDisagree",
            "auditRetainedN", "auditRejectedN",
            "auditRetainedNConsensus", "auditRejectedNConsensus",
        ):
            lines.append(rf"\newcommand{{\{name}}}{{{todo}}}")
        return "\n".join(lines) + "\n"

    def _add_rate(prefix: str, pos: int, n: int) -> None:
        p, lo, hi = _wilson(pos, n)
        lines.append(rf"\newcommand{{\{prefix}}}{{{_pct(p)}\%}}")
        lines.append(
            rf"\newcommand{{\{prefix}CI}}{{[{_pct(lo)}, {_pct(hi)}]\%}}"
        )

    # Primary (paper body): adjudicated when present, else consensus.
    lines.append(rf"\newcommand{{\auditRetainedN}}{{{stats['ret_p_n']}}}")
    lines.append(rf"\newcommand{{\auditRejectedN}}{{{stats['rej_p_n']}}}")
    _add_rate("auditPrec", int(stats["ret_p_pos"]), int(stats["ret_p_n"]))
    _add_rate("auditFOR", int(stats["rej_p_pos"]), int(stats["rej_p_n"]))

    # Consensus-only (footnote / transparency).
    lines.append(
        rf"\newcommand{{\auditRetainedNConsensus}}{{{stats['ret_c_n']}}}"
    )
    lines.append(
        rf"\newcommand{{\auditRejectedNConsensus}}{{{stats['rej_c_n']}}}"
    )
    _add_rate(
        "auditPrecConsensus", int(stats["ret_c_pos"]), int(stats["ret_c_n"])
    )
    _add_rate(
        "auditFORConsensus", int(stats["rej_c_pos"]), int(stats["rej_c_n"])
    )

    # Backward-compatible aliases (old FNR name -> FOR).
    lines.append(r"\newcommand{\auditFNR}{\auditFOR}")
    lines.append(r"\newcommand{\auditFNRCI}{\auditFORCI}")

    strat = stats["strat_primary"]  # type: ignore[assignment]
    pool_prec = _pct(float(strat["p"]))  # type: ignore[index]
    pool_lo = _pct(float(strat["lo"]))  # type: ignore[index]
    pool_hi = _pct(float(strat["hi"]))  # type: ignore[index]
    lines.append(rf"\newcommand{{\auditPoolPrec}}{{{pool_prec}\%}}")
    lines.append(
        rf"\newcommand{{\auditPoolPrecCI}}{{[{pool_lo}, {pool_hi}]\%}}"
    )

    lines.append(rf"\newcommand{{\auditDisagree}}{{{stats['disagree']}}}")
    if stats["kappa"] is None:
        lines.append(rf"\newcommand{{\auditKappa}}{{{todo}}}")
    else:
        lines.append(
            rf"\newcommand{{\auditKappa}}{{{float(stats['kappa']):.2f}}}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="docs/precision_audit_sample.csv")
    parser.add_argument("--filtered-dir", default="ingestion/filtered")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="docs/msr_paper/audit_macros.tex")
    args = parser.parse_args()

    stats = compute(args.sample, args.filtered_dir, args.config)
    text = render_macros(stats)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text)

    if not stats["labelled"]:
        print(
            f"No labels found in {args.sample}; wrote pending macros to "
            f"{args.out}."
        )
        return

    sc = stats["strat_consensus"]
    sp = stats["strat_primary"]
    print(
        f"Wrote audit macros to {args.out}\n"
        f"  consensus retained prec {stats['ret_c_pos']}/{stats['ret_c_n']}; "
        f"FOR {stats['rej_c_pos']}/{stats['rej_c_n']}\n"
        f"  primary   retained prec {stats['ret_p_pos']}/{stats['ret_p_n']}; "
        f"FOR {stats['rej_p_pos']}/{stats['rej_p_n']} "
        f"(adjudicated cells={stats['n_adjudicated']})\n"
        f"  kappa={stats['kappa']}, disagreements={stats['disagree']}/"
        f"{stats['both_n']}\n"
        f"  stratified pool (primary): N={sp['N']} N1={sp['N1']} N2={sp['N2']}; "
        f"p1={sp['pos1']}/{sp['n1']} p2={sp['pos2']}/{sp['n2']}; "
        f"P={100*float(sp['p']):.1f}% CI=[{100*float(sp['lo']):.1f}, "
        f"{100*float(sp['hi']):.1f}]%\n"
        f"  stratified arithmetic: "
        f"P=(N1/N)*p1+(N2/N)*p2 = "
        f"({sp['N1']}/{sp['N']})*{float(sp['p1']):.4f} + "
        f"({sp['N2']}/{sp['N']})*{float(sp['p2']):.4f}\n"
        f"  consensus stratified P={100*float(sc['p']):.1f}% "
        f"[{100*float(sc['lo']):.1f}, {100*float(sc['hi']):.1f}]%"
    )


if __name__ == "__main__":
    main()

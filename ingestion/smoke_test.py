"""Step 8 (verification): smoke-test one mined case across all three adapters.

Runs a single mined case through MARE, iReDev, and QUARE under the cheapest
setting (single_agent, fixed seed) and records runtime, token estimates, and
estimated cost. This is a smoke test (does it run end to end?), not a full
matrix run.

Token counts: the harness run records do not store provider token usage, so
input tokens are estimated from the requirement text and output tokens from the
generated phase artifacts, using a simple ~4-chars-per-token heuristic (or
tiktoken if available). Costs use gpt-4o-mini pricing and are clearly labelled
as estimates.

Usage:
    uv run python ingestion/smoke_test.py --case <case_name> \
        [--model gpt-4o-mini-2024-07-18] [--setting single_agent]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _common import LOGGER
from _common import read_json
from _common import resolve_path
from _common import utc_timestamp
from _common import write_json

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ("mare", "iredev", "quare")

# gpt-4o-mini-2024-07-18 pricing (USD per 1M tokens) at packaging time.
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60


def _estimate_tokens(text: str) -> int:
    """Estimate token count; use tiktoken if present, else ~4 chars/token."""

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001 - tiktoken optional
        return max(1, len(text) // 4)


def _artifact_text(artifacts_dir: Path) -> str:
    """Concatenate all generated phase JSON as a proxy for output tokens."""

    chunks: list[str] = []
    for path in sorted(artifacts_dir.glob("phase*.json")):
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def run_one(case_path: Path, system: str, setting: str, seed: int, model: str) -> dict:
    """Run one adapter on the case and collect timing/token/cost estimates."""

    out_dir = REPO_ROOT / "artifacts" / "smoke" / f"{case_path.stem}-{system}"
    run_record = out_dir / "run_record.json"
    cmd = [
        "uv",
        "run",
        "openre_bench",
        "--run-case",
        "--case-input",
        str(case_path),
        "--artifacts-dir",
        str(out_dir),
        "--run-record",
        str(run_record),
        "--system",
        system,
        "--setting",
        setting,
        "--seed",
        str(seed),
    ]
    LOGGER.info("Running %s on %s ...", system, case_path.name)
    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    ok = completed.returncode == 0 and run_record.exists()
    if not ok:
        LOGGER.error("%s FAILED (exit %d):\n%s", system, completed.returncode, completed.stderr[-1500:])

    case_payload = read_json(case_path) if case_path.exists() else {}
    input_tokens = _estimate_tokens(case_payload.get("requirement", ""))
    output_tokens = _estimate_tokens(_artifact_text(out_dir)) if ok else 0
    cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_M + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M

    validation_passed = None
    if run_record.exists():
        try:
            validation_passed = read_json(run_record).get("validation_passed")
        except Exception:  # noqa: BLE001
            validation_passed = None

    return {
        "system": system,
        "setting": setting,
        "seed": seed,
        "model": model,
        "ok": ok,
        "runtime_seconds": round(elapsed, 3),
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "cost_usd_est": round(cost, 6),
        "validation_passed": validation_passed,
        "artifacts_dir": str(out_dir.relative_to(REPO_ROOT)),
    }


def _write_report(results: list[dict], case_name: str, model: str) -> Path:
    report_dir = REPO_ROOT / "artifacts" / "smoke"
    report_dir.mkdir(parents=True, exist_ok=True)
    md = report_dir / "cost_report.md"

    lines = [
        f"# Smoke Test Report: {case_name}\n",
        f"Generated: {utc_timestamp()}  |  Model (label): `{model}`\n",
        "Token counts are estimates (see script docstring); costs use "
        "gpt-4o-mini pricing and are illustrative, not billed values.\n",
        "| System | Setting | OK | Runtime (s) | Input tok (est) | Output tok (est) | "
        "Cost USD (est) | Validation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['system']} | {row['setting']} | {'✅' if row['ok'] else '❌'} | "
            f"{row['runtime_seconds']} | {row['input_tokens_est']} | {row['output_tokens_est']} | "
            f"{row['cost_usd_est']} | {row['validation_passed']} |"
        )
    total_cost = sum(r["cost_usd_est"] for r in results)
    total_time = sum(r["runtime_seconds"] for r in results)
    passed = sum(1 for r in results if r["ok"])
    lines.append(
        f"\n**Summary:** {passed}/{len(results)} adapters ran end to end. "
        f"Total est. cost ${total_cost:.6f}, total runtime {total_time:.1f}s."
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(report_dir / "cost_report.json", {
        "case_name": case_name,
        "model": model,
        "generated_at": utc_timestamp(),
        "results": results,
    })
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one mined case across adapters.")
    parser.add_argument("--case", required=True, help="Case name (without _input.json).")
    parser.add_argument("--cases-dir", default="data/case_studies")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--setting", default="single_agent")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    case_path = resolve_path(args.cases_dir) / f"{args.case}_input.json"
    if not case_path.exists():
        raise SystemExit(f"Case not found: {case_path}")

    results = [
        run_one(case_path, system, args.setting, args.seed, args.model) for system in SYSTEMS
    ]
    report = _write_report(results, args.case, args.model)
    LOGGER.info("Wrote smoke report: %s", report)

    if not all(r["ok"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

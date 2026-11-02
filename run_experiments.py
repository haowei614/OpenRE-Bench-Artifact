#!/usr/bin/env python3
"""Run the multi-agent comparison matrix on the mined cases.

Thin wrapper around ``openre_bench --run-comparison-matrix`` that:
  * enforces socket-level httpx timeouts (litellm's ``timeout`` is ignored by
    Python's ssl.recv() in some environments), mirroring ``run_matrix.py``;
  * sets ``litellm.drop_params=True`` so provider-unsupported parameters (e.g.
    Anthropic has no ``seed``) are dropped instead of raising;
  * defaults to the mined-case input set and a low-cost model.

Any argument passed on the command line overrides the matching default
(argparse keeps the last occurrence), e.g.::

    # full run: 5 mined cases x 3 systems x 3 seeds
    uv run python run_experiments.py --systems mare,iredev,quare

    # cheap pilot: one case, one system, one seed
    uv run python run_experiments.py --system quare \
        --cases-dir data/_pilot --matrix-seeds 101 \
        --output-dir artifacts/pilot_claude

The API key is read by the harness from ``.api_key`` / ``OPENAI_API_KEY``; its
value is passed straight to litellm, so an Anthropic ``sk-ant-...`` key works
when the model is a Claude model.
"""
from __future__ import annotations

import os
import sys

# Patch httpx default timeout before any OpenAI/litellm import.
os.environ.setdefault("HTTPX_TIMEOUT", "120")

import httpx  # noqa: E402

_TIMEOUT = httpx.Timeout(120.0, connect=15.0)
_orig_client_init = httpx.Client.__init__


def _patched_client_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs.setdefault("timeout", _TIMEOUT)
    _orig_client_init(self, *args, **kwargs)


httpx.Client.__init__ = _patched_client_init  # type: ignore[method-assign]

import litellm  # noqa: E402

litellm.request_timeout = 120
# Anthropic (and some others) reject params like ``seed``; drop rather than fail.
litellm.drop_params = True

# --- Token / cost accounting -------------------------------------------------
# run_record.json does not persist LLM usage, so we wrap litellm.completion to
# accumulate prompt/completion tokens and an estimated USD cost across the run.
_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
_orig_completion = litellm.completion


def _accounting_completion(*args, **kwargs):  # type: ignore[no-untyped-def]
    resp = _orig_completion(*args, **kwargs)
    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            _USAGE["calls"] += 1
            _USAGE["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            _USAGE["completion_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )
        try:  # litellm knows per-model pricing
            _USAGE["cost_usd"] += float(litellm.completion_cost(resp) or 0.0)
        except Exception:  # noqa: BLE001 - pricing table may lack a model
            pass
    except Exception:  # noqa: BLE001 - never let accounting break a run
        pass
    return resp


litellm.completion = _accounting_completion  # type: ignore[assignment]

import atexit  # noqa: E402
import json as _json  # noqa: E402


def _dump_usage() -> None:
    if _USAGE["calls"] == 0:
        return
    out_dir = None
    argv = sys.argv
    if "--output-dir" in argv:
        # Use the LAST occurrence: CLI overrides the built-in default (argparse
        # keeps the last, so must we).
        idx = len(argv) - 1 - argv[::-1].index("--output-dir")
        out_dir = argv[idx + 1]
    report = dict(_USAGE)
    print(
        f"\n[usage] calls={report['calls']} "
        f"prompt_tokens={report['prompt_tokens']} "
        f"completion_tokens={report['completion_tokens']} "
        f"est_cost_usd={report['cost_usd']:.4f}"
    )
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "usage_report.json"), "w") as fh:
                _json.dump(report, fh, indent=2)
        except Exception:  # noqa: BLE001
            pass


atexit.register(_dump_usage)
# -----------------------------------------------------------------------------

from openre_bench.cli import main  # noqa: E402

DEFAULTS = [
    "--run-comparison-matrix",
    "--cases-dir", "data/mined_cases",
    "--rag-corpus-dir", "data/knowledge_base",
    "--matrix-seeds", "101,202,303",
    "--model", "claude-haiku-4-5-20251001",
    "--temperature", "0.7",
    "--round-cap", "3",
    "--max-tokens", "4000",
    "--output-dir", "experiment_outputs/mined-claude",
]

if __name__ == "__main__":
    sys.argv = ["openre_bench"] + DEFAULTS + sys.argv[1:]
    raise SystemExit(main())

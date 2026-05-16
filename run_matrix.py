#!/usr/bin/env python3
"""Wrapper to run comparison matrices with reliable httpx timeouts.

The default litellm/openai client may hang on SSL socket reads because
Python's ssl.SSLSocket.recv() ignores litellm's ``timeout`` parameter
in some environments.  This wrapper patches httpx's default transport
to enforce a 60-second connect+read timeout at the socket level.
"""
from __future__ import annotations

import sys
import os

# Patch httpx default timeout before any OpenAI/litellm import
os.environ.setdefault("HTTPX_TIMEOUT", "60")

import httpx  # noqa: E402
_TIMEOUT = httpx.Timeout(60.0, connect=15.0)
_orig_client_init = httpx.Client.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs.setdefault("timeout", _TIMEOUT)
    _orig_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init

# Now import litellm and set its timeout too
import litellm  # noqa: E402
litellm.request_timeout = 60

from openre_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [
        "openre_bench",
        "--run-comparison-matrix",
        "--cases-dir", "data/case_studies",
        "--rag-corpus-dir", "data/knowledge_base",
        "--matrix-seeds", "101,202,303",
        "--model", "gpt-4o-mini",
        "--temperature", "0.7",
        "--round-cap", "3",
        "--max-tokens", "4000",
    ] + sys.argv[1:]  # allow --system and --output-dir from CLI
    raise SystemExit(main())

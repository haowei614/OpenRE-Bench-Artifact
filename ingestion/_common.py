"""Shared utilities for the OpenRE-Bench case-ingestion pipeline.

Contains configuration loading, structured logging, a thin GitHub REST client
built on the standard library (no third-party HTTP dependency), token handling
that never persists secrets, and deterministic JSON snapshot I/O.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Repo root is the parent of the ingestion/ directory.
INGESTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = INGESTION_DIR.parent
DEFAULT_CONFIG_PATH = INGESTION_DIR / "config.yaml"
ENV_LOCAL_PATH = INGESTION_DIR / ".env.local"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a configured stdout logger with a stable format."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOGGER = get_logger("ingestion")


# ---------------------------------------------------------------------------
# Timestamps and JSON I/O
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """Return a stable UTC timestamp string (matches the harness format)."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_stamp_compact() -> str:
    """Return a filesystem-safe UTC timestamp for snapshot filenames."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    """Load JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, pretty JSON to disk (sorted keys, UTF-8)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} did not parse to a mapping.")
    return config


def resolve_path(raw: str) -> Path:
    """Resolve a config path relative to the repo root when not absolute."""

    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


# ---------------------------------------------------------------------------
# Token handling (never persisted, never logged)
# ---------------------------------------------------------------------------


def _load_env_local() -> dict[str, str]:
    """Parse ingestion/.env.local (gitignored) into a dict, if it exists."""

    values: dict[str, str] = {}
    if not ENV_LOCAL_PATH.exists():
        return values
    for line in ENV_LOCAL_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_github_token(config: dict[str, Any]) -> str | None:
    """Read the GitHub token from the environment or ingestion/.env.local.

    The token is never written to snapshots, logs, or returned in any record.
    """

    env_name = config.get("github", {}).get("token_env", "GITHUB_TOKEN")
    token = os.environ.get(env_name)
    if token:
        return token.strip()
    token = _load_env_local().get(env_name)
    return token.strip() if token else None


def scrub_secret(text: str, token: str | None) -> str:
    """Redact a token if it ever appears in text destined for logs/snapshots."""

    if token and token in text:
        return text.replace(token, "***REDACTED***")
    return text


# ---------------------------------------------------------------------------
# GitHub REST client (stdlib urllib, with rate-limit + retry handling)
# ---------------------------------------------------------------------------


@dataclass
class GitHubClient:
    """Minimal authenticated GitHub REST client with rate-limit awareness."""

    token: str | None
    api_base: str = "https://api.github.com"
    rate_limit_delay: float = 0.8
    retry_limit: int = 4
    per_page: int = 100
    user_agent: str = "openre-bench-ingestion/0.1"
    logger: logging.Logger = LOGGER

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self.user_agent,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _sleep_for_rate_limit(self, response_headers: Any) -> None:
        """Sleep until reset when the primary rate-limit is exhausted."""

        remaining = response_headers.get("X-RateLimit-Remaining")
        reset = response_headers.get("X-RateLimit-Reset")
        if remaining is not None and reset is not None:
            try:
                remaining_int = int(remaining)
                reset_int = int(reset)
            except (TypeError, ValueError):
                return
            if remaining_int <= 0:
                wait = max(0.0, reset_int - time.time()) + 1.0
                self.logger.warning(
                    "Rate limit exhausted; sleeping %.0fs until reset.", wait
                )
                time.sleep(wait)

    def request(self, url: str, params: dict[str, Any] | None = None) -> tuple[Any, Any]:
        """Perform a GET request, returning (parsed_json, response_headers).

        Retries on transient errors and secondary rate limits with backoff.
        """

        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{url}?{query}" if "?" not in url else f"{url}&{query}"
        else:
            full_url = url

        last_error: Exception | None = None
        for attempt in range(self.retry_limit + 1):
            request = urllib.request.Request(full_url, headers=self._headers())
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read().decode("utf-8")
                    headers = response.headers
                    time.sleep(self.rate_limit_delay)
                    self._sleep_for_rate_limit(headers)
                    return json.loads(body) if body else None, headers
            except urllib.error.HTTPError as error:
                headers = error.headers
                # Secondary / primary rate limit -> wait and retry.
                if error.code in (403, 429):
                    retry_after = headers.get("Retry-After") if headers else None
                    if retry_after is not None:
                        wait = float(retry_after) + 1.0
                    else:
                        reset = headers.get("X-RateLimit-Reset") if headers else None
                        wait = (
                            max(0.0, int(reset) - time.time()) + 1.0
                            if reset is not None
                            else 2.0 ** attempt
                        )
                    self.logger.warning(
                        "HTTP %s on attempt %d; backing off %.0fs.",
                        error.code,
                        attempt + 1,
                        wait,
                    )
                    time.sleep(min(wait, 900.0))
                    last_error = error
                    continue
                if error.code >= 500:
                    wait = 2.0 ** attempt
                    self.logger.warning(
                        "HTTP %s on attempt %d; retrying in %.0fs.",
                        error.code,
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                    last_error = error
                    continue
                # Non-retryable (404, 401, 422, ...): surface immediately.
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"GitHub API error {error.code} for {full_url}: {detail[:300]}"
                ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                wait = 2.0 ** attempt
                self.logger.warning(
                    "Network error on attempt %d (%s); retrying in %.0fs.",
                    attempt + 1,
                    error,
                    wait,
                )
                time.sleep(wait)
                last_error = error
                continue

        raise RuntimeError(f"Exhausted retries for {full_url}: {last_error}")

    def get_repo(self, repo: str) -> dict[str, Any]:
        """Fetch repository metadata (stars, language, issue counts, license)."""

        payload, _ = self.request(f"{self.api_base}/repos/{repo}")
        return payload

    def get_search_issue_count(self, repo: str, query_suffix: str) -> int:
        """Return total_count for a repo-scoped issue search query."""

        query = f"repo:{repo} {query_suffix}"
        payload, _ = self.request(
            f"{self.api_base}/search/issues", params={"q": query, "per_page": 1}
        )
        return int(payload.get("total_count", 0)) if payload else 0

    def _iter_issues_single_label(
        self,
        repo: str,
        label: str | None,
        state: str,
        max_issues: int,
    ) -> list[dict[str, Any]]:
        """Page through /issues for one label (or all issues if label is None)."""

        collected: list[dict[str, Any]] = []
        page = 1
        while len(collected) < max_issues:
            params: dict[str, Any] = {
                "state": state,
                "per_page": self.per_page,
                "page": page,
                "sort": "created",
                # Most-recent first: cases should reflect current requirements
                # of active repos, not just a project's earliest issues.
                "direction": "desc",
            }
            if label:
                params["labels"] = label
            payload, _ = self.request(f"{self.api_base}/repos/{repo}/issues", params=params)
            if not payload:
                break
            page_issues = [item for item in payload if "pull_request" not in item]
            collected.extend(page_issues)
            self.logger.info(
                "  %s [label=%s]: page %d (%d issues, %d cumulative)",
                repo,
                label or "*",
                page,
                len(page_issues),
                len(collected),
            )
            if len(payload) < self.per_page:
                break
            page += 1
        return collected[:max_issues]

    def iter_issues(
        self,
        repo: str,
        labels: list[str],
        state: str = "all",
        max_issues: int = 600,
    ) -> list[dict[str, Any]]:
        """Fetch issues (excluding PRs) for a repo, matching ANY of `labels`.

        The GitHub /issues `labels` parameter is AND (an issue must carry every
        listed label). To get OR semantics across the configured labels, each
        label is queried separately and the results are unioned (deduplicated
        by issue number, then sorted by number for deterministic ordering).
        Pull requests are removed (they carry a `pull_request` key).
        """

        if not labels:
            return self._iter_issues_single_label(repo, None, state, max_issues)

        by_number: dict[int, dict[str, Any]] = {}
        for label in labels:
            for issue in self._iter_issues_single_label(repo, label, state, max_issues):
                number = issue.get("number")
                if number is not None and number not in by_number:
                    by_number[number] = issue
        merged = sorted(by_number.values(), key=lambda i: int(i.get("number", 0)))
        return merged[:max_issues]


def build_client(config: dict[str, Any]) -> GitHubClient:
    """Construct a GitHubClient from config and the environment token."""

    github_cfg = config.get("github", {})
    token = get_github_token(config)
    if not token:
        LOGGER.warning(
            "No GitHub token found (env %s or ingestion/.env.local). "
            "Unauthenticated access is limited to 60 requests/hour and will "
            "likely fail. See ingestion/README.md.",
            github_cfg.get("token_env", "GITHUB_TOKEN"),
        )
    return GitHubClient(
        token=token,
        api_base=github_cfg.get("api_base", "https://api.github.com"),
        rate_limit_delay=float(github_cfg.get("rate_limit_delay", 0.8)),
        retry_limit=int(github_cfg.get("retry_limit", 4)),
        per_page=int(github_cfg.get("per_page", 100)),
        user_agent=github_cfg.get("user_agent", "openre-bench-ingestion/0.1"),
    )

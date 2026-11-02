"""Text normalization and deduplication helpers for ingestion.

Deterministic given a fixed seed; used by the filter and aggregate steps.
"""

from __future__ import annotations

import re

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")

# GitHub @-mention: "@handle" (1-39 chars, alnum with single internal hyphens),
# not preceded by a word char, "/", "@", or "." so emails and paths are skipped.
_MENTION = re.compile(r"(?<![\w/@.])@([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})")

# Tokens that look like @-mentions but are code/markup (Javadoc tags, decorators,
# CSS at-rules, Elasticsearch metadata fields), not GitHub usernames. Kept verbatim
# so technical requirement text is not corrupted by anonymization.
MENTION_ALLOWLIST = frozenset(
    {
        "timestamp", "param", "params", "parameter", "return", "returns",
        "throws", "see", "link", "code", "author", "since", "deprecated",
        "override", "media", "keyframes", "import", "charset", "namespace",
        "property", "staticmethod", "classmethod", "dataclass",
        "cached_property", "pytest", "fixture", "patch", "mock",
    }
)


def anonymize_mentions(
    text: str | None,
    allowlist: frozenset[str] = MENTION_ALLOWLIST,
) -> str:
    """Replace GitHub @-mentions of users with ``@USER`` (de-identification).

    Applies everywhere in the text, including inside code spans, so author
    handles are not redistributed. Tokens in ``allowlist`` (case-insensitive,
    e.g. ``@timestamp``, ``@param``) are code/markup and left verbatim.
    Deterministic and idempotent (``@USER`` is itself not re-matched by name).
    """

    if not text:
        return text or ""

    def _sub(match: re.Match[str]) -> str:
        handle = match.group(1)
        if handle.lower() in allowlist:
            return match.group(0)
        return "@USER"

    return _MENTION.sub(_sub, text)


def normalize_issue_body(text: str | None) -> str:
    """Strip issue-template boilerplate, code fences, markup, and collapse space.

    Produces clean natural-language requirement text suitable for RE input.
    """

    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _HTML_COMMENT.sub(" ", cleaned)      # template instructions
    cleaned = _FENCED_CODE.sub(" ", cleaned)       # code blocks (noise for NL RE)
    cleaned = _MD_IMAGE.sub(" ", cleaned)          # images/badges
    cleaned = _MD_LINK.sub(r"\1", cleaned)         # keep link text, drop target
    cleaned = _URL.sub(" ", cleaned)               # bare URLs
    cleaned = _HTML_TAG.sub(" ", cleaned)          # residual HTML
    cleaned = cleaned.replace("#", "").replace("*", "").replace("`", "")
    cleaned = _MULTISPACE.sub(" ", cleaned)
    cleaned = _MULTINEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""

    return len(text.split())


def truncate_words(text: str, max_words: int) -> str:
    """Truncate to the first `max_words` words, appending an ellipsis if cut."""

    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def deduplicate(
    texts: list[str],
    threshold: float,
    seed: int,
) -> list[int]:
    """Return indices of texts to KEEP after near-duplicate removal.

    Greedy: iterate in input order, keep an item unless it is >= `threshold`
    cosine-similar (TF-IDF) to an already-kept item. Deterministic.
    """

    if len(texts) <= 1:
        return list(range(len(texts)))

    # Local import keeps the module importable without sklearn for pure-text use.
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    np.random.seed(seed)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Empty vocabulary (all stop-words/empty) -> keep everything.
        return list(range(len(texts)))

    similarity = cosine_similarity(matrix)
    kept: list[int] = []
    for i in range(len(texts)):
        is_duplicate = any(similarity[i, j] >= threshold for j in kept)
        if not is_duplicate:
            kept.append(i)
    return kept

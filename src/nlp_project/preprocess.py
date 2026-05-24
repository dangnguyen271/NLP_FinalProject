"""Text cleaning and segmentation for news articles."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from bs4 import BeautifulSoup


_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*(share this article|read more|advertisement|sponsored)\s*[:\-]?\s*",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bclick here to .*?(\.|$)", re.IGNORECASE),
    re.compile(r"\bcopyright\s+©.*?(\.|$)", re.IGNORECASE),
)


def strip_html(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if "<" not in text:
        return text
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def normalize_article(value: object) -> str:
    """Deterministic normalisation suitable for summarisation models.

    Unlike a classification preprocessor, this preserves casing and punctuation
    because both are signals the summariser uses for fluency. We only:
      - strip Unicode control characters
      - normalise unicode (NFKC) so curly quotes and ligatures behave predictably
      - drop common boilerplate phrases
      - collapse repeated whitespace
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _CONTROL_RE.sub(" ", text)
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Lightweight regex sentence splitter.

    Avoids the NLTK punkt download in CI/offline environments. Handles common
    abbreviations (Mr., Mrs., Dr., U.S., e.g., i.e.) by joining the trailing
    period back into the preceding token before splitting.
    """

    if not text:
        return []
    masked = text
    abbreviations = (
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
        "St.", "U.S.", "U.K.", "e.g.", "i.e.", "etc.", "vs.", "No.",
    )
    for abbr in abbreviations:
        masked = masked.replace(abbr, abbr.replace(".", "<<DOT>>"))
    # Split on sentence terminators followed by whitespace and a capital letter
    # OR the end of the string.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", masked)
    return [part.replace("<<DOT>>", ".").strip() for part in parts if part.strip()]


def deduplicate(rows: Iterable[dict]) -> list[dict]:
    """Drop exact-article duplicates while preserving order."""

    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        key = (row.get("article") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped

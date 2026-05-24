from __future__ import annotations

from collections.abc import Iterable
import re
import string
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_text(
    value: object,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = False,
) -> str:
    """Normalize text deterministically without external resources."""

    if value is None:
        return ""
    text = str(value)
    text = "".join(" " if unicodedata.category(char)[0] == "C" else char for char in text)
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(_PUNCT_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", text).strip()


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer wrapping the project text normalizer."""

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def fit(self, texts: Iterable[object], y: object = None) -> "TextNormalizer":
        return self

    def transform(self, texts: Iterable[object]) -> list[str]:
        return [
            normalize_text(
                text,
                lowercase=self.lowercase,
                remove_punctuation=self.remove_punctuation,
            )
            for text in texts
        ]

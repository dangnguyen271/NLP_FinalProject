from __future__ import annotations

from nlp_project.preprocess import (
    deduplicate,
    normalize_article,
    split_sentences,
    strip_html,
)


def test_strip_html_removes_tags_and_scripts():
    html = "<p>Hello <script>alert(1)</script><b>world</b></p>"
    text = strip_html(html)
    assert "alert" not in text
    assert "Hello" in text
    assert "world" in text
    assert "<" not in text


def test_normalize_preserves_case_and_punctuation():
    text = normalize_article("Hello,  World!  How are you?")
    assert text == "Hello, World! How are you?"


def test_normalize_drops_boilerplate():
    text = normalize_article("Share this article: a quick note. Click here to subscribe.")
    assert "share this article" not in text.lower()


def test_split_sentences_handles_common_abbreviations():
    sentences = split_sentences(
        "Dr. Smith works at the U.S. office. He flew to Tokyo last week."
    )
    assert len(sentences) == 2
    assert sentences[0].startswith("Dr. Smith")


def test_deduplicate_drops_exact_articles():
    rows = [
        {"article": "Repeated story.", "highlights": "a"},
        {"article": "repeated STORY.", "highlights": "b"},
        {"article": "Different story.", "highlights": "c"},
    ]
    deduped = deduplicate(rows)
    assert len(deduped) == 2

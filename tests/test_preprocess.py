from __future__ import annotations

from nlp_project.preprocess import normalize_text


def test_trims_and_lowercases_text():
    assert normalize_text("  Helpful   TEXT  ") == "helpful text"


def test_removes_control_characters():
    assert normalize_text("Hello\x00there\nOK") == "hello there ok"


def test_keeps_punctuation_by_default():
    assert normalize_text("Useful, clear!") == "useful, clear!"


def test_can_remove_punctuation_when_requested():
    assert normalize_text("Useful, clear!", remove_punctuation=True) == "useful clear"


def test_handles_empty_and_none_safely():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""

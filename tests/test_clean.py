"""Financial-notation cleaning (section 4).

Naive cleaning corrupts the notation these tests pin down, and the corruption is
invisible downstream: ``(1,234)`` becoming ``1,234`` produces a report where a
loss reads as a profit.
"""

from __future__ import annotations

import pytest

from affa.ingestion.clean import (
    NOTATION_PROBES,
    clean_text,
    dehyphenate,
    normalize_unicode,
    normalize_whitespace,
)


@pytest.mark.parametrize("probe", NOTATION_PROBES)
def test_financial_notation_survives_cleaning(probe: str) -> None:
    cleaned = clean_text(f"The statement reads {probe} for the period.")
    assert probe in cleaned, f"cleaning destroyed {probe!r}: {cleaned!r}"


def test_parenthesised_negative_keeps_its_parentheses() -> None:
    # The parenthesis IS the minus sign in a filing.
    assert "(1,234)" in clean_text("Net loss of (1,234) for the year.")
    assert "$(2,345.6)" in clean_text("Operating loss $(2,345.6) in the segment.")


def test_ligatures_are_folded() -> None:
    assert clean_text("The ﬁnancial eﬀect was signiﬁcant.") == (
        "The financial effect was significant."
    )


def test_smart_quotes_and_dashes_become_ascii() -> None:
    out = clean_text("“Revenue” rose 4–5% — the company’s best year.")
    assert '"Revenue"' in out
    assert "company's" in out
    assert "–" not in out and "—" not in out


def test_unicode_minus_becomes_parsable_hyphen() -> None:
    # U+2212 breaks float(); filings use it for negatives.
    assert normalize_unicode("−3.4") == "-3.4"


def test_nfkc_is_not_used_so_footnote_markers_stay_separate() -> None:
    """Superscript footnote markers must not fuse into the adjacent figure.

    NFKC folds superscript one to "1", turning "Revenue<sup>1</sup> 1,234" into
    "Revenue1 1,234". NFC plus the explicit tables avoids that.
    """
    assert "¹" in normalize_unicode("Revenue¹ 1,234")


def test_dehyphenation_joins_broken_words_only() -> None:
    assert dehyphenate("total reve-\nnue was") == "total revenue was"
    # A real compound must keep its hyphen.
    assert dehyphenate("non-\nGAAP measures") == "non-\nGAAP measures"
    assert "10-K" in clean_text("this 10-K report")


def test_layout_preserved_for_tables() -> None:
    """Column alignment is the only thing binding a label to its figure."""
    table = "Revenue      1,234    5,678\nCost          234      567"
    preserved = normalize_whitespace(table, preserve_layout=True)
    assert "      " in preserved
    collapsed = normalize_whitespace(table, preserve_layout=False)
    assert "      " not in collapsed


def test_page_furniture_removed() -> None:
    assert "Page 12 of 340" not in clean_text("Some text.\nPage 12 of 340\nMore text.")


def test_empty_input_is_safe() -> None:
    assert clean_text("") == ""

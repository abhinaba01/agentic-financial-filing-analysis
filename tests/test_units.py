"""Number parsing, scale and percentage conventions (section 6)."""

from __future__ import annotations

import pytest

from affa.kpi.units import (
    MatchOutcome,
    PercentConvention,
    compare_values,
    detect_scale,
    normalize_percent,
    normalize_scale,
    parse_financial_number,
)
from affa.schema import Scale


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.5", 1234.5),
        ("1,234", 1234.0),
        ("(1,234)", -1234.0),  # parenthesised negative
        ("$(2,345.6)", -2345.6),  # currency inside the parenthesis
        ("(0.15)", -0.15),  # negative EPS
        ("-3.4", -3.4),
        ("45%", 45.0),
        ("1.2x", 1.2),
        ("250 bps", 2.5),  # basis points normalised to percent
    ],
)
def test_parse_financial_number(raw: str, expected: float) -> None:
    parsed = parse_financial_number(raw)
    assert parsed is not None, f"failed to parse {raw!r}"
    assert parsed.value == pytest.approx(expected)


@pytest.mark.parametrize("nil", ["-", "--", "n/a", "N/A", "nil", "", "—"])
def test_nil_markers_are_not_zero(nil: str) -> None:
    """A dash means "no value", not "zero". Parsing it as 0.0 invents data."""
    assert parse_financial_number(nil) is None


def test_footnote_marker_is_not_read_as_the_figure() -> None:
    """Regression: "(see Note 3) 1,234" parsed as -3.0.

    A parenthesised footnote reference was read as a parenthesised negative,
    producing a wrong value with a flipped sign.
    """
    parsed = parse_financial_number("(see Note 3) 1,234")
    assert parsed is not None
    assert parsed.value == 1234.0

    parsed = parse_financial_number("Note 3 (1,234)")
    assert parsed is not None
    assert parsed.value == -1234.0


def test_percent_and_multiple_flags() -> None:
    assert parse_financial_number("45%").is_percent
    assert parse_financial_number("1.2x").is_multiple
    assert parse_financial_number("$1,234").is_currency


def test_inline_scale_word_is_captured() -> None:
    parsed = parse_financial_number("$1,234.5 million")
    assert parsed.scale_hint is Scale.MILLIONS
    assert parsed.in_units() == pytest.approx(1_234_500_000.0)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("(In millions, except per share data)", Scale.MILLIONS),
        ("amounts in thousands", Scale.THOUSANDS),
        ("$ in bn", Scale.BILLIONS),
        ("no scale stated here", Scale.UNITS),
    ],
)
def test_detect_scale(header: str, expected: Scale) -> None:
    assert detect_scale(header) is expected


def test_normalize_scale_round_trip() -> None:
    assert normalize_scale(383285.0, Scale.MILLIONS) == pytest.approx(383_285_000_000.0)
    assert normalize_scale(1.0, Scale.BILLIONS, Scale.MILLIONS) == pytest.approx(1000.0)


def test_declared_percent_convention_is_exact() -> None:
    frac = normalize_percent(0.42, convention=PercentConvention.FRACTION)
    assert frac.value == pytest.approx(42.0)
    assert frac.converted and not frac.ambiguous

    pts = normalize_percent(42.0, convention=PercentConvention.POINTS)
    assert pts.value == pytest.approx(42.0)
    assert not pts.converted


def test_undeclared_percent_inside_band_is_flagged_ambiguous() -> None:
    """0.42 is genuinely undecidable, and the result says so rather than guessing quietly."""
    result = normalize_percent(0.42)
    assert result.ambiguous is True
    assert result.value == pytest.approx(42.0)

    # Outside the band there is no ambiguity.
    assert normalize_percent(42.0).ambiguous is False


def test_compare_values_separates_unit_errors_from_real_misses() -> None:
    """A scale mismatch and a wrong answer need different fixes, so they are different outcomes."""
    scale_miss = compare_values(383285.0, 383_285_000_000.0)
    assert scale_miss.outcome is MatchOutcome.UNIT_ERROR

    sign_miss = compare_values(1234.0, -1234.0)
    assert sign_miss.outcome is MatchOutcome.UNIT_ERROR
    assert "sign" in sign_miss.detail

    real_miss = compare_values(500.0, 1234.0)
    assert real_miss.outcome is MatchOutcome.MISMATCH

    assert compare_values(44.10, 44.13).outcome is MatchOutcome.MATCH
    assert compare_values(None, 1.0).outcome is MatchOutcome.MISSING

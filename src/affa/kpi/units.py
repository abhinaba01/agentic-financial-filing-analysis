"""Number parsing, scale normalisation and percentage conventions (section 6).

Three failure modes this module exists to prevent, all of which produce numbers
that look completely reasonable in a report:

* ``(1,234)`` read as ``1234`` - the parenthesis convention for negatives is
  universal in filings, and dropping it flips the sign of a line item.
* ``383,285`` from a statement headed "in millions" stored as 383285 - three
  orders of magnitude off, and every derived ratio built on it is wrong.
* ``0.42`` compared against ``42.0`` - the same margin in two conventions,
  scored as a miss.

On the last one: the fix is to normalise the *convention* and count the
conversion, never to rescale ground truth until the metric improves
(anti-pattern #12). :func:`compare_values` returns *why* two values differ so a
unit mismatch is reported as a unit error rather than laundered into accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from affa.schema import SCALE_MULTIPLIER, Scale

# Values a filing uses to mean "nothing here". Parsing these as 0.0 invents data.
NIL_TOKENS = {"", "-", "--", "---", "n/a", "na", "nil", "none", "*", "—", "–"}


class PercentConvention(str, Enum):
    POINTS = "points"  # 42.0 means 42%
    FRACTION = "fraction"  # 0.42 means 42%
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParsedNumber:
    """A number read from a filing, with everything the raw text told us."""

    value: float
    raw: str
    is_percent: bool = False
    is_multiple: bool = False
    is_currency: bool = False
    from_parentheses: bool = False
    scale_hint: Scale | None = None

    def in_units(self, scale: Scale = Scale.UNITS) -> float:
        """Absolute value under ``scale``, or the number's own hint if it has one."""
        effective = self.scale_hint or scale
        return self.value * SCALE_MULTIPLIER[effective]


# Matches: $1,234.5  (1,234)  $(2,345.6)  1.2x  45%  -3.4  1,234
_NUMBER_RE = re.compile(
    r"""
    (?P<open_paren>\()?
    \s*(?P<currency>[$€£¥])?
    \s*(?P<sign>[-+])?
    \s*(?P<digits>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    \s*(?P<close_paren>\))?
    \s*(?P<suffix>%|x\b|bps\b)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "(in millions, except per share amounts)" / "amounts in thousands" / "$ in bn"
_SCALE_RE = re.compile(
    r"\b(?:in|amounts\s+in|expressed\s+in|\$\s*in)\s+"
    r"(thousands?|millions?|billions?|mm|bn|k)\b",
    re.IGNORECASE,
)
_SCALE_WORDS = {
    "k": Scale.THOUSANDS,
    "thousand": Scale.THOUSANDS,
    "thousands": Scale.THOUSANDS,
    "mm": Scale.MILLIONS,
    "million": Scale.MILLIONS,
    "millions": Scale.MILLIONS,
    "bn": Scale.BILLIONS,
    "billion": Scale.BILLIONS,
    "billions": Scale.BILLIONS,
}

# A magnitude word attached to a single figure: "$1,234.5 million"
_INLINE_SCALE_RE = re.compile(
    r"\b(thousand|million|billion)s?\b",
    re.IGNORECASE,
)


_FOOTNOTE_CUE_RE = re.compile(r"\b(see|notes?|items?|footnotes?|refer(?:\s+to)?|part)\b[\s(]*$")


def _in_parentheses(raw: str, m: re.Match[str]) -> bool:
    """True when the matched figure is enclosed in parentheses.

    Handles both the case where the regex captured the parens itself
    (``(1,234)``) and the case where the opening paren sits further left, before
    intervening words (``(see Note 3)``), which is how the engine sees a match
    that starts mid-parenthetical.
    """
    if m.group("open_paren") and m.group("close_paren"):
        return True
    if not m.group("close_paren"):
        return False
    before = raw[: m.start()]
    # An unmatched "(" to the left means this figure is inside it.
    return "(" in before and before.rindex("(") > before.rfind(")")


def _is_footnote_marker(raw: str, m: re.Match[str]) -> bool:
    """True when a small integer is a footnote reference, not a figure.

    ``(see Note 3) 1,234`` must yield 1,234, not -3: filings mark footnotes with
    a parenthesised integer, and reading one as a line item is wrong *and*
    sign-flipped, which is the worst combination because it looks plausible.
    """
    digits = m.group("digits")
    if "," in digits or "." in digits or len(digits) > 2:
        return False
    preceding = raw[max(0, m.start() - 24) : m.start()].lower()
    if _FOOTNOTE_CUE_RE.search(preceding):
        return True
    # A bare "(3)" with more numbers after it is a marker, not the value.
    return _in_parentheses(raw, m) and bool(
        _NUMBER_RE.search(raw[m.end() :]) and raw[m.end() :].strip()
    )


def parse_financial_number(text: str, *, default_scale: Scale | None = None) -> ParsedNumber | None:
    """Parse one financial figure. Returns ``None`` for nil markers and non-numbers.

    Negatives are recognised from a leading minus *or* from surrounding
    parentheses, including the ``$(2,345)`` form where the currency symbol sits
    inside the parenthesis.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if raw.lower() in NIL_TOKENS:
        return None

    m = None
    for candidate in _NUMBER_RE.finditer(raw):
        if not candidate.group("digits"):
            continue
        if _is_footnote_marker(raw, candidate):
            continue
        m = candidate
        break
    if m is None:
        return None

    digits = m.group("digits").replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None

    # Parentheses mean negative, but only when they actually enclose the figure.
    from_parens = _in_parentheses(raw, m)
    negative = m.group("sign") == "-" or from_parens
    if negative:
        value = -abs(value)

    suffix = (m.group("suffix") or "").lower()
    is_percent = suffix == "%"
    is_multiple = suffix == "x"
    if suffix == "bps":
        # Basis points are percent/100; normalise immediately so downstream code
        # only ever sees one percent representation.
        value = value / 100.0
        is_percent = True

    scale_hint = None
    tail = raw[m.end() : m.end() + 24]
    if sm := _INLINE_SCALE_RE.search(tail):
        scale_hint = _SCALE_WORDS[sm.group(1).lower()]
    elif default_scale is not None:
        scale_hint = default_scale

    return ParsedNumber(
        value=value,
        raw=raw,
        is_percent=is_percent,
        is_multiple=is_multiple,
        is_currency=bool(m.group("currency")),
        from_parentheses=from_parens,
        scale_hint=scale_hint,
    )


def detect_scale(context: str, default: Scale = Scale.UNITS) -> Scale:
    """Read the reporting scale off a statement header.

    Financial statements declare it once, in a parenthetical above the table
    ("(In millions, except per share data)"), and every figure below inherits it.
    """
    if not context:
        return default
    m = _SCALE_RE.search(context)
    if m:
        return _SCALE_WORDS.get(m.group(1).lower(), default)
    return default


def normalize_scale(value: float, from_scale: Scale, to_scale: Scale = Scale.UNITS) -> float:
    """Convert a figure between reporting scales."""
    return value * SCALE_MULTIPLIER[from_scale] / SCALE_MULTIPLIER[to_scale]


@dataclass(frozen=True)
class PercentNormalization:
    value: float
    converted: bool
    ambiguous: bool
    assumed: PercentConvention


def normalize_percent(
    value: float,
    *,
    convention: PercentConvention = PercentConvention.UNKNOWN,
    ambiguity_band: tuple[float, float] = (0.0, 1.0),
) -> PercentNormalization:
    """Express a percentage in canonical points (42.0 means 42%).

    With a declared ``convention`` the conversion is exact. Without one, values
    inside ``ambiguity_band`` are genuinely undecidable - 0.42 is both "42%" in
    fraction convention and "0.42%" in points convention - so the result is
    flagged ``ambiguous`` and the caller decides whether to trust it. The band is
    never silently widened to make more values convert cleanly.
    """
    if convention is PercentConvention.POINTS:
        return PercentNormalization(value, False, False, PercentConvention.POINTS)
    if convention is PercentConvention.FRACTION:
        return PercentNormalization(value * 100.0, True, False, PercentConvention.FRACTION)

    lo, hi = ambiguity_band
    if lo <= abs(value) <= hi:
        # Heuristic: assume fraction, but say so. Most margins and growth rates
        # land well outside [0,1] in points convention, so a value inside it is
        # much more likely a fraction - "much more likely" is not "certainly".
        return PercentNormalization(value * 100.0, True, True, PercentConvention.FRACTION)
    return PercentNormalization(value, False, False, PercentConvention.POINTS)


class MatchOutcome(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNIT_ERROR = "unit_error"  # right figure, wrong scale or convention
    MISSING = "missing"


@dataclass(frozen=True)
class Comparison:
    outcome: MatchOutcome
    relative_error_pct: float | None
    detail: str = ""

    @property
    def is_match(self) -> bool:
        return self.outcome is MatchOutcome.MATCH


# Ratios a scale mismatch would produce. Checked so a 1000x or 100x miss is
# reported as a unit error rather than as an ordinary wrong answer - they need
# different fixes and lumping them together hides which one you have.
_UNIT_ERROR_RATIOS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100.0, 0.01)


def compare_values(
    predicted: float | None,
    gold: float | None,
    *,
    tolerance_pct: float = 1.0,
) -> Comparison:
    """Tolerance-aware comparison that separates unit errors from real misses.

    Ground truth is never rescaled here. When the prediction is off by exactly a
    scale factor, the outcome is ``UNIT_ERROR`` - counted, reported, and not
    quietly folded into the accuracy numerator.
    """
    if predicted is None or gold is None:
        return Comparison(MatchOutcome.MISSING, None, "value absent")

    if gold == 0:
        if abs(predicted) <= 1e-9:
            return Comparison(MatchOutcome.MATCH, 0.0)
        return Comparison(MatchOutcome.MISMATCH, None, "gold is zero, prediction is not")

    rel = abs(predicted - gold) / abs(gold) * 100.0
    if rel <= tolerance_pct:
        return Comparison(MatchOutcome.MATCH, rel)

    if predicted != 0:
        ratio = gold / predicted
        for candidate in _UNIT_ERROR_RATIOS:
            if abs(ratio - candidate) / candidate <= 0.01:
                return Comparison(
                    MatchOutcome.UNIT_ERROR,
                    rel,
                    f"off by {candidate:g}x - scale or percent convention mismatch",
                )
        # Sign-only difference: the parenthesised-negative bug, specifically.
        if abs(abs(predicted) - abs(gold)) / abs(gold) * 100.0 <= tolerance_pct:
            return Comparison(
                MatchOutcome.UNIT_ERROR, rel, "sign flipped - check parenthesised negatives"
            )

    return Comparison(MatchOutcome.MISMATCH, rel)

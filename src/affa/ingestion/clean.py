"""Text cleaning that does not destroy financial notation (section 4).

The tokens below must survive cleaning byte-for-byte, because every one of them
carries meaning the KPI extractor depends on:

    $1,234.5      currency with thousands separators
    (1,234)       parenthesised negative - the standard filing convention
    1.2x          a multiple (leverage, coverage)
    45%           a percentage
    FY2023        a fiscal-period label
    (0.15)        a negative EPS
    3.5 - 4.0%    a guidance range

``tests/test_clean.py`` asserts each of these explicitly. Aggressive whitespace
or punctuation stripping silently turns ``(1,234)`` into ``1,234`` and flips the
sign of a line item, which is the kind of error that survives all the way into a
report and looks completely plausible.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures. PDF extraction emits these constantly; leaving them in breaks both
# tokenisation and string matching against the source text during verification.
LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
    "œ": "oe",
    "Œ": "OE",
    "æ": "ae",
    "Æ": "AE",
}

# Quotes, dashes and spaces. The minus sign U+2212 matters: filings use it for
# negatives and a naive float() call chokes on it.
PUNCTUATION = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",
    "″": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",
    "−": "-",  # MINUS SIGN -> ASCII hyphen, so numbers parse
    "­": "",  # soft hyphen: invisible, and corrupts words if kept
    " ": " ",  # non-breaking space
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    "​": "",  # zero-width space
    "‌": "",
    "‍": "",
    "﻿": "",
    "…": "...",
}

_LIGATURE_RE = re.compile("|".join(map(re.escape, LIGATURES)))
_PUNCT_RE = re.compile("|".join(map(re.escape, PUNCTUATION)))

# Word split across a line break by PDF layout: "reve-\nnue" -> "revenue".
# Restricted to lowercase-to-lowercase so real compounds keep their hyphen:
# "non-\nGAAP" stays "non-GAAP", and "FY2023-\n2024" is untouched.
_HYPHEN_LINEBREAK_RE = re.compile(r"([a-z])-\s*\n\s*([a-z])")

# Page furniture that adds nothing and pollutes chunks.
_PAGE_ARTIFACT_RE = re.compile(
    r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|-\s*\d+\s*-)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_HORIZONTAL_WS_RE = re.compile(r"[ \t\f\v]+")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
# Runs of dot/underscore leaders from tables of contents: ".........  42"
_LEADER_RE = re.compile(r"[.·_]{4,}")


def normalize_unicode(text: str) -> str:
    """NFC plus explicit ligature and punctuation folding.

    Deliberately *not* NFKC. NFKC also folds superscripts, so a footnote marker
    in ``Revenue(1) 1,234`` becomes a digit and can fuse into the adjacent
    figure. NFC leaves numerals alone; the ligature and punctuation tables above
    cover what NFKC was wanted for.
    """
    text = unicodedata.normalize("NFC", text)
    text = _LIGATURE_RE.sub(lambda m: LIGATURES[m.group(0)], text)
    text = _PUNCT_RE.sub(lambda m: PUNCTUATION[m.group(0)], text)
    return text


def dehyphenate(text: str) -> str:
    """Rejoin words broken across a line break. Leaves genuine hyphens alone."""
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def strip_page_artifacts(text: str) -> str:
    return _PAGE_ARTIFACT_RE.sub("", text)


def normalize_whitespace(text: str, *, preserve_layout: bool = False) -> str:
    """Collapse whitespace.

    ``preserve_layout=True`` keeps runs of spaces and single newlines, which is
    required for tables: column alignment is the only thing separating a label
    from its figure once a financial statement has been flattened to text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_layout:
        text = _TRAILING_WS_RE.sub("", text)
        return _BLANK_RUN_RE.sub("\n\n", text).strip("\n")
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def clean_text(
    text: str,
    *,
    normalize_unicode_chars: bool = True,
    dehyphenate_linebreaks: bool = True,
    preserve_layout: bool = False,
    strip_leaders: bool = True,
) -> str:
    """Full cleaning pass. Order matters.

    Unicode folding runs first so the hyphenation and whitespace rules see ASCII
    hyphens and ASCII spaces; running them the other way round misses words
    broken with a non-breaking hyphen.
    """
    if not text:
        return ""
    if normalize_unicode_chars:
        text = normalize_unicode(text)
    if strip_leaders:
        text = _LEADER_RE.sub(" ", text)
    if dehyphenate_linebreaks:
        text = dehyphenate(text)
    text = strip_page_artifacts(text)
    return normalize_whitespace(text, preserve_layout=preserve_layout)


# --- notation guard -------------------------------------------------------
# Used by tests and by the ingestion smoke check. Kept in the module rather than
# the test file so a caller can assert the property on their own documents.

NOTATION_PROBES: tuple[str, ...] = (
    "$1,234.5",
    "(1,234)",
    "1.2x",
    "45%",
    "FY2023",
    "(0.15)",
    "$(2,345)",
    "10-K",
    "3.5%-4.0%",
    "2023 vs. 2022",
)


def notation_survives(probe: str, **clean_kwargs: object) -> bool:
    """True when ``probe`` is still present verbatim after cleaning."""
    return probe in clean_text(f"Line before. {probe} Line after.", **clean_kwargs)  # type: ignore[arg-type]

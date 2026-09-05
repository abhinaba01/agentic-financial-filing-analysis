"""Query reformulation for the bounded re-query loop.

Section 3: **every retry attempt must actually change the query.** A loop that
re-runs an identical search is a no-op that burns its budget and produces
identical results three times, which looks like "the document just doesn't
contain it" rather than "the retry never did anything".

So each attempt applies a *different* strategy, and :func:`reformulate` verifies
the result differs from every query already tried, falling back through the
remaining strategies until it finds one that does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

# Vocabulary that actually appears in filings, used to move a colloquial query
# toward the register of the source text.
_FILING_SYNONYMS: dict[str, str] = {
    "revenue": "net sales total revenues",
    "profit": "net income gross profit operating income",
    "earnings": "net income earnings per share",
    "debt": "long-term debt borrowings notes payable",
    "cash": "cash and cash equivalents operating activities",
    "risk": "risk factors uncertainties adverse effect",
    "growth": "increase compared to prior year change",
    "margin": "gross profit percentage of net sales",
    "outlook": "forward-looking expectations guidance",
}

_STATEMENT_TERMS = (
    "consolidated statements of operations "
    "consolidated balance sheets "
    "consolidated statements of cash flows"
)


def _expand_synonyms(query: str) -> str:
    """Attempt 1: add filing vocabulary for terms the query uses colloquially."""
    additions = [
        expansion
        for term, expansion in _FILING_SYNONYMS.items()
        if term in query.lower() and expansion not in query
    ]
    return f"{query} {' '.join(additions)}".strip() if additions else f"{query} {_STATEMENT_TERMS}"


def _target_statements(query: str) -> str:
    """Attempt 2: point at the primary financial statements explicitly."""
    return f"{query} {_STATEMENT_TERMS} total amounts fiscal year"


def _keywords_only(query: str) -> str:
    """Attempt 3: strip question words down to content terms.

    Dense retrievers are distracted by interrogative framing; the nouns are the
    signal. This is the most aggressive rewrite, so it goes last.
    """
    stop = {
        "what",
        "was",
        "were",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "for",
        "in",
        "on",
        "to",
        "and",
        "or",
        "how",
        "much",
        "many",
        "did",
        "does",
        "do",
        "company",
        "please",
        "tell",
        "me",
        "about",
        "this",
        "that",
        "its",
    }
    words = [w.strip("?.,;:") for w in query.split()]
    kept = [w for w in words if w.lower() not in stop and w.strip()]
    return " ".join(kept) if kept else query


# Ordered by attempt number. Index 0 is used for the first retry (attempt 1).
STRATEGIES: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("synonym_expansion", _expand_synonyms),
    ("statement_targeting", _target_statements),
    ("keyword_extraction", _keywords_only),
)


def reformulate(
    original_query: str,
    attempt: int,
    already_tried: Sequence[str] = (),
) -> tuple[str, str]:
    """Produce a query for retry ``attempt`` that differs from everything tried.

    ``attempt`` is 1-based: attempt 1 is the first *retry*, after the initial
    search. Returns ``(query, strategy_name)``.

    The guarantee is checked, not assumed. If the strategy for this attempt
    happens to reproduce an earlier query, the remaining strategies are tried in
    turn, and only if all of them collide does the query get a disambiguating
    suffix - so the loop can never spend an attempt on a search it already ran.
    """
    tried = {q.strip().lower() for q in already_tried}
    # Wraps, so an attempt number beyond the strategy list stays in range rather
    # than indexing off the end.
    index = max(0, attempt - 1) % len(STRATEGIES)

    order = list(range(index, len(STRATEGIES))) + list(range(0, index))
    for i in order:
        name, fn = STRATEGIES[i]
        candidate = " ".join(fn(original_query).split())
        if candidate.strip().lower() not in tried:
            return candidate, name

    # Every strategy collided. Rather than repeat a search, vary it explicitly.
    name, fn = STRATEGIES[index]
    candidate = " ".join(f"{fn(original_query)} attempt {attempt}".split())
    return candidate, f"{name}+disambiguated"

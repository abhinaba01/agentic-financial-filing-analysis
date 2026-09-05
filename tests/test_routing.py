"""Routing thresholds, retry budget and query reformulation (section 3).

These cover anti-patterns #1, #2 and #3 directly: an unreachable threshold, two
competing retry mechanisms, and retries that re-run the same search.
"""

from __future__ import annotations

import dataclasses

import pytest

from affa.agent.reformulate import STRATEGIES, reformulate
from affa.agent.routing import assess_sufficiency, route_after_retrieval, stop_reason
from affa.config import ConfigError, validate_threshold_reachability


def test_retry_threshold_must_exceed_the_similarity_floor() -> None:
    """Anti-pattern #1: a condition an upstream filter guarantees is false.

    Retrieval discards everything below the floor, so mean similarity is always
    >= floor. A retry rule at or below it can never fire.
    """
    with pytest.raises(ConfigError, match="unreachable"):
        validate_threshold_reachability(min_similarity=0.45, retry_below=0.45)
    with pytest.raises(ConfigError, match="unreachable"):
        validate_threshold_reachability(min_similarity=0.60, retry_below=0.45)
    # The shipped relationship is fine.
    validate_threshold_reachability(min_similarity=0.25, retry_below=0.45)


def test_shipped_config_satisfies_the_invariant(cfg) -> None:
    assert cfg.routing.retry_below_mean_similarity > cfg.retrieval.min_similarity


def test_importing_routing_asserts_the_invariant() -> None:
    """The check runs at import time, not on the first bad document."""
    import affa.agent.routing as routing

    assert routing._DEFAULTS is not None


def test_config_construction_rejects_an_unreachable_threshold(cfg) -> None:
    bad_routing = dataclasses.replace(cfg.routing, retry_below_mean_similarity=0.10)
    with pytest.raises(ConfigError, match="unreachable"):
        dataclasses.replace(cfg, routing=bad_routing)


def test_sufficiency_verdicts(cfg) -> None:
    assert assess_sufficiency(0, 0.0, cfg).sufficient is False
    assert assess_sufficiency(2, 0.9, cfg).sufficient is False  # too few chunks
    assert assess_sufficiency(5, 0.30, cfg).sufficient is False  # below retry threshold
    assert assess_sufficiency(5, 0.70, cfg).sufficient is True


def test_router_stops_at_the_retry_budget(cfg) -> None:
    """Anti-pattern #2: the budget lives in exactly one place."""
    state = {"sufficient": False, "retrieval_attempts": cfg.routing.max_retrieval_attempts}
    assert route_after_retrieval(state, cfg) == "proceed"
    assert "budget exhausted" in stop_reason(state, cfg)


def test_router_retries_while_budget_remains(cfg) -> None:
    assert route_after_retrieval({"sufficient": False, "retrieval_attempts": 1}, cfg) == "retry"


def test_router_proceeds_once_sufficient(cfg) -> None:
    assert route_after_retrieval({"sufficient": True, "retrieval_attempts": 1}, cfg) == "proceed"
    assert stop_reason({"sufficient": True}, cfg) == "sufficient evidence"


def test_every_retry_changes_the_query() -> None:
    """Anti-pattern #3: a retry that re-runs the same search burns its budget."""
    original = "What was the company's revenue and profit for the year?"
    tried = [original]
    for attempt in range(1, len(STRATEGIES) + 3):
        query, strategy = reformulate(original, attempt, tried)
        assert query not in tried, f"attempt {attempt} repeated a query ({strategy})"
        assert query.strip(), "reformulation produced an empty query"
        tried.append(query)
    assert len(set(tried)) == len(tried)


def test_reformulation_varies_by_attempt_number() -> None:
    """Different attempts use different strategies, not the same one twice."""
    original = "revenue growth and margins"
    first, first_name = reformulate(original, 1, [original])
    second, second_name = reformulate(original, 2, [original, first])
    assert first != second
    assert first_name != second_name


def test_reformulation_never_returns_the_original() -> None:
    original = "total revenue"
    query, _ = reformulate(original, 1, [original])
    assert query.strip().lower() != original.strip().lower()

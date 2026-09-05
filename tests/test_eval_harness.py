"""Evaluation harness contracts (section 9).

The rule these tests hold in place: a metric without a baseline measured the
same way is not a result, and published figures must never sit where they read
as this repo's own numbers.
"""

from __future__ import annotations

import json

import pytest

from affa.eval.harness import (
    COMPONENTS,
    EvaluationResult,
    LiteratureReference,
    base_parser,
)
from affa.eval.metrics import (
    accuracy,
    macro_f1,
    ndcg_at_k,
    overlap_count,
    reciprocal_rank,
)


def result(**overrides) -> EvaluationResult:
    base = dict(
        component="test",
        dataset="d",
        split="test",
        n_examples=100,
        metrics={"f1": 0.9},
        baseline_name="base",
        baseline_metrics={"f1": 0.7},
        model_name="mine",
    )
    base.update(overrides)
    return EvaluationResult(**base)


def test_a_metric_without_a_baseline_is_refused() -> None:
    with pytest.raises(ValueError, match="not a result"):
        result(baseline_metrics={})


def test_a_missing_baseline_can_be_declared_explicitly() -> None:
    """Sometimes no baseline exists. It has to be stated, not silently omitted."""
    r = result(baseline_metrics={}, baseline_absent_reason="no comparable prior system")
    assert "no comparable prior system" in r.render()


def test_deltas_are_computed_against_the_baseline() -> None:
    assert result().deltas() == {"f1": pytest.approx(0.2)}


def test_sampling_is_flagged_in_the_output() -> None:
    """Subset numbers are not comparable to published figures, and must say so."""
    r = result(n_examples=100, subset_of=10_000)
    assert r.is_sampled
    rendered = r.render()
    assert "sampled from 10000" in rendered
    assert "NOT to published figures" in rendered


def test_full_runs_are_not_flagged_as_sampled() -> None:
    assert not result(n_examples=100, subset_of=100).is_sampled


def test_literature_is_rendered_under_its_own_heading() -> None:
    """Anti-pattern #9: published numbers placed where they read as ours."""
    r = result(
        literature=[
            LiteratureReference(
                source="FiNER-139 paper",
                metric="micro-F1",
                value=0.892,
                conditions="full splits",
            )
        ]
    )
    rendered = r.render()
    heading = rendered.index("published figures")
    assert "NOT produced by this repo" in rendered
    # The paper's number must appear only after the separating heading.
    assert rendered.index("0.892") > heading


def test_result_serialises_with_provenance() -> None:
    payload = json.loads(json.dumps(result().to_dict()))
    assert payload["seed"] == 42
    assert payload["affa_version"]
    assert payload["python_version"]
    assert payload["deltas"] == {"f1": 0.2}


def test_all_harnesses_share_the_documented_flags() -> None:
    parser = base_parser("x", "y")
    flags = {opt for action in parser._actions for opt in action.option_strings}
    assert {"--test-set", "--output", "--limit", "--run-agent", "--baseline"} <= flags


def test_every_component_is_loadable() -> None:
    """Each name in COMPONENTS must resolve to a real evaluator."""
    from affa.eval.harness import _load

    for component in COMPONENTS:
        evaluator = _load(component)
        assert evaluator.name
        assert evaluator.default_baseline, f"{component} declares no default baseline"


def test_unknown_component_is_a_clear_error() -> None:
    from affa.eval.harness import _load

    with pytest.raises(ValueError, match="unknown component"):
        _load("nonsense")


def test_retrieval_harness_refuses_the_stub_embedder(monkeypatch) -> None:
    """A lexical stub would produce a meaningless NDCG that looks entirely real."""
    import argparse

    from affa.eval.retrieval_eval import RetrievalEvaluator

    monkeypatch.setenv("AFFA_FORCE_STUB_EMBEDDER", "1")
    args = argparse.Namespace(
        config=None,
        test_set=None,
        model=None,
        baseline=None,
        limit=None,
        seed=42,
        top_k=10,
        corpus_sample=100,
    )
    with pytest.raises(SystemExit, match="meaningless"):
        RetrievalEvaluator().run(args)


# --- metric implementations ----------------------------------------------


def test_ndcg_rewards_higher_ranks() -> None:
    relevance = {"a": 1.0}
    assert ndcg_at_k(["a", "b", "c"], relevance, 10) > ndcg_at_k(["b", "c", "a"], relevance, 10)
    assert ndcg_at_k(["b", "c"], relevance, 10) == 0.0
    # No relevant document: 0.0 rather than a division by zero.
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)
    assert reciprocal_rank(["a", "b"], {"b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_macro_f1_penalises_ignoring_a_minority_class() -> None:
    """Exactly why macro-F1 is reported for the skewed sentiment split."""
    y_true = [0, 1, 1, 1, 2]
    all_majority = [1, 1, 1, 1, 1]
    assert accuracy(y_true, all_majority) == pytest.approx(0.6)
    assert macro_f1(y_true, all_majority, 3) < 0.3


def test_overlap_count_reports_zero_explicitly() -> None:
    """Section 2: the count is reported even when it is zero."""
    count, examples = overlap_count(["a b c"], ["x y z"])
    assert count == 0
    assert examples == []

    count, examples = overlap_count(["Revenue  ROSE"], ["revenue rose"])
    assert count == 1, "overlap check must normalise whitespace and case"

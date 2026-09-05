"""The recommendation rubric (section 7).

The requirements being enforced: the verdict is deterministic, the rubric is
versioned and loaded from disk, and ``insufficient_evidence`` is a genuinely
reachable outcome rather than a branch that exists only in the enum.
"""

from __future__ import annotations

import pytest

from affa.config import load_rubric
from affa.recommend.rubric import assessment_from_score, evaluate, score_from_bands
from affa.schema import (
    Assessment,
    Disagreement,
    ExtractedMetric,
    ExtractionMethod,
    FinancialMetrics,
    Scale,
    SourceRef,
)


def metric(name: str, value: float, chunk: str = "c1", page: int = 31) -> ExtractedMetric:
    return ExtractedMetric(
        name=name,
        value=value,
        scale=Scale.UNITS,
        source=SourceRef(chunk_id=chunk, page=page),
        method=ExtractionMethod.RULE_BASED,
        confidence=0.7,
    )


def healthy_metrics() -> FinancialMetrics:
    from affa.schema import DerivedMetric

    src = [SourceRef(chunk_id="c1", page=31)]
    return FinancialMetrics(
        extracted=[metric("revenue", 4.8126e9), metric("net_income", 7.314e8)],
        derived=[
            DerivedMetric(
                name="net_margin_pct", value=15.2, formula="f", operands={"a": 1.0}, sources=src
            ),
            DerivedMetric(
                name="debt_to_equity", value=0.48, formula="f", operands={"a": 1.0}, sources=src
            ),
            DerivedMetric(
                name="free_cash_flow", value=8.358e8, formula="f", operands={"a": 1.0}, sources=src
            ),
            DerivedMetric(
                name="current_ratio", value=1.61, formula="f", operands={"a": 1.0}, sources=src
            ),
            DerivedMetric(
                name="ocf_to_net_income", value=1.51, formula="f", operands={"a": 1.0}, sources=src
            ),
        ],
        yoy_changes={"revenue_yoy_pct": 11.87, "net_income_yoy_pct": 24.35},
    )


def test_rubric_is_loaded_from_disk_and_versioned(cfg) -> None:
    rubric = load_rubric(cfg)
    assert rubric["version"] == "1.0"
    assert set(rubric["factors"]) == {
        "profitability",
        "growth",
        "leverage",
        "cash_generation",
        "risk",
        "tone",
    }
    # Weights are visible and adjustable, per section 7.
    assert sum(f["weight"] for f in rubric["factors"].values()) == pytest.approx(1.0)


def test_band_scoring_is_ordered_and_open_ended() -> None:
    bands = [
        {"max": 0.0, "score": -1.0, "label": "loss"},
        {"max": 12.0, "score": 0.3, "label": "moderate"},
        {"max": None, "score": 1.0, "label": "exceptional"},
    ]
    assert score_from_bands(-5.0, bands)[0] == -1.0
    assert score_from_bands(8.0, bands)[0] == 0.3
    assert score_from_bands(99.0, bands)[0] == 1.0


def test_verdict_is_deterministic(cfg) -> None:
    """Same inputs, same verdict - every time. That is why the LLM does not decide it."""
    first = evaluate(healthy_metrics(), cfg=cfg, sentiment_score=0.4, risk_severity_index=0.3)
    second = evaluate(healthy_metrics(), cfg=cfg, sentiment_score=0.4, risk_severity_index=0.3)
    assert first.recommendation.assessment == second.recommendation.assessment
    assert first.recommendation.aggregate_score == second.recommendation.aggregate_score
    assert first.recommendation.factor_scores == second.recommendation.factor_scores


def test_healthy_filing_scores_favorable(cfg) -> None:
    outcome = evaluate(healthy_metrics(), cfg=cfg, sentiment_score=0.4, risk_severity_index=0.3)
    rec = outcome.recommendation
    assert rec.assessment is Assessment.FAVORABLE
    assert rec.aggregate_score is not None
    assert rec.rubric_version == "1.0"
    assert rec.disclaimer


def test_insufficient_evidence_is_reachable(cfg) -> None:
    """Section 7: a system that always produces a verdict is not analyzing anything."""
    outcome = evaluate(FinancialMetrics(), cfg=cfg)
    rec = outcome.recommendation
    assert rec.assessment is Assessment.INSUFFICIENT_EVIDENCE
    # Publishing a score here would imply a verdict the evidence does not support.
    assert rec.aggregate_score is None
    assert rec.factors_missing


def test_insufficient_evidence_fires_on_partial_coverage(cfg) -> None:
    """Two factors out of six is not enough, however good those two look."""
    partial = FinancialMetrics(
        extracted=[metric("revenue", 1e9)],
        yoy_changes={"revenue_yoy_pct": 25.0},
    )
    outcome = evaluate(partial, cfg=cfg, sentiment_score=0.9, risk_severity_index=0.0)
    assert outcome.recommendation.assessment is Assessment.INSUFFICIENT_EVIDENCE
    assert any("factors could be scored" in n or "rubric weight" in n for n in outcome.notes)


def test_factor_requires_all_its_required_metrics(cfg) -> None:
    """An optional metric must not rescue a factor whose required input is missing."""
    from affa.schema import DerivedMetric

    src = [SourceRef(chunk_id="c1", page=1)]
    metrics = FinancialMetrics(
        extracted=[metric("revenue", 1e9)],
        derived=[
            # gross_margin_pct is optional for profitability; net_margin_pct is required.
            DerivedMetric(
                name="gross_margin_pct", value=65.0, formula="f", operands={"a": 1.0}, sources=src
            ),
        ],
    )
    outcome = evaluate(metrics, cfg=cfg)
    assert "profitability" in outcome.recommendation.factors_missing


def test_rationale_items_carry_citations(cfg) -> None:
    outcome = evaluate(healthy_metrics(), cfg=cfg, sentiment_score=0.4, risk_severity_index=0.3)
    cited = [r for r in outcome.recommendation.rationale if r.citations]
    assert cited, "no rationale item carried a citation"
    assert all(c.chunk_id for r in cited for c in r.citations)


def test_confidence_reflects_coverage_not_verdict_strength(cfg) -> None:
    full = evaluate(healthy_metrics(), cfg=cfg, sentiment_score=0.4, risk_severity_index=0.3)
    thin = evaluate(FinancialMetrics(), cfg=cfg)
    assert full.recommendation.confidence > thin.recommendation.confidence


def test_disagreements_reduce_confidence(cfg) -> None:
    clean = evaluate(healthy_metrics(), cfg=cfg, sentiment_score=0.4, risk_severity_index=0.3)
    noisy = evaluate(
        healthy_metrics(),
        cfg=cfg,
        sentiment_score=0.4,
        risk_severity_index=0.3,
        disagreements=[Disagreement(name="revenue", xbrl_model=1.0, rule_based=2.0)],
    )
    assert noisy.recommendation.confidence < clean.recommendation.confidence


def test_assessment_bands_cover_the_range(cfg) -> None:
    bands = load_rubric(cfg)["assessment_bands"]
    assert assessment_from_score(-0.9, bands) is Assessment.UNFAVORABLE
    assert assessment_from_score(0.0, bands) is Assessment.MIXED
    assert assessment_from_score(0.9, bands) is Assessment.FAVORABLE

"""KPI extraction, derivation and the reconciliation of the two extractors."""

from __future__ import annotations

import pytest

from affa.kpi.derive import derive_metrics, pe_ratio, yoy_change_pct
from affa.kpi.extract import extract_kpis, normalize_external_percents
from affa.kpi.rules import extract_rule_based, match_metric
from affa.schema import ExtractionMethod, Scale


def test_match_metric_prefers_the_most_specific_label() -> None:
    """ "Total cost of revenue" must not be claimed by ``revenue``."""
    assert match_metric("Total cost of revenue").name == "cost_of_revenue"
    assert match_metric("Total net sales").name == "revenue"
    assert match_metric("Gross profit").name == "gross_profit"
    assert match_metric("Total shareholders equity").name == "shareholders_equity"


def test_match_metric_rejects_incidental_mentions() -> None:
    assert match_metric("Deferred revenue recognised during the period was") is None
    assert match_metric("") is None


def test_extraction_reads_the_current_period_column(income_statement_chunk) -> None:
    hits = extract_rule_based([income_statement_chunk])
    revenue = [h for h in hits if h.metric == "revenue"]
    assert revenue, "revenue not extracted"
    current = [h for h in revenue if h.is_current_period]
    assert current[0].value == pytest.approx(4812.6)
    assert current[0].scale is Scale.MILLIONS


def test_parenthesised_negative_extracted_as_negative(income_statement_chunk) -> None:
    hits = extract_rule_based([income_statement_chunk])
    interest = [h for h in hits if h.metric == "interest_expense" and h.is_current_period]
    assert interest[0].value == pytest.approx(-48.6)


def test_eps_is_not_scaled_by_the_statement_header(income_statement_chunk, cfg) -> None:
    """Regression: "(In millions, except per share data)" was applied to EPS.

    Diluted EPS of $3.64 became $3.64 million, which silently drove every P/E
    computed from it to zero.
    """
    outcome = extract_kpis([income_statement_chunk], cfg)
    eps = outcome.metrics.get_extracted("eps_diluted")
    assert eps is not None
    assert eps.value_in_units == pytest.approx(3.64)
    assert eps.scale is Scale.UNITS

    revenue = outcome.metrics.get_extracted("revenue")
    assert revenue.value_in_units == pytest.approx(4_812_600_000.0)


def test_derived_metrics_carry_formula_and_operands(financial_chunks, cfg) -> None:
    outcome = extract_kpis(financial_chunks, cfg)
    gm = outcome.metrics.get_derived("gross_margin_pct")
    assert gm is not None
    assert gm.value == pytest.approx(65.80, abs=0.05)
    assert gm.formula == "gross_profit / revenue * 100"
    assert set(gm.operands) == {"gross_profit", "revenue"}


def test_free_cash_flow_handles_capex_sign(financial_chunks, cfg) -> None:
    """Capex is an outflow whether the filing prints it as (268.4) or 268.4."""
    outcome = extract_kpis(financial_chunks, cfg)
    fcf = outcome.metrics.get_derived("free_cash_flow")
    assert fcf.value == pytest.approx(1_104_200_000 - 268_400_000, rel=1e-6)


def test_yoy_is_computed_in_points_and_not_reinterpreted(financial_chunks, cfg) -> None:
    """Regression: small YoY values were re-scaled by percent-convention inference.

    A true -0.96% change lies inside the undecidable band, so running convention
    inference over a value the pipeline itself computed turned it into -96%.
    """
    outcome = extract_kpis(financial_chunks, cfg)
    yoy = outcome.metrics.yoy_changes
    assert yoy["revenue_yoy_pct"] == pytest.approx(11.87, abs=0.02)
    assert yoy["net_income_yoy_pct"] == pytest.approx(24.35, abs=0.02)
    # Small magnitudes must survive untouched.
    assert yoy["eps_diluted_yoy_pct"] == pytest.approx(24.23, abs=0.05)


def test_external_percent_normalisation_is_reported_not_silent(cfg) -> None:
    values, notes = normalize_external_percents({"net_margin_pct": 0.42}, cfg.kpi)
    assert values["net_margin_pct"] == pytest.approx(42.0)
    assert any("AMBIGUOUS" in n for n in notes), "an undecidable conversion must be flagged"


def test_derivation_refuses_meaningless_domains() -> None:
    """Negative equity would score as conservative leverage if simply divided."""
    metrics, notes = derive_metrics(
        {"total_debt": 100.0, "shareholders_equity": -50.0, "revenue": 1000.0}
    )
    assert all(m.name != "debt_to_equity" for m in metrics)
    assert any("debt_to_equity" in n for n in notes)


def test_division_by_zero_yields_a_note_not_infinity() -> None:
    metrics, notes = derive_metrics({"current_assets": 10.0, "current_liabilities": 0.0})
    assert all(m.name != "current_ratio" for m in metrics)
    assert any("current_ratio" in n for n in notes)


def test_pe_ratio_requires_an_external_price() -> None:
    """A 10-K contains no share price; a P/E inferred from one would be fabricated."""
    assert pe_ratio(92.5, 3.64).value == pytest.approx(25.41, abs=0.01)
    assert pe_ratio(92.5, -1.0) is None


def test_pe_absent_without_price_and_the_reason_is_recorded(financial_chunks, cfg) -> None:
    outcome = extract_kpis(financial_chunks, cfg)
    assert outcome.metrics.get_derived("pe_ratio") is None
    assert any("pe_ratio" in n and "market price" in n for n in outcome.notes)


def test_yoy_refuses_negative_or_zero_base() -> None:
    assert yoy_change_pct(100.0, 0.0) is None
    assert yoy_change_pct(100.0, -50.0) is None
    assert yoy_change_pct(110.0, 100.0) == pytest.approx(10.0)


def test_extraction_method_is_recorded(financial_chunks, cfg) -> None:
    """Section 6: the report must say which extractor produced each value."""
    outcome = extract_kpis(financial_chunks, cfg)
    assert outcome.metrics.extracted
    assert all(m.method is ExtractionMethod.RULE_BASED for m in outcome.metrics.extracted)
    assert outcome.tagger_used is False
    assert any("rule-based" in n for n in outcome.notes)

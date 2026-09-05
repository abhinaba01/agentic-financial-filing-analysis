"""Derived metrics: computed, never guessed (section 6).

Every derived value ships with the formula and the operands that produced it, so
a reader can redo the arithmetic without trusting the pipeline. That is also why
each rule below is a small pure function over a metric dictionary rather than an
LLM call: there is no reason to ask a language model what gross margin is.

Guards that matter:

* division by zero returns nothing rather than infinity;
* negative shareholders' equity makes debt-to-equity meaningless (a company with
  negative book value scores as low-leverage if you just divide), so it is
  skipped with a warning instead of published;
* capital expenditure appears as a negative in the cash-flow statement and as a
  positive in an MD&A sentence, so free cash flow uses its magnitude.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from affa.schema import DerivedMetric, ExtractionMethod, SourceRef

Values = Mapping[str, float]


@dataclass(frozen=True)
class DerivationRule:
    name: str
    formula: str
    inputs: tuple[str, ...]
    unit: str
    fn: Callable[[Values], float | None]
    note: str = ""


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _margin(part: str, whole: str) -> Callable[[Values], float | None]:
    def compute(v: Values) -> float | None:
        # Margins against a negative or zero revenue base are not interpretable.
        if v[whole] <= 0:
            return None
        return v[part] / v[whole] * 100.0

    return compute


def _ebitda(v: Values) -> float | None:
    return v["operating_income"] + v["depreciation_amortization"]


def _free_cash_flow(v: Values) -> float | None:
    # Capex is a cash outflow however the filing signs it.
    return v["operating_cash_flow"] - abs(v["capital_expenditure"])


def _debt_to_equity(v: Values) -> float | None:
    equity = v["shareholders_equity"]
    if equity <= 0:
        # Negative equity would produce a negative ratio that reads as
        # conservative leverage in the rubric. Refuse to emit it.
        return None
    return v["total_debt"] / equity


def _current_ratio(v: Values) -> float | None:
    return _safe_div(v["current_assets"], v["current_liabilities"])


def _roe(v: Values) -> float | None:
    equity = v["shareholders_equity"]
    if equity <= 0:
        return None
    return v["net_income"] / equity * 100.0


def _roa(v: Values) -> float | None:
    if v["total_assets"] <= 0:
        return None
    return v["net_income"] / v["total_assets"] * 100.0


def _ocf_to_net_income(v: Values) -> float | None:
    ni = v["net_income"]
    if ni <= 0:
        # The ratio is an earnings-quality check; with a loss it has no meaning.
        return None
    return v["operating_cash_flow"] / ni


def _interest_coverage(v: Values) -> float | None:
    interest = abs(v["interest_expense"])
    return _safe_div(v["operating_income"], interest)


DERIVATION_RULES: tuple[DerivationRule, ...] = (
    DerivationRule(
        "gross_margin_pct",
        "gross_profit / revenue * 100",
        ("gross_profit", "revenue"),
        "percent",
        _margin("gross_profit", "revenue"),
    ),
    DerivationRule(
        "operating_margin_pct",
        "operating_income / revenue * 100",
        ("operating_income", "revenue"),
        "percent",
        _margin("operating_income", "revenue"),
    ),
    DerivationRule(
        "net_margin_pct",
        "net_income / revenue * 100",
        ("net_income", "revenue"),
        "percent",
        _margin("net_income", "revenue"),
    ),
    DerivationRule(
        "ebitda",
        "operating_income + depreciation_amortization",
        ("operating_income", "depreciation_amortization"),
        "USD",
        _ebitda,
        note="EBITDA approximated from operating income; excludes non-recurring items.",
    ),
    DerivationRule(
        "free_cash_flow",
        "operating_cash_flow - abs(capital_expenditure)",
        ("operating_cash_flow", "capital_expenditure"),
        "USD",
        _free_cash_flow,
    ),
    DerivationRule(
        "debt_to_equity",
        "total_debt / shareholders_equity",
        ("total_debt", "shareholders_equity"),
        "ratio",
        _debt_to_equity,
    ),
    DerivationRule(
        "current_ratio",
        "current_assets / current_liabilities",
        ("current_assets", "current_liabilities"),
        "ratio",
        _current_ratio,
    ),
    DerivationRule(
        "roe_pct",
        "net_income / shareholders_equity * 100",
        ("net_income", "shareholders_equity"),
        "percent",
        _roe,
    ),
    DerivationRule(
        "roa_pct",
        "net_income / total_assets * 100",
        ("net_income", "total_assets"),
        "percent",
        _roa,
    ),
    DerivationRule(
        "ocf_to_net_income",
        "operating_cash_flow / net_income",
        ("operating_cash_flow", "net_income"),
        "ratio",
        _ocf_to_net_income,
    ),
    DerivationRule(
        "interest_coverage",
        "operating_income / abs(interest_expense)",
        ("operating_income", "interest_expense"),
        "ratio",
        _interest_coverage,
    ),
)

# Depends on ebitda, which is itself derived, so it runs in a second pass.
SECOND_PASS_RULES: tuple[DerivationRule, ...] = (
    DerivationRule(
        "ebitda_margin_pct",
        "ebitda / revenue * 100",
        ("ebitda", "revenue"),
        "percent",
        _margin("ebitda", "revenue"),
    ),
    DerivationRule(
        "fcf_margin_pct",
        "free_cash_flow / revenue * 100",
        ("free_cash_flow", "revenue"),
        "percent",
        _margin("free_cash_flow", "revenue"),
    ),
)


def derive_metrics(
    values: dict[str, float],
    sources: Mapping[str, SourceRef] | None = None,
) -> tuple[list[DerivedMetric], list[str]]:
    """Compute every derivable metric from ``values``.

    Returns the metrics and a list of human-readable notes about what could not
    be computed and why - a metric silently absent is indistinguishable from a
    metric the pipeline forgot about.
    """
    sources = sources or {}
    out: list[DerivedMetric] = []
    notes: list[str] = []
    available = dict(values)

    for rules in (DERIVATION_RULES, SECOND_PASS_RULES):
        for rule in rules:
            missing = [i for i in rule.inputs if i not in available]
            if missing:
                notes.append(f"{rule.name}: not computed, missing {', '.join(missing)}")
                continue
            operands = {i: available[i] for i in rule.inputs}
            try:
                value = rule.fn(operands)
            except (ZeroDivisionError, TypeError) as exc:
                notes.append(f"{rule.name}: not computed ({exc})")
                continue
            if value is None:
                notes.append(
                    f"{rule.name}: not computed, inputs out of valid domain "
                    f"({', '.join(f'{k}={v:g}' for k, v in operands.items())})"
                )
                continue
            refs = [sources[i] for i in rule.inputs if i in sources]
            out.append(
                DerivedMetric(
                    name=rule.name,
                    value=round(value, 6),
                    unit=rule.unit,
                    formula=rule.formula,
                    operands=operands,
                    sources=refs,
                    method=ExtractionMethod.DERIVED,
                )
            )
            available[rule.name] = value
    return out, notes


def pe_ratio(
    market_price_per_share: float, eps_diluted: float, source: SourceRef | None = None
) -> DerivedMetric | None:
    """P/E, which needs a market price.

    Section 6 is explicit that the price is an optional *input*, never inferred
    from the filing: a 10-K does not contain the current share price, and any
    P/E derived without one is fabricated.
    """
    if eps_diluted <= 0:
        return None
    value = market_price_per_share / eps_diluted
    return DerivedMetric(
        name="pe_ratio",
        value=round(value, 4),
        unit="ratio",
        formula="market_price_per_share / eps_diluted",
        operands={
            "market_price_per_share": market_price_per_share,
            "eps_diluted": eps_diluted,
        },
        sources=[source] if source else [],
        method=ExtractionMethod.DERIVED,
    )


def yoy_change_pct(current: float, prior: float) -> float | None:
    """Year-over-year percentage change.

    Undefined when the prior period is zero, and unstable in sign when the prior
    period is negative (a loss shrinking looks like a large negative "growth"),
    so both are refused rather than reported.
    """
    if prior == 0 or prior < 0:
        return None
    return (current - prior) / abs(prior) * 100.0

"""Rubric-based assessment (section 7).

The verdict is a deterministic aggregation over signals the system extracted and
the ``verify`` node checked. The LLM writes the prose around it and never
chooses it, which is what makes the recommendation reproducible: the same filing
and the same rubric version give the same verdict every time.

What this module deliberately does *not* do: predict returns. There is no ground
truth for "should I invest in this company", so labelling filings with subsequent
price movement and training on it would produce a confident non-generaliser.
What is offered instead is an explicit, versioned, adjustable opinion whose
weights and thresholds are in ``configs/rubric_v1.yaml`` where a reader can
disagree with them.

``insufficient_evidence`` is a real outcome, not a courtesy. When too few factors
can be scored from verified, cited metrics, the system says so and publishes no
aggregate score - the schema refuses to serialise one in that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from affa.config import AffaConfig, get_config, load_rubric
from affa.kpi.catalog import metric_label
from affa.schema import (
    Assessment,
    Disagreement,
    FinancialMetrics,
    RationaleItem,
    Recommendation,
    SourceRef,
)


@dataclass
class FactorScore:
    name: str
    score: float
    weight: float
    components: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    citations: list[SourceRef] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return not self.missing and bool(self.components)


def score_from_bands(value: float, bands: list[dict[str, Any]]) -> tuple[float, str]:
    """Map a value onto its band. Bands are ordered; ``max: null`` is open-ended."""
    for band in bands:
        ceiling = band.get("max")
        if ceiling is None or value <= float(ceiling):
            return float(band["score"]), str(band.get("label", ""))
    last = bands[-1]
    return float(last["score"]), str(last.get("label", ""))


def _collect_value(name: str, values: dict[str, float]) -> float | None:
    return values.get(name)


def score_factor(
    name: str,
    spec: dict[str, Any],
    values: dict[str, float],
    sources: dict[str, SourceRef],
) -> FactorScore:
    """Score one rubric factor from the metrics available.

    A factor is only *scored* when all of its ``required_metrics`` are present.
    Optional metrics refine the score when available but never rescue a factor
    whose required inputs are missing - that is what keeps ``insufficient_evidence``
    reachable instead of decorative.
    """
    weight = float(spec.get("weight", 0.0))
    required = list(spec.get("required_metrics", []))
    optional = list(spec.get("optional_metrics", []))
    scoring = spec.get("scoring", {})

    missing = [m for m in required if _collect_value(m, values) is None]
    components: dict[str, float] = {}
    labels: dict[str, str] = {}
    citations: list[SourceRef] = []

    for metric in required + optional:
        value = _collect_value(metric, values)
        if value is None or metric not in scoring:
            continue
        sub_score, label = score_from_bands(value, scoring[metric]["bands"])
        components[metric] = sub_score
        labels[metric] = label
        if metric in sources:
            citations.append(sources[metric])

    score = sum(components.values()) / len(components) if components else 0.0
    return FactorScore(
        name=name,
        score=round(max(-1.0, min(1.0, score)), 4),
        weight=weight,
        components=components,
        labels=labels,
        citations=citations,
        missing=missing,
    )


def assessment_from_score(aggregate: float, bands: list[dict[str, Any]]) -> Assessment:
    for band in bands:
        ceiling = band.get("max")
        if ceiling is None or aggregate <= float(ceiling):
            return Assessment(band["assessment"])
    return Assessment(bands[-1]["assessment"])


def _statement(factor: FactorScore) -> str:
    """Plain-language summary of a factor, built from the rubric's own band labels.

    Deterministic on purpose: this is the sentence that appears in the report's
    rationale, so it has to say exactly what the rubric did. The LLM's narrative
    sits alongside it, not in place of it.
    """
    detail = ", ".join(
        f"{metric_label(m)}: {factor.labels.get(m) or f'{s:+.2f}'}"
        for m, s in factor.components.items()
    )
    if factor.score > 0.15:
        direction = "supports"
    elif factor.score < -0.15:
        direction = "weighs against"
    else:
        direction = "is neutral for"
    return f"{factor.name.replace('_', ' ').capitalize()} {direction} the assessment ({detail})."


@dataclass
class RubricOutcome:
    recommendation: Recommendation
    factor_scores: list[FactorScore]
    notes: list[str] = field(default_factory=list)


def build_values(
    metrics: FinancialMetrics,
    *,
    sentiment_score: float | None = None,
    risk_severity_index: float | None = None,
) -> tuple[dict[str, float], dict[str, SourceRef]]:
    """Flatten everything the rubric can read into one name -> value mapping."""
    values: dict[str, float] = {}
    sources: dict[str, SourceRef] = {}

    for m in metrics.extracted:
        values[m.name] = m.value_in_units
        sources[m.name] = m.source
    for d in metrics.derived:
        values[d.name] = d.value
        if d.sources:
            sources[d.name] = d.sources[0]
    values.update(metrics.yoy_changes)
    for name in metrics.yoy_changes:
        base = name.removesuffix("_yoy_pct")
        if base in sources:
            sources[name] = sources[base]

    if sentiment_score is not None:
        values["sentiment_score"] = sentiment_score
    if risk_severity_index is not None:
        values["risk_severity_index"] = risk_severity_index
    return values, sources


def evaluate(
    metrics: FinancialMetrics,
    *,
    cfg: AffaConfig | None = None,
    rubric: dict[str, Any] | None = None,
    sentiment_score: float | None = None,
    risk_severity_index: float | None = None,
    disagreements: list[Disagreement] | None = None,
    verified_citations: bool = True,
) -> RubricOutcome:
    """Run the rubric and produce a :class:`Recommendation`."""
    cfg = cfg or get_config()
    rubric = rubric or load_rubric(cfg)
    disagreements = disagreements or []

    values, sources = build_values(
        metrics, sentiment_score=sentiment_score, risk_severity_index=risk_severity_index
    )

    factors_spec: dict[str, Any] = rubric["factors"]
    scores = [score_factor(name, spec, values, sources) for name, spec in factors_spec.items()]
    scored = [f for f in scores if f.scored]
    unscored = [f for f in scores if not f.scored]

    total_weight = sum(float(s.get("weight", 0.0)) for s in factors_spec.values()) or 1.0
    covered_weight = sum(f.weight for f in scored)
    weight_coverage = covered_weight / total_weight

    sufficiency = rubric.get("sufficiency", {})
    min_factors = int(sufficiency.get("min_factors_scored", 0))
    min_weight = float(sufficiency.get("min_weight_covered", 0.0))
    needs_citations = bool(sufficiency.get("require_verified_citations", False))

    notes: list[str] = []
    for f in unscored:
        notes.append(f"factor {f.name!r} not scored: missing {', '.join(f.missing) or 'inputs'}")

    citation_coverage = sum(1 for f in scored if f.citations) / len(scored) if scored else 0.0

    insufficient_reasons: list[str] = []
    if len(scored) < min_factors:
        insufficient_reasons.append(
            f"only {len(scored)} of {len(scores)} factors could be scored, "
            f"rubric requires {min_factors}"
        )
    if weight_coverage < min_weight:
        insufficient_reasons.append(
            f"scored factors cover {weight_coverage:.0%} of rubric weight, "
            f"requires {min_weight:.0%}"
        )
    if needs_citations and not verified_citations:
        insufficient_reasons.append("no verified citations available for the scored factors")
    if needs_citations and scored and citation_coverage == 0.0:
        insufficient_reasons.append("no scored factor carries a source citation")

    conf_spec = rubric.get("confidence", {})
    disagreement_fraction = (
        len(disagreements) / max(len(metrics.extracted), 1) if metrics.extracted else 0.0
    )
    confidence = (
        float(conf_spec.get("base", 0.3))
        + float(conf_spec.get("weight_coverage_bonus", 0.4)) * weight_coverage
        + float(conf_spec.get("citation_coverage_bonus", 0.2)) * citation_coverage
        - float(conf_spec.get("disagreement_penalty", 0.15)) * disagreement_fraction
    )
    confidence = max(0.0, min(float(conf_spec.get("max", 0.95)), confidence))

    if insufficient_reasons:
        notes.extend(insufficient_reasons)
        return RubricOutcome(
            recommendation=Recommendation(
                assessment=Assessment.INSUFFICIENT_EVIDENCE,
                # Confidence in the *verdict*, and the verdict is "we cannot say".
                confidence=round(min(confidence, 0.35), 4),
                rubric_version=str(rubric["version"]),
                factor_scores={f.name: f.score for f in scored},
                factors_scored=[f.name for f in scored],
                factors_missing=[f.name for f in unscored],
                weight_covered=round(weight_coverage, 4),
                aggregate_score=None,  # schema forbids one here, by design
                rationale=[],
                disclaimer=cfg.report.disclaimer,
            ),
            factor_scores=scores,
            notes=notes,
        )

    # Normalised by covered weight, not total weight: an unscored factor should
    # not drag the verdict toward neutral as if it had been measured at zero.
    # Coverage is expressed through confidence and the sufficiency gate instead.
    aggregate = sum(f.score * f.weight for f in scored) / covered_weight

    rationale = [
        RationaleItem(
            factor=f.name,
            statement=_statement(f),
            score=f.score,
            citations=f.citations[:3],
        )
        for f in scored
    ]

    return RubricOutcome(
        recommendation=Recommendation(
            assessment=assessment_from_score(aggregate, rubric["assessment_bands"]),
            confidence=round(confidence, 4),
            rubric_version=str(rubric["version"]),
            factor_scores={f.name: f.score for f in scored},
            factors_scored=[f.name for f in scored],
            factors_missing=[f.name for f in unscored],
            weight_covered=round(weight_coverage, 4),
            aggregate_score=round(aggregate, 4),
            rationale=rationale,
            disclaimer=cfg.report.disclaimer,
        ),
        factor_scores=scores,
        notes=notes,
    )

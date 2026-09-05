"""Combine rule-based and model-based extraction into the report's metric block.

Section 6 requires recording *which method produced each value* and surfacing
disagreements rather than resolving them behind the reader's back. The two
extractors are run independently over the same chunks and reconciled here:

* both present and agreeing  -> ``both_agree`` (highest confidence)
* both present and differing -> a ``Disagreement`` entry is emitted, the value
  is resolved by an explicit, stated policy, and the report shows both numbers
* one present                -> that one, labelled with its method
"""

from __future__ import annotations

from dataclasses import dataclass, field

from affa.config import AffaConfig, KpiConfig
from affa.ingestion.types import Chunk
from affa.kpi.catalog import PERCENT_METRICS, YOY_METRICS
from affa.kpi.derive import derive_metrics, pe_ratio, yoy_change_pct
from affa.kpi.rules import RuleHit, extract_rule_based
from affa.kpi.units import PercentConvention, compare_values, normalize_percent
from affa.kpi.xbrl import TagHit, XBRLTagger
from affa.schema import (
    Disagreement,
    ExtractedMetric,
    ExtractionMethod,
    FinancialMetrics,
    Scale,
    SourceRef,
)


@dataclass
class ExtractionOutcome:
    metrics: FinancialMetrics
    notes: list[str] = field(default_factory=list)
    tagger_used: bool = False
    prior_period_values: dict[str, float] = field(default_factory=dict)

    @property
    def current_values(self) -> dict[str, float]:
        """Absolute-unit values keyed by metric name, for derivation and the rubric."""
        return {m.name: m.value_in_units for m in self.metrics.extracted}


def _best_rule_hit(hits: list[RuleHit]) -> RuleHit | None:
    """Pick the most trustworthy hit for a metric in the current period.

    Table rows beat prose, higher confidence beats lower, and the first numeric
    column is the current period. Ties break toward the earliest occurrence,
    which in a filing is the primary statement rather than a later restatement
    or a segment note.
    """
    current = [h for h in hits if h.is_current_period] or hits
    if not current:
        return None
    return max(
        current,
        key=lambda h: (h.confidence, h.raw_text.count("|"), -abs(h.column_index)),
    )


def _prior_value(hits: list[RuleHit]) -> float | None:
    """Second numeric column: the prior comparative period."""
    priors = [h for h in hits if h.column_index == 1]
    if not priors:
        return None
    best = max(priors, key=lambda h: h.confidence)
    return best.value * _scale_multiplier(best.scale)


def _scale_multiplier(scale: Scale) -> float:
    from affa.schema import SCALE_MULTIPLIER

    return SCALE_MULTIPLIER[scale]


def reconcile(
    rule_hits: list[RuleHit],
    tag_hits: list[TagHit],
    *,
    kpi_cfg: KpiConfig,
) -> tuple[list[ExtractedMetric], list[Disagreement], dict[str, float]]:
    """Merge the two extractors, recording provenance and disagreements."""
    by_metric_rules: dict[str, list[RuleHit]] = {}
    for h in rule_hits:
        by_metric_rules.setdefault(h.metric, []).append(h)
    by_metric_tags: dict[str, list[TagHit]] = {}
    for t in tag_hits:
        by_metric_tags.setdefault(t.metric, []).append(t)

    extracted: list[ExtractedMetric] = []
    disagreements: list[Disagreement] = []
    priors: dict[str, float] = {}

    for metric in sorted(set(by_metric_rules) | set(by_metric_tags)):
        rule = _best_rule_hit(by_metric_rules.get(metric, []))
        tags = by_metric_tags.get(metric, [])
        tag = max(tags, key=lambda t: t.confidence) if tags else None

        if (prior := _prior_value(by_metric_rules.get(metric, []))) is not None:
            priors[metric] = prior

        if rule is not None and tag is not None:
            rule_abs = rule.value * _scale_multiplier(rule.scale)
            tag_abs = tag.value * _scale_multiplier(tag.scale)
            cmp = compare_values(tag_abs, rule_abs, tolerance_pct=kpi_cfg.tolerance_pct)
            if cmp.is_match:
                extracted.append(_to_metric(rule, ExtractionMethod.BOTH_AGREE, confidence=0.95))
                continue
            disagreements.append(
                Disagreement(
                    name=metric,
                    xbrl_model=tag_abs,
                    rule_based=rule_abs,
                    relative_difference_pct=cmp.relative_error_pct,
                    # Stated policy: trust the tagger, because it reads the number
                    # in context while the rule matches a nearby label. Both
                    # numbers stay in the report so the reader can disagree.
                    resolved_to=ExtractionMethod.XBRL_MODEL,
                )
            )
            extracted.append(_tag_to_metric(tag, confidence=min(0.85, tag.confidence)))
            continue

        if rule is not None:
            extracted.append(_to_metric(rule, ExtractionMethod.RULE_BASED, rule.confidence))
        elif tag is not None:
            extracted.append(_tag_to_metric(tag, confidence=tag.confidence))

    return extracted, disagreements, priors


def _to_metric(hit: RuleHit, method: ExtractionMethod, confidence: float) -> ExtractedMetric:
    return ExtractedMetric(
        name=hit.metric,
        value=hit.value,
        unit=hit.unit,
        scale=hit.scale,
        period=hit.period_hint,
        source=SourceRef(chunk_id=hit.chunk_id, page=hit.page, raw_text=hit.raw_text),
        method=method,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
    )


def _tag_to_metric(hit: TagHit, confidence: float) -> ExtractedMetric:
    return ExtractedMetric(
        name=hit.metric,
        value=hit.value,
        unit="USD",
        scale=hit.scale,
        source=SourceRef(chunk_id=hit.chunk_id, page=hit.page, raw_text=hit.raw_text),
        method=ExtractionMethod.XBRL_MODEL,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
    )


def normalize_external_percents(
    values: dict[str, float], kpi_cfg: KpiConfig
) -> tuple[dict[str, float], list[str]]:
    """Put percentage metrics *of unknown provenance* into the canonical convention.

    Only for values arriving from outside the pipeline - hand-labelled gold
    files, user-supplied overrides, a third-party feed - where the convention was
    never declared. Every conversion is returned as a note so it is visible in
    the report and in the evaluation output.

    Deliberately **not** applied to values this pipeline derives. Those are in
    points by construction (:mod:`affa.kpi.derive` multiplies by 100), and
    running convention inference over them corrupts exactly the small ones:
    a true -0.96% YoY change sits inside the undecidable band and would be
    "corrected" to -96%. Guessing a convention you already know is how a correct
    number becomes a wrong one, and it is the same class of error as
    anti-pattern #12.
    """
    declared = (
        PercentConvention.POINTS
        if kpi_cfg.percent_canonical == "points"
        else PercentConvention.FRACTION
    )
    out = dict(values)
    notes: list[str] = []
    for name in list(out):
        if name not in PERCENT_METRICS:
            continue
        result = normalize_percent(
            out[name],
            convention=PercentConvention.UNKNOWN,
            ambiguity_band=kpi_cfg.percent_ambiguity_band,
        )
        if result.converted:
            out[name] = result.value
            notes.append(
                f"{name}: converted {values[name]:g} -> {result.value:g} assuming "
                f"{result.assumed.value} convention"
                + (
                    " (AMBIGUOUS - value lay inside the undecidable band)"
                    if result.ambiguous
                    else ""
                )
            )
        elif declared is PercentConvention.FRACTION:
            notes.append(f"{name}: left as {out[name]:g}; canonical convention is fraction")
    return out, notes


def extract_kpis(
    chunks: list[Chunk],
    cfg: AffaConfig,
    *,
    tagger: XBRLTagger | None = None,
    market_price_per_share: float | None = None,
) -> ExtractionOutcome:
    """Full KPI extraction: rules + model, reconciliation, derivation, YoY."""
    rule_hits = extract_rule_based(chunks)

    tagger = tagger if tagger is not None else XBRLTagger(cfg.models.xbrl_tagger)
    tagger_used = tagger.load()
    tag_hits = tagger.tag_chunks(chunks) if tagger_used else []

    extracted, disagreements, priors = reconcile(rule_hits, tag_hits, kpi_cfg=cfg.kpi)

    notes: list[str] = []
    if not tagger_used:
        notes.append(
            "XBRL tagger not loaded; all values come from the rule-based extractor. "
            "The extraction-method field on each metric records this."
        )

    values = {m.name: m.value_in_units for m in extracted}
    sources = {m.name: m.source for m in extracted}

    derived, derive_notes = derive_metrics(values, sources)
    notes.extend(derive_notes)

    if market_price_per_share is not None:
        eps = values.get("eps_diluted")
        if eps is not None:
            pe = pe_ratio(market_price_per_share, eps, sources.get("eps_diluted"))
            if pe is not None:
                derived.append(pe)
            else:
                notes.append("pe_ratio: not computed, diluted EPS is not positive")
        else:
            notes.append("pe_ratio: not computed, diluted EPS was not extracted")
    else:
        notes.append(
            "pe_ratio: not computed, no market price supplied "
            "(a filing does not contain one; it is an explicit optional input)"
        )

    yoy: dict[str, float] = {}
    for metric in YOY_METRICS:
        current, prior = values.get(metric), priors.get(metric)
        if current is None or prior is None:
            continue
        change = yoy_change_pct(current, prior)
        if change is None:
            notes.append(
                f"{metric}_yoy_pct: not computed, prior period is zero or negative "
                f"(prior={prior:g})"
            )
            continue
        # Already in percentage points: yoy_change_pct multiplies by 100. No
        # convention inference here - see normalize_external_percents.
        yoy[f"{metric}_yoy_pct"] = round(change, 4)

    return ExtractionOutcome(
        metrics=FinancialMetrics(
            extracted=extracted,
            derived=derived,
            yoy_changes=yoy,
            disagreements=disagreements,
        ),
        notes=notes,
        tagger_used=tagger_used,
        prior_period_values=priors,
    )

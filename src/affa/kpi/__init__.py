"""KPI extraction: rules + XBRL model, derivation, unit handling."""

from affa.kpi.catalog import EXTRACTED_METRICS, METRICS_BY_NAME, MetricSpec
from affa.kpi.derive import derive_metrics, pe_ratio, yoy_change_pct
from affa.kpi.extract import ExtractionOutcome, extract_kpis
from affa.kpi.rules import extract_rule_based, match_metric
from affa.kpi.units import compare_values, detect_scale, parse_financial_number
from affa.kpi.xbrl import XBRLTagger

__all__ = [
    "EXTRACTED_METRICS",
    "METRICS_BY_NAME",
    "ExtractionOutcome",
    "MetricSpec",
    "XBRLTagger",
    "compare_values",
    "derive_metrics",
    "detect_scale",
    "extract_kpis",
    "extract_rule_based",
    "match_metric",
    "parse_financial_number",
    "pe_ratio",
    "yoy_change_pct",
]

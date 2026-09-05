"""Rule-based KPI extraction with provenance.

This is the baseline the fine-tuned XBRL tagger (section 5.1) is measured
against, and the fallback when that model is not loaded. It is deliberately
conservative: it would rather return nothing than return a number it matched to
the wrong label, because a missing metric shows up as ``insufficient_evidence``
whereas a wrong one shows up as a confident, cited, incorrect report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from affa.ingestion.types import Chunk
from affa.kpi.catalog import EXTRACTED_METRICS, MetricSpec
from affa.kpi.units import ParsedNumber, detect_scale, parse_financial_number
from affa.schema import ExtractionMethod, Scale

# A statement row: label, then one or more period columns.
#   "Total net sales | 383,285 | 394,328"
#   "Total net sales    383,285   394,328"
_CELL_SPLIT_RE = re.compile(r"\s*\|\s*|\s{2,}|\t")

# Column headers that identify which period a column holds.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class RuleHit:
    """One extracted value plus everything needed to trace and score it."""

    metric: str
    value: float
    scale: Scale
    unit: str
    chunk_id: str
    page: int | None
    raw_text: str
    column_index: int = 0
    period_hint: str | None = None
    confidence: float = 0.6
    method: ExtractionMethod = ExtractionMethod.RULE_BASED

    @property
    def is_current_period(self) -> bool:
        """Filings put the most recent period in the first numeric column."""
        return self.column_index == 0


def _normalize_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9%&'\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_metric(
    label: str, specs: tuple[MetricSpec, ...] = EXTRACTED_METRICS
) -> MetricSpec | None:
    """Match a row label to a metric.

    Patterns are checked longest-first across all metrics, not metric-by-metric,
    so "total cost of revenue" cannot be claimed by ``revenue`` just because
    ``revenue`` is earlier in the catalogue. Matching the wrong metric is the
    expensive failure here; matching nothing is cheap.
    """
    norm = _normalize_label(label)
    if not norm or len(norm) > 90:
        return None

    candidates: list[tuple[int, MetricSpec, str]] = []
    for spec in specs:
        for pat in spec.patterns:
            p = _normalize_label(pat)
            if p and p in norm:
                candidates.append((len(p), spec, p))
    if not candidates:
        return None

    candidates.sort(key=lambda t: -t[0])
    best_len, best_spec, best_pat = candidates[0]
    # Reject a match that covers only a sliver of a long label: "revenue" inside
    # "deferred revenue recognised during the period" is not total revenue.
    if best_len / max(len(norm), 1) < 0.34 and len(norm) > 28:
        return None
    return best_spec


def _row_cells(line: str) -> tuple[str, list[str]]:
    parts = [p.strip() for p in _CELL_SPLIT_RE.split(line) if p.strip()]
    if not parts:
        return "", []
    return parts[0], parts[1:]


def extract_from_chunk(chunk: Chunk, *, default_scale: Scale | None = None) -> list[RuleHit]:
    """Extract every metric this chunk states, in document order."""
    scale = default_scale or detect_scale(chunk.text, Scale.UNITS)
    hits: list[RuleHit] = []

    for line in chunk.text.splitlines():
        if not line.strip():
            continue
        label, cells = _row_cells(line)
        spec = match_metric(label)
        if spec is None:
            continue

        # Parsed WITHOUT a default scale, so `scale_hint` means "this figure
        # carried its own magnitude word" and nothing else. Applying the
        # statement-level scale is a separate decision made per metric below.
        numbers: list[tuple[int, ParsedNumber]] = []
        for i, cell in enumerate(cells):
            parsed = parse_financial_number(cell)
            if parsed is not None:
                numbers.append((i, parsed))

        if not numbers and not cells:
            # Prose form: "Total net sales were $383,285 million in 2023."
            parsed = parse_financial_number(line[len(label) :])
            if parsed is not None:
                numbers.append((0, parsed))

        for col, num in enumerate(n for _, n in numbers):
            if spec.is_per_share and abs(num.value) > 1000:
                # A per-share figure in the thousands is a share count that drifted
                # into an EPS row, not an EPS.
                continue
            if not spec.negative_ok and num.value < 0:
                continue
            value = num.value
            if spec.is_per_share:
                # "(In millions, except per share data)" - the header's scale
                # explicitly does not reach EPS. Applying it turns $6.13 of
                # earnings per share into $6.13 million, and every P/E built on
                # it collapses to zero.
                effective_scale = num.scale_hint or Scale.UNITS
            elif spec.unit == "shares":
                # Share counts state their own magnitude when they have one.
                effective_scale = num.scale_hint or Scale.UNITS
            else:
                effective_scale = num.scale_hint or scale
            hits.append(
                RuleHit(
                    metric=spec.name,
                    value=value,
                    scale=effective_scale,
                    unit=spec.unit,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    raw_text=line.strip()[:300],
                    column_index=col,
                    confidence=0.7 if chunk.chunk_type == "table" else 0.5,
                )
            )
    return hits


def extract_rule_based(chunks: list[Chunk], *, default_scale: Scale | None = None) -> list[RuleHit]:
    """Run rule extraction across a document's chunks."""
    out: list[RuleHit] = []
    for chunk in chunks:
        out.extend(extract_from_chunk(chunk, default_scale=default_scale))
    return out


def period_columns(chunk_text: str) -> list[str]:
    """Period labels for a statement's numeric columns, if the header names them."""
    for line in chunk_text.splitlines()[:4]:
        years = _YEAR_RE.findall(line)
        if len(years) >= 2:
            return [m.group(0) for m in _YEAR_RE.finditer(line)]
    return []

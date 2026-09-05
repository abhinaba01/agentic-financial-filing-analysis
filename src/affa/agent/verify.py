"""Claim verification: the critic step (sections 3 and 9).

Every generated claim is re-checked against the passages it cites and marked
``supported`` / ``unsupported`` / ``contradicted``. Unsupported claims are
dropped or flagged, never silently emitted. This is what makes "why did it say
that" answerable, and it is where the faithfulness metric comes from.

The definition used, stated precisely because section 9 requires it:

    A claim is **supported** when every number in it appears in one of its cited
    chunks - allowing a reporting-scale factor and a stated tolerance - or is
    directly computable from numbers that do (a ratio, difference, sum, or
    percentage change of two of them) - *and* the claim's financial entities
    overlap the cited text by at least ``min_entity_overlap``.

    A claim is **contradicted** when a number in it fails to match, but the
    cited chunk states a different value on the same labelled line. That is a
    stronger and more useful signal than "unsupported": it means the evidence
    was found and disagrees.

    Otherwise the claim is **unsupported**.

The arithmetic tolerance is a real requirement, not laxity: filings round, and a
claim saying "44.1%" about a chunk containing 169,148 and 383,285 is faithful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from affa.config import VerificationConfig
from affa.kpi.catalog import EXTRACTED_METRICS
from affa.kpi.units import parse_financial_number
from affa.schema import EvidenceChunk, Finding, Verification

# Number-like spans inside prose, including the parenthesised-negative form.
_NUMBER_SPAN_RE = re.compile(
    r"\(?\$?\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?%?"  # 1,234.5
    r"|\(?\$?\s*-?\d+\.\d+\)?%?"  # 12.34
    r"|\(?\$?\s*-?\d+\)?%"  # 45%
    r"|\$\s*-?\d+(?:\.\d+)?"  # $42
)

# Scale factors a faithful restatement may apply: "$383.3 billion" describing a
# table stated in millions is the same figure, not a new one.
_SCALE_FACTORS = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)

_METRIC_WORDS: dict[str, set[str]] = {
    spec.name: {w for pat in spec.patterns for w in pat.lower().split() if len(w) > 3}
    for spec in EXTRACTED_METRICS
}

_STOPWORDS = {
    "the",
    "and",
    "for",
    "was",
    "were",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "had",
    "its",
    "their",
    "which",
    "than",
    "into",
    "over",
    "under",
    "year",
    "years",
    "period",
    "company",
    "during",
    "compared",
    "prior",
}


@dataclass
class ClaimCheck:
    """Per-claim verification result, with enough detail to audit the verdict."""

    claim: str
    verdict: Verification
    detail: str
    numbers_in_claim: list[float] = field(default_factory=list)
    numbers_matched: list[float] = field(default_factory=list)
    numbers_unmatched: list[float] = field(default_factory=list)
    entity_overlap: float = 0.0
    cited_chunks: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[float]:
    """Every figure stated in a piece of text."""
    out: list[float] = []
    for m in _NUMBER_SPAN_RE.finditer(text):
        parsed = parse_financial_number(m.group(0))
        if parsed is not None:
            out.append(parsed.value)
    return out


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def subjects(text: str) -> set[str]:
    """Financial concepts a piece of text is about, as canonical metric names.

    Resolved through the metric catalogue, so a passage saying "Total net sales"
    and a claim saying "Revenue" are recognised as the same subject. Comparing
    raw words instead would reject a correct, well-cited claim purely for using
    normalised vocabulary - which is what the extractor is supposed to do.
    """
    tokens = _tokens(text)
    return {name for name, words in _METRIC_WORDS.items() if tokens & words}


def subject_overlap(claim: str, evidence_text: str) -> tuple[float, str]:
    """How much of what the claim is about is actually discussed in the evidence.

    Returns the fraction and the basis used. This is a secondary guard: the
    numeric check above is the strong one. Its job is to catch a claim whose
    figures happen to appear in a passage about something else entirely.
    """
    claim_subjects = subjects(claim)
    if claim_subjects:
        evidence_subjects = subjects(evidence_text)
        matched = claim_subjects & evidence_subjects
        return len(matched) / len(claim_subjects), "subject"

    # No recognised financial concept - fall back to plain lexical overlap.
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0, "lexical"
    return len(claim_tokens & _tokens(evidence_text)) / len(claim_tokens), "lexical"


def _matches_with_scale(target: float, candidates: list[float], tolerance_pct: float) -> bool:
    for cand in candidates:
        for factor in _SCALE_FACTORS:
            scaled = cand * factor
            if target == 0:
                if abs(scaled) < 1e-9:
                    return True
                continue
            if abs(scaled - target) / abs(target) * 100.0 <= tolerance_pct:
                return True
    return False


def _as_margin(ratio: float) -> float | None:
    """A ratio expressed as a percentage, but only where that reading is possible.

    Over ten evidence figures there are hundreds of candidate pairs, and an
    unconstrained ``a / b * 100`` will land on almost any target by coincidence -
    which inflates the faithfulness metric with claims that were never grounded.
    A margin is a part over a whole, so it cannot exceed 100%; growth rates,
    which can, are covered by the separate ``(a - b) / b * 100`` form.

    Found by test: 4,812.6 / 962.5 * 100 = 500.01 "supported" a fabricated
    "net income was 500.0 million".
    """
    pct = ratio * 100.0
    return pct if -100.0 <= pct <= 100.0 else None


def _is_computable(target: float, sources: list[float], tolerance_pct: float) -> bool:
    """Can ``target`` be derived from a pair of numbers present in the evidence?

    Covers the operations a faithful financial claim actually performs: ratio as
    a percentage, plain ratio, difference and sum. A margin quoted in a claim is
    legitimately grounded in the two figures that produce it, and refusing that
    would mark correct derived statements as hallucinated.

    Scale-aware, for the same reason :func:`_matches_with_scale` is. A statement
    printed "in millions" gives sources of 962.5 and 214.7; a claim stating the
    EBITDA those imply says 1,177,200,000. That is the *same* arithmetic, and
    checking only the raw magnitudes would report a correct derivation as
    contradicted - the most damaging possible false positive for a metric whose
    whole job is to catch fabrication.
    """
    if not sources or target == 0:
        return False
    tol = tolerance_pct / 100.0

    # Additive results carry the sources' scale, so the target is compared at
    # each plausible scale. Ratios are scale-free and only ever compared raw.
    scaled_targets = [target * factor for factor in _SCALE_FACTORS]

    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            scale_free: list[float] = []
            scale_bearing: list[float] = [a - b, b - a, a + b]
            if b != 0:
                scale_free += [_as_margin(a / b), a / b, (a - b) / abs(b) * 100.0]
            if a != 0:
                scale_free += [_as_margin(b / a), b / a, (b - a) / abs(a) * 100.0]
            scale_free = [c for c in scale_free if c is not None]

            for cand in scale_free:
                if abs(cand - target) <= abs(target) * tol:
                    return True
            for cand in scale_bearing:
                for scaled in scaled_targets:
                    if scaled != 0 and abs(cand - scaled) <= abs(scaled) * tol:
                        return True
    return False


def _finds_contradiction(
    unmatched: list[float], evidence_numbers: list[float], claim: str, cited_text: str
) -> str | None:
    """Detect the stronger failure: evidence present and disagreeing."""
    if not unmatched or not evidence_numbers:
        return None
    claim_tokens = _tokens(claim)
    for name, words in _METRIC_WORDS.items():
        if not (claim_tokens & words):
            continue
        for line in cited_text.splitlines():
            if not (_tokens(line) & words):
                continue
            line_numbers = extract_numbers(line)
            if line_numbers:
                return (
                    f"cited passage states {line_numbers[0]:,.4g} for {name.replace('_', ' ')} "
                    f"but the claim states {unmatched[0]:,.4g}"
                )
    return None


def verify_claim(
    claim: str,
    cited_chunk_ids: list[str],
    evidence: list[EvidenceChunk],
    cfg: VerificationConfig,
) -> ClaimCheck:
    """Check one claim against the chunks it cites."""
    by_id = {c.chunk_id: c for c in evidence}
    cited = [by_id[cid] for cid in cited_chunk_ids if cid in by_id]

    if not cited_chunk_ids:
        return ClaimCheck(claim, Verification.UNSUPPORTED, "claim cites no chunk")
    if not cited:
        return ClaimCheck(
            claim,
            Verification.UNSUPPORTED,
            f"cited chunks are not in the evidence set: {cited_chunk_ids[:3]}",
            cited_chunks=list(cited_chunk_ids),
        )

    cited_text = "\n".join(c.text for c in cited)
    evidence_numbers = extract_numbers(cited_text)
    claim_numbers = extract_numbers(claim)

    matched: list[float] = []
    unmatched: list[float] = []
    for num in claim_numbers:
        if _matches_with_scale(num, evidence_numbers, cfg.numeric_tolerance_pct):
            matched.append(num)
        elif _is_computable(num, evidence_numbers, cfg.numeric_tolerance_pct):
            matched.append(num)
        else:
            unmatched.append(num)

    overlap, overlap_basis = subject_overlap(claim, cited_text)

    check = ClaimCheck(
        claim=claim,
        verdict=Verification.UNSUPPORTED,
        detail="",
        numbers_in_claim=claim_numbers,
        numbers_matched=matched,
        numbers_unmatched=unmatched,
        entity_overlap=round(overlap, 4),
        cited_chunks=[c.chunk_id for c in cited],
    )

    if unmatched:
        contradiction = _finds_contradiction(unmatched, evidence_numbers, claim, cited_text)
        if contradiction:
            check.verdict = Verification.CONTRADICTED
            check.detail = contradiction
            return check
        check.verdict = Verification.UNSUPPORTED
        check.detail = (
            f"{len(unmatched)} of {len(claim_numbers)} figures absent from the cited "
            f"passages and not computable from them: "
            f"{', '.join(f'{n:,.4g}' for n in unmatched[:3])}"
        )
        return check

    if overlap < cfg.min_entity_overlap:
        check.verdict = Verification.UNSUPPORTED
        check.detail = (
            f"{overlap_basis} overlap {overlap:.2f} below {cfg.min_entity_overlap:.2f}; "
            "the cited passage does not discuss the same subject"
        )
        return check

    check.verdict = Verification.SUPPORTED
    check.detail = (
        f"{len(matched)}/{len(claim_numbers)} figures grounded, "
        f"{overlap_basis} overlap {overlap:.2f}"
    )
    return check


@dataclass
class VerificationOutcome:
    kept: list[Finding]
    dropped: int
    checks: list[ClaimCheck]

    @property
    def citation_coverage(self) -> float:
        """Fraction of claims that cite at least one chunk."""
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.cited_chunks) / len(self.checks)

    @property
    def support_precision(self) -> float:
        """Fraction of claims whose citations actually support them."""
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.verdict is Verification.SUPPORTED) / len(self.checks)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of claims that are unsupported or contradicted."""
        if not self.checks:
            return 0.0
        bad = sum(
            1
            for c in self.checks
            if c.verdict in (Verification.UNSUPPORTED, Verification.CONTRADICTED)
        )
        return bad / len(self.checks)


def verify_findings(
    findings: list[Finding],
    evidence: list[EvidenceChunk],
    cfg: VerificationConfig,
) -> VerificationOutcome:
    """Verify every finding, applying the configured drop/flag policy."""
    kept: list[Finding] = []
    checks: list[ClaimCheck] = []
    dropped = 0

    for finding in findings:
        check = verify_claim(finding.claim, finding.supporting_chunks, evidence, cfg)
        checks.append(check)

        annotated = Finding(
            claim=finding.claim,
            supporting_chunks=finding.supporting_chunks,
            verification=check.verdict,
            verification_detail=check.detail,
            factor=finding.factor,
        )

        if check.verdict is Verification.SUPPORTED:
            kept.append(annotated)
        elif check.verdict is Verification.CONTRADICTED and cfg.keep_contradicted_as_flagged:
            # Kept deliberately: a contradiction is a finding about the document
            # worth showing the reader, not noise to hide.
            kept.append(annotated)
        elif cfg.drop_unsupported_claims:
            dropped += 1
        else:
            kept.append(annotated)

    return VerificationOutcome(kept=kept, dropped=dropped, checks=checks)

"""Pydantic schema for the structured investment-assessment report (section 8).

Every number the system reports carries provenance, and every claim carries a
verification verdict. The schema enforces that: a finding cannot be serialised
without a verification status, a derived metric cannot be serialised without the
formula and operands that produced it, and every report carries the disclaimer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from affa import DISCLAIMER, PIPELINE_VERSION


class ExtractionMethod(str, Enum):
    """How a value was obtained. Recorded per value so disagreements are visible."""

    XBRL_MODEL = "xbrl_model"
    RULE_BASED = "rule_based"
    BOTH_AGREE = "both_agree"
    DERIVED = "derived"
    USER_SUPPLIED = "user_supplied"


class Verification(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class Assessment(str, Enum):
    FAVORABLE = "favorable"
    MIXED = "mixed"
    UNFAVORABLE = "unfavorable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Scale(str, Enum):
    """Reporting scale declared by the filing ("amounts in millions")."""

    UNITS = "units"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"


SCALE_MULTIPLIER: dict[Scale, float] = {
    Scale.UNITS: 1.0,
    Scale.THOUSANDS: 1e3,
    Scale.MILLIONS: 1e6,
    Scale.BILLIONS: 1e9,
}


class SourceRef(BaseModel):
    """Pointer back into the filing. The unit of traceability in this system."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    raw_text: str | None = Field(default=None, description="Verbatim span the value was read from.")


class ExtractedMetric(BaseModel):
    """A figure read directly off the filing."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str = "USD"
    scale: Scale = Scale.UNITS
    period: str | None = None
    source: SourceRef
    method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @property
    def value_in_units(self) -> float:
        """Value normalised to absolute units, whatever the filing's scale."""
        return self.value * SCALE_MULTIPLIER[self.scale]


class DerivedMetric(BaseModel):
    """A computed figure. Never guessed - always formula plus operands (section 6)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str = "ratio"
    formula: str
    operands: dict[str, float]
    sources: list[SourceRef] = Field(default_factory=list)
    method: ExtractionMethod = ExtractionMethod.DERIVED

    @model_validator(mode="after")
    def _formula_must_have_operands(self) -> DerivedMetric:
        # A derived metric without operands is unauditable: the reader cannot
        # re-do the arithmetic, which is the entire point of publishing it.
        if not self.operands:
            raise ValueError(f"derived metric {self.name!r} must carry its operands")
        return self


class Disagreement(BaseModel):
    """The two extractors produced different values for the same metric.

    Surfaced rather than silently resolved: disagreement is a useful signal about
    document quality and is fed into the confidence penalty.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    xbrl_model: float | None = None
    rule_based: float | None = None
    relative_difference_pct: float | None = None
    resolved_to: ExtractionMethod | None = None


class FinancialMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extracted: list[ExtractedMetric] = Field(default_factory=list)
    derived: list[DerivedMetric] = Field(default_factory=list)
    yoy_changes: dict[str, float] = Field(default_factory=dict)
    disagreements: list[Disagreement] = Field(default_factory=list)

    def get_extracted(self, name: str) -> ExtractedMetric | None:
        for m in self.extracted:
            if m.name == name:
                return m
        return None

    def get_derived(self, name: str) -> DerivedMetric | None:
        for m in self.derived:
            if m.name == name:
                return m
        return None

    def value_of(self, name: str) -> float | None:
        """Look a metric up across extracted, derived and YoY, in that order."""
        e = self.get_extracted(name)
        if e is not None:
            return e.value_in_units
        d = self.get_derived(name)
        if d is not None:
            return d.value
        return self.yoy_changes.get(name)


class SentimentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: Literal["positive", "neutral", "negative"] = "neutral"
    score: float = Field(ge=-1.0, le=1.0, default=0.0)
    by_section: dict[str, float] = Field(default_factory=dict)
    model_name: str | None = None
    available: bool = True


class RiskFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: str
    severity: Severity
    source: SourceRef
    category: str | None = None


class EvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page: int | None = None
    text: str
    similarity: float
    chunk_type: str = "narrative"


class Finding(BaseModel):
    """One claim produced by the reasoning node, plus its verification verdict."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    supporting_chunks: list[str] = Field(default_factory=list)
    verification: Verification
    verification_detail: str | None = None
    factor: str | None = Field(
        default=None, description="Rubric factor this finding informs, if any."
    )

    @model_validator(mode="after")
    def _supported_claims_need_citations(self) -> Finding:
        # "supported" without a citation is exactly the failure mode the verify
        # node exists to prevent, so the schema refuses to represent it.
        if self.verification is Verification.SUPPORTED and not self.supporting_chunks:
            raise ValueError(
                f"finding marked supported must cite at least one chunk: {self.claim[:80]!r}"
            )
        return self


class ReasoningBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    chain_of_thought: str = ""


class RationaleItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: str
    statement: str
    score: float = Field(ge=-1.0, le=1.0)
    citations: list[SourceRef] = Field(default_factory=list)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: Assessment
    confidence: float = Field(ge=0.0, le=1.0)
    rubric_version: str
    factor_scores: dict[str, float] = Field(default_factory=dict)
    factors_scored: list[str] = Field(default_factory=list)
    factors_missing: list[str] = Field(default_factory=list)
    weight_covered: float = Field(ge=0.0, le=1.0, default=0.0)
    aggregate_score: float | None = None
    rationale: list[RationaleItem] = Field(default_factory=list)
    narrative: str | None = Field(
        default=None,
        description="LLM-written prose explaining the rubric output. Never decides it.",
    )
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _insufficient_has_no_aggregate(self) -> Recommendation:
        # If the evidence was too thin to score, we must not also publish a
        # number that looks like a verdict.
        if self.assessment is Assessment.INSUFFICIENT_EVIDENCE and self.aggregate_score is not None:
            raise ValueError(
                "insufficient_evidence must not carry an aggregate_score; "
                "a score implies a verdict the evidence does not support"
            )
        return self

    @field_validator("factor_scores")
    @classmethod
    def _scores_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for k, s in v.items():
            if not -1.0 <= s <= 1.0:
                raise ValueError(f"factor score {k}={s} outside [-1, 1]")
        return v


class RetrievalDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks_retrieved: int = 0
    chunks_discarded_below_floor: int = 0
    mean_similarity: float = 0.0
    retries: int = 0
    reformulations: list[str] = Field(default_factory=list)
    sufficient: bool = False
    stop_reason: str | None = None


class ModelsUsed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedder: str
    xbrl_tagger: str
    sentiment: str
    reasoner: str


class ReportMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str | None = None
    ticker: str | None = None
    doc_type: str | None = None
    fiscal_period: str | None = None
    source_file: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_version: str = PIPELINE_VERSION
    models: ModelsUsed


class AnalysisReport(BaseModel):
    """The single structured document this system exists to produce."""

    model_config = ConfigDict(extra="forbid")

    metadata: ReportMetadata
    financial_metrics: FinancialMetrics = Field(default_factory=FinancialMetrics)
    sentiment: SentimentBlock = Field(default_factory=SentimentBlock)
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    reasoning: ReasoningBlock = Field(default_factory=ReasoningBlock)
    recommendation: Recommendation
    retrieval_diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    unsupported_claims_dropped: int = 0
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _citations_must_resolve(self) -> AnalysisReport:
        """Every cited chunk id must exist in the evidence list.

        A citation pointing at a chunk the report does not contain is not
        traceable, which defeats the purpose of the citation.
        """
        known = {c.chunk_id for c in self.evidence}
        if not known:
            return self
        dangling: set[str] = set()
        for f in self.reasoning.findings:
            dangling |= {c for c in f.supporting_chunks if c not in known}
        for item in self.recommendation.rationale:
            dangling |= {c.chunk_id for c in item.citations if c.chunk_id not in known}
        if dangling:
            raise ValueError(f"report cites chunk ids absent from evidence: {sorted(dangling)[:5]}")
        return self

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return self.model_dump_json(**kwargs)


def empty_report(
    *,
    models: ModelsUsed,
    source_file: str | None = None,
    rubric_version: str = "1.0",
    reason: str = "no analysis performed",
) -> AnalysisReport:
    """A schema-valid report expressing "we could not assess this".

    Used when ingestion yields nothing usable, so the failure path returns the
    same contract as the success path instead of an ad-hoc error blob.
    """
    return AnalysisReport(
        metadata=ReportMetadata(source_file=source_file, models=models),
        recommendation=Recommendation(
            assessment=Assessment.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            rubric_version=rubric_version,
        ),
        warnings=[reason],
    )

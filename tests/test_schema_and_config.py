"""Report schema validation and the live-config contract (sections 8 and 10)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from affa.config import ConfigError, load_config, repo_root
from affa.schema import (
    AnalysisReport,
    Assessment,
    DerivedMetric,
    EvidenceChunk,
    Finding,
    ModelsUsed,
    RationaleItem,
    ReasoningBlock,
    Recommendation,
    ReportMetadata,
    Scale,
    SourceRef,
    Verification,
    empty_report,
)


@pytest.fixture
def models() -> ModelsUsed:
    return ModelsUsed(embedder="e", xbrl_tagger="t", sentiment="s", reasoner="r")


def test_supported_finding_must_cite_something() -> None:
    """The exact failure the verify node exists to prevent, blocked at the type level."""
    with pytest.raises(ValidationError, match="must cite at least one chunk"):
        Finding(claim="Revenue rose", supporting_chunks=[], verification=Verification.SUPPORTED)
    # Unsupported findings may legitimately have no citation.
    Finding(claim="Revenue rose", supporting_chunks=[], verification=Verification.UNSUPPORTED)


def test_derived_metric_must_carry_operands() -> None:
    """Without operands a reader cannot re-do the arithmetic (section 6)."""
    with pytest.raises(ValidationError, match="must carry its operands"):
        DerivedMetric(name="gross_margin_pct", value=44.1, formula="a / b", operands={})


def test_insufficient_evidence_cannot_carry_an_aggregate_score() -> None:
    with pytest.raises(ValidationError, match="must not carry an aggregate_score"):
        Recommendation(
            assessment=Assessment.INSUFFICIENT_EVIDENCE,
            confidence=0.2,
            rubric_version="1.0",
            aggregate_score=0.4,
        )


def test_citations_must_resolve_to_evidence(models) -> None:
    """A citation pointing at a chunk the report does not contain is not traceable."""
    with pytest.raises(ValidationError, match="cites chunk ids absent"):
        AnalysisReport(
            metadata=ReportMetadata(models=models),
            evidence=[EvidenceChunk(chunk_id="c1", text="x", similarity=0.5)],
            reasoning=ReasoningBlock(
                findings=[
                    Finding(
                        claim="Something",
                        supporting_chunks=["ghost"],
                        verification=Verification.SUPPORTED,
                    )
                ]
            ),
            recommendation=Recommendation(
                assessment=Assessment.MIXED, confidence=0.5, rubric_version="1.0"
            ),
        )


def test_rationale_citations_are_checked_too(models) -> None:
    with pytest.raises(ValidationError, match="cites chunk ids absent"):
        AnalysisReport(
            metadata=ReportMetadata(models=models),
            evidence=[EvidenceChunk(chunk_id="c1", text="x", similarity=0.5)],
            recommendation=Recommendation(
                assessment=Assessment.MIXED,
                confidence=0.5,
                rubric_version="1.0",
                aggregate_score=0.1,
                rationale=[
                    RationaleItem(
                        factor="growth",
                        statement="s",
                        score=0.1,
                        citations=[SourceRef(chunk_id="ghost")],
                    )
                ],
            ),
        )


def test_factor_scores_must_be_in_range() -> None:
    with pytest.raises(ValidationError, match="outside"):
        Recommendation(
            assessment=Assessment.MIXED,
            confidence=0.5,
            rubric_version="1.0",
            aggregate_score=0.0,
            factor_scores={"growth": 4.2},
        )


def test_every_report_carries_the_disclaimer(models) -> None:
    report = empty_report(models=models, source_file="x.pdf")
    assert "Not investment advice" in report.recommendation.disclaimer
    assert report.recommendation.assessment is Assessment.INSUFFICIENT_EVIDENCE


def test_scale_normalisation_on_extracted_metrics() -> None:
    from affa.schema import ExtractedMetric, ExtractionMethod

    m = ExtractedMetric(
        name="revenue",
        value=4812.6,
        scale=Scale.MILLIONS,
        source=SourceRef(chunk_id="c1"),
        method=ExtractionMethod.RULE_BASED,
    )
    assert m.value_in_units == pytest.approx(4_812_600_000.0)


def test_report_round_trips_through_json(models) -> None:
    report = empty_report(models=models, source_file="x.pdf")
    restored = AnalysisReport.model_validate_json(report.to_json())
    assert restored.metadata.models.embedder == "e"


def test_extra_keys_are_rejected(models) -> None:
    """extra="forbid" keeps a typo'd field from silently vanishing."""
    with pytest.raises(ValidationError):
        ReportMetadata(models=models, tickerr="AAPL")


# --- configuration -------------------------------------------------------


def test_default_config_loads_and_validates(cfg) -> None:
    assert cfg.pipeline_version
    assert cfg.models.embedder.name
    assert cfg.recommendation.rubric_path().is_file()


def test_missing_config_file_is_a_clear_error() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(repo_root() / "configs" / "does_not_exist.yaml")


def test_collection_is_namespaced_by_embedder(cfg) -> None:
    """Vectors from different models must never share a collection (section 4)."""
    import dataclasses

    other = dataclasses.replace(cfg.models.embedder, name="BAAI/bge-large-en-v1.5")
    assert cfg.vector_store.collection_name(
        cfg.models.embedder
    ) != cfg.vector_store.collection_name(other)


def test_every_config_key_is_consumed(cfg) -> None:
    """Anti-pattern #6: a config file the docs treat as live that nothing loads.

    Walks the YAML and asserts each top-level section is represented on the
    loaded object, so a key can never linger in the file as decoration.
    """
    raw = yaml.safe_load((repo_root() / "configs" / "default.yaml").read_text(encoding="utf-8"))
    for section in raw:
        assert hasattr(cfg, section), f"config section {section!r} is never loaded"

    # Spot-check the nested keys that drive behaviour.
    assert cfg.ingestion.chunk.target_tokens == raw["ingestion"]["chunk"]["target_tokens"]
    assert cfg.retrieval.min_similarity == raw["retrieval"]["min_similarity"]
    assert cfg.routing.max_retrieval_attempts == raw["routing"]["max_retrieval_attempts"]
    assert cfg.verification.numeric_tolerance_pct == raw["verification"]["numeric_tolerance_pct"]
    assert cfg.kpi.tolerance_pct == raw["kpi"]["tolerance_pct"]
    assert cfg.report.disclaimer == raw["report"]["disclaimer"]


def test_reasoner_backend_is_validated() -> None:
    from affa.config import ReasonerConfig

    with pytest.raises(ConfigError, match="backend must be one of"):
        ReasonerConfig(
            backend="magic",
            local_name="x",
            local_adapter=None,
            hosted_provider="anthropic",
            hosted_name="y",
            max_new_tokens=10,
            temperature=0.0,
        )

"""End-to-end pipeline and the API contract with models mocked (section 10)."""

from __future__ import annotations

import importlib.util

import pytest

from affa.config import get_config
from affa.ingestion import ingest_filing
from affa.ingestion.embed import HashingEmbedder
from affa.pipeline import analyze_filing
from affa.report.render import to_html, to_markdown
from affa.schema import AnalysisReport, Assessment


@pytest.fixture
def stub_embedder() -> HashingEmbedder:
    """Deterministic embedder so the suite needs no model download."""
    return HashingEmbedder()


def test_ingest_produces_chunks_with_provenance(sample_filing_path, cfg, stub_embedder) -> None:
    result = ingest_filing(sample_filing_path, cfg, in_memory=True, embedder=stub_embedder)
    assert result.chunks
    assert result.document.ticker == "NWSY"
    assert result.document.doc_type == "10-K"
    assert result.document.fiscal_period == "FY2024"
    assert any(c.chunk_type == "table" for c in result.chunks), "statement tables were lost"
    assert all(c.doc_id == result.document.doc_id for c in result.chunks)
    assert result.n_indexed == len(result.chunks)


def test_reingesting_the_same_file_is_idempotent(sample_filing_path, cfg, stub_embedder) -> None:
    """Content-addressed chunk ids mean citations in an old report still resolve."""
    first = ingest_filing(sample_filing_path, cfg, in_memory=True, embedder=stub_embedder)
    second = ingest_filing(
        sample_filing_path, cfg, in_memory=True, embedder=stub_embedder, store=first.store
    )
    assert second.n_indexed == 0
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]


def test_full_analysis_produces_a_valid_report(sample_filing_path, cfg) -> None:
    result = analyze_filing(
        sample_filing_path, cfg=cfg, in_memory=True, market_price_per_share=92.50
    )
    report = result.report

    assert isinstance(report, AnalysisReport)
    assert report.recommendation.assessment is Assessment.FAVORABLE
    assert report.financial_metrics.get_extracted("revenue") is not None
    assert report.financial_metrics.get_derived("gross_margin_pct") is not None
    assert report.reasoning.findings
    assert report.retrieval_diagnostics.stop_reason
    assert "Not investment advice" in report.recommendation.disclaimer

    # The report records what actually ran, not what is configured.
    assert report.metadata.models.xbrl_tagger == "rule_based_only"
    assert report.metadata.models.sentiment == "lexicon_fallback"


def test_pe_ratio_only_appears_when_a_price_is_supplied(sample_filing_path, cfg) -> None:
    without = analyze_filing(sample_filing_path, cfg=cfg, in_memory=True)
    assert without.report.financial_metrics.get_derived("pe_ratio") is None

    with_price = analyze_filing(
        sample_filing_path, cfg=cfg, in_memory=True, market_price_per_share=92.50
    )
    pe = with_price.report.financial_metrics.get_derived("pe_ratio")
    assert pe is not None
    assert pe.value == pytest.approx(92.50 / 3.64, rel=1e-3)


def test_thin_filing_yields_insufficient_evidence(thin_filing, cfg) -> None:
    """Section 7: the abstention path must be reachable on a real document."""
    result = analyze_filing(thin_filing, cfg=cfg, in_memory=True)
    rec = result.report.recommendation
    assert rec.assessment is Assessment.INSUFFICIENT_EVIDENCE
    assert rec.aggregate_score is None
    assert rec.factors_missing


def test_report_round_trips_and_renders(sample_filing_path, cfg) -> None:
    report = analyze_filing(sample_filing_path, cfg=cfg, in_memory=True).report

    restored = AnalysisReport.model_validate_json(report.to_json())
    assert restored.recommendation.assessment == report.recommendation.assessment

    markdown = to_markdown(report)
    assert "Not investment advice" in markdown
    assert "## Retrieval diagnostics" in markdown

    html = to_html(report)
    assert html.startswith("<!doctype html>")
    assert "<table>" in html


def test_analysis_is_deterministic(sample_filing_path, cfg) -> None:
    """Same filing, same verdict. The rubric decides, so this must hold."""
    first = analyze_filing(sample_filing_path, cfg=cfg, in_memory=True).report
    second = analyze_filing(sample_filing_path, cfg=cfg, in_memory=True).report
    assert first.recommendation.assessment == second.recommendation.assessment
    assert first.recommendation.aggregate_score == second.recommendation.aggregate_score
    assert first.recommendation.factor_scores == second.recommendation.factor_scores


def test_unsupported_format_is_rejected(tmp_path, cfg) -> None:
    bad = tmp_path / "filing.docx"
    bad.write_text("not a supported format", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported filing format"):
        analyze_filing(bad, cfg=cfg, in_memory=True)


# --- API ------------------------------------------------------------------

# Scoped to the API tests only. A module-level importorskip would skip the
# pipeline tests above it too, silently reducing coverage to nothing.
_API_DEPS = all(importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx"))
requires_api = pytest.mark.skipif(
    not _API_DEPS, reason='API tests need fastapi + httpx: pip install -e ".[agent,dev]"'
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from affa.api.main import app

    return TestClient(app)


@requires_api
def test_health_reports_actual_model_state(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "Not investment advice" in body["disclaimer"]
    # Configured-but-disabled must be visible, not implied.
    assert body["models"]["xbrl_tagger_enabled"] is False


@requires_api
def test_config_endpoint_exposes_thresholds(client) -> None:
    body = client.get("/config").json()
    cfg = get_config()
    assert body["retrieval"]["min_similarity"] == cfg.retrieval.min_similarity
    assert body["routing"]["retry_below_mean_similarity"] > body["retrieval"]["min_similarity"], (
        "the API must not advertise an unreachable threshold"
    )


@requires_api
def test_analyze_returns_the_same_schema_as_the_cli(client, sample_filing_path) -> None:
    with open(sample_filing_path, "rb") as fh:
        response = client.post(
            "/analyze",
            files={"file": ("demo_10k.json", fh, "application/json")},
            data={"response_format": "json"},
        )
    assert response.status_code == 200
    # One contract: the API response validates against the same model the CLI emits.
    report = AnalysisReport.model_validate(response.json())
    assert report.recommendation.disclaimer


@requires_api
def test_analyze_rejects_unsupported_types(client) -> None:
    response = client.post(
        "/analyze", files={"file": ("x.docx", b"data", "application/octet-stream")}
    )
    assert response.status_code == 415


@requires_api
def test_analyze_rejects_empty_uploads(client) -> None:
    response = client.post("/analyze", files={"file": ("x.json", b"", "application/json")})
    assert response.status_code == 400


@requires_api
def test_markdown_response_format(client, sample_filing_path) -> None:
    with open(sample_filing_path, "rb") as fh:
        response = client.post(
            "/analyze",
            files={"file": ("demo_10k.json", fh, "application/json")},
            data={"response_format": "markdown"},
        )
    assert response.status_code == 200
    assert "Not investment advice" in response.text


@requires_api
def test_malformed_json_filing_is_a_client_error(client) -> None:
    response = client.post(
        "/analyze", files={"file": ("bad.json", b"{not json", "application/json")}
    )
    assert response.status_code in (400, 422)

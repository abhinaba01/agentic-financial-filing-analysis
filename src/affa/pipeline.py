"""End-to-end entry point: filing in, validated report out."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affa.agent.graph import build_graph, models_used
from affa.agent.state import new_state
from affa.config import AffaConfig, get_config
from affa.ingestion import ingest_filing
from affa.kpi.xbrl import XBRLTagger
from affa.llm import build_llm
from affa.schema import AnalysisReport, empty_report

log = logging.getLogger(__name__)

DEFAULT_QUESTION = (
    "What were the company's revenue, profitability, cash generation, leverage "
    "and principal risks for the fiscal year, and how did they change?"
)


@dataclass
class PipelineResult:
    report: AnalysisReport
    n_chunks: int
    backend: str
    used_stub_embedder: bool


def analyze_filing(
    path: str | Path,
    *,
    cfg: AffaConfig | None = None,
    question: str = DEFAULT_QUESTION,
    ticker: str | None = None,
    company: str | None = None,
    fiscal_period: str | None = None,
    market_price_per_share: float | None = None,
    in_memory: bool = False,
    llm: Any | None = None,
) -> PipelineResult:
    """Ingest a filing and run the agent graph over it."""
    cfg = cfg or get_config()

    ingested = ingest_filing(
        path,
        cfg,
        ticker=ticker,
        company=company,
        fiscal_period=fiscal_period,
        in_memory=in_memory,
    )

    tagger = XBRLTagger(cfg.models.xbrl_tagger)
    tagger.load()
    reasoner = llm if llm is not None else build_llm(cfg)

    resolved_models = models_used(cfg, tagger_used=tagger.available, sentiment_used=False)
    resolved_models = resolved_models.model_copy(
        update={
            "embedder": getattr(ingested.embedder, "name", cfg.models.embedder.name),
            "reasoner": getattr(reasoner, "name", "stub"),
        }
    )

    if not ingested.chunks:
        return PipelineResult(
            report=empty_report(
                models=resolved_models,
                source_file=str(path),
                reason="no text could be extracted from the filing",
            ),
            n_chunks=0,
            backend="none",
            used_stub_embedder=ingested.used_stub_embedder,
        )

    bundle = build_graph(
        cfg,
        store=ingested.store,
        document=ingested.document,
        llm=reasoner,
        tagger=tagger,
        models=resolved_models,
    )

    state = new_state(
        doc_id=ingested.document.doc_id,
        source_file=str(path),
        chunks=ingested.chunks,
        question=question,
        market_price_per_share=market_price_per_share,
    )

    # Recursion limit sits above the retry budget so the routing edge is what
    # ends the loop, not the framework's guard rail surfacing as an error.
    limit = cfg.routing.max_retrieval_attempts * 2 + 12
    final = bundle.app.invoke(state, {"recursion_limit": limit})

    report: AnalysisReport = final["report"]
    if ingested.used_stub_embedder:
        report = report.model_copy(
            update={
                "warnings": report.warnings
                + [
                    "retrieval used the hashing stub embedder (lexical, not semantic); "
                    "similarity scores are not comparable to a real embedding model"
                ]
            }
        )

    return PipelineResult(
        report=report,
        n_chunks=len(ingested.chunks),
        backend=bundle.backend,
        used_stub_embedder=ingested.used_stub_embedder,
    )

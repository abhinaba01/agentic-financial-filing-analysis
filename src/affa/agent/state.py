"""Shared typed state for the LangGraph pipeline.

The rule that shapes this module (section 3): **concurrent branches return only
their own state keys, never the whole state.** LangGraph runs the fan-out
branches in one superstep; if two of them return the full state, both write every
key in that superstep and the framework rejects it as a conflicting update. The
branch that only meant to set ``sentiment`` ends up also writing ``chunks``,
``evidence`` and everything else it happened to read.

Two things enforce it here:

* :data:`BRANCH_OUTPUT_KEYS` declares exactly which keys each parallel node owns,
  and ``tests/test_nodes_delta.py`` asserts every branch node's return is a
  subset of its own declaration;
* ``warnings`` is the one key more than one node writes, so it carries an
  ``operator.add`` reducer - concurrent appends merge instead of clobbering.

Branches also treat the incoming state as read-only. Mutating a list that arrived
in the state mutates the object other branches are reading in the same superstep.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from affa.ingestion.types import Chunk
from affa.schema import (
    AnalysisReport,
    Disagreement,
    EvidenceChunk,
    FinancialMetrics,
    Finding,
    Recommendation,
    RiskFactor,
    SentimentBlock,
)


class AgentState(TypedDict, total=False):
    """State threaded through the graph. Every key is optional until written."""

    # --- set by ingest ---
    doc_id: str
    source_file: str
    chunks: list[Chunk]
    question: str
    market_price_per_share: float | None

    # --- fan-out branch outputs (one owner each) ---
    financial_metrics: FinancialMetrics
    extraction_notes: list[str]
    tagger_used: bool
    disagreements: list[Disagreement]
    sentiment: SentimentBlock
    risk_factors: list[RiskFactor]
    risk_severity_index: float
    doc_metadata: dict[str, Any]

    # --- retrieval loop ---
    query: str
    queries_tried: list[str]
    retrieval_attempts: int
    evidence: list[EvidenceChunk]
    # Count of chunks retrieval actually returned. Kept separate from
    # len(evidence), which also includes provenance chunks added by `reason`.
    n_retrieved: int
    mean_similarity: float
    chunks_discarded_below_floor: int
    sufficient: bool
    stop_reason: str

    # --- reason / verify ---
    findings: list[Finding]
    chain_of_thought: str
    verified_findings: list[Finding]
    unsupported_claims_dropped: int

    # --- recommend / synthesize ---
    recommendation: Recommendation
    report: AnalysisReport

    # Written by several nodes, so it needs a merge reducer rather than a
    # last-write-wins assignment.
    warnings: Annotated[list[str], operator.add]


# Exact key ownership for the four concurrent branches. A branch returning
# anything outside its set is a bug, and the test suite says so by name.
BRANCH_OUTPUT_KEYS: dict[str, frozenset[str]] = {
    "extract_kpis": frozenset(
        {"financial_metrics", "extraction_notes", "tagger_used", "disagreements", "warnings"}
    ),
    "analyze_sentiment": frozenset({"sentiment", "warnings"}),
    "extract_risks": frozenset({"risk_factors", "risk_severity_index", "warnings"}),
    "extract_doc_metadata": frozenset({"doc_metadata", "warnings"}),
}

BRANCH_NODES: tuple[str, ...] = tuple(BRANCH_OUTPUT_KEYS)


def new_state(
    *,
    doc_id: str,
    source_file: str,
    chunks: list[Chunk],
    question: str,
    market_price_per_share: float | None = None,
) -> AgentState:
    """Initial state for a run."""
    return AgentState(
        doc_id=doc_id,
        source_file=source_file,
        chunks=chunks,
        question=question,
        market_price_per_share=market_price_per_share,
        queries_tried=[],
        retrieval_attempts=0,
        warnings=[],
    )

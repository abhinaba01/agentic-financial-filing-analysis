"""Graph nodes.

Every node returns a **delta** - only the keys it owns - and treats the incoming
state as read-only. The four fan-out branches declare their keys in
:data:`affa.agent.state.BRANCH_OUTPUT_KEYS` and the test suite holds them to it.

Retrieval and generation are separate nodes on purpose (section 3): the re-query
loop wraps ``retrieve`` alone, so a retry re-searches with a reformulated query
and ``reason`` runs once, after the loop settles. An LLM call inside the retry
loop would be paid for on every discarded attempt.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from affa.agent.reformulate import reformulate
from affa.agent.routing import assess_sufficiency, stop_reason
from affa.agent.state import AgentState
from affa.agent.verify import verify_findings
from affa.config import AffaConfig
from affa.ingestion.types import Chunk
from affa.kpi.catalog import metric_label
from affa.kpi.extract import extract_kpis
from affa.llm import LLMClient, extract_json
from affa.recommend.rubric import evaluate as run_rubric
from affa.schema import (
    AnalysisReport,
    Assessment,
    EvidenceChunk,
    FinancialMetrics,
    Finding,
    ModelsUsed,
    ReasoningBlock,
    ReportMetadata,
    RetrievalDiagnostics,
    RiskFactor,
    SentimentBlock,
    Severity,
    SourceRef,
    Verification,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- fan-out


def make_extract_kpis_node(cfg: AffaConfig, tagger: Any | None = None):
    """Branch 1: XBRL/KPI extraction."""

    def node(state: AgentState) -> dict[str, Any]:
        chunks: list[Chunk] = state.get("chunks", [])
        outcome = extract_kpis(
            chunks,
            cfg,
            tagger=tagger,
            market_price_per_share=state.get("market_price_per_share"),
        )
        # Only this branch's keys. Returning `state` here is anti-pattern #10.
        return {
            "financial_metrics": outcome.metrics,
            "extraction_notes": outcome.notes,
            "tagger_used": outcome.tagger_used,
            "disagreements": outcome.metrics.disagreements,
            "warnings": [] if outcome.tagger_used else ["XBRL tagger not loaded"],
        }

    return node


_POSITIVE_TONE = {
    "growth",
    "increase",
    "increased",
    "strong",
    "record",
    "improved",
    "improvement",
    "expansion",
    "favorable",
    "gain",
    "gains",
    "higher",
    "exceeded",
    "robust",
    "profitability",
    "outperformed",
    "momentum",
}
_NEGATIVE_TONE = {
    "decline",
    "declined",
    "decrease",
    "decreased",
    "loss",
    "losses",
    "weak",
    "weakness",
    "adverse",
    "unfavorable",
    "impairment",
    "restructuring",
    "lower",
    "shortfall",
    "litigation",
    "downturn",
    "deterioration",
    "headwinds",
}


def make_sentiment_node(cfg: AffaConfig, classifier: Any | None = None):
    """Branch 2: management tone.

    Uses the fine-tuned classifier from section 5.3 when it is loaded. The
    lexicon fallback is explicitly labelled in the report as a lexicon, never as
    a model score, because the two are not comparable and the rubric weights tone
    on the assumption it came from a classifier.
    """

    def node(state: AgentState) -> dict[str, Any]:
        chunks: list[Chunk] = state.get("chunks", [])
        text_chunks = [c for c in chunks if c.chunk_type != "table"][:60]

        if classifier is not None and getattr(classifier, "available", False):
            block = classifier.score_chunks(text_chunks)
            return {"sentiment": block, "warnings": []}

        pos = neg = 0
        for chunk in text_chunks:
            words = set(re.findall(r"[a-z]+", chunk.text.lower()))
            pos += len(words & _POSITIVE_TONE)
            neg += len(words & _NEGATIVE_TONE)
        total = pos + neg
        score = (pos - neg) / total if total else 0.0
        overall = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return {
            "sentiment": SentimentBlock(
                overall=overall,
                score=round(score, 4),
                by_section={},
                model_name="lexicon_fallback",
                available=False,
            ),
            "warnings": [
                "sentiment from a word lexicon, not the fine-tuned classifier; "
                "treat the tone factor as indicative only"
            ],
        }

    return node


# Risk-factor cues, ordered severity-high first.
_RISK_CUES: tuple[tuple[str, Severity], ...] = (
    ("material adverse effect", Severity.HIGH),
    ("going concern", Severity.HIGH),
    ("substantial doubt", Severity.HIGH),
    ("could materially", Severity.HIGH),
    ("significant risk", Severity.HIGH),
    ("we may not be able", Severity.MEDIUM),
    ("could adversely affect", Severity.MEDIUM),
    ("may adversely affect", Severity.MEDIUM),
    ("subject to risks", Severity.MEDIUM),
    ("uncertain", Severity.LOW),
    ("competition", Severity.LOW),
    ("fluctuat", Severity.LOW),
)
_SEVERITY_WEIGHT = {Severity.HIGH: 1.0, Severity.MEDIUM: 0.55, Severity.LOW: 0.25}


def make_risk_node(cfg: AffaConfig, max_risks: int = 25):
    """Branch 3: risk-factor extraction and a severity index for the rubric."""

    def node(state: AgentState) -> dict[str, Any]:
        chunks: list[Chunk] = state.get("chunks", [])
        risks: list[RiskFactor] = []
        seen: set[str] = set()

        for chunk in chunks:
            if chunk.chunk_type == "table":
                continue
            lowered = chunk.text.lower()
            for cue, severity in _RISK_CUES:
                if cue not in lowered:
                    continue
                idx = lowered.index(cue)
                sentence = _sentence_around(chunk.text, idx)
                key = sentence[:80].lower()
                if key in seen or len(sentence) < 25:
                    continue
                seen.add(key)
                risks.append(
                    RiskFactor(
                        risk=sentence[:400],
                        severity=severity,
                        source=SourceRef(
                            chunk_id=chunk.chunk_id, page=chunk.page, raw_text=sentence[:200]
                        ),
                        category=chunk.section,
                    )
                )
                break
            if len(risks) >= max_risks:
                break

        if risks:
            index = sum(_SEVERITY_WEIGHT[r.severity] for r in risks) / len(risks)
            # Scale by how much of the document flags risk at all, so a filing
            # with three severe risks does not score the same as one with thirty.
            density = min(1.0, len(risks) / max(max_risks * 0.6, 1))
            index = round(min(1.0, index * (0.6 + 0.4 * density)), 4)
        else:
            index = 0.0

        return {"risk_factors": risks, "risk_severity_index": index, "warnings": []}

    return node


def _sentence_around(text: str, index: int) -> str:
    start = max(text.rfind(".", 0, index) + 1, 0)
    end = text.find(".", index)
    end = len(text) if end == -1 else end + 1
    return text[start:end].strip()


def make_doc_metadata_node(cfg: AffaConfig, document: Any | None = None):
    """Branch 4: document metadata (period, ticker, company, doc type)."""

    def node(state: AgentState) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "doc_id": state.get("doc_id"),
            "source_file": state.get("source_file"),
            "n_chunks": len(state.get("chunks", [])),
        }
        if document is not None:
            meta.update(
                {
                    "company": document.company,
                    "ticker": document.ticker,
                    "doc_type": document.doc_type,
                    "fiscal_period": document.fiscal_period,
                    "n_pages": document.n_pages,
                }
            )
        warnings: list[str] = []
        if not meta.get("fiscal_period"):
            warnings.append("fiscal period could not be determined from the filing")
        if not meta.get("ticker"):
            warnings.append("ticker could not be determined; retrieval filtering disabled")
        return {"doc_metadata": meta, "warnings": warnings}

    return node


# ------------------------------------------------------------- retrieval


def make_retrieve_node(cfg: AffaConfig, store: Any):
    """Retrieval only. No generation here - that is the point of the split."""

    def node(state: AgentState) -> dict[str, Any]:
        attempt = state.get("retrieval_attempts", 0)
        tried = list(state.get("queries_tried", []))
        base_query = state.get("question", "")

        if attempt == 0:
            query, strategy = base_query, "original"
        else:
            # Guaranteed distinct from everything already tried.
            query, strategy = reformulate(base_query, attempt, tried)

        where: dict[str, Any] = {}
        if cfg.retrieval.filter_by_ticker:
            ticker = (state.get("doc_metadata") or {}).get("ticker")
            if ticker:
                where["ticker"] = ticker

        raw = store.query(query, top_k=cfg.retrieval.top_k, where=where or None)
        kept = [r for r in raw if r.similarity >= cfg.retrieval.min_similarity]
        discarded = len(raw) - len(kept)

        evidence = [
            EvidenceChunk(
                chunk_id=r.chunk_id,
                page=r.page,
                text=r.text,
                similarity=round(float(r.similarity), 4),
                chunk_type=r.chunk_type,
            )
            for r in kept
        ]
        mean_sim = sum(e.similarity for e in evidence) / len(evidence) if evidence else 0.0
        verdict = assess_sufficiency(len(evidence), mean_sim, cfg)

        return {
            "query": query,
            "queries_tried": tried + [query],
            # The node counts attempts; it does not decide to stop. That call
            # belongs to route_after_retrieval and nowhere else.
            "retrieval_attempts": attempt + 1,
            "evidence": evidence,
            "n_retrieved": len(evidence),
            "mean_similarity": round(mean_sim, 4),
            "chunks_discarded_below_floor": discarded,
            "sufficient": verdict.sufficient,
            "warnings": []
            if verdict.sufficient
            else [f"retrieval attempt {attempt + 1} ({strategy}): {verdict.reason}"],
        }

    return node


# ---------------------------------------------------------------- reason

_REASON_SYSTEM = (
    "You are a financial analyst reading SEC filing excerpts. Produce factual "
    "claims that are stated in or directly computable from the excerpts. Every "
    "claim must cite the chunk ids it came from. Never state a figure that is "
    "not in the excerpts. Respond with JSON only."
)


def _augment_with_provenance(
    retrieved: list[EvidenceChunk],
    metrics: FinancialMetrics,
    chunks: list[Chunk],
) -> list[EvidenceChunk]:
    """Add every chunk a metric cites to the evidence set.

    Retrieved chunks keep their similarity and their order. Provenance-only
    chunks are appended with similarity 0.0, which is honest: they were not
    retrieved for this query, they are here because a published figure came from
    them and the reader must be able to look it up.
    """
    known = {e.chunk_id for e in retrieved}
    by_id = {c.chunk_id: c for c in chunks}

    cited: list[str] = []
    for m in metrics.extracted:
        cited.append(m.source.chunk_id)
    for d in metrics.derived:
        cited.extend(s.chunk_id for s in d.sources)

    out = list(retrieved)
    for chunk_id in cited:
        if chunk_id in known or chunk_id not in by_id:
            continue
        known.add(chunk_id)
        chunk = by_id[chunk_id]
        out.append(
            EvidenceChunk(
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                text=chunk.text,
                similarity=0.0,
                chunk_type=chunk.chunk_type,
            )
        )
    return out


def _fmt_claim_value(value: float) -> str:
    """Format a figure for a claim without destroying it.

    ``{:,.0f}`` turns $3.71 of earnings per share into "4", and the verifier then
    checks the wrong number against the evidence. Small magnitudes keep their
    decimals; large ones do not need them.
    """
    if abs(value) < 1000:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:,.0f}"


def _deterministic_findings(
    metrics: FinancialMetrics, evidence: list[EvidenceChunk]
) -> list[Finding]:
    """Findings built directly from extracted metrics and their real sources.

    Used when no LLM is configured. These are grounded by construction, which is
    what makes the pipeline runnable and measurable without a model - and it is
    the honest fallback: the alternative would be an empty reasoning block or,
    much worse, invented prose.
    """
    known = {c.chunk_id for c in evidence}
    findings: list[Finding] = []

    for m in metrics.extracted:
        if m.source.chunk_id not in known:
            continue
        findings.append(
            Finding(
                claim=(
                    f"{metric_label(m.name)} is reported as "
                    f"{_fmt_claim_value(m.value_in_units)} {m.unit}."
                ),
                supporting_chunks=[m.source.chunk_id],
                verification=Verification.UNSUPPORTED,  # verify node decides
                factor=None,
            )
        )

    for d in metrics.derived:
        cited = [s.chunk_id for s in d.sources if s.chunk_id in known]
        if not cited:
            continue
        findings.append(
            Finding(
                claim=(f"{metric_label(d.name)} computes to {d.value:,.2f} ({d.formula})."),
                supporting_chunks=cited,
                verification=Verification.UNSUPPORTED,
                factor=None,
            )
        )
    return findings


def make_reason_node(cfg: AffaConfig, llm: LLMClient | None = None):
    """Evidence -> findings with citations. Runs once, after the loop settles."""

    def node(state: AgentState) -> dict[str, Any]:
        retrieved: list[EvidenceChunk] = state.get("evidence", [])
        metrics: FinancialMetrics = state.get("financial_metrics") or FinancialMetrics()

        # KPI extraction reads the whole document, so a metric's source chunk is
        # often not one the retriever surfaced. Those chunks still have to be in
        # the report's evidence list, or the citations pointing at them do not
        # resolve and the report stops being traceable - which the schema
        # refuses outright. The citation set is therefore
        # `retrieved evidence` union `metric provenance`, and retrieval
        # diagnostics keep counting only what retrieval actually returned.
        evidence = _augment_with_provenance(retrieved, metrics, state.get("chunks", []))

        findings = _deterministic_findings(metrics, evidence)
        chain = (
            f"Retrieved {len(evidence)} passages over "
            f"{state.get('retrieval_attempts', 0)} attempt(s); "
            f"derived {len(findings)} candidate claims from extracted metrics."
        )
        warnings: list[str] = []

        if llm is not None and not getattr(llm, "is_stub", False) and evidence:
            excerpt = "\n\n".join(
                f"[{e.chunk_id}] (page {e.page}) {e.text[:1200]}" for e in evidence[:8]
            )
            prompt = (
                f"Question: {state.get('question', '')}\n\n"
                f"Excerpts:\n{excerpt}\n\n"
                'Return JSON: [{"claim": "...", "supporting_chunks": ["<chunk id>"]}]'
            )
            try:
                response = llm.complete(prompt, system=_REASON_SYSTEM)
                parsed = extract_json(response.text)
                known = {e.chunk_id for e in evidence}
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict) or not item.get("claim"):
                            continue
                        cited = [c for c in item.get("supporting_chunks", []) if c in known]
                        findings.append(
                            Finding(
                                claim=str(item["claim"])[:600],
                                supporting_chunks=cited,
                                verification=Verification.UNSUPPORTED,
                                factor=item.get("factor"),
                            )
                        )
                    chain += f" LLM ({llm.name}) proposed {len(parsed)} additional claims."
                else:
                    warnings.append(
                        f"reasoner {llm.name} returned unparseable output; "
                        "kept deterministic findings only"
                    )
            except Exception as exc:  # pragma: no cover - runtime robustness
                warnings.append(f"reasoner call failed ({exc}); kept deterministic findings")

        return {
            "findings": findings,
            "evidence": evidence,
            "chain_of_thought": chain,
            "warnings": warnings,
        }

    return node


# ---------------------------------------------------------------- verify


def make_verify_node(cfg: AffaConfig):
    """The critic. Mandatory, and the source of the faithfulness metric."""

    def node(state: AgentState) -> dict[str, Any]:
        findings: list[Finding] = state.get("findings", [])
        evidence: list[EvidenceChunk] = state.get("evidence", [])
        outcome = verify_findings(findings, evidence, cfg.verification)

        warnings: list[str] = []
        if outcome.dropped:
            warnings.append(f"{outcome.dropped} claim(s) dropped as unsupported by their citations")
        contradicted = sum(1 for c in outcome.checks if c.verdict is Verification.CONTRADICTED)
        if contradicted:
            warnings.append(f"{contradicted} claim(s) contradicted by the cited passages")

        return {
            "verified_findings": outcome.kept,
            "unsupported_claims_dropped": outcome.dropped,
            "warnings": warnings,
        }

    return node


# ------------------------------------------------------------- recommend

_NARRATIVE_SYSTEM = (
    "You explain a rubric's output in plain language for a research report. "
    "The verdict and the factor scores are given to you and are final: do not "
    "change, re-weight, or second-guess them. Write two short paragraphs. "
    "Do not introduce any figure that is not in the material provided."
)


def make_recommend_node(cfg: AffaConfig, llm: LLMClient | None = None):
    """Deterministic rubric verdict, with an optional LLM-written narrative."""

    def node(state: AgentState) -> dict[str, Any]:
        metrics: FinancialMetrics = state.get("financial_metrics") or FinancialMetrics()
        sentiment: SentimentBlock = state.get("sentiment") or SentimentBlock()
        verified: list[Finding] = state.get("verified_findings", [])

        outcome = run_rubric(
            metrics,
            cfg=cfg,
            sentiment_score=sentiment.score,
            risk_severity_index=state.get("risk_severity_index"),
            disagreements=metrics.disagreements,
            # A factor may only enter the recommendation on the back of evidence
            # the verify node accepted.
            verified_citations=any(f.verification is Verification.SUPPORTED for f in verified),
        )
        recommendation = outcome.recommendation
        warnings = list(outcome.notes)

        if llm is not None and not getattr(llm, "is_stub", False):
            summary = {
                "assessment": recommendation.assessment.value,
                "confidence": recommendation.confidence,
                "factor_scores": recommendation.factor_scores,
                "rationale": [r.statement for r in recommendation.rationale],
            }
            try:
                response = llm.complete(
                    f"Rubric output:\n{summary}\n\nWrite the explanation.",
                    system=_NARRATIVE_SYSTEM,
                )
                if response.text.strip():
                    recommendation = recommendation.model_copy(
                        update={"narrative": response.text.strip()}
                    )
            except Exception as exc:  # pragma: no cover - runtime robustness
                warnings.append(f"narrative generation failed ({exc})")

        return {"recommendation": recommendation, "warnings": warnings}

    return node


# ------------------------------------------------------------ synthesize


def make_synthesize_node(cfg: AffaConfig, models: ModelsUsed):
    """Assemble the validated report."""

    def node(state: AgentState) -> dict[str, Any]:
        meta = state.get("doc_metadata") or {}
        metrics: FinancialMetrics = state.get("financial_metrics") or FinancialMetrics()
        evidence: list[EvidenceChunk] = state.get("evidence", [])
        recommendation = state.get("recommendation")

        if recommendation is None:  # pragma: no cover - defensive
            from affa.schema import Recommendation

            recommendation = Recommendation(
                assessment=Assessment.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                rubric_version="1.0",
                disclaimer=cfg.report.disclaimer,
            )

        report = AnalysisReport(
            metadata=ReportMetadata(
                company=meta.get("company"),
                ticker=meta.get("ticker"),
                doc_type=meta.get("doc_type"),
                fiscal_period=meta.get("fiscal_period"),
                source_file=state.get("source_file"),
                pipeline_version=cfg.pipeline_version,
                models=models,
            ),
            financial_metrics=metrics,
            sentiment=state.get("sentiment") or SentimentBlock(),
            risk_factors=state.get("risk_factors", []),
            evidence=evidence,
            reasoning=ReasoningBlock(
                findings=state.get("verified_findings", []),
                chain_of_thought=(
                    state.get("chain_of_thought", "") if cfg.report.include_chain_of_thought else ""
                ),
            ),
            recommendation=recommendation,
            retrieval_diagnostics=RetrievalDiagnostics(
                chunks_retrieved=state.get("n_retrieved", len(evidence)),
                chunks_discarded_below_floor=state.get("chunks_discarded_below_floor", 0),
                mean_similarity=state.get("mean_similarity", 0.0),
                retries=max(0, state.get("retrieval_attempts", 0) - 1),
                reformulations=state.get("queries_tried", []),
                sufficient=state.get("sufficient", False),
                stop_reason=stop_reason(state, cfg),
            ),
            unsupported_claims_dropped=state.get("unsupported_claims_dropped", 0),
            warnings=list(dict.fromkeys(state.get("warnings", []))),
        )
        return {"report": report}

    return node

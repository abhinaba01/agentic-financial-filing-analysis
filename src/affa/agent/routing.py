"""Routing decisions and the threshold-reachability assertion.

Two invariants live here, both from section 3.

**Reachability (anti-pattern #1).** Retrieval discards chunks below
``retrieval.min_similarity`` (f). The mean similarity of the survivors is
therefore always >= f, so a rule of "retry when mean similarity < t" is dead code
unless ``t > f``. The assertion runs at *import* time against the shipped
defaults, so a config edit that breaks it fails the moment anything imports the
router, not on some later document that happened to retrieve badly.

**One retry mechanism (anti-pattern #2).** The decision to retry lives in this
module and nowhere else. Nodes may count attempts; they may not decide to stop.
Two retry mechanisms with different trigger conditions fight each other, and the
symptom is a loop that stops early on some documents and never on others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from affa.agent.state import AgentState
from affa.config import AffaConfig, get_config, validate_threshold_reachability

# --- import-time reachability check ---------------------------------------
# Deliberately at module scope. If configs/default.yaml ever sets a retry
# threshold at or below the similarity floor, importing affa.agent.routing
# raises ConfigError with an explanation instead of shipping a dead branch.
_DEFAULTS = get_config()
validate_threshold_reachability(
    min_similarity=_DEFAULTS.retrieval.min_similarity,
    retry_below=_DEFAULTS.routing.retry_below_mean_similarity,
)

RetrievalDecision = Literal["retry", "proceed"]


@dataclass(frozen=True)
class SufficiencyVerdict:
    """Why retrieval was or was not judged sufficient. Ends up in the report."""

    sufficient: bool
    reason: str
    mean_similarity: float
    n_chunks: int


def assess_sufficiency(
    evidence_count: int, mean_similarity: float, cfg: AffaConfig | None = None
) -> SufficiencyVerdict:
    """Judge the evidence set. Pure function of its inputs, so it is testable."""
    cfg = cfg or get_config()
    routing = cfg.routing

    if evidence_count == 0:
        return SufficiencyVerdict(False, "no chunks survived the similarity floor", 0.0, 0)
    if evidence_count < routing.min_chunks_for_sufficiency:
        return SufficiencyVerdict(
            False,
            f"only {evidence_count} chunks retrieved, need {routing.min_chunks_for_sufficiency}",
            mean_similarity,
            evidence_count,
        )
    if mean_similarity < routing.retry_below_mean_similarity:
        return SufficiencyVerdict(
            False,
            f"mean similarity {mean_similarity:.3f} below "
            f"{routing.retry_below_mean_similarity:.3f}",
            mean_similarity,
            evidence_count,
        )
    return SufficiencyVerdict(
        True,
        f"{evidence_count} chunks at mean similarity {mean_similarity:.3f}",
        mean_similarity,
        evidence_count,
    )


def route_after_retrieval(state: AgentState, cfg: AffaConfig | None = None) -> RetrievalDecision:
    """The only place the retry loop can be continued or stopped.

    Wraps retrieval alone (section 3): a retry re-searches with a reformulated
    query, and generation runs once, after the loop settles. Putting an LLM call
    inside this loop would pay for a generation on every discarded attempt.
    """
    cfg = cfg or get_config()
    if state.get("sufficient", False):
        return "proceed"
    if state.get("retrieval_attempts", 0) >= cfg.routing.max_retrieval_attempts:
        return "proceed"  # budget exhausted; downstream reports thin evidence
    return "retry"


def stop_reason(state: AgentState, cfg: AffaConfig | None = None) -> str:
    """Human-readable explanation of how the loop terminated."""
    cfg = cfg or get_config()
    if state.get("sufficient", False):
        return "sufficient evidence"
    attempts = state.get("retrieval_attempts", 0)
    if attempts >= cfg.routing.max_retrieval_attempts:
        return (
            f"retry budget exhausted after {attempts} attempts; "
            "proceeding with the evidence available"
        )
    return "loop exited without a sufficiency verdict"

"""LangGraph ``StateGraph`` assembly (section 3).

    START -> [extract_kpis | analyze_sentiment | extract_risks | doc_metadata]
          -> (fan-in) -> retrieve -> {retry -> retrieve | proceed}
          -> reason -> verify -> recommend -> synthesize -> END

Notes on the shape, each from the design rules:

* The four branches run concurrently and each returns only its own keys.
* The re-query loop wraps ``retrieve`` alone. ``reason`` sits outside it and runs
  once, so a retry costs a vector search and not a generation.
* The retry decision lives in ``route_after_retrieval`` and nowhere else.
* ``verify`` is not optional and has no bypass edge.

A pure-Python fallback executor is provided for environments without langgraph
installed. It runs the identical node functions in the identical order and is
used by the tests that assert branch isolation, so the delta discipline is
verified even where langgraph is absent. It is *not* a substitute for langgraph
in production: it runs the branches sequentially and does no checkpointing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from affa.agent.nodes import (
    make_doc_metadata_node,
    make_extract_kpis_node,
    make_reason_node,
    make_recommend_node,
    make_retrieve_node,
    make_risk_node,
    make_sentiment_node,
    make_synthesize_node,
    make_verify_node,
)
from affa.agent.routing import route_after_retrieval
from affa.agent.state import BRANCH_NODES, AgentState
from affa.config import AffaConfig, get_config
from affa.llm import LLMClient
from affa.schema import ModelsUsed

log = logging.getLogger(__name__)


def models_used(cfg: AffaConfig, *, tagger_used: bool, sentiment_used: bool) -> ModelsUsed:
    """Record which models *actually ran*, not which ones are configured.

    Anti-pattern #4 is documentation naming a model that was swapped out. The
    report metadata is generated from runtime facts for the same reason.
    """
    return ModelsUsed(
        embedder=cfg.models.embedder.name,
        xbrl_tagger=cfg.models.xbrl_tagger.active_name if tagger_used else "rule_based_only",
        sentiment=cfg.models.sentiment.active_name if sentiment_used else "lexicon_fallback",
        reasoner=cfg.models.reasoner.active_name,
    )


@dataclass
class GraphBundle:
    """The compiled graph plus the node callables, so tests can poke at either."""

    app: Any
    nodes: dict[str, Any]
    cfg: AffaConfig
    backend: str


def build_nodes(
    cfg: AffaConfig,
    *,
    store: Any,
    document: Any | None = None,
    llm: LLMClient | None = None,
    tagger: Any | None = None,
    classifier: Any | None = None,
    models: ModelsUsed | None = None,
) -> dict[str, Any]:
    resolved_models = models or models_used(
        cfg,
        tagger_used=bool(tagger is not None and getattr(tagger, "available", False)),
        sentiment_used=bool(classifier is not None and getattr(classifier, "available", False)),
    )
    return {
        "extract_kpis": make_extract_kpis_node(cfg, tagger=tagger),
        "analyze_sentiment": make_sentiment_node(cfg, classifier=classifier),
        "extract_risks": make_risk_node(cfg),
        "extract_doc_metadata": make_doc_metadata_node(cfg, document=document),
        "retrieve": make_retrieve_node(cfg, store),
        "reason": make_reason_node(cfg, llm=llm),
        "verify": make_verify_node(cfg),
        "recommend": make_recommend_node(cfg, llm=llm),
        "synthesize": make_synthesize_node(cfg, resolved_models),
    }


def build_graph(
    cfg: AffaConfig | None = None,
    *,
    store: Any,
    document: Any | None = None,
    llm: LLMClient | None = None,
    tagger: Any | None = None,
    classifier: Any | None = None,
    models: ModelsUsed | None = None,
    force_fallback: bool = False,
) -> GraphBundle:
    """Compile the analysis graph."""
    cfg = cfg or get_config()
    nodes = build_nodes(
        cfg,
        store=store,
        document=document,
        llm=llm,
        tagger=tagger,
        classifier=classifier,
        models=models,
    )

    if not force_fallback:
        try:
            return GraphBundle(
                app=_compile_langgraph(cfg, nodes), nodes=nodes, cfg=cfg, backend="langgraph"
            )
        except ImportError:
            log.warning(
                "langgraph not installed; using the sequential fallback executor. "
                'Install it with: pip install -e ".[agent]"'
            )

    return GraphBundle(
        app=SequentialExecutor(cfg, nodes), nodes=nodes, cfg=cfg, backend="sequential"
    )


def _compile_langgraph(cfg: AffaConfig, nodes: dict[str, Any]) -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # Fan-out: four concurrent branches from START, each writing only its keys.
    for branch in BRANCH_NODES:
        graph.add_edge(START, branch)
    # Fan-in: retrieve waits for all four.
    for branch in BRANCH_NODES:
        graph.add_edge(branch, "retrieve")

    # Bounded re-query loop around retrieval alone.
    graph.add_conditional_edges(
        "retrieve",
        lambda state: route_after_retrieval(state, cfg),
        {"retry": "retrieve", "proceed": "reason"},
    )

    graph.add_edge("reason", "verify")
    graph.add_edge("verify", "recommend")
    graph.add_edge("recommend", "synthesize")
    graph.add_edge("synthesize", END)

    # The loop can re-enter `retrieve` at most max_retrieval_attempts times; the
    # recursion limit is set above that so the routing budget is what stops the
    # loop, not the framework's guard rail firing as an error.
    return graph.compile()


class SequentialExecutor:
    """Fallback executor with the same semantics, minus concurrency."""

    def __init__(self, cfg: AffaConfig, nodes: dict[str, Any]) -> None:
        self.cfg = cfg
        self.nodes = nodes

    @staticmethod
    def _merge(state: AgentState, delta: dict[str, Any]) -> AgentState:
        merged = dict(state)
        for key, value in delta.items():
            # Mirrors the operator.add reducer declared on `warnings`.
            if key == "warnings":
                merged["warnings"] = list(merged.get("warnings", [])) + list(value)
            else:
                merged[key] = value
        return merged  # type: ignore[return-value]

    def invoke(self, state: AgentState, config: dict[str, Any] | None = None) -> AgentState:
        current: AgentState = dict(state)  # type: ignore[assignment]

        # Branches read the same input state, exactly as in a real superstep.
        branch_input: AgentState = dict(current)  # type: ignore[assignment]
        for name in BRANCH_NODES:
            current = self._merge(current, self.nodes[name](branch_input))

        while True:
            current = self._merge(current, self.nodes["retrieve"](current))
            if route_after_retrieval(current, self.cfg) == "proceed":
                break

        for name in ("reason", "verify", "recommend", "synthesize"):
            current = self._merge(current, self.nodes[name](current))
        return current

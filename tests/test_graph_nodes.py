"""Graph node contracts (section 3, anti-pattern #10).

The central rule: concurrent branches return only their own state keys. A branch
that returns the whole state makes every branch write every key in one
superstep, which LangGraph rejects - and which is invisible until you add the
second parallel node.
"""

from __future__ import annotations

import pytest

from affa.agent.graph import build_graph, build_nodes
from affa.agent.state import BRANCH_NODES, BRANCH_OUTPUT_KEYS, new_state
from affa.schema import ModelsUsed


@pytest.fixture
def models() -> ModelsUsed:
    return ModelsUsed(
        embedder="test-embedder",
        xbrl_tagger="rule_based_only",
        sentiment="lexicon_fallback",
        reasoner="stub",
    )


@pytest.fixture
def nodes(cfg, stub_store, parsed_document, models):
    return build_nodes(cfg, store=stub_store, document=parsed_document, models=models)


@pytest.fixture
def base_state(financial_chunks):
    return new_state(
        doc_id="TESTDOC",
        source_file="test.json",
        chunks=financial_chunks,
        question="What were revenue, margins, leverage and cash generation?",
    )


@pytest.mark.parametrize("branch", BRANCH_NODES)
def test_branch_returns_only_its_own_keys(branch, nodes, base_state) -> None:
    delta = nodes[branch](base_state)
    assert isinstance(delta, dict)
    extra = set(delta) - BRANCH_OUTPUT_KEYS[branch]
    assert not extra, f"{branch} wrote keys it does not own: {sorted(extra)}"


@pytest.mark.parametrize("branch", BRANCH_NODES)
def test_branch_does_not_mutate_incoming_state(branch, nodes, base_state) -> None:
    """Branches share one input object during a superstep; mutating it corrupts siblings."""
    before_keys = set(base_state)
    before_chunks = list(base_state["chunks"])
    nodes[branch](base_state)
    assert set(base_state) == before_keys
    assert base_state["chunks"] == before_chunks


def test_branches_write_disjoint_keys() -> None:
    """No two branches may own the same key, except the reduced ``warnings``."""
    seen: dict[str, str] = {}
    for branch, keys in BRANCH_OUTPUT_KEYS.items():
        for key in keys - {"warnings"}:
            assert key not in seen, f"{branch} and {seen[key]} both own {key!r}"
            seen[key] = branch


def test_retrieve_node_counts_but_does_not_stop(nodes, base_state, cfg) -> None:
    """Anti-pattern #2: a node may count attempts, but only the router may stop."""
    state = dict(base_state)
    for expected in range(1, 4):
        delta = nodes["retrieve"](state)
        assert delta["retrieval_attempts"] == expected
        assert "stop" not in delta
        state.update(delta)


def test_retrieve_discards_below_the_similarity_floor(nodes, base_state, cfg) -> None:
    delta = nodes["retrieve"](base_state)
    assert all(e.similarity >= cfg.retrieval.min_similarity for e in delta["evidence"])


def test_retrieve_reformulates_on_the_second_attempt(nodes, base_state) -> None:
    state = dict(base_state)
    state.update(nodes["retrieve"](state))
    first = state["query"]
    state.update(nodes["retrieve"](state))
    assert state["query"] != first
    assert len(set(state["queries_tried"])) == len(state["queries_tried"])


def test_reason_node_makes_no_llm_call_inside_the_retry_loop(
    cfg, stub_store, parsed_document, models, base_state
) -> None:
    """Section 3: retrieval and generation are separate nodes.

    The loop wraps retrieval alone, so retries cost a vector search - never a
    generation. This counts calls to prove it.
    """
    calls: list[str] = []

    class CountingLLM:
        name = "counting"
        is_stub = False

        def complete(self, prompt, *, system=None):
            calls.append(prompt)
            from affa.llm.base import LLMResponse

            return LLMResponse(text="[]", model=self.name)

    graph_nodes = build_nodes(
        cfg, store=stub_store, document=parsed_document, llm=CountingLLM(), models=models
    )
    state = dict(base_state)
    for _ in range(3):
        state.update(graph_nodes["retrieve"](state))
    assert calls == [], "retrieval retries must not invoke the reasoner"

    state.update(graph_nodes["extract_kpis"](state))
    graph_nodes["reason"](state)
    assert len(calls) == 1, "generation should run exactly once, after the loop"


def test_full_graph_produces_a_valid_report(
    cfg, stub_store, parsed_document, models, base_state
) -> None:
    bundle = build_graph(cfg, store=stub_store, document=parsed_document, models=models)
    final = bundle.app.invoke(base_state, {"recursion_limit": 25})
    report = final["report"]
    assert report.recommendation.disclaimer
    assert report.retrieval_diagnostics.stop_reason
    assert report.metadata.models.embedder == "test-embedder"


def test_sequential_fallback_matches_langgraph_shape(
    cfg, stub_store, parsed_document, models, base_state
) -> None:
    """The fallback executor runs the same nodes in the same order."""
    fallback = build_graph(
        cfg, store=stub_store, document=parsed_document, models=models, force_fallback=True
    )
    assert fallback.backend == "sequential"
    final = fallback.app.invoke(dict(base_state))
    assert final["report"].recommendation.rubric_version == "1.0"


def test_warnings_accumulate_across_nodes(
    cfg, stub_store, parsed_document, models, base_state
) -> None:
    """``warnings`` uses an add-reducer; concurrent appends must not clobber."""
    bundle = build_graph(
        cfg, store=stub_store, document=parsed_document, models=models, force_fallback=True
    )
    final = bundle.app.invoke(dict(base_state))
    assert len(final["warnings"]) >= 2

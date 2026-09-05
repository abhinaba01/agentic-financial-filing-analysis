"""Retrieval evaluation: NDCG@10, Hit@1/5, MRR (sections 5.2 and 9).

Two corpora, deliberately:

* ``BeIR/fiqa`` - retail-investor forum discussion,
* ``PatronusAI/financebench`` - SEC filing passages.

Section 5.2's evidence-based instruction is to train on 10-K QA rather than
FiQA, because fine-tuning on FiQA in a prior project improved FiQA NDCG@10 by
2.3% and *cost* 11.6% Hit@1 on filing retrieval. Evaluating on both is how that
trade-off stays visible instead of being hidden behind one favourable number.

This harness makes no LLM calls, so before/after comparisons are free - run it
often.
"""

from __future__ import annotations

import argparse
import logging
import os
import random

from affa.config import get_config
from affa.eval.harness import EvaluationResult, Evaluator, LiteratureReference
from affa.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank

log = logging.getLogger(__name__)


class RetrievalEvaluator(Evaluator):
    """Retrieval quality against a stock embedder on identical data."""

    name = "retrieval"
    default_dataset = "BeIR/fiqa"
    default_baseline = "BAAI/bge-base-en-v1.5"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default=None,
            help="Fine-tuned embedder to evaluate (default: models.embedder from config)",
        )
        parser.add_argument(
            "--top-k", type=int, default=10, help="Cutoff for NDCG/Hit (default 10)"
        )
        parser.add_argument(
            "--corpus-sample",
            type=int,
            default=5000,
            help="Corpus documents to index. Sampling makes absolute numbers "
            "incomparable to published figures; only the model-vs-baseline delta is valid.",
        )

    def run(self, args: argparse.Namespace) -> EvaluationResult:
        cfg = get_config(args.config) if args.config else get_config()

        # The hashing stub is lexical, not semantic. A retrieval number produced
        # with it would be meaningless while looking exactly like a real one, so
        # this harness refuses rather than reporting it.
        if os.environ.get("AFFA_FORCE_STUB_EMBEDDER") == "1":
            raise SystemExit(
                "AFFA_FORCE_STUB_EMBEDDER=1 is set. The stub embedder scores lexical "
                "overlap, not semantic similarity, so any NDCG/Hit/MRR from it is "
                "meaningless. Unset the variable and install '.[ingest]' to benchmark "
                "a real embedding model."
            )

        dataset = args.test_set or self.default_dataset
        model_name = args.model or cfg.models.embedder.name
        baseline_name = args.baseline or self.default_baseline

        queries, corpus, qrels = _load_ir_dataset(dataset, limit=args.limit, seed=args.seed)
        if args.corpus_sample and len(corpus) > args.corpus_sample:
            corpus = _sample_corpus(corpus, qrels, args.corpus_sample, args.seed)

        notes = [
            f"corpus indexed: {len(corpus)} documents",
            f"queries evaluated: {len(queries)}",
        ]
        if model_name == baseline_name:
            notes.append(
                "model and baseline are the same checkpoint: this run measures the "
                "harness, not a fine-tune. Set --model to your fine-tuned embedder."
            )

        metrics = _score_model(model_name, queries, corpus, qrels, args.top_k, cfg)
        baseline_metrics = _score_model(baseline_name, queries, corpus, qrels, args.top_k, cfg)

        literature = []
        if "fiqa" in dataset.lower():
            literature.append(
                LiteratureReference(
                    source="BEIR leaderboard",
                    metric="NDCG@10 (bge-base-en-v1.5, full corpus)",
                    value=0.406,
                    conditions="full 57k-document corpus, no sampling; not this run's setup",
                )
            )

        return EvaluationResult(
            component="retrieval",
            dataset=dataset,
            split="test",
            n_examples=len(queries),
            metrics=metrics,
            baseline_name=baseline_name,
            baseline_metrics=baseline_metrics,
            model_name=model_name,
            notes=notes,
            literature=literature,
            subset_of=args.corpus_sample if args.corpus_sample else None,
            seed=args.seed,
        )


def _sample_corpus(
    corpus: dict[str, str], qrels: dict[str, dict[str, float]], n: int, seed: int
) -> dict[str, str]:
    """Sample the corpus but always keep every gold document.

    Dropping a gold passage would make the metric measure sampling luck rather
    than retrieval, and would depress both model and baseline unequally.
    """
    gold = {doc_id for rels in qrels.values() for doc_id in rels}
    keep = {doc_id: corpus[doc_id] for doc_id in gold if doc_id in corpus}
    rest = [doc_id for doc_id in corpus if doc_id not in keep]
    rng = random.Random(seed)
    rng.shuffle(rest)
    for doc_id in rest[: max(0, n - len(keep))]:
        keep[doc_id] = corpus[doc_id]
    return keep


def _load_ir_dataset(
    name: str, *, limit: int | None, seed: int
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, float]]]:
    """Load queries, corpus and qrels for an IR dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            'retrieval evaluation needs `datasets`. Install: pip install -e ".[eval]"'
        ) from exc

    if "financebench" in name.lower():
        return _load_financebench(limit, seed)

    corpus_ds = load_dataset(name, "corpus", split="corpus")
    queries_ds = load_dataset(name, "queries", split="queries")
    qrels_ds = load_dataset(f"{name}-qrels", split="test")

    corpus = {str(r["_id"]): f"{r.get('title', '')} {r.get('text', '')}".strip() for r in corpus_ds}
    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_ds:
        qid, cid = str(row["query-id"]), str(row["corpus-id"])
        qrels.setdefault(qid, {})[cid] = float(row["score"])

    queries = {str(r["_id"]): str(r["text"]) for r in queries_ds if str(r["_id"]) in qrels}
    if limit:
        keys = sorted(queries)
        random.Random(seed).shuffle(keys)
        queries = {k: queries[k] for k in keys[:limit]}
        qrels = {k: v for k, v in qrels.items() if k in queries}
    return queries, corpus, qrels


def _load_financebench(
    limit: int | None, seed: int
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, float]]]:
    from datasets import load_dataset

    ds = load_dataset("PatronusAI/financebench", split="train")
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, float]] = {}

    for i, row in enumerate(ds):
        qid = f"q{i}"
        evidence = row.get("evidence") or []
        texts = [
            e.get("evidence_text", "")
            for e in evidence
            if isinstance(e, dict) and e.get("evidence_text")
        ] or [row.get("evidence_text", "")]
        texts = [t for t in texts if t]
        if not texts or not row.get("question"):
            continue
        queries[qid] = str(row["question"])
        for j, text in enumerate(texts):
            cid = f"d{i}_{j}"
            corpus[cid] = str(text)
            qrels.setdefault(qid, {})[cid] = 1.0

    if limit:
        keys = sorted(queries)
        random.Random(seed).shuffle(keys)
        queries = {k: queries[k] for k in keys[:limit]}
        qrels = {k: v for k, v in qrels.items() if k in queries}
    return queries, corpus, qrels


def _score_model(
    model_name: str,
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels: dict[str, dict[str, float]],
    top_k: int,
    cfg,
) -> dict[str, float]:
    """Index the corpus with ``model_name`` and score every query."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import semantic_search

    model = SentenceTransformer(model_name)
    doc_ids = list(corpus)
    # Same instruction policy as inference: configs/default.yaml sets it to null,
    # so queries and documents are encoded identically. A mismatch here would
    # measure the prefix, not the model.
    instruction = cfg.models.embedder.query_instruction or ""

    doc_emb = model.encode(
        [corpus[d] for d in doc_ids],
        batch_size=cfg.models.embedder.batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    query_ids = list(queries)
    query_emb = model.encode(
        [f"{instruction}{queries[q]}" for q in query_ids],
        batch_size=cfg.models.embedder.batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    hits = semantic_search(query_emb, doc_emb, top_k=max(top_k, 10))

    ndcgs, h1, h5, mrrs = [], [], [], []
    for qid, result in zip(query_ids, hits, strict=True):
        ranked = [doc_ids[h["corpus_id"]] for h in result]
        relevance = qrels.get(qid, {})
        relevant = {d for d, s in relevance.items() if s > 0}
        ndcgs.append(ndcg_at_k(ranked, relevance, top_k))
        h1.append(hit_at_k(ranked, relevant, 1))
        h5.append(hit_at_k(ranked, relevant, 5))
        mrrs.append(reciprocal_rank(ranked, relevant))

    n = max(len(query_ids), 1)
    return {
        f"ndcg@{top_k}": round(sum(ndcgs) / n, 4),
        "hit@1": round(sum(h1) / n, 4),
        "hit@5": round(sum(h5) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI shim
    from affa.eval.harness import base_parser, emit

    evaluator = RetrievalEvaluator()
    parser = base_parser("affa-eval retrieval", RetrievalEvaluator.__doc__ or "")
    evaluator.add_arguments(parser)
    args = parser.parse_args(argv)
    emit(evaluator.run(args), args.output)
    return 0

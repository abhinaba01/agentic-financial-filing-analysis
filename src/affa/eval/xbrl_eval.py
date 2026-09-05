"""XBRL numeric tagger evaluation on FiNER-139 (sections 5.1 and 9).

Scored with ``seqeval`` at the span level. Token accuracy is meaningless here:
``O`` dominates so heavily that predicting it everywhere scores above 95%.

The per-concept breakdown is not optional. The tag distribution is severely
skewed, and micro-F1 hides that most of the 139 concepts are too rare to learn
from a subset - a headline number that looks close to the paper can sit on top
of a model that has learned six concepts and ignored the rest.
"""

from __future__ import annotations

import argparse
import logging

from affa.config import get_config
from affa.eval.harness import EvaluationResult, Evaluator, LiteratureReference

log = logging.getLogger(__name__)

FINER_PAPER = LiteratureReference(
    source="FiNER-139 paper (Loukas et al., 2022), sec-bert-base row",
    metric="micro-F1",
    value=0.892,
    conditions="full 900k/112k/108k splits, 2 epochs; NOT produced by this repo",
)


class XBRLEvaluator(Evaluator):
    """Span-level F1 for the FiNER-139 numeric tagger."""

    name = "xbrl"
    default_dataset = "nlpaueb/finer-139"
    default_baseline = "nlpaueb/sec-bert-base (no fine-tuning)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", default=None, help="Fine-tuned tagger to evaluate")
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--max-length", type=int, default=256)
        parser.add_argument(
            "--top-concepts",
            type=int,
            default=25,
            help="How many per-concept rows to include (default 25)",
        )

    def run(self, args: argparse.Namespace) -> EvaluationResult:
        cfg = get_config(args.config) if args.config else get_config()
        model_name = args.model or cfg.models.xbrl_tagger.active_name
        baseline_name = args.baseline or self.default_baseline

        dataset, label_names, total = _load_finer(args.test_set, args.limit, args.seed)

        notes = [
            f"label set: {len(label_names)} tags (139 concepts x B-/I- plus O)",
            "scored with seqeval at span level; token accuracy is not reported "
            "because O dominates the distribution",
        ]
        if args.limit:
            notes.append(
                f"SUBSET: {len(dataset)} of {total} test examples. Absolute numbers are "
                "not comparable to the paper - only the model-vs-baseline delta below is."
            )

        metrics, per_concept = _score(
            model_name, dataset, label_names, args.batch_size, args.max_length
        )
        baseline_metrics, _ = _score(
            _baseline_checkpoint(baseline_name, cfg),
            dataset,
            label_names,
            args.batch_size,
            args.max_length,
        )

        notes.append(
            "baseline is the base encoder with an untrained tagging head: a measured "
            "floor on this exact data, not a strong competitor. The paper's number is "
            "listed separately below and was produced under different conditions."
        )

        ranked = sorted(per_concept.items(), key=lambda kv: -kv[1]["support"])
        for concept, scores in ranked[: args.top_concepts]:
            notes.append(
                f"per-concept {concept}: P={scores['precision']:.3f} "
                f"R={scores['recall']:.3f} F1={scores['f1']:.3f} n={scores['support']}"
            )
        unlearned = [c for c, s in per_concept.items() if s["support"] > 0 and s["f1"] == 0.0]
        if unlearned:
            notes.append(
                f"{len(unlearned)} concepts present in the test set scored F1=0. "
                "Micro-F1 does not show this."
            )

        return EvaluationResult(
            component="xbrl",
            dataset=args.test_set or self.default_dataset,
            split="test",
            n_examples=len(dataset),
            metrics=metrics,
            baseline_name=baseline_name,
            baseline_metrics=baseline_metrics,
            model_name=model_name,
            notes=notes,
            literature=[FINER_PAPER],
            subset_of=total,
            seed=args.seed,
        )


def _baseline_checkpoint(baseline_name: str, cfg) -> str:
    return baseline_name.split(" ")[0] if " " in baseline_name else baseline_name


def _load_finer(test_set: str | None, limit: int | None, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError('needs `datasets`. Install: pip install -e ".[eval]"') from exc

    name = test_set or "nlpaueb/finer-139"
    try:
        # finer-139 is a loading-script dataset. datasets>=4.0 removed script
        # execution, and the third-party parquet mirrors are not equivalent -
        # at least one is deduplicated, which changes the splits and breaks
        # comparability with the paper.
        ds = load_dataset(name, trust_remote_code=True)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"could not load {name}. It is a loading-script dataset requiring "
            "datasets>=2.19,<4.0 and trust_remote_code=True. Do not substitute a "
            f"parquet mirror: the splits differ. Underlying error: {exc}"
        ) from exc

    split = ds["test"]
    total = len(split)
    label_names = split.features["ner_tags"].feature.names
    if limit:
        split = split.shuffle(seed=seed).select(range(min(limit, total)))
    return split, label_names, total


def _score(model_name: str, dataset, label_names, batch_size: int, max_length: int):
    """Run the tagger and score spans with seqeval."""
    import torch
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_names)
    ).eval()

    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        encoded = tok(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        predictions = logits.argmax(-1)

        for i, gold in enumerate(batch["ner_tags"]):
            word_ids = encoded.word_ids(batch_index=i)
            seen: set[int] = set()
            t_seq: list[str] = []
            p_seq: list[str] = []
            for pos, word_id in enumerate(word_ids):
                # Only the first sub-word of a word carries a tag; continuations
                # and specials are ignored. Getting this wrong scores a
                # different task and produces a meaningless F1.
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                if word_id >= len(gold):
                    continue
                t_seq.append(label_names[gold[word_id]])
                p_seq.append(label_names[int(predictions[i][pos])])
            if t_seq:
                true_seqs.append(t_seq)
                pred_seqs.append(p_seq)

    metrics = {
        "micro_f1": round(float(f1_score(true_seqs, pred_seqs, average="micro")), 4),
        "micro_precision": round(float(precision_score(true_seqs, pred_seqs, average="micro")), 4),
        "micro_recall": round(float(recall_score(true_seqs, pred_seqs, average="micro")), 4),
        "macro_f1": round(float(f1_score(true_seqs, pred_seqs, average="macro")), 4),
    }

    report = classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    per_concept = {
        concept: {
            "precision": float(scores["precision"]),
            "recall": float(scores["recall"]),
            "f1": float(scores["f1-score"]),
            "support": int(scores["support"]),
        }
        for concept, scores in report.items()
        if isinstance(scores, dict) and concept not in {"micro avg", "macro avg", "weighted avg"}
    }
    return metrics, per_concept

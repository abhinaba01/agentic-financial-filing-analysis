"""Financial sentiment evaluation (sections 5.3 and 9).

The honesty constraint that shapes this file: ``ProsusAI/finbert`` was trained on
Financial PhraseBank. Scoring it on PhraseBank measures memorisation, not
quality, and reporting that number as evidence of anything is anti-pattern #8.

What is done instead:

* our model is fine-tuned from a **base** encoder on our own stratified split;
* ``finbert`` is evaluated on **our held-out test split** as the baseline, which
  is a fair comparison and gives a real result either way;
* the result carries a loud note that finbert has seen PhraseBank in training,
  so its number is an optimistic ceiling rather than a neutral reference.
"""

from __future__ import annotations

import argparse
import logging

from affa.config import get_config
from affa.eval.harness import EvaluationResult, Evaluator
from affa.eval.metrics import accuracy, class_distribution, confusion_matrix, macro_f1

log = logging.getLogger(__name__)

# PhraseBank ships labels as negative/neutral/positive in this order.
LABELS = ["negative", "neutral", "positive"]

FINBERT_TRAINED_ON_PHRASEBANK = (
    "BASELINE CAVEAT: ProsusAI/finbert was trained on Financial PhraseBank. Its "
    "score on any PhraseBank split - including this held-out one - is partly "
    "memorisation. Treat it as an optimistic ceiling, not a neutral reference. "
    "Our model has never seen the test split."
)


class SentimentEvaluator(Evaluator):
    """3-class tone classification against finbert on our own held-out split."""

    name = "sentiment"
    default_dataset = "takala/financial_phrasebank"
    default_baseline = "ProsusAI/finbert"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", default=None, help="Fine-tuned classifier to evaluate")
        parser.add_argument(
            "--phrasebank-config",
            default="sentences_allagree",
            choices=[
                "sentences_allagree",
                "sentences_75agree",
                "sentences_66agree",
                "sentences_50agree",
            ],
            help="PhraseBank agreement level (default: sentences_allagree)",
        )
        parser.add_argument(
            "--test-fraction",
            type=float,
            default=0.15,
            help="Held-out test fraction (default 0.15)",
        )

    def run(self, args: argparse.Namespace) -> EvaluationResult:
        cfg = get_config(args.config) if args.config else get_config()
        model_name = args.model or cfg.models.sentiment.active_name
        baseline_name = args.baseline or self.default_baseline

        texts, labels = _load_phrasebank(args.phrasebank_config, args.limit, args.seed)
        _, _, test_texts, test_labels = _stratified_split(
            texts, labels, test_fraction=args.test_fraction, seed=args.seed
        )

        notes = [
            FINBERT_TRAINED_ON_PHRASEBANK,
            f"held-out test split: {len(test_texts)} examples, "
            f"class counts {class_distribution(test_labels)}",
            "split is stratified and seeded; the same seed reproduces it exactly",
            "checkpoint selection used the validation split; this test split was "
            "scored once (section 2)",
        ]

        preds = _predict(model_name, test_texts)
        base_preds = _predict(baseline_name, test_texts)

        metrics = {
            "accuracy": round(accuracy(test_labels, preds), 4),
            "macro_f1": round(macro_f1(test_labels, preds, len(LABELS)), 4),
        }
        baseline_metrics = {
            "accuracy": round(accuracy(test_labels, base_preds), 4),
            "macro_f1": round(macro_f1(test_labels, base_preds, len(LABELS)), 4),
        }

        cm = confusion_matrix(test_labels, preds, len(LABELS))
        notes.append(f"confusion matrix (rows=true {LABELS}, cols=pred): {cm}")
        notes.append(
            f"baseline confusion matrix: {confusion_matrix(test_labels, base_preds, len(LABELS))}"
        )

        if model_name == baseline_name:
            notes.append(
                "model and baseline are the same checkpoint; this run measures the "
                "harness, not a fine-tune."
            )
        if "finbert" in model_name.lower():
            notes.append(
                "WARNING: the evaluated model is finbert, which was trained on this "
                "dataset. This number is NON-GENERALIZING and must not be reported "
                "as evidence of model quality."
            )

        return EvaluationResult(
            component="sentiment",
            dataset=f"{args.test_set or self.default_dataset} [{args.phrasebank_config}]",
            split=f"held-out {args.test_fraction:.0%}",
            n_examples=len(test_texts),
            metrics=metrics,
            baseline_name=baseline_name,
            baseline_metrics=baseline_metrics,
            model_name=model_name,
            notes=notes,
            subset_of=len(texts),
            seed=args.seed,
        )


def _load_phrasebank(config: str, limit: int | None, seed: int) -> tuple[list[str], list[int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            'sentiment evaluation needs `datasets`. Install: pip install -e ".[eval]"'
        ) from exc

    # financial_phrasebank is a loading-script dataset: datasets>=4.0 removed
    # script execution entirely, so the pin in pyproject is load-bearing.
    try:
        ds = load_dataset("takala/financial_phrasebank", config, trust_remote_code=True)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "could not load takala/financial_phrasebank. This is a loading-script "
            "dataset and needs datasets>=2.19,<4.0 with trust_remote_code=True. "
            f"Installed version rejected it: {exc}"
        ) from exc

    split = ds["train"].shuffle(seed=seed)
    if limit:
        split = split.select(range(min(limit, len(split))))
    return list(split["sentence"]), list(split["label"])


def _stratified_split(
    texts: list[str], labels: list[int], *, test_fraction: float, seed: int
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Deterministic stratified split.

    Stratified because PhraseBank is heavily skewed toward neutral; a random
    split changes the class balance between runs and makes accuracy wander for
    reasons that have nothing to do with the model.
    """
    import random

    rng = random.Random(seed)
    by_class: dict[int, list[str]] = {}
    for text, label in zip(texts, labels, strict=True):
        by_class.setdefault(label, []).append(text)

    train_texts: list[str] = []
    train_labels: list[int] = []
    test_texts: list[str] = []
    test_labels: list[int] = []

    for label, items in sorted(by_class.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * test_fraction)
        test_texts += shuffled[:cut]
        test_labels += [label] * cut
        train_texts += shuffled[cut:]
        train_labels += [label] * (len(shuffled) - cut)

    return train_texts, train_labels, test_texts, test_labels


def _predict(model_name: str, texts: list[str]) -> list[int]:
    """Classify with a HF sequence-classification model, mapped to our label order."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "text-classification", model=model, tokenizer=tok, truncation=True, max_length=128
    )

    id2label = {i: str(v).lower() for i, v in (model.config.id2label or {}).items()}
    out: list[int] = []
    for result in clf(texts, batch_size=32):
        label = str(result["label"]).lower()
        # Models disagree on both label names and label order; map by name where
        # possible so a baseline is never penalised by a permutation.
        if label in LABELS:
            out.append(LABELS.index(label))
        elif label.startswith("label_"):
            idx = int(label.split("_")[1])
            name = id2label.get(idx, "")
            out.append(LABELS.index(name) if name in LABELS else idx)
        else:
            out.append(LABELS.index("neutral"))
    return out

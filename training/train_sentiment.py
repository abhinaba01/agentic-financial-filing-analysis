"""Fine-tune the financial sentiment / tone classifier (section 5.3).

    Base    nlpaueb/sec-bert-base (or distilroberta-base) - NOT ProsusAI/finbert
    Data    takala/financial_phrasebank, config sentences_allagree
    T4      batch 32, max_len 128, fp16, 3-4 epochs, lr 2e-5

**Do not fine-tune finbert here.** ``ProsusAI/finbert`` was already trained on
Financial PhraseBank, so any score it gets on this dataset is train-set scoring
and proves nothing (anti-pattern #8). The script refuses a finbert base model
unless you explicitly acknowledge it.

The fair comparison, which this setup makes available: our model fine-tuned from
a base encoder on our own stratified split, versus finbert evaluated on *our
held-out split*. That is a real result either way, and it is what
``affa-eval sentiment`` reports.

Run::

    python training/train_sentiment.py --output-dir /content/drive/MyDrive/affa/sentiment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.common import (  # noqa: E402
    RunConfig,
    assert_checkpoint_selection_is_valid,
    require_datasets_below_4,
    resolve_checkpoint_dir,
    resume_checkpoint,
    set_global_seed,
    training_arguments,
)

BASE_MODEL = "nlpaueb/sec-bert-base"
DATASET = "takala/financial_phrasebank"
LABELS = ["negative", "neutral", "positive"]

CONTAMINATED_BASES = ("prosusai/finbert", "yiyanghkust/finbert")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Checkpoint dir (put it on Drive)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--phrasebank-config",
        default="sentences_allagree",
        choices=[
            "sentences_allagree",
            "sentences_75agree",
            "sentences_66agree",
            "sentences_50agree",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--allow-ephemeral", action="store_true")
    parser.add_argument(
        "--i-know-this-base-saw-phrasebank",
        action="store_true",
        help="Acknowledge that the chosen base was trained on this dataset. The result "
        "is then labelled non-generalizing and must be reported that way.",
    )
    parser.add_argument("--final-test", action="store_true", help="Score TEST once, at the end")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if any(bad in args.base_model.lower() for bad in CONTAMINATED_BASES):
        if not args.i_know_this_base_saw_phrasebank:
            raise SystemExit(
                f"{args.base_model} was trained on Financial PhraseBank. Fine-tuning it "
                "on the same dataset and reporting the score is anti-pattern #8: the "
                "number measures memorisation, not quality.\n\n"
                "Use a base encoder (nlpaueb/sec-bert-base, distilroberta-base) and "
                "evaluate finbert on YOUR held-out split as the baseline instead - "
                "`affa-eval sentiment` does exactly that.\n\n"
                "To proceed anyway, pass --i-know-this-base-saw-phrasebank; the result "
                "must then be labelled NON-GENERALIZING wherever it appears."
            )
        print(
            "\n*** WARNING: base model saw this dataset in pre-training. Every number "
            "from this run is NON-GENERALIZING and must be labelled as such. ***\n"
        )

    require_datasets_below_4()
    set_global_seed(args.seed)

    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
    )

    ds = load_dataset(DATASET, args.phrasebank_config, trust_remote_code=True)["train"]
    print(f"[data] {len(ds)} sentences, config={args.phrasebank_config}")

    splits = _stratified_three_way(ds, args.seed, args.val_fraction, args.test_fraction)
    for name, split in splits.items():
        counts = {LABELS[i]: sum(1 for label in split["label"] if label == i) for i in range(3)}
        print(f"[data] {name}: {len(split)} {counts}")

    # PhraseBank rows repeat verbatim across agreement configs; a duplicate that
    # straddles the split boundary would leak the test answer into training.
    from affa.eval.metrics import overlap_count

    leak, examples = overlap_count(splits["train"]["sentence"], splits["test"]["sentence"])
    print(f"[overlap] train/test exact duplicates: {leak}")
    for example in examples[:3]:
        print(f"[overlap]   {example[:110]}")
    if leak:
        keep = [
            i
            for i, sentence in enumerate(splits["test"]["sentence"])
            if " ".join(sentence.lower().split())
            not in {" ".join(s.lower().split()) for s in splits["train"]["sentence"]}
        ]
        splits["test"] = splits["test"].select(keep)
        print(f"[overlap] dropped {leak} leaked rows from test; {len(splits['test'])} remain")

    assert_checkpoint_selection_is_valid(selection_split="validation", reporting_split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)},
    )

    def tokenize(batch):
        return tokenizer(
            batch["sentence"], truncation=True, max_length=args.max_length, padding=False
        )

    encoded = {
        name: split.map(tokenize, batched=True, remove_columns=["sentence"])
        for name, split in splits.items()
    }

    def compute_metrics(eval_pred):
        import numpy as np

        from affa.eval.metrics import accuracy, macro_f1

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1).tolist()
        labels = list(labels)
        return {
            "accuracy": accuracy(labels, predictions),
            "f1": macro_f1(labels, predictions, 3),
        }

    output_dir = resolve_checkpoint_dir(args.output_dir, allow_ephemeral=args.allow_ephemeral)
    run_config = RunConfig(
        task="sentiment",
        base_model=args.base_model,
        dataset=f"{DATASET}:{args.phrasebank_config}",
        seed=args.seed,
        train_samples=len(splits["train"]),
        eval_samples=len(splits["validation"]),
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        extra={
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "stratified": True,
        },
    )

    trainer = Trainer(
        model=model,
        args=training_arguments(
            output_dir=output_dir,
            save_steps=args.save_steps,
            seed=args.seed,
            metric_for_best_model="f1",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            per_device_eval_batch_size=args.batch_size,
        ),
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    last = resume_checkpoint(output_dir, run_config)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print("\n[validation] (selected the checkpoint - not a headline number)")
    print(trainer.evaluate())

    if args.final_test:
        print("\n[test] scoring the held-out test split - ONCE")
        print(trainer.evaluate(eval_dataset=encoded["test"], metric_key_prefix="test"))
        print(
            "\nFor the fair finbert comparison on this same held-out split:\n"
            f"  affa-eval sentiment --model {output_dir / 'final'} "
            "--baseline ProsusAI/finbert --output eval_results/sentiment.json"
        )

    if args.push_to_hub:
        trainer.push_to_hub()
    return 0


def _stratified_three_way(ds, seed: int, val_fraction: float, test_fraction: float):
    """Stratified split. PhraseBank is skewed toward neutral, so proportions are held."""
    by_class: dict[int, list[int]] = {}
    for i, label in enumerate(ds["label"]):
        by_class.setdefault(int(label), []).append(i)

    import random

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        n_test = int(len(indices) * test_fraction)
        n_val = int(len(indices) * val_fraction)
        test_idx += indices[:n_test]
        val_idx += indices[n_test : n_test + n_val]
        train_idx += indices[n_test + n_val :]

    return {
        "train": ds.select(sorted(train_idx)),
        "validation": ds.select(sorted(val_idx)),
        "test": ds.select(sorted(test_idx)),
    }


if __name__ == "__main__":
    raise SystemExit(main())

"""Fine-tune the FiNER-139 XBRL numeric tagger (section 5.1).

    Base     nlpaueb/sec-bert-base
    Data     nlpaueb/finer-139 (900,384 / 112,494 / 108,378)
    Task     token classification, 279 labels (139 concepts x B-/I-, plus O)
    T4       batch 32, max_len 256, fp16, 2 epochs, lr 3e-5

Plain ``sec-bert-base`` on purpose, not ``sec-bert-num`` or ``sec-bert-shape``,
so the comparison against the paper's row is clean.

Two things that train happily while producing meaningless numbers, both handled
explicitly below:

* **Label alignment.** Only the first sub-word of a word carries the tag;
  continuation sub-words and specials get ``-100``. Getting this wrong trains
  without complaint and yields an F1 that measures a different task.
* **Scoring.** ``seqeval`` at span level. Token accuracy is meaningless here
  because ``O`` dominates - predicting it everywhere scores above 95%.

Run::

    python training/train_xbrl_tagger.py --output-dir /content/drive/MyDrive/affa/xbrl \\
        --train-samples 200000 --eval-samples 10000
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
DATASET = "nlpaueb/finer-139"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Checkpoint dir (put it on Drive)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=None, help="Subset size, or all")
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--allow-ephemeral", action="store_true")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Score the TEST split. Do this exactly once, at the very end.",
    )
    return parser.parse_args(argv)


def align_labels(examples, tokenizer, max_length: int):
    """Tokenize and align word-level tags to sub-word tokens.

    The first sub-word of each word keeps the tag; every continuation sub-word
    and every special token gets ``-100`` so the loss ignores it.
    """
    encoded = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    aligned: list[list[int]] = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = encoded.word_ids(batch_index=i)
        previous_word = None
        labels: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word:
                labels.append(tags[word_id])
            else:
                labels.append(-100)
            previous_word = word_id
        aligned.append(labels)
    encoded["labels"] = aligned
    return encoded


def build_metrics(label_names):
    from seqeval.metrics import f1_score, precision_score, recall_score

    def compute(eval_pred):
        import numpy as np

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_seqs, pred_seqs = [], []
        for pred_row, label_row in zip(predictions, labels, strict=True):
            true_seq, pred_seq = [], []
            for predicted, gold in zip(pred_row, label_row, strict=True):
                # -100 is HF's ignore index: continuation sub-words and specials.
                if gold == -100:
                    continue
                true_seq.append(label_names[gold])
                pred_seq.append(label_names[predicted])
            if true_seq:
                true_seqs.append(true_seq)
                pred_seqs.append(pred_seq)
        return {
            "f1": float(f1_score(true_seqs, pred_seqs, average="micro")),
            "precision": float(precision_score(true_seqs, pred_seqs, average="micro")),
            "recall": float(recall_score(true_seqs, pred_seqs, average="micro")),
        }

    return compute


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_datasets_below_4()
    set_global_seed(args.seed)

    from datasets import load_dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
    )

    ds = load_dataset(DATASET, trust_remote_code=True)
    label_names = ds["train"].features["ner_tags"].feature.names
    print(f"[data] {len(label_names)} labels; splits: {[(k, len(v)) for k, v in ds.items()]}")

    train = ds["train"].shuffle(seed=args.seed)
    if args.train_samples:
        train = train.select(range(min(args.train_samples, len(train))))
    # Selection happens on validation. The test split stays untouched until the
    # single --final-test run.
    validation = ds["validation"].shuffle(seed=args.seed)
    if args.eval_samples:
        validation = validation.select(range(min(args.eval_samples, len(validation))))

    assert_checkpoint_selection_is_valid(selection_split="validation", reporting_split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: i for i, name in enumerate(label_names)},
    )

    tokenized_train = train.map(
        lambda b: align_labels(b, tokenizer, args.max_length),
        batched=True,
        remove_columns=train.column_names,
        desc="tokenizing train",
    )
    tokenized_val = validation.map(
        lambda b: align_labels(b, tokenizer, args.max_length),
        batched=True,
        remove_columns=validation.column_names,
        desc="tokenizing validation",
    )

    output_dir = resolve_checkpoint_dir(args.output_dir, allow_ephemeral=args.allow_ephemeral)
    run_config = RunConfig(
        task="xbrl_tagger",
        base_model=args.base_model,
        dataset=DATASET,
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        extra={"num_labels": len(label_names)},
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_metrics(label_names),
    )

    # Idempotent: re-running this after a crash resumes, with no code edit.
    last = resume_checkpoint(output_dir, run_config)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print("\n[validation] (this split selected the checkpoint - not a headline number)")
    print(trainer.evaluate())

    if args.final_test:
        print("\n[test] scoring the held-out test split - do this ONCE")
        tokenized_test = ds["test"].map(
            lambda b: align_labels(b, tokenizer, args.max_length),
            batched=True,
            remove_columns=ds["test"].column_names,
            desc="tokenizing test",
        )
        print(trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test"))
        print(
            "\nFor the per-concept breakdown (required - micro-F1 hides that most of "
            "the 139 concepts are too rare to learn from a subset), run:\n"
            "  affa-eval xbrl --model <this checkpoint> --output eval_results/xbrl.json"
        )

    if args.push_to_hub:
        trainer.push_to_hub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

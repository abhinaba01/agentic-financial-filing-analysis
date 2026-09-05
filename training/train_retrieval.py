"""Fine-tune the retrieval embedder (section 5.2).

    Base    BAAI/bge-base-en-v1.5 (bge-large if VRAM allows)
    Data    virattt/financial-qa-10K - question -> context pairs over 10-K filings
    Loss    CachedMultipleNegativesRankingLoss (GradCache)
    T4      effective batch 64, mini_batch_size 8-16, 1 epoch, lr 2e-5

**Train on 10-K QA, not FiQA.** This is the single most important choice here
and it is evidence-based: in a prior project, fine-tuning ``bge-large`` on FiQA
improved FiQA NDCG@10 by 2.3% and *cost* 11.6% Hit@1 on filing retrieval. FiQA
is retail-investor forum discussion; filings are SEC prose. Train in-domain, and
evaluate on both to show you know the difference.

Details that decide whether this works:

* **Batch size is the in-batch negative count** for this loss, so it matters
  more than epochs. GradCache is what lets a 16GB T4 hold an effective 64.
* ``BatchSamplers.NO_DUPLICATES`` - two rows sharing a positive in one batch
  trains the model against a true positive.
* **The BGE query instruction prefix is used in neither training nor
  inference.** A mismatch between the two is worse than skipping it, so the
  choice is made once here and mirrored by ``models.embedder.query_instruction:
  null`` in ``configs/default.yaml``.
* **FinanceBench overlap is checked and dropped before training**, and the count
  is reported even when it is zero.

Run::

    python training/train_retrieval.py --output-dir /content/drive/MyDrive/affa/embedder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Both the repo root and src/ must be importable: `training.common` lives at the
# root, `affa` lives under src/. Running this file directly puts only
# training/ on sys.path, so neither resolves without this.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from training.common import (  # noqa: E402
    RunConfig,
    assert_checkpoint_selection_is_valid,
    require_accelerate,
    resolve_checkpoint_dir,
    resume_checkpoint,
    set_global_seed,
)

BASE_MODEL = "BAAI/bge-base-en-v1.5"
DATASET = "virattt/financial-qa-10K"

# Documented once, here and in configs/default.yaml. Do not set one without the
# other: a prefix used at training time but not inference time (or vice versa)
# shifts the query embedding away from what the model was tuned for.
QUERY_INSTRUCTION: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Checkpoint dir (put it on Drive)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Effective batch = in-batch negative count. Matters more than epochs.",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=8,
        help="GradCache sub-batch actually resident on the GPU (8-16 on a T4)",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--allow-ephemeral", action="store_true")
    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help="Skip the FinanceBench overlap check (not recommended)",
    )
    return parser.parse_args(argv)


def drop_financebench_overlap(dataset, *, skip: bool = False):
    """Remove training rows whose context appears in FinanceBench gold passages.

    Section 5.2 requires this check, and section 2 requires the count to be
    reported even when it is zero. Training on an evaluation passage turns the
    benchmark into a memorisation test.
    """
    if skip:
        print("[overlap] SKIPPED by flag - any FinanceBench number from this run is suspect")
        return dataset

    from affa.eval.metrics import overlap_count

    try:
        from datasets import load_dataset

        bench = load_dataset("PatronusAI/financebench", split="train")
    except Exception as exc:
        print(f"[overlap] could not load FinanceBench ({exc}); overlap NOT verified")
        return dataset

    gold: list[str] = []
    for row in bench:
        evidence = row.get("evidence") or []
        if isinstance(evidence, list):
            gold += [
                e.get("evidence_text", "")
                for e in evidence
                if isinstance(e, dict) and e.get("evidence_text")
            ]
        if row.get("evidence_text"):
            gold.append(str(row["evidence_text"]))
    gold = [g for g in gold if g]

    contexts = list(dataset["context"])
    count, examples = overlap_count(gold, contexts)
    print(f"[overlap] FinanceBench gold vs training contexts: {count} exact matches")
    for example in examples[:3]:
        print(f"[overlap]   dropping: {example[:110]}")

    if count == 0:
        return dataset

    normalized_gold = {" ".join(g.lower().split()) for g in gold}
    keep = [
        i
        for i, ctx in enumerate(contexts)
        if " ".join(str(ctx).lower().split()) not in normalized_gold
    ]
    print(f"[overlap] keeping {len(keep)} of {len(contexts)} training rows")
    return dataset.select(keep)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_accelerate()
    set_global_seed(args.seed)

    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    raw = load_dataset(DATASET, split="train").shuffle(seed=args.seed)
    raw = drop_financebench_overlap(raw, skip=args.skip_overlap_check)

    pairs = raw.select_columns(["question", "context"]).rename_columns(
        {"question": "anchor", "context": "positive"}
    )
    n_eval = min(args.eval_samples, max(len(pairs) // 10, 1))
    validation = pairs.select(range(n_eval))
    train = pairs.select(range(n_eval, len(pairs)))
    if args.train_samples:
        train = train.select(range(min(args.train_samples, len(train))))

    print(f"[data] train={len(train)} validation={len(validation)}")
    assert_checkpoint_selection_is_valid(selection_split="validation", reporting_split="test")
    print(
        "[splits] checkpoint selection uses this validation split. Reporting happens on "
        "BeIR/fiqa and FinanceBench via `affa-eval retrieval`, which this run never sees."
    )
    if QUERY_INSTRUCTION is None:
        print("[prefix] no query instruction in training; configs/default.yaml matches (null)")

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_length

    # GradCache: gradients are computed in mini_batch_size chunks, so the
    # effective batch (and therefore the in-batch negative count) can be 64 on a
    # T4 that could not hold 64 at once.
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)

    output_dir = resolve_checkpoint_dir(args.output_dir, allow_ephemeral=args.allow_ephemeral)
    run_config = RunConfig(
        task="retrieval_embedder",
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
        extra={
            "mini_batch_size": args.mini_batch_size,
            "query_instruction": QUERY_INSTRUCTION,
            "loss": "CachedMultipleNegativesRankingLoss",
        },
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        fp16=True,
        # Two rows sharing a positive in one batch would train the model against
        # a true positive, which is the opposite of the intended signal.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=max(args.save_steps // 10, 10),
        report_to=[],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint" if args.push_to_hub else "every_save",
        hub_private_repo=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        loss=loss,
    )

    last = resume_checkpoint(output_dir, run_config)
    trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(str(output_dir / "final"))

    print(
        "\nNext: measure it on BOTH corpora, because the point of training in-domain "
        "is a trade-off you have to be able to see.\n"
        f"  affa-eval retrieval --test-set BeIR/fiqa --model {output_dir / 'final'} "
        "--output eval_results/retrieval_fiqa.json\n"
        f"  affa-eval retrieval --test-set PatronusAI/financebench --model "
        f"{output_dir / 'final'} --output eval_results/retrieval_financebench.json"
    )
    if args.push_to_hub:
        trainer.push_to_hub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared training utilities: durable checkpointing, resume, and split hygiene.

Section 5.5, implemented once for all four fine-tunes. The failure being
defended against is specific: a Colab runtime dies at minute 39 of a 40-minute
epoch, and everything is lost because the checkpoints were on ephemeral disk or
the resume path was never tested.

Three things this module enforces:

1. **Checkpoints go somewhere that survives the VM.** ``/content`` and any local
   ``output_dir`` vanish with the runtime, which is precisely the event being
   defended against. :func:`resolve_checkpoint_dir` refuses an ephemeral path
   unless the caller explicitly opts in.

2. **Resume is idempotent and guarded.** :func:`resume_checkpoint` finds the
   last checkpoint with no code edit. :func:`RunConfig` is written into the
   checkpoint directory and re-checked on resume: changing the seed, the subset
   size or the base model after a crash means the global step now indexes into
   different data, and the resumed run is silently meaningless
   (anti-pattern #14). That is refused rather than warned about.

3. **The test split is touched once.** :func:`split_train_val_test` produces
   three splits and :func:`assert_checkpoint_selection_is_valid` refuses a
   configuration that selects checkpoints on the split it intends to report
   (anti-pattern #7).
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RUN_CONFIG_FILENAME = "affa_run_config.json"

# Paths that do not survive a Colab runtime restart.
EPHEMERAL_PREFIXES = ("/content", "/tmp", "/var/tmp", "./", "outputs", "results")


class ResumeMismatchError(RuntimeError):
    """Raised when a resume would continue a run under different settings."""


@dataclass
class RunConfig:
    """Everything that must not change across a resume.

    Written into the checkpoint directory as JSON. On resume the current
    settings are compared field by field, and a mismatch raises rather than
    training on quietly by accident.
    """

    task: str
    base_model: str
    dataset: str
    seed: int
    train_samples: int | None
    eval_samples: int | None
    max_length: int
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / RUN_CONFIG_FILENAME
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> RunConfig | None:
        path = Path(directory) / RUN_CONFIG_FILENAME
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def differences(self, other: RunConfig) -> dict[str, tuple[Any, Any]]:
        mine, theirs = asdict(self), asdict(other)
        return {k: (theirs[k], mine[k]) for k in mine if mine[k] != theirs[k]}


def resolve_checkpoint_dir(
    path: str | Path,
    *,
    allow_ephemeral: bool = False,
) -> Path:
    """Validate that checkpoints will outlive the runtime.

    Writing checkpoints to ephemeral storage looks like working checkpointing
    right up until the run you needed it for (anti-pattern #13), so an ephemeral
    path has to be opted into explicitly.
    """
    resolved = Path(path)
    text = str(resolved).replace("\\", "/")
    durable = (
        "/drive/" in text
        or text.startswith("/content/drive")
        or os.environ.get("AFFA_ALLOW_EPHEMERAL_CKPT") == "1"
    )
    if not durable and not allow_ephemeral:
        looks_ephemeral = any(text.startswith(p) for p in EPHEMERAL_PREFIXES)
        if looks_ephemeral:
            raise ValueError(
                f"checkpoint dir {resolved} is on ephemeral storage and will vanish "
                "with the Colab VM - which is the exact failure checkpointing is for. "
                "Mount Drive and point output_dir inside it, or push to the Hub with "
                "hub_strategy='checkpoint'. Pass allow_ephemeral=True only when you "
                "are deliberately running a throwaway job."
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resume_checkpoint(
    checkpoint_dir: str | Path, current: RunConfig, *, strict: bool = True
) -> str | None:
    """Find the checkpoint to resume from, refusing an invalid resume.

    Idempotent: re-running the calling cell after a crash resumes automatically,
    with no code edit. Returns ``None`` for a fresh run.
    """
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        current.save(directory)
        return None

    try:
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError('needs transformers. Install: pip install -e ".[train]"') from exc

    last = get_last_checkpoint(str(directory))
    previous = RunConfig.load(directory)

    if last is None:
        current.save(directory)
        return None

    if previous is None:
        message = (
            f"found checkpoint {last} but no {RUN_CONFIG_FILENAME} beside it, so the "
            "settings it was trained under are unknown. Resuming could silently "
            "continue a different run."
        )
        if strict:
            raise ResumeMismatchError(message)
        log.warning("%s Continuing anyway because strict=False.", message)
        current.save(directory)
        return last

    diffs = current.differences(previous)
    if diffs:
        detail = "\n".join(
            f"  {key}: checkpoint={was!r} -> current={now!r}" for key, (was, now) in diffs.items()
        )
        raise ResumeMismatchError(
            f"refusing to resume {last}: settings changed since the checkpoint was "
            f"written.\n{detail}\n\n"
            "The global step points into a specific data order. Changing the seed, "
            "the subset size, or the base model makes the resumed run index into "
            "different data, and its loss curve will not join up with the first half. "
            "Either restore the original settings, or start a fresh run in a new "
            "checkpoint directory."
        )

    log.info("resuming from %s", last)
    return last


def training_arguments(
    *,
    output_dir: str | Path,
    save_steps: int,
    seed: int,
    metric_for_best_model: str,
    learning_rate: float,
    per_device_train_batch_size: int,
    num_train_epochs: float,
    gradient_accumulation_steps: int = 1,
    fp16: bool = True,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    greater_is_better: bool = True,
    **extra: Any,
):
    """``TrainingArguments`` with the section 5.5 checkpointing policy baked in.

    * ``save_strategy="steps"`` - an epoch here is 40+ minutes and a disconnect
      at minute 39 loses all of it.
    * ``eval_strategy`` must equal ``save_strategy`` when
      ``load_best_model_at_end=True`` or ``Trainer`` raises.
    * ``save_total_limit=2`` is mandatory, not tidiness: a full checkpoint is
      3-4x model size (fp32 weights plus two AdamW moments) and Drive's free
      tier is 15GB.
    * ``load_best_model_at_end`` selects on the **validation** split. Never point
      it at the split you intend to report (anti-pattern #7).
    """
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        seed=seed,
        data_seed=seed,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        logging_steps=max(save_steps // 10, 10),
        report_to=[],
    )
    if push_to_hub:
        # Durable without consuming Drive quota, and the private repo keeps an
        # in-progress fine-tune from being published by accident.
        kwargs.update(
            push_to_hub=True,
            hub_model_id=hub_model_id,
            hub_strategy="checkpoint",
            hub_private_repo=True,
        )
    kwargs.update(extra)
    return TrainingArguments(**kwargs)


def assert_checkpoint_selection_is_valid(*, selection_split: str, reporting_split: str) -> None:
    """Refuse to report a score on the split that chose the checkpoint.

    Anti-pattern #7. ``load_best_model_at_end`` picks the checkpoint that scored
    best on whatever it evaluated; reporting that same split turns a selection
    statistic into a headline number.
    """
    if selection_split == reporting_split:
        raise ValueError(
            f"checkpoint selection and reporting both use the {selection_split!r} split. "
            "Select on validation, report on test, and touch test exactly once."
        )


def split_train_val_test(
    dataset,
    *,
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
):
    """Deterministic three-way split for datasets that ship only a train split."""
    shuffled = dataset.shuffle(seed=seed)
    n = len(shuffled)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    return {
        "test": shuffled.select(range(n_test)),
        "validation": shuffled.select(range(n_test, n_test + n_val)),
        "train": shuffled.select(range(n_test + n_val, n)),
    }


def set_global_seed(seed: int) -> None:
    """Seed every RNG that affects data order or initialisation."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require_datasets_below_4() -> None:
    """Fail loudly when ``datasets>=4.0`` would break a loading-script dataset.

    ``finer-139``, ``financial_phrasebank`` and ``finqa`` are loading-script
    datasets. ``datasets>=4.0`` removed script execution entirely, so they do not
    load at all - and the third-party parquet mirrors are not equivalent (at
    least one is deduplicated, which changes the splits and breaks comparability
    with the published numbers).
    """
    import datasets

    major = int(datasets.__version__.split(".")[0])
    if major >= 4:
        raise RuntimeError(
            f"datasets {datasets.__version__} is installed, but the loading-script "
            "datasets this project trains on need datasets>=2.19,<4.0 "
            "(script execution was removed in 4.0).\n\n"
            '    pip install "datasets>=2.19,<4.0"\n\n'
            "Do not substitute a parquet mirror: at least one is deduplicated, which "
            "changes the splits and makes the numbers incomparable to the paper."
        )


def report_overlap(train_texts, eval_texts, *, label: str = "train/eval") -> int:
    """Check and report train/eval overlap, even when it is zero.

    Section 2 requires the count in the output either way: a stated zero is
    evidence the check ran; silence is not.
    """
    from affa.eval.metrics import overlap_count

    count, examples = overlap_count(list(train_texts), list(eval_texts))
    print(
        f"[overlap] {label}: {count} exact-match overlaps out of {len(list(eval_texts))} eval rows"
    )
    for example in examples[:5]:
        print(f"[overlap]   {example[:110]}")
    return count


def time_one_save(trainer, output_dir: str | Path) -> float:
    """Time a checkpoint write so ``save_steps`` can be set from measurement.

    Drive writes are slow enough that saving every few hundred steps can spend
    more wall clock on checkpointing than on training. Section 5.5's rule is to
    time one save and set ``save_steps`` so checkpointing costs under ~5% of
    runtime; this returns the number that rule needs.
    """
    import time

    start = time.time()
    trainer.save_model(str(Path(output_dir) / "timing-probe"))
    elapsed = time.time() - start
    print(f"[checkpoint] one save took {elapsed:.1f}s")
    print(f"[checkpoint] for <5% overhead, one save per >= {elapsed * 20:.0f}s of training")
    return elapsed

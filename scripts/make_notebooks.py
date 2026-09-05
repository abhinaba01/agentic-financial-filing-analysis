"""Generate the four Colab notebooks in ``notebooks/``.

The notebooks are generated rather than hand-edited so the parts that must be
identical across all four - the Drive mount, the repo-freshness check, the
``datasets<4.0`` pin, and the resumable-checkpoint cell from section 5.5 - stay
identical. Edit this file and re-run it; do not edit the .ipynb files directly.

    python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/abhinaba01/agentic-financial-filing-analysis.git"
# Derived, not hand-typed: a separate REPO_DIR constant can silently drift from
# the actual repo name in REPO_URL (it did - "agentic-financial-filing-analyst"
# vs the real "...-analysis" - and notebooks that were never manually patched
# after generation would fail their very first cell trying to `cd` into a
# directory `git clone` never created).
REPO_DIR = REPO_URL.rsplit("/", 1)[-1].removesuffix(".git")
NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# --- cells shared by every notebook --------------------------------------

GPU_CHECK = code(
    """
# Confirm we actually have the T4 this recipe is written for.
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

import torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU."
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""
)

DRIVE_MOUNT = code(
    """
# Checkpoints MUST live somewhere that survives the VM (section 5.5).
# /content is ephemeral - it vanishes with the runtime, which is exactly the
# failure checkpointing exists to defend against.
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = '/content/drive/MyDrive/affa'
os.makedirs(DRIVE_ROOT, exist_ok=True)
print('checkpoints ->', DRIVE_ROOT)
"""
)


def repo_cell() -> dict:
    return code(
        f"""
# Clone or update the repo, and verify it is current. Re-running this notebook
# from the top after a disconnect must not silently train an old revision.
import os, subprocess

REPO_URL = {REPO_URL!r}
REPO_DIR = '/content/{REPO_DIR}'

if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'fetch', '--all'], check=True)

local  = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', 'HEAD'],
                        capture_output=True, text=True).stdout.strip()
remote = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', '@{{u}}'],
                        capture_output=True, text=True).stdout.strip()

if remote and local != remote:
    print(f'repo is BEHIND origin (local {{local[:8]}} != remote {{remote[:8]}})')
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=True)
    print('pulled; RESTART THE RUNTIME so the new code is imported')
else:
    print(f'repo is current at {{local[:8]}}')

os.chdir(REPO_DIR)
"""
    )


INSTALL = code(
    """
# datasets<4.0 is REQUIRED, not a preference: finer-139, financial_phrasebank
# and finqa are loading-script datasets, and datasets>=4.0 removed script
# execution entirely. The parquet mirrors are NOT equivalent - at least one is
# deduplicated, which changes the splits and breaks comparability.
%pip install -q -e ".[train,eval]"
%pip install -q "datasets>=2.19,<4.0"

import datasets, transformers
print('datasets', datasets.__version__, '| transformers', transformers.__version__)
assert int(datasets.__version__.split('.')[0]) < 4, (
    'datasets>=4.0 cannot execute loading scripts; pin datasets>=2.19,<4.0'
)
"""
)

RESUME_MD = md(
    """
## Checkpointing and resume

Colab runtimes disconnect, get recycled, and hit idle timeouts. Everything below
is built so a crash costs minutes, not the whole run.

**What resume restores:** model weights, optimizer moments, LR-scheduler
position, RNG state, global step, and dataloader position. That is why we resume
rather than "just train again from the saved weights" — restarting the optimizer
and the LR schedule from scratch is a *different run*, and its loss curve will
not join up with the first half.

**The cell below is idempotent.** Re-run it after a crash and it resumes
automatically, with no code edit.

**Determinism is a precondition.** `SEED`, `TRAIN_SAMPLES` and `EVAL_SAMPLES` are
written into the checkpoint directory as JSON, and the resume path *refuses* to
continue if they no longer match. Changing any of them after a crash means the
global step now points into different data and the resumed run is silently
meaningless (anti-pattern #14).

**Disk:** a full checkpoint is roughly 3–4× model size — fp32 weights plus two
AdamW moments — so `save_total_limit=2` is required, not tidiness, against
Drive's 15GB free tier. `save_steps` is set for ~15–20 minutes of training, not
per epoch: an epoch here is 40+ minutes and a disconnect at minute 39 loses all
of it.
"""
)

TEST_RESUME_MD = md(
    """
## Test the resume path — do not assume it

Untested resume logic is usually broken resume logic, and the moment you find
out is the moment you have already lost the run.

1. Run the training cell above and let it write at least two checkpoints.
2. **Runtime → Interrupt execution** (or just let the runtime die).
3. Re-run the training cell *unchanged*.

What you should see: `resuming from .../checkpoint-N`, and the loss continuing
from where it stopped rather than restarting near its initial value. If step
numbering restarts at 0, resume is not working — fix that before starting the
real run.
"""
)


def hub_cell(task: str, notes: str) -> dict:
    return code(
        f"""
# Push the model and a card carrying the REAL numbers and the subset size.
# A model card with aspirational numbers is worse than no card.
from huggingface_hub import notebook_login
notebook_login()

HUB_ID = 'YOUR_USERNAME/affa-{task}'

card = \"\"\"---
license: apache-2.0
tags: [finance, sec-filings, affa]
---

# affa-{task}

Fine-tuned for the Agentic Financial Filing Analyst.

{notes}

## Measured results

Fill these in from the evaluation cell above. Report the **test** split score,
the **baseline measured on the same data with the same protocol**, and the
training subset size. Do not paste a number from a paper here.

| metric | this model | baseline | notes |
|---|---:|---:|---|
| (fill in) | | | |

- Training subset: `TRAIN_SAMPLES` (state the number actually used)
- Seed: `SEED`
- Checkpoint selected on: validation split
- Test split touched: once

## Not financial advice

Research and educational use only.
\"\"\"

import pathlib
pathlib.Path(f'{{CKPT_DIR}}/final/README.md').write_text(card, encoding='utf-8')
print('model card written; review it before pushing')
"""
    )


def build_xbrl() -> dict:
    return notebook(
        [
            md(
                """
# 01 — XBRL numeric tagger (FiNER-139)

Fine-tunes `nlpaueb/sec-bert-base` for token classification over 139 XBRL
concepts, so KPI extraction can decide whether a given number *is* `Revenues` or
`NetIncomeLoss` rather than matching a nearby label with a regex.

| | |
|---|---|
| Base | `nlpaueb/sec-bert-base` (plain — not `-num`/`-shape`, so the paper comparison is clean) |
| Data | `nlpaueb/finer-139` — 900,384 / 112,494 / 108,378 |
| Task | token classification, 279 labels (139 concepts × B-/I-, plus `O`) |
| T4 | batch 32, max_len 256, fp16, 2 epochs, lr 3e-5 |

**Published reference (not this repo's result):** the FiNER-139 paper reports
89.2% micro-F1 for `sec-bert-base` on the full splits. If you train on a subset,
your number is not comparable to it — quote your own baseline delta instead.

Two ways this trains happily while producing meaningless numbers:
- **Label alignment** — only the first sub-word of a word carries the tag;
  continuations and specials get `-100`.
- **Scoring** — `seqeval` at span level. Token accuracy is meaningless because
  `O` dominates; predicting it everywhere scores above 95%.
"""
            ),
            GPU_CHECK,
            DRIVE_MOUNT,
            repo_cell(),
            INSTALL,
            code(
                """
# Run configuration. These values are written into the checkpoint directory and
# re-checked on resume - changing one after a crash invalidates the run.
SEED          = 42
TRAIN_SAMPLES = 200_000   # None = all 900k (does not fit one 12h T4 session)
EVAL_SAMPLES  = 10_000
MAX_LENGTH    = 256
BATCH_SIZE    = 32
EPOCHS        = 2
LR            = 3e-5
SAVE_STEPS    = 500       # ~15-20 min of training on a T4 at this batch size

CKPT_DIR = f'{DRIVE_ROOT}/xbrl_tagger'
print(CKPT_DIR)
"""
            ),
            RESUME_MD,
            code(
                """
# Idempotent: re-run after a crash and it resumes from the last checkpoint.
!python training/train_xbrl_tagger.py \\
    --output-dir "{CKPT_DIR}" \\
    --seed {SEED} \\
    --train-samples {TRAIN_SAMPLES} \\
    --eval-samples {EVAL_SAMPLES} \\
    --max-length {MAX_LENGTH} \\
    --batch-size {BATCH_SIZE} \\
    --epochs {EPOCHS} \\
    --learning-rate {LR} \\
    --save-steps {SAVE_STEPS}
"""
            ),
            TEST_RESUME_MD,
            md(
                """
## Evaluate against a measured baseline

The harness reports the per-concept breakdown as well as micro-F1. That is not
optional: the tag distribution is severely skewed, and a headline number close to
the paper's can sit on top of a model that learned six concepts and ignored the
other 133.
"""
            ),
            code(
                """
# Scores the TEST split once, against the base encoder as a measured floor.
# The paper's 89.2% appears in the output under its own heading, labelled as a
# published figure produced under different conditions.
!affa-eval xbrl \\
    --model "{CKPT_DIR}/final" \\
    --limit 5000 \\
    --output eval_results/xbrl.json

import json
print(json.dumps(json.load(open('eval_results/xbrl.json'))['metrics'], indent=2))
"""
            ),
            hub_cell(
                "xbrl-tagger",
                "Token classification over FiNER-139 (139 US-GAAP concepts, B-/I- tagging).",
            ),
        ]
    )


def build_retrieval() -> dict:
    return notebook(
        [
            md(
                """
# 02 — Retrieval embedder (10-K QA)

Fine-tunes `BAAI/bge-base-en-v1.5` on question→context pairs drawn from real
10-K filings.

| | |
|---|---|
| Base | `BAAI/bge-base-en-v1.5` (`bge-large` if VRAM allows) |
| Data | `virattt/financial-qa-10K` |
| Loss | `CachedMultipleNegativesRankingLoss` (GradCache) |
| T4 | effective batch 64, `mini_batch_size` 8–16, 1 epoch, lr 2e-5 |

**Train on 10-K QA, not FiQA.** This is the single most important choice here and
it is evidence-based: fine-tuning `bge-large` on FiQA in a prior project improved
FiQA NDCG@10 by 2.3% and *cost* 11.6% Hit@1 on filing retrieval. FiQA is
retail-investor forum discussion; filings are SEC prose. We train in-domain and
evaluate on **both**, so the trade-off is visible instead of hidden behind the
one favourable number.

Things that decide whether this works:
- **Batch size is the in-batch negative count** for this loss — it matters more
  than epochs, and GradCache is what lets a 16GB T4 hold an effective 64.
- `BatchSamplers.NO_DUPLICATES`: two rows sharing a positive in one batch trains
  the model against a true positive.
- The BGE query-instruction prefix is used in **neither** training nor inference.
  A mismatch is worse than skipping it; `configs/default.yaml` mirrors this.
- FinanceBench overlap is checked and dropped before training, and the count is
  printed even when it is zero.
"""
            ),
            GPU_CHECK,
            DRIVE_MOUNT,
            repo_cell(),
            INSTALL,
            code(
                """
SEED            = 42
TRAIN_SAMPLES   = None    # the dataset is small enough to use in full
EVAL_SAMPLES    = 500
MAX_LENGTH      = 384
BATCH_SIZE      = 64      # effective batch = in-batch negative count
MINI_BATCH_SIZE = 8       # what actually sits on the T4 at once (GradCache)
EPOCHS          = 1
LR              = 2e-5
SAVE_STEPS      = 200

CKPT_DIR = f'{DRIVE_ROOT}/retrieval_embedder'
print(CKPT_DIR)
"""
            ),
            RESUME_MD,
            code(
                """
# SentenceTransformerTrainer subclasses HF Trainer, so the same resume applies.
!python training/train_retrieval.py \\
    --output-dir "{CKPT_DIR}" \\
    --seed {SEED} \\
    --eval-samples {EVAL_SAMPLES} \\
    --max-length {MAX_LENGTH} \\
    --batch-size {BATCH_SIZE} \\
    --mini-batch-size {MINI_BATCH_SIZE} \\
    --epochs {EPOCHS} \\
    --learning-rate {LR} \\
    --save-steps {SAVE_STEPS}
"""
            ),
            TEST_RESUME_MD,
            md(
                """
## Evaluate on both corpora

Retrieval evaluation makes **no LLM calls**, so before/after comparisons are
free — run them often.

Report both tables. If in-domain training helped filings and hurt FiQA, that is
the expected result and the interesting one; reporting only the corpus that
improved would be exactly the kind of selective reporting this project exists to
avoid.
"""
            ),
            code(
                """
# FiQA: forum discussion. Expect this to move less, or to regress.
!affa-eval retrieval \\
    --test-set BeIR/fiqa \\
    --model "{CKPT_DIR}/final" \\
    --baseline BAAI/bge-base-en-v1.5 \\
    --limit 500 \\
    --output eval_results/retrieval_fiqa.json
"""
            ),
            code(
                """
# FinanceBench: SEC filing prose. This is the corpus the product actually serves.
!affa-eval retrieval \\
    --test-set PatronusAI/financebench \\
    --model "{CKPT_DIR}/final" \\
    --baseline BAAI/bge-base-en-v1.5 \\
    --output eval_results/retrieval_financebench.json
"""
            ),
            hub_cell(
                "retrieval-embedder",
                "Bi-encoder fine-tuned on 10-K question/context pairs for filing retrieval.",
            ),
        ]
    )


def build_sentiment() -> dict:
    return notebook(
        [
            md(
                """
# 03 — Financial sentiment / tone classifier

| | |
|---|---|
| Base | `nlpaueb/sec-bert-base` (or `distilroberta-base`) — **not** `ProsusAI/finbert` |
| Data | `takala/financial_phrasebank`, `sentences_allagree` |
| Task | 3-class sequence classification |
| T4 | batch 32, max_len 128, fp16, 3–4 epochs, lr 2e-5 |

**Why not fine-tune finbert:** `ProsusAI/finbert` was already trained on
Financial PhraseBank. Fine-tuning it on the same dataset and reporting the score
measures memorisation, not quality (anti-pattern #8). The training script refuses
a finbert base unless you explicitly acknowledge it.

**The fair comparison instead:** our model fine-tuned from a base encoder on our
own stratified split, versus `finbert` evaluated on *our held-out split*. That is
a real result either way — including the way where finbert wins.
"""
            ),
            GPU_CHECK,
            DRIVE_MOUNT,
            repo_cell(),
            INSTALL,
            code(
                """
SEED          = 42
BASE_MODEL    = 'nlpaueb/sec-bert-base'
PB_CONFIG     = 'sentences_allagree'   # also try sentences_75agree
MAX_LENGTH    = 128
BATCH_SIZE    = 32
EPOCHS        = 4
LR            = 2e-5
SAVE_STEPS    = 50                     # small dataset; steps come quickly
VAL_FRACTION  = 0.15
TEST_FRACTION = 0.15

CKPT_DIR = f'{DRIVE_ROOT}/sentiment'
print(CKPT_DIR)
"""
            ),
            RESUME_MD,
            code(
                """
# Stratified split (PhraseBank is heavily skewed toward neutral), train/test
# duplicate check, checkpoint selection on validation.
!python training/train_sentiment.py \\
    --output-dir "{CKPT_DIR}" \\
    --base-model {BASE_MODEL} \\
    --phrasebank-config {PB_CONFIG} \\
    --seed {SEED} \\
    --max-length {MAX_LENGTH} \\
    --batch-size {BATCH_SIZE} \\
    --epochs {EPOCHS} \\
    --learning-rate {LR} \\
    --save-steps {SAVE_STEPS} \\
    --val-fraction {VAL_FRACTION} \\
    --test-fraction {TEST_FRACTION} \\
    --final-test
"""
            ),
            TEST_RESUME_MD,
            code(
                """
# finbert scored on OUR held-out split, as the baseline. The output carries the
# caveat that finbert saw PhraseBank in training, so its number is an optimistic
# ceiling rather than a neutral reference.
!affa-eval sentiment \\
    --model "{CKPT_DIR}/final" \\
    --baseline ProsusAI/finbert \\
    --phrasebank-config {PB_CONFIG} \\
    --test-fraction {TEST_FRACTION} \\
    --output eval_results/sentiment.json

import json
r = json.load(open('eval_results/sentiment.json'))
print('ours    ', r['metrics'])
print('finbert ', r['baseline_metrics'])
print('delta   ', r['deltas'])
"""
            ),
            md(
                """
### If your model loses to finbert

Record it and say so. A fine-tune that fails to beat its baseline stays in the
README with the measurement — that is a real finding about how much headroom
there is over a strong in-domain model, and deleting it would make every other
number in the repo less trustworthy.
"""
            ),
            hub_cell(
                "sentiment",
                "3-class financial tone classifier, fine-tuned from a base encoder "
                "(not from finbert) on a stratified PhraseBank split.",
            ),
        ]
    )


def build_finqa() -> dict:
    return notebook(
        [
            md(
                """
# 04 — Numerical-reasoning LLM (FinQA, QLoRA) — stretch goal

| | |
|---|---|
| Base | `Qwen2.5-3B-Instruct` (7B only once 3B is comfortable) |
| Data | `ibm/finqa` |
| Method | QLoRA, 4-bit NF4, LoRA r=16 α=32, attention + MLP projections |
| T4 | batch 1, grad-accum 8–16, max_len 1024, gradient checkpointing, paged AdamW |

FinQA supplies multi-step reasoning programs over financial tables — the right
supervision for the derived-KPI and reasoning steps.

**The result worth reporting is the three-way comparison** on identical prompts:
base zero-shot, this QLoRA model, and a hosted frontier model. "Reached X% of the
hosted model's execution accuracy at zero marginal cost" is a strong, honest
claim; "our model scored X%" on its own is not.

QLoRA is the cheap checkpointing case: only adapter weights are saved (~50–100MB),
so this can checkpoint frequently at negligible cost.
"""
            ),
            GPU_CHECK,
            DRIVE_MOUNT,
            repo_cell(),
            code(
                """
%pip install -q -e ".[train,eval,hosted]"
%pip install -q "datasets>=2.19,<4.0" bitsandbytes peft accelerate

import datasets, transformers, peft
print('datasets', datasets.__version__, '| transformers', transformers.__version__,
      '| peft', peft.__version__)
assert int(datasets.__version__.split('.')[0]) < 4
"""
            ),
            code(
                """
SEED          = 42
BASE_MODEL    = 'Qwen/Qwen2.5-3B-Instruct'
TRAIN_SAMPLES = None
EVAL_SAMPLES  = 200
MAX_LENGTH    = 1024
BATCH_SIZE    = 1
GRAD_ACCUM    = 16     # effective batch 16
EPOCHS        = 2
LR            = 2e-4   # higher than full fine-tuning: standard for LoRA
SAVE_STEPS    = 100    # adapters are tiny, so save often

CKPT_DIR = f'{DRIVE_ROOT}/finqa_qlora'
print(CKPT_DIR)
"""
            ),
            RESUME_MD,
            code(
                """
!python training/train_finqa_qlora.py \\
    --output-dir "{CKPT_DIR}" \\
    --base-model {BASE_MODEL} \\
    --seed {SEED} \\
    --eval-samples {EVAL_SAMPLES} \\
    --max-length {MAX_LENGTH} \\
    --batch-size {BATCH_SIZE} \\
    --grad-accum {GRAD_ACCUM} \\
    --epochs {EPOCHS} \\
    --learning-rate {LR} \\
    --save-steps {SAVE_STEPS}
"""
            ),
            TEST_RESUME_MD,
            md(
                """
## The three-way comparison

The architecture puts the local fine-tune and the hosted API model behind one
interface, selectable by config — so this is a flag, not a rewrite.

`--hosted` costs API calls. Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) first,
and keep `--limit` small while you are iterating.
"""
            ),
            code(
                """
# Point the config at the adapter you just trained.
import yaml, pathlib

cfg_path = pathlib.Path('configs/default.yaml')
cfg = yaml.safe_load(cfg_path.read_text())
cfg['models']['reasoner']['backend'] = 'local'
cfg['models']['reasoner']['local_adapter'] = f'{CKPT_DIR}/final'
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(yaml.safe_dump(cfg['models']['reasoner'], sort_keys=False))
"""
            ),
            code(
                """
import os
os.environ['ANTHROPIC_API_KEY'] = ''  # paste yours, or drop --hosted below

# Measures: this adapter, the base model zero-shot, and the hosted model,
# on identical prompts and the same split.
!affa-eval finqa --hosted --limit 200 --output eval_results/finqa.json

import json
r = json.load(open('eval_results/finqa.json'))
print('adapter ', r['metrics'])
print('base    ', r['baseline_metrics'])
for note in r['notes']:
    print(' -', note)
"""
            ),
            hub_cell(
                "finqa-qlora",
                "QLoRA adapter for `Qwen2.5-3B-Instruct`, trained on FinQA multi-step "
                "numerical reasoning over financial tables.",
            ),
        ]
    )


BUILDERS = {
    "01_xbrl_tagger.ipynb": build_xbrl,
    "02_retrieval_embedder.ipynb": build_retrieval,
    "03_sentiment.ipynb": build_sentiment,
    "04_finqa_qlora.ipynb": build_finqa,
}


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in BUILDERS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(builder(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

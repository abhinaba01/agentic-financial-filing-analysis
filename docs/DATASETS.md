# Datasets: what they are, and what will bite you

Every dataset this project touches, why it was chosen, and the specific failure
mode that comes with it.

---

## The `datasets<4.0` pin

Three of the five datasets below are **loading-script datasets**:

- `nlpaueb/finer-139`
- `takala/financial_phrasebank`
- `ibm/finqa`

`datasets>=4.0` removed script execution entirely, so these do not load at all —
not "slower", not "with a warning". The `eval` and `train` extras pin
`datasets>=2.19,<4.0`, and `training/common.py::require_datasets_below_4()`
raises with instructions if a newer version is installed.

All three also need `trust_remote_code=True`.

### Do not substitute a parquet mirror

Third-party parquet mirrors of these datasets exist and will load under
`datasets>=4.0`. **At least one is deduplicated**, which changes the split sizes.
A model evaluated on a deduplicated test split is not comparable to the published
number, and the discrepancy looks like a modelling result rather than a data
difference. If you must use a mirror, verify the split cardinalities against the
numbers below first and say in the README which mirror you used.

---

## `nlpaueb/finer-139` — XBRL numeric tagging

| | |
|---|---|
| Task | token classification, 279 labels (139 XBRL concepts × B-/I-, plus `O`) |
| Splits | 900,384 train / 112,494 validation / 108,378 test |
| Used by | §5.1 tagger, `affa-eval xbrl` |

**Label alignment is the trap.** Only the first sub-word of each word carries the
tag; continuation sub-words and special tokens must get `-100`. Getting this
wrong trains happily, converges, and produces an F1 that measures a different
task. `training/train_xbrl_tagger.py::align_labels` does it correctly.

**Score with `seqeval`, at span level.** Token accuracy is meaningless: `O`
dominates so heavily that predicting it everywhere scores above 95%.

**Report the per-concept breakdown.** The tag distribution is severely skewed.
Micro-F1 hides that most of the 139 concepts are too rare to learn from a
subset, so a headline number near the paper's can sit on top of a model that
learned six concepts. `affa-eval xbrl` prints the breakdown and counts how many
concepts scored exactly zero.

**Published reference:** 89.2% micro-F1 for `sec-bert-base` on the full splits
(FiNER-139 paper, Loukas et al. 2022). That is not this repo's number, and if you
train on a subset it is not comparable to yours either.

---

## `virattt/financial-qa-10K` — retrieval training

| | |
|---|---|
| Task | question → context pairs over real 10-K filings |
| Used by | §5.2 embedder |

**Train on this, not on FiQA.** Evidence: fine-tuning `bge-large` on FiQA in a
prior project improved FiQA NDCG@10 by 2.3% and *cost* 11.6% Hit@1 on filing
retrieval. FiQA is retail-investor forum discussion; filings are SEC prose.

**Check FinanceBench overlap before training.** Training on a passage that later
appears as evaluation gold turns the benchmark into a memorisation test.
`training/train_retrieval.py::drop_financebench_overlap` does the check, drops
the overlap, and prints the count **even when it is zero** — a stated zero is
evidence the check ran, silence is not.

**Batch size is the in-batch negative count** for
`CachedMultipleNegativesRankingLoss`, so it matters more than epochs. GradCache
is what lets a 16GB T4 hold an effective batch of 64. Use
`BatchSamplers.NO_DUPLICATES`: two rows sharing a positive in one batch trains
the model against a true positive.

---

## `BeIR/fiqa` + `BeIR/fiqa-qrels` — retrieval evaluation

| | |
|---|---|
| Task | IR benchmark over financial forum posts |
| Used by | `affa-eval retrieval` |

Evaluated **alongside** FinanceBench, never instead of it. The point of training
in-domain is a trade-off, and reporting only the corpus that improved would hide
exactly the thing worth knowing.

Sampling the corpus (`--corpus-sample`) makes absolute numbers incomparable to
the BEIR leaderboard. The harness keeps every gold document when it samples —
dropping one would make the metric measure sampling luck — and prints a warning
next to any sampled figure.

---

## `PatronusAI/financebench` — retrieval evaluation on filings

| | |
|---|---|
| Task | questions with gold evidence passages from SEC filings |
| Used by | `affa-eval retrieval` |

This is the corpus the product actually serves. It is also the overlap source
for the training-set check above, which is why the two must not be run without
the check in between.

---

## `takala/financial_phrasebank` — sentiment

| | |
|---|---|
| Task | 3-class tone classification |
| Configs | `sentences_allagree` (cleanest) through `sentences_50agree` |
| Used by | §5.3 classifier, `affa-eval sentiment` |

**`ProsusAI/finbert` was trained on this dataset.** Fine-tuning it here and
reporting the score measures memorisation, not quality — the training script
refuses a finbert base unless you explicitly acknowledge it, and the result is
then labelled non-generalizing.

The fair comparison, which the harness implements: our model fine-tuned from a
base encoder on our own stratified split, versus finbert evaluated on **our
held-out split**. That is a real result either way, including the way where
finbert wins.

**The split must be stratified.** PhraseBank is heavily skewed toward neutral, so
a random split changes the class balance run to run and accuracy wanders for
reasons unrelated to the model.

**Check for duplicates across the split boundary.** Sentences repeat verbatim
between agreement configs; a duplicate straddling train and test leaks the
answer. `training/train_sentiment.py` checks and drops them, reporting the count.

**Known gap:** `yiyanghkust/finbert-tone` fails to load on current
`transformers` — its `config.json` has no `model_type`. A working tone model is a
genuine gap worth filling if you can source tone-labelled data.

---

## `ibm/finqa` — numerical reasoning

| | |
|---|---|
| Task | multi-step reasoning programs over financial tables |
| Used by | §5.4 QLoRA fine-tune, `affa-eval finqa` |

**Score executed answers, not program strings.** Many correct programs produce
one answer, so string matching understates every model unpredictably.

**Answers mix conventions** — the same value appears as `14.1%` and as `0.141`.
The harness accepts both. That is convention handling, not ground-truth
rescaling: nothing about the gold value is changed, the comparison just knows
both readings exist.

**Mask the prompt during training.** Loss is computed on the answer only, so the
model learns to answer rather than to reproduce the table it was given.

---

## `data/kpi_gold/` — the hand-labelled set

Not a public dataset. It is the only thing in this project that measures the
actual product, and it has to be built by hand — see
[`data/kpi_gold/README.md`](../data/kpi_gold/README.md) for the format and the
labelling rules.

Currently it holds **one synthetic document**. Section 9 asks for 10–20 real
filings. Until those exist, `affa-eval kpi` verifies the harness, not the
product, and the README says so.

---

## Overlap checking generally

Section 2's rule: any dataset used for training is checked for overlap against
evaluation data, and **the overlap count is reported even when it is zero**.

`affa.eval.metrics.overlap_count` does exact-match comparison after whitespace
and case normalisation. It will not catch paraphrase or near-duplicate overlap —
that is a real limitation, stated here rather than left implicit. If you need
near-duplicate detection, MinHash over shingles is the next step, and the count
it produces should be reported alongside the exact count rather than replacing
it.

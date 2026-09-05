---
license: apache-2.0
language: en
tags: [finance, sec-filings, sentiment, affa]
base_model: nlpaueb/sec-bert-base
datasets: [takala/financial_phrasebank]
---

# affa-sentiment

Fine-tuned for the [Agentic Financial Filing Analyst](https://github.com/abhinaba01/agentic-financial-filing-analysis).

## What it does

3-class tone classifier (negative / neutral / positive) for sentences drawn
from SEC filing prose — MD&A, risk factors, earnings narrative. Feeds the
`tone` factor in the recommendation rubric (`configs/rubric_v1.yaml`), which is
deliberately the lowest-weighted factor (0.10) since management tone is the
softest signal in the rubric.

## Training

| | |
|---|---|
| Base model | `nlpaueb/sec-bert-base` |
| Dataset | `takala/financial_phrasebank`, `sentences_allagree` config |
| Training examples | 2,264 sentences total, split 70/15/15 (train/validation/test), stratified by class |
| Epochs | 4 |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Max sequence length | 128 |
| Precision | fp16 |
| Hardware | single Colab T4 (16GB) |
| Seed | 42 |
| Checkpoint selected on | **validation** split |
| Test split touched | **once** |

## Results

Measured on the held-out **test** split (338 sentences, class counts
`{negative: 45, neutral: 208, positive: 85}`), against `ProsusAI/finbert`
measured on the **same** 338 sentences with the same protocol.

| Metric | affa-sentiment | ProsusAI/finbert | Delta |
|---|---:|---:|---:|
| Accuracy | 0.9911 | 0.9615 | +0.0296 |
| Macro-F1 | 0.9829 | 0.9514 | +0.0315 |

Confusion matrix (rows = true, columns = predicted; order
negative/neutral/positive):

```
affa-sentiment        ProsusAI/finbert
[[43,  0,  2],         [[44,  0,  1],
 [ 0, 208,  0],          [ 3, 199,  6],
 [ 1,  0, 84]]           [ 2,  1, 82]]
```

Three misclassifications total, all between negative and positive, none
touching the neutral class — expected on `sentences_allagree`, the subset where
100% of the original PhraseBank annotators agreed on the label, making it the
cleanest and easiest split of the dataset.

### Why beating finbert here is notable, not just expected

**`ProsusAI/finbert` was trained on Financial PhraseBank.** Its score on any
PhraseBank split — including this held-out one — is partly memorisation: it may
have seen these exact 338 sentences, with their labels, during its own
training. `affa-sentiment` has never seen them; this test split was carved out
of the data before training started and scored exactly once. Despite finbert's
structural advantage on this specific evaluation, `affa-sentiment` still scores
higher on both metrics. Treat finbert's number here as an optimistic ceiling
for what memorisation on this exact test set buys, not as a neutral baseline —
and the fact that a genuinely held-out fine-tune clears that ceiling is the
actual result.

This card follows the project's own rule: `ProsusAI/finbert` was deliberately
**not** used as the base model to fine-tune from, since doing so and reporting
the score would measure memorisation rather than quality. It is used only as
the (structurally advantaged) baseline.

## Limitations

- Trained and evaluated only on `sentences_allagree` (2,264 sentences), the
  smallest and cleanest PhraseBank configuration. Performance on
  `sentences_75agree` or noisier real-world filing text has not been measured.
- PhraseBank sentences are short, single-topic, and drawn from financial news —
  not full filing paragraphs. Performance on longer, multi-clause MD&A
  sentences with mixed sentiment has not been separately evaluated.
- The class distribution is skewed toward neutral (208/338 in the test split);
  macro-F1 is reported specifically because accuracy alone would understate
  errors on the minority classes.
- Not validated against the fine-tuned XBRL tagger, retrieval embedder, or
  FinQA reasoner in an end-to-end report yet — those three have not been
  fine-tuned as of this card being written.

## Not financial advice

Research and educational use only. This model classifies the tone of text. It
does not predict prices and must not be used as the basis for an investment
decision.

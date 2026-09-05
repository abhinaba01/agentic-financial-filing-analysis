# Model cards

One card per pushed model, carrying **the real evaluation numbers and the
training subset size**.

## Status: no models pushed yet

No fine-tune has been run, so there are no cards here yet — only the template
below. A card with aspirational numbers is worse than no card, so none is
written until there is a measurement to put in it.

The notebooks in [`notebooks/`](../../notebooks/) write a card into the
checkpoint directory at push time; fill in the numbers from the evaluation cell
before pushing, and copy the finished card here.

---

## Template

```markdown
---
license: apache-2.0
language: en
tags: [finance, sec-filings, affa]
base_model: <exact base checkpoint>
datasets: [<exact dataset id>]
---

# <model name>

Fine-tuned for the [Agentic Financial Filing Analyst](<repo url>).

## What it does

One paragraph. What input, what output, what it is for.

## Training

| | |
|---|---|
| Base model | `<exact checkpoint>` |
| Dataset | `<exact id>`, `<config>` |
| Training examples | **<actual number used>** (of <full split size>) |
| Epochs | |
| Learning rate | |
| Batch size | (effective, if gradient accumulation or GradCache is used) |
| Precision | fp16 / 4-bit NF4 |
| Hardware | single Colab T4 (16GB) |
| Seed | 42 |
| Checkpoint selected on | **validation** split |
| Test split touched | **once** |

## Results

Measured on the held-out **test** split, against a baseline measured on the same
data with the same protocol.

| Metric | This model | Baseline (`<what>`) | Delta |
|---|---:|---:|---:|
| | | | |

<If the subset was sampled: state the subset size and that absolute numbers are
not comparable to published figures - only the delta against the baseline row.>

<If this model did NOT beat its baseline: say so here, plainly, with the
numbers. A fine-tune that fails to beat its baseline stays documented.>

### Published figures for orientation

Not produced by this model, measured under different conditions:

| Source | Figure | Conditions |
|---|---|---|
| | | |

## Limitations

- What it was not trained on.
- Where it is known to fail.
- Whether the base model had prior exposure to the evaluation data, and what
  that means for the number above.

## Not financial advice

Research and educational use only. This model analyzes documents. It does not
predict prices and must not be used as the basis for an investment decision.
```

---

## Rules for filling this in

1. **Never paste a number from a paper into the results table.** Papers go in
   the "Published figures" section, labelled, with their conditions.
2. **State the training subset size**, not the full split size, whenever they
   differ. "89% F1" on 5% of the data is a different claim.
3. **Report the test split, not the validation split.** Validation selected the
   checkpoint; reporting it turns a selection statistic into a headline.
4. **Name the baseline and how it was measured.** "Beats the baseline" without
   saying which baseline, on what data, is not a result.
5. **If the base model saw the evaluation data**, say so at the top of
   Limitations, not the bottom.

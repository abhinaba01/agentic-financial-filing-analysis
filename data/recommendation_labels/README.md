# Recommendation labels

Filings labelled **by hand against the rubric**, used by
`affa-eval recommendation`.

## What this measures, and what it does not

It measures whether the rubric is **implemented as written** — whether the code
reaches the same verdict a person does when applying
[`configs/rubric_v1.yaml`](../../configs/rubric_v1.yaml) to the same filing.

It does **not** measure whether the rubric predicts anything about returns. No
such claim is made anywhere in this project, and none should be made from a high
agreement number here. There is no ground truth for "should I invest in this
company"; the rubric is an explicit, arguable opinion, and this harness only
checks that the implementation matches the stated opinion.

## Status: empty

Section 9 asks for **20–30 filings**. None are labelled yet, so
`affa-eval recommendation` exits with a message rather than producing a number.

## Format

One JSON file per filing:

```json
{
  "source_file": "data/filings/AAPL_10K_2023.pdf",
  "ticker": "AAPL",
  "fiscal_period": "FY2023",
  "assessment": "favorable",
  "labeller": "initials or name",
  "labelled_at": "2026-01-15",
  "notes": "Strong margins and cash generation; leverage elevated but well covered.",
  "factor_scores": {
    "profitability": 0.8,
    "growth": -0.3,
    "leverage": 0.5,
    "cash_generation": 0.7,
    "risk": -0.4,
    "tone": 0.1
  }
}
```

`assessment` must be one of `favorable`, `mixed`, `unfavorable`,
`insufficient_evidence`.

`factor_scores` are optional but valuable: they give factor-level agreement,
which localises a disagreement to the factor that caused it instead of leaving
you with one number and no idea where it went wrong. Only the **sign** is
compared (positive / neutral / negative), because exact score matching would
measure band boundaries rather than judgement.

## How to label

1. Read the filing. Apply the rubric in `configs/rubric_v1.yaml` yourself —
   the weights, the bands, the sufficiency rule.
2. **Do not run the system first.** Labelling after seeing the output measures
   your agreement with the system's output, not the system's agreement with a
   human, and the number it produces is worthless.
3. Include filings you expect to score `insufficient_evidence`. If that outcome
   never appears in the label set, the harness cannot tell whether the
   sufficiency gate works, and it prints a warning saying so.
4. Include at least a few `unfavorable` filings. A set of only healthy companies
   makes the always-"mixed" baseline look competitive and tells you nothing.
5. Record who labelled it and when. If two people label the same filing, keep
   both files and report inter-labeller agreement — it is the ceiling on any
   agreement number the system can meaningfully achieve.

## Running it

```bash
affa-eval recommendation --output eval_results/recommendation.json
```

The baseline is the majority class (always answer "mixed"). A system that cannot
beat that is not analysing anything.

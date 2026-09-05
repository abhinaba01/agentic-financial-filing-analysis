# Hand-labelled KPI set

This is the most valuable dataset in the project. It is the only thing that
measures the actual product — every other benchmark measures a component against
someone else's task. Nobody else has this set, and building it is manual work
that cannot be skipped or generated.

## What is here now

| File | Filing | Status |
|---|---|---|
| `demo_10k.gold.json` | `data/samples/demo_10k.json` | **Synthetic.** A fictional company, written for this repo. It exercises the harness end-to-end; it is not evidence about real filings. |

**The set contains no real filings yet.** Section 9 asks for 10–20 real ones.
Until those exist, any KPI number produced by `affa-eval kpi` measures the
pipeline against a document written to be easy for it, and the README's results
table must say so rather than quoting the figure.

## Format

One JSON file per filing:

```json
{
  "source_file": "data/filings/AAPL_10K_2023.pdf",
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "fiscal_period": "FY2023",
  "notes": "Labelled from the consolidated statements, pp. 31-36.",
  "metrics": {
    "revenue":            {"value": 383285000000, "unit": "USD", "page": 31},
    "gross_profit":       {"value": 169148000000, "unit": "USD", "page": 31},
    "net_income":         {"value":  96995000000, "unit": "USD", "page": 31},
    "eps_diluted":        {"value": 6.13,         "unit": "USD/share", "page": 31}
  }
}
```

Rules for labelling, each of which exists because breaking it corrupts the
measurement:

1. **Values are in absolute units.** Not the filing's reporting scale. A
   statement headed "in millions" showing `383,285` is labelled
   `383285000000`. The scale conversion is one of the things being tested, so
   the gold file must not encode the same assumption the extractor makes.
2. **Per-share amounts are never scaled.** `eps_diluted` is `6.13`, even when
   the statement header says "in millions, except per share data".
3. **Negatives keep their sign.** A `(10,959)` capital-expenditure line is
   labelled `-10959000000`.
4. **Label the consolidated total, not a segment.** If the filing reports both,
   the gold value is the consolidated one.
5. **Record the page.** It is how a disagreement gets adjudicated later.
6. **Never adjust a gold value because the extractor disagrees.** That is
   anti-pattern #12. If the extractor is off by exactly 1000x, the harness will
   report it as a `unit_error` — fix the converter.

Metric names must come from `affa.kpi.catalog.EXTRACTED_METRICS`; anything else
is silently ignored by the harness.

## Running it

```bash
affa-eval kpi --output eval_results/kpi.json
```

The harness reports `value_accuracy`, `extraction_recall` and `unit_error_rate`
separately, against the rule-based extractor as the measured baseline.

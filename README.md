# Agentic Financial Filing Analyst

An agentic RAG system that ingests SEC filings (10-K/10-Q as PDF, HTML, text or
JSON), extracts financial KPIs with provenance, reasons over retrieved evidence,
**verifies every claim against the passages it cites**, and emits a structured
investment assessment in which every statement is traceable to a specific
passage in the source document.

> **Research and educational use only. Not investment advice.** The system
> analyzes documents. It does not predict prices, and it is not a return
> predictor — see [The recommendation](#the-recommendation-is-a-rubric-not-a-prediction).

---

## Status, honestly

**One of the four models has been fine-tuned so far: sentiment.** The pipeline
runs end to end today using the stock `bge-base-en-v1.5` embedder, a rule-based
KPI extractor, and the fine-tuned sentiment classifier below. XBRL tagging and
numerical reasoning are still on their rule-based / stub fallbacks. The four
fine-tunes described in [Models](#models-to-fine-tune) have training scripts,
Colab notebooks and evaluation harnesses; three of the four have not been run
yet, so most of the results table below still says so rather than showing a
number.

That is deliberate. This project's whole premise is that a number without a
measured baseline is not a result, so the alternative — filling the table with
plausible figures before a fine-tune actually runs — would defeat the point. See
[Results](#results) and [What is not done](#what-is-not-done).

---

## Architecture

```
                    ingest (parse → clean → chunk → embed)
                                   ↓
                                 START
          ┌───────────────┬────────┴────────┬──────────────────┐
          ↓               ↓                 ↓                  ↓
     XBRL/KPI        Sentiment         Risk-factor        Doc metadata
     extraction      analysis          extraction         (period, ticker)
          └───────────────┴────────┬────────┴──────────────────┘
                                   ↓  (fan-in)
                              retrieve  ←──────┐
                                   ↓           │ bounded re-query
                          sufficient? ─────────┘   (≤ 3 attempts)
                                   ↓
                               reason  (evidence → findings, with citations)
                                   ↓
                               verify  (critic: every claim supported?)
                                   ↓
                            recommend  (rubric over verified findings)
                                   ↓
                             synthesize → structured report
```

A LangGraph `StateGraph` over a shared typed state
([`src/affa/agent/graph.py`](src/affa/agent/graph.py)). Four design rules are
enforced in code rather than described in prose:

| Rule | Where it is enforced |
|---|---|
| Concurrent branches return only their own keys, never the whole state | [`state.py:BRANCH_OUTPUT_KEYS`](src/affa/agent/state.py), asserted per-branch in [`test_graph_nodes.py`](tests/test_graph_nodes.py) |
| Retrieval and generation are separate nodes; the retry loop wraps retrieval alone | [`graph.py`](src/affa/agent/graph.py); a call-counting test proves a retry costs no generation |
| Every retry actually changes the query | [`reformulate.py`](src/affa/agent/reformulate.py) — three strategies, and the result is checked against every query already tried |
| Routing thresholds must be reachable | [`routing.py`](src/affa/agent/routing.py) asserts `retry_below > min_similarity` **at import time** |

The retry budget lives in exactly one place — the routing edge. Nodes count
attempts; they never decide to stop.

### The `verify` node

This is the differentiator. Every generated claim is re-checked against the
passages it cites and marked `supported` / `unsupported` / `contradicted`.
Unsupported claims are dropped or flagged, never silently emitted.

A claim is **supported** when every number in it appears in one of its cited
chunks — allowing a reporting-scale factor and a stated tolerance — or is
directly computable from numbers that do, **and** the claim's financial subjects
are discussed in the cited text. It is **contradicted** when a figure fails to
match but the cited passage states a different value on the same labelled line;
that is a stronger signal than mere absence, so it is reported separately and
kept in the report rather than dropped.

Subject matching goes through the metric catalogue, so a passage saying "Total
net sales" and a claim saying "Revenue" are recognised as the same subject.
Comparing raw words would reject correct, well-cited claims for using the
extractor's normalised vocabulary.

---

## Quick start

```bash
pip install -e ".[dev]"
```

```bash
python -m affa.cli data/samples/demo_10k.json --format markdown --in-memory
```

The bundled sample is **synthetic** — Northwind Systems is a fictional company
written for this repo. It exercises the pipeline; it is not evidence about real
filings.

For PDF/HTML parsing, the vector store and the agent graph:

```bash
pip install -e ".[ingest,agent]"
```

Full walkthrough from a clean checkout: [RUNNING.md](RUNNING.md).

### CLI

```
affa-analyze FILING [options]
```

| Flag | Meaning |
|---|---|
| `--config PATH` | Config YAML (default `configs/default.yaml`) |
| `--output, -o PATH` | Write the report here (default stdout) |
| `--format {json,markdown,html}` | Output format (default `json`) |
| `--question TEXT` | Analysis question driving retrieval |
| `--ticker TEXT` | Override the ticker sniffed from the filing |
| `--company TEXT` | Override the company name |
| `--fiscal-period TEXT` | Override the fiscal period, e.g. `FY2024` |
| `--market-price FLOAT` | Share price, for P/E. A filing contains none, so it is an explicit input |
| `--in-memory` | Use an in-memory vector store instead of persistent Chroma |
| `--verbose, -v` | Debug logging |

[`tests/test_docs_contract.py`](tests/test_docs_contract.py) asserts this table
and `argparse` agree in both directions, so a documented flag that does not exist
fails the suite.

### API and UI

```bash
uvicorn affa.api.main:app --reload      # POST /analyze, GET /health, GET /config
streamlit run src/affa/ui/app.py        # filing on the left, cited evidence on the right
```

---

## Sample output

Real output from the bundled synthetic filing, generated by the command above.
Full versions: [`docs/sample_output/`](docs/sample_output/).

```
**Favorable** (confidence 0.83, rubric v1.0)
Aggregate score `+0.621` over 6 factors covering 100% of rubric weight.

| Factor          | Score | Rationale                                                     |
|-----------------|------:|---------------------------------------------------------------|
| profitability   | +0.77 | Net margin: strong, Gross margin: software-like [p.31]        |
| growth          | +0.75 | Revenue yoy: growing, Net income yoy: strong growth [p.31]    |
| leverage        | +0.90 | Debt to equity: conservative, Current ratio: comfortable [p.33]|
| cash generation | +0.70 | Free cash flow: cash generative, OCF/NI: cash-backed [p.36]   |
| risk            | -0.60 | Risk severity index: elevated risks                            |
| tone            | +0.60 | Sentiment score: positive tone                                 |
```

Every derived metric ships with the arithmetic behind it:

```json
{ "name": "gross_margin_pct", "value": 65.8036,
  "formula": "gross_profit / revenue * 100",
  "operands": { "gross_profit": 3166800000.0, "revenue": 4812600000.0 } }
```

and every claim carries its verification verdict:

```
[supported] Gross profit is reported as 3,166,800,000 USD.  (chunk: …:0003:…)
[supported] Ebitda computes to 1,177,200,000.00 (operating_income + depreciation_amortization).
```

### `insufficient_evidence` is a real outcome

Running the pipeline on a filing with prose but no financial statements:

```
**Insufficient Evidence** (confidence 0.35, rubric v1.0)
Factors not scored: profitability, growth, leverage, cash_generation.
```

No aggregate score is published — the schema
[refuses to serialise one](src/affa/schema.py) in that case, because a score
implies a verdict the evidence does not support. This path is covered by
[`test_rubric.py`](tests/test_rubric.py).

---

## The recommendation is a rubric, not a prediction

There is no ground-truth dataset for "should I invest in this company".

**This system does not predict stock returns.** Labelling filings with subsequent
price movement and training on it produces a return predictor that will not
generalize, and claiming otherwise is not a claim this project makes.

What it does instead: a deterministic, versioned, adjustable aggregation over
signals it has actually extracted and verified. The weights and thresholds live
in [`configs/rubric_v1.yaml`](configs/rubric_v1.yaml) where you can read them and
disagree with them.

| Factor | Weight | Required inputs |
|---|---:|---|
| Profitability | 0.22 | net margin |
| Growth | 0.20 | revenue YoY |
| Leverage & liquidity | 0.18 | debt-to-equity |
| Cash generation | 0.20 | free cash flow |
| Risk-factor severity | 0.10 | risk severity index |
| Management tone | 0.10 | sentiment score |

The output is graded — `favorable | mixed | unfavorable | insufficient_evidence`
— with a confidence score and the per-factor scores that produced it. Confidence
reflects **evidence coverage and agreement**, never how strong the verdict is; a
confident "mixed" is a normal outcome.

**The LLM writes the narrative explanation. It does not decide the verdict.**
That keeps the recommendation reproducible and auditable, and keeps the model in
the job it is good at.

---

## Models to fine-tune

All four are written to train on a **single Colab T4 (16GB, fp16, ~12h)**.
Notebooks: [`notebooks/`](notebooks/). Scripts: [`training/`](training/).

| # | Model | Base | Data | Technique |
|---|---|---|---|---|
| 5.1 | XBRL numeric tagger | `nlpaueb/sec-bert-base` | `nlpaueb/finer-139` | token classification, 279 labels |
| 5.2 | Retrieval embedder | `BAAI/bge-base-en-v1.5` | `virattt/financial-qa-10K` | `CachedMultipleNegativesRankingLoss` (GradCache) |
| 5.3 | Sentiment / tone | `nlpaueb/sec-bert-base` | `takala/financial_phrasebank` | 3-class classification |
| 5.4 | Numerical reasoning | `Qwen2.5-3B-Instruct` | `ibm/finqa` | QLoRA, 4-bit NF4, r=16 |

Three choices worth stating up front, because each is a decision rather than a
default:

- **Train retrieval on 10-K QA, not FiQA.** Evidence-based: in a prior project,
  fine-tuning `bge-large` on FiQA improved FiQA NDCG@10 by 2.3% and *cost* 11.6%
  Hit@1 on filing retrieval. FiQA is retail-investor forum discussion; filings
  are SEC prose. We train in-domain and evaluate on both.
- **Do not fine-tune `ProsusAI/finbert` on Financial PhraseBank.** It was already
  trained on that dataset, so the score would be train-set scoring. The training
  script refuses it. `finbert` is instead used as the *baseline*, evaluated on
  our held-out split.
- **The BGE query-instruction prefix is used in neither training nor
  inference.** A mismatch between the two is worse than skipping it, so the
  choice is made once and mirrored in `configs/default.yaml`.

### Checkpointing and resume

Colab runtimes disconnect. Every training run is resumable, and that is a
requirement of the notebooks rather than a nice-to-have —
[`training/common.py`](training/common.py):

- checkpoints must live on Drive or the Hub; an ephemeral `output_dir` is
  **refused**, because that is exactly the failure checkpointing defends against;
- the resume cell is idempotent — re-run it after a crash and it resumes with no
  code edit;
- the run config (seed, subset sizes, base model, hyperparameters) is written
  into the checkpoint directory, and resume **refuses to continue** if any of it
  changed: the global step indexes into a specific data order, so a changed seed
  or subset size makes the resumed run silently meaningless;
- `save_total_limit=2` is required, not tidiness — a full checkpoint is 3–4×
  model size and Drive's free tier is 15GB;
- checkpoint selection uses **validation**; the test split is touched exactly
  once, and [a guard](training/common.py) refuses a config that reports on the
  split that selected the checkpoint.

These guards are covered by
[`tests/test_training_common.py`](tests/test_training_common.py) — untested
resume logic is usually broken resume logic.

---

## Results

### Measured in this repo

| Component | Metric | This repo | Baseline (same data, same protocol) | Status |
|---|---|---|---|---|
| XBRL tagger | seqeval micro-F1 | — | — | **not run** — no fine-tune yet |
| Retrieval (FiQA) | NDCG@10 | — | — | **not run** |
| Retrieval (FinanceBench) | Hit@1 | — | — | **not run** |
| Sentiment | accuracy / macro-F1 | **0.9911 / 0.9829** | 0.9615 / 0.9514 (`ProsusAI/finbert`, same held-out split) | **run** — see caveat and [model card](docs/model_cards/sentiment.md) |
| KPI extraction | value accuracy | 1.000 (19/19) | 1.000, rule-based | **synthetic filing only** — see caveat |
| Numerical reasoning | execution accuracy | — | — | **not run** |
| RAG faithfulness | claim-support precision | — | — | **needs a real labelled set** |
| Recommendation | rubric agreement | — | — | **needs labelled filings** |

**The sentiment row is a real result, with a caveat on the baseline.**
`nlpaueb/sec-bert-base` was fine-tuned from scratch on a stratified 70/15/15
split of `financial_phrasebank` (`sentences_allagree`, 2,264 sentences, seed 42)
and scored once on the 338-sentence held-out test split — it never saw that
split during training. `ProsusAI/finbert` is evaluated on the same 338
sentences as the baseline, but finbert was *trained* on Financial PhraseBank, so
its score there is partly memorisation — treat it as an optimistic ceiling, not
a neutral baseline (`affa-eval sentiment` prints this same caveat every run). Our
model beating that ceiling (+0.030 accuracy, +0.032 macro-F1) is the notable
part: it means a from-scratch fine-tune on a disjoint split outperforms a model
with a memorisation advantage on the exact sentences being scored.

**The KPI number is not a result.** It is measured on
`data/kpi_gold/demo_10k.gold.json`, a filing this repo wrote itself, with the
XBRL tagger disabled — so the "model" and the "baseline" are the same rule-based
extractor, and 19/19 says the harness works, not that the product does. Section 9
asks for 10–20 **real** filings; see [`data/kpi_gold/README.md`](data/kpi_gold/README.md).

Every harness refuses to emit a metric without a baseline measured the same way
— `EvaluationResult` raises unless you supply one or explicitly state why none
exists ([`harness.py`](src/affa/eval/harness.py)).

### Published figures — not produced by this repo

Listed separately, under different conditions, for orientation only. **These are
not this repo's results and must never be read as such.**

| Source | Figure | Conditions |
|---|---|---|
| FiNER-139 paper (Loukas et al., 2022), `sec-bert-base` row | 89.2% micro-F1 | full 900k/112k/108k splits |
| BEIR leaderboard, `bge-base-en-v1.5` | 0.406 NDCG@10 on FiQA | full 57k-document corpus |

### Negative results

None recorded yet, because no fine-tune has been run. When one fails to beat its
baseline it stays here with the measurement rather than being deleted.

---

## Evaluation

One CLI, one interface per component:

```bash
affa-eval <component> [--test-set ...] [--output ...] [--limit N] [--run-agent] [--baseline NAME]
```

| Component | Dataset | Metrics | Baseline |
|---|---|---|---|
| `xbrl` | `nlpaueb/finer-139` test | seqeval micro-F1, per-concept P/R/F1 | base encoder, untrained head |
| `retrieval` | `BeIR/fiqa`, `PatronusAI/financebench` | NDCG@10, Hit@1/5, MRR | stock `bge-base`, same corpus + seed |
| `sentiment` | `financial_phrasebank` held-out | accuracy, macro-F1, confusion matrix | `ProsusAI/finbert` on the same split |
| `kpi` | hand-labelled filings | value accuracy, extraction recall, **unit-error rate** | rule-based extractor |
| `finqa` | `ibm/finqa` test | execution accuracy, per-operator | base zero-shot **and** hosted model |
| `faithfulness` | your labelled set | citation coverage, claim-support precision, hallucination rate | ungrounded generation |
| `recommendation` | 20–30 labelled filings | rubric agreement, `insufficient_evidence` rate | always-"mixed" majority class |

Notes that matter:

- Retrieval evaluation makes **no LLM calls**, so before/after comparisons are
  free. Run them often.
- `--limit` records the subset size and the renderer prints a warning next to any
  sampled number: sampling makes absolute figures incomparable to published
  ones, and only your own before/after delta stays valid.
- Unit errors are reported **separately** from wrong values. A 1000× scale
  mismatch and a wrong answer need different fixes, and folding them together
  hides which one you have. Ground truth is never rescaled to improve a metric.
- `affa-eval faithfulness --hand-check N` samples claims to a JSON file for
  manual review, because an unvalidated automated faithfulness metric is not
  evidence.

---

## Configuration

[`configs/default.yaml`](configs/default.yaml) is **live** — every key in it is
read by running code, and
[`test_schema_and_config.py`](tests/test_schema_and_config.py) walks the file to
prove it. A config file the docs treat as authoritative that nothing loads is a
trap this repo does not set.

Two thresholds are related by an invariant:

```yaml
retrieval:
  min_similarity: 0.25              # f - chunks below this are discarded
routing:
  retry_below_mean_similarity: 0.45 # t - must be > f, or the retry can never fire
```

Retrieval discards everything below `f`, so the mean similarity of what survives
is always ≥ `f`. A retry rule at or below `f` is dead code that still looks live
in the diagram. `affa.agent.routing` asserts `t > f` **at import**.

---

## Testing

```bash
pytest
```

232 tests, under 2 minutes, no network and no GPU. Coverage follows the spec's list:
financial-notation cleaning, unit/scale conversion, negative-number parsing,
chunker termination, threshold reachability, delta-returning graph nodes, schema
validation, and the API contract with models mocked.

Every bug found during development has a named regression test. Eight of them
came from this build — the last four surfaced only when a real training run
was attempted on real hardware (Windows locally, then a T4 in Colab), not under
pytest:

| Bug | Test |
|---|---|
| `"(In millions, except per share data)"` scaled EPS, so `$3.64` became `$3.64M` and every P/E collapsed to zero | `test_eps_is_not_scaled_by_the_statement_header` |
| Percent-convention inference ran over YoY values the pipeline had already computed in points, turning a true −0.96% into −96% | `test_yoy_is_computed_in_points_and_not_reinterpreted` |
| `(see Note 3) 1,234` parsed the footnote marker as a parenthesised negative, yielding −3 | `test_footnote_marker_is_not_read_as_the_figure` |
| `4,812.6 / 962.5 × 100 = 500.01` let an unconstrained ratio "ground" a fabricated claim | `test_contradiction_is_distinguished_from_absence` |
| All four training scripts died at import when run as `python training/train_x.py` — only pytest's injected `pythonpath` made them work | `test_script_imports_when_run_directly` |
| On Windows, `pyarrow` (pulled in by `datasets`) loading before `torch` broke `torch`'s DLL init (`c10.dll`) — reproduced running the XBRL tagger script for real | `test_common_imports_torch_before_anything_else_can` |
| A missing `accelerate` surfaced as a `transformers` internals traceback four frames deep instead of a one-line fix | `test_require_accelerate_message_is_actionable_when_missing` |
| The GPU-check cell called `.total_mem`, which does not exist on `torch`'s device-properties object (the real name is `.total_memory`) — CPU-only CI cannot execute this cell, so it only failed on an actual T4 in Colab | `test_gpu_check_cell_uses_the_real_torch_attribute` |

---

## What is not done

Stated plainly, because a roadmap that only lists finished work is not a roadmap.

- **Three of four models have not been fine-tuned.** All four training scripts
  and notebooks exist and the guards are tested; sentiment has been run on a T4
  and evaluated (see [Results](#results)), XBRL, retrieval and FinQA have not.
- **The fine-tuned sentiment model is not wired into the default config or
  pushed to the Hub yet.** The checkpoint exists on Drive and is evaluated, but
  `configs/default.yaml` still ships `models.sentiment.enabled: false` because
  no Hub id exists to point it at. Analysis reports still say
  `lexicon_fallback` until that happens.
- **No real filings are labelled.** `data/kpi_gold/` contains one synthetic
  document. The hand-labelled set over 10–20 real filings — the most valuable
  dataset in the project, and the only thing that measures the actual product —
  has not been built.
- **No recommendation labels.** `affa-eval recommendation` needs 20–30 filings
  labelled by hand against the rubric; none exist yet.
- **Faithfulness is automated but not human-validated.** The metric is
  implemented and the hand-check sampler exists; nobody has filled in the
  human verdicts, so agreement between the automated metric and a human is
  unknown.
- **The XBRL tagger is disabled by default** (`enabled: false`), so KPI
  extraction is rule-based until it is fine-tuned. Labelled as such in every
  report it touches.
- **Concurrency is not measured.** The four branches fan out in the graph, but
  nobody has verified that this is faster than running them sequentially — CPU
  thread oversubscription can make parallel slower, and GPU branches share one
  device. The claim is not made.
- **Table extraction is shallow.** `pdfplumber`'s default strategy handles ruled
  tables; borderless financial statements in some filings will need tuning.

---

## Repository layout

```
src/affa/
  config.py          live YAML config + threshold invariant
  schema.py          Pydantic report schema (section 8)
  pipeline.py        filing in → validated report out
  cli.py             affa-analyze
  ingestion/         parse · clean · chunk · embed
  kpi/               catalog · units · rules · xbrl · derive · extract
  agent/             state · routing · reformulate · nodes · verify · graph
  recommend/         versioned rubric
  report/            markdown + HTML rendering
  llm/               one interface, three backends (stub · local · hosted)
  eval/              seven harnesses, one CLI
  api/  ui/          FastAPI service, Streamlit UI
training/            four fine-tunes + shared checkpoint/resume
notebooks/           one Colab notebook per fine-tune (generated by scripts/)
configs/             default.yaml, rubric_v1.yaml
data/                samples, kpi_gold (hand-labelled set)
tests/               148 tests, no network, no GPU
```

## License

Apache-2.0. See [LICENSE](LICENSE).

**Research and educational use only. Not investment advice.**

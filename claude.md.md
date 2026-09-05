# Build Prompt: Agentic Financial Filing Analyst

Paste this whole document as the opening prompt for a new repository. It is written to be acted on directly, and the constraints in it are not decorative — several encode bugs that were found by measurement in a previous project, and reproducing them would silently invalidate the results.

---

## 1. What to build

An agentic RAG system that ingests a company's financial filings (10-K/10-Q PDF, HTML, or text), extracts key financial metrics, and produces a **structured investment assessment report** in which every claim is traceable to a specific passage in the source document.

The system must:

1. Parse, clean, chunk, and embed uploaded filings into a vector store
2. Extract KPIs — revenue, net income/loss, gross profit, operating income, EBITDA, EPS, P/E ratio, margins, debt-to-equity, free cash flow, and YoY changes for each
3. Retrieve evidence and reason over it with a multi-agent graph
4. Emit a **recommendation** (see §7 — read the constraint there before designing this) with explicit, cited reasoning
5. Output one structured JSON report containing metrics, evidence, reasoning, risks, and the recommendation
6. Evaluate every component against real benchmarks with published or measured baselines

Four models are fine-tuned as part of the project (§5). Everything must train and run on a **single Google Colab T4 (16GB)**.

---

## 2. Hard constraints

**Hardware.** All training must fit a T4: 16GB VRAM, fp16, roughly 12h sessions. Where a model does not fit natively, use the technique that makes it fit (GradCache for contrastive training, QLoRA 4-bit for the LLM) and say so in the docs. Do not specify an A100-only recipe.

**Honesty of measurement.** These are non-negotiable and are the point of the project:

- No metric is ever reported without a baseline measured on the same data with the same protocol.
- No number from a paper is presented as a number from this repo. Keep them in visibly separate sections.
- Checkpoint selection uses a **validation** split. The test split is touched exactly once, at the end. Never set `load_best_model_at_end` against the split you intend to report.
- Any dataset used for training is checked for overlap against evaluation data, and the overlap count is reported even when it is zero.
- If a model is fine-tuned on a dataset it was already trained on, that is stated loudly and the number is labelled as non-generalizing. (`ProsusAI/finbert` was trained on Financial PhraseBank — scoring it there measures nothing.)
- Negative results are recorded, not deleted. A fine-tune that fails to beat its baseline stays in the README with the measurement.

**Not financial advice.** The report is a research/education artifact. Every output carries a disclaimer. The system analyzes documents; it does not predict prices (see §7).

---

## 3. Architecture

A LangGraph `StateGraph` over a shared typed state. Required structure:

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
                          sufficient? ─────────┘
                                   ↓
                               reason  (evidence → findings, with citations)
                                   ↓
                               verify  (critic: every claim supported?)
                                   ↓
                            recommend  (rubric over verified findings)
                                   ↓
                             synthesize → structured report
```

Design rules, each of which exists because violating it caused a real bug:

- **Concurrent branches return only their own state keys**, never the whole state. Returning the full state from parallel nodes makes every branch write every key in one superstep, which LangGraph rejects. Branches must also treat the incoming state as read-only.
- **Retrieval and generation are separate nodes.** The re-query loop wraps retrieval alone, so a retry re-searches with a reformulated query and generation runs once, after the loop settles. Do not put an LLM call inside the retry loop.
- **Every retry attempt must actually change the query.** A loop that re-runs an identical search is a no-op that burns its budget. Reformulate per attempt, and vary the reformulation by attempt number.
- **Routing thresholds must be reachable.** If retrieval discards chunks below similarity `f`, then a routing rule of "retry when mean similarity < `t`" can never fire unless `t > f`. Assert this relationship in code at import time.
- **One retry mechanism, not two.** Retry policy lives in the routing edge only. Nodes may count attempts; they may not decide to stop.
- **The `verify` node is mandatory and is the project's differentiator.** It re-checks each generated claim against the retrieved passages and marks it supported / unsupported / contradicted. Unsupported claims are dropped or flagged in the report, never silently emitted. This is what makes "why did it say that" auditable, and it produces the faithfulness metric in §9.

---

## 4. Ingestion pipeline

- **Parse** — PDF via `pdfplumber` (text *and* tables; financial statements are tables), HTML via BeautifulSoup, plus plain text/JSON. Preserve page numbers and table structure; the report cites them.
- **Clean** — normalize whitespace, smart quotes, ligatures, and hyphenation. Do not destroy financial notation: `$1,234.5`, `(1,234)` for negatives, `1.2x`, `45%`, `FY2023`. Test each of these explicitly — naive regex cleaning corrupts them.
- **Chunk** — token-bounded (~512 tokens, ~64 overlap) with sentence-aware boundaries via spaCy. Keep tables intact as single chunks where possible. **Guard the sliding window against non-advancing offsets**, which is an easy infinite loop on documents longer than one chunk.
- **Embed** — the fine-tuned retrieval model from §5.2, into ChromaDB with persistent storage. Store `doc_id`, `ticker`, `fiscal_period`, `page_number`, `chunk_type` as metadata for filtered retrieval.

Vectors written by one embedding model are meaningless when queried by another, and the vector store will return confident nonsense rather than erroring. Namespace collections by model, and re-index when the model changes.

---

## 5. Models to fine-tune

Four, in this order. The first three are core; the fourth is the stretch goal.

### 5.1 XBRL numeric tagger (drives KPI extraction)

| | |
|---|---|
| **Base** | `nlpaueb/sec-bert-base` |
| **Dataset** | `nlpaueb/finer-139` |
| **Task** | Token classification, 279 labels (139 XBRL concepts × B-/I-, plus `O`) |
| **Splits** | 900,384 train / 112,494 validation / 108,378 test |
| **Baseline** | 89.2% micro-F1 (FiNER-139 paper, `sec-bert-base`, full split) |
| **T4 config** | batch 32, max_len 256, fp16, 2 epochs, lr 3e-5 |

This replaces regex KPI extraction with a model that decides whether a given number *is* `Revenues` or `NetIncomeLoss`. Use plain `sec-bert-base`, not `sec-bert-num`/`sec-bert-shape`, so the comparison to the paper's row is clean.

Gotchas:
- `finer-139` is a **loading-script dataset** — pin `datasets>=2.19,<4.0` and pass `trust_remote_code=True`. `datasets>=4.0` removed script execution entirely. Parquet mirrors exist but are third-party and at least one is deduplicated, which changes the splits and breaks comparability.
- Label alignment: the first subword of a word carries the tag, continuation subwords and specials get `-100`. Getting this wrong trains happily and produces meaningless F1.
- Score with `seqeval` (span-level). Token accuracy is meaningless here — `O` dominates.
- Report the **per-concept** breakdown. The tag distribution is severely skewed; micro-F1 hides that most of the 139 concepts are too rare to learn from a subset.

### 5.2 Retrieval embedder

| | |
|---|---|
| **Base** | `BAAI/bge-base-en-v1.5` (or `bge-large-en-v1.5` if VRAM allows) |
| **Train data** | `virattt/financial-qa-10K` — question→context pairs over 10-K filings |
| **Benchmarks** | `BeIR/fiqa` + `BeIR/fiqa-qrels`; `PatronusAI/financebench` |
| **Loss** | `CachedMultipleNegativesRankingLoss` (GradCache) |
| **T4 config** | effective batch 64, `mini_batch_size` 8–16, 1 epoch, lr 2e-5 |

**Train on 10-K QA, not FiQA.** This is the single most important choice here, and it is evidence-based: in a prior project, fine-tuning `bge-large` on FiQA improved FiQA NDCG@10 by 2.3% and *cost* 11.6% Hit@1 on filing retrieval. FiQA is retail-investor forum discussion; filings are SEC prose. Train in-domain, and evaluate on both to demonstrate you understand the difference.

Gotchas:
- Batch size **is** the in-batch negative count for this loss — it matters more than epochs. GradCache is what lets a T4 hold 64.
- Use `BatchSamplers.NO_DUPLICATES`; two rows sharing a positive in one batch trains the model against a true positive.
- BGE recommends a query instruction prefix. Either use it in both training and inference, or neither — a mismatch is worse than skipping it. Document the choice.
- Check `virattt/financial-qa-10K` passages against FinanceBench gold passages and drop overlaps before training. Report the count.

### 5.3 Financial sentiment / tone classifier

| | |
|---|---|
| **Base** | `nlpaueb/sec-bert-base` or `distilroberta-base` — **not** `ProsusAI/finbert` |
| **Dataset** | `takala/financial_phrasebank`, config `sentences_allagree` (also try `sentences_75agree`) |
| **Task** | 3-class sequence classification (positive / neutral / negative) |
| **T4 config** | batch 32, max_len 128, fp16, 3–4 epochs, lr 2e-5 |

Fine-tune from a **base** model with your own stratified train/val/test split. Do not fine-tune `finbert` here: it was already trained on this dataset, so any number you get is train-set scoring and proves nothing. Report your model against `finbert` evaluated on *your held-out split* as the baseline — that is a fair comparison and gives you a real result either way.

Gotcha: `financial_phrasebank` is also a loading-script dataset — same `datasets<4.0` + `trust_remote_code=True` constraint.

Optional extension: `yiyanghkust/finbert-tone` fails to load on current `transformers` (its `config.json` has no `model_type`), so a working tone model is a genuine gap worth filling if you can source tone-labelled data.

### 5.4 Numerical-reasoning LLM (stretch)

| | |
|---|---|
| **Base** | `Qwen2.5-3B-Instruct` (7B only if 3B is comfortable first) |
| **Dataset** | `ibm/finqa` (`trust_remote_code=True`) |
| **Method** | QLoRA, 4-bit NF4, LoRA r=16, α=32, target attention + MLP projections |
| **T4 config** | batch 1, grad-accum 8–16, max_len 1024, gradient checkpointing, paged AdamW |
| **Baseline** | the same base model zero-shot, **and** a hosted model (e.g. `gpt-4o`) on the same split |

FinQA supplies multi-step reasoning programs over financial tables — the right supervision for the derived-KPI and reasoning steps. The genuinely interesting result is the three-way comparison: base zero-shot vs. your QLoRA model vs. a frontier hosted model, on identical prompts. "Reached X% of gpt-4o's execution accuracy at zero marginal cost" is a strong, honest claim.

The architecture must support both a local fine-tuned model and a hosted API model behind one interface, selectable by config, so the comparison is a flag rather than a rewrite.

### 5.5 Checkpointing and resume — applies to all four

Colab runtimes disconnect, get recycled, and hit idle timeouts. Every training run must be resumable, and this is a requirement of the notebooks, not a nice-to-have.

**Checkpoints must survive the runtime dying.** `/content` and any local `output_dir` are on ephemeral disk — they vanish with the VM, which is precisely the failure being defended against. Use one of:

- **Google Drive**, mounted at the top of the notebook, with `output_dir` pointing inside it
- **HuggingFace Hub**, via `push_to_hub=True` with `hub_strategy="checkpoint"` and a private repo — durable, and it doesn't consume Drive quota

All four models train through HF `Trainer` (`SentenceTransformerTrainer` subclasses it), so the same configuration applies to every one:

```python
args = TrainingArguments(
    output_dir=CKPT_DIR,  # on Drive, or paired with push_to_hub
    save_strategy="steps",
    save_steps=SAVE_STEPS,  # ~15-20 min of training, not per epoch
    save_total_limit=2,  # mandatory - see disk note below
    eval_strategy="steps",  # must match save_strategy when
    eval_steps=SAVE_STEPS,  #   load_best_model_at_end=True, or Trainer errors
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=SEED,
)
```

Epoch-level saving is not enough. An epoch here is 40+ minutes; a disconnect at minute 39 loses all of it.

**The resume cell must be idempotent** — re-running it after a crash resumes automatically, with no code edit:

```python
import os
from transformers.trainer_utils import get_last_checkpoint

last = get_last_checkpoint(CKPT_DIR) if os.path.isdir(CKPT_DIR) else None
if last:
    print(f"resuming from {last}")
trainer.train(resume_from_checkpoint=last)
```

**What resume restores, and why it matters:** model weights, optimizer moments, LR-scheduler position, RNG state, global step, and dataloader position. This is why you resume rather than "just train again from the saved weights" — restarting the optimizer and LR schedule from scratch is a different run, and its loss curve will not join up with the first half.

**Determinism is a precondition.** Resume is only meaningful if the run is reproducible:

- Fix `SEED` and use it for `TrainingArguments(seed=...)` and every `.shuffle(seed=SEED)`
- Keep subset selection (`TRAIN_SAMPLES`, `EVAL_SAMPLES`) identical across a resume — changing it after a crash means the global step now points into different data, and the resumed run is silently invalid
- Write the run config (seed, subset sizes, base model, hyperparameters) into the checkpoint directory as JSON, and have the resume path refuse to continue if it doesn't match the current cell's settings

**Disk and I/O:**

- A full checkpoint is roughly 3–4× model size — fp32 weights plus two AdamW moments. Around 1.5GB for a 110M encoder, 4–5GB for `bge-large`. Drive's free tier is 15GB, so `save_total_limit=2` is required, not tidiness.
- Drive writes are slow. Saving every few hundred steps can spend more wall clock on checkpoints than on training. Time one save, then set `save_steps` so checkpointing costs under ~5% of runtime.
- **QLoRA is the cheap case**: only adapter weights are saved, ~50–100MB, so the 3B model in §5.4 can checkpoint frequently at negligible cost.

**Test the resume path; do not assume it.** Kill one run deliberately partway through, re-run the cell, and confirm the loss continues from where it stopped instead of restarting. Untested resume logic is usually broken resume logic, and the moment you find out is the moment you have already lost the run.

---

## 6. KPI extraction

Extract, with provenance (page, chunk id, raw text span) for each:

**Directly extracted** — revenue, cost of revenue, gross profit, operating income, net income/loss, EBITDA, EPS (basic/diluted), total assets, total liabilities, shareholders' equity, operating cash flow, capital expenditure, shares outstanding.

**Derived** (computed, never guessed) — gross margin %, operating margin %, net margin %, EBITDA margin %, free cash flow, debt-to-equity, current ratio, ROE, ROA, P/E (needs a market price — treat as an explicit optional input, not something inferred from the filing), YoY change for each extractable metric.

Requirements:

- Combine the §5.1 tagger with rule-based extraction and record **which method produced each value**. They disagree sometimes, and that disagreement is a useful signal to surface.
- Every derived metric carries its formula and operands in the output, so a reader can check the arithmetic.
- **Units and scale are the main source of silent error.** Filings mix "in millions" / "in thousands" / absolute, and percentages appear as both `0.42` and `42%`. Normalize explicitly, store the unit, and test the conversions. In a prior project a KPI evaluation scored 0.466 purely because of a percentage-convention mismatch, and the fix was to handle the convention, not to rescale ground truth to make the metric look better.
- Negative numbers in filings appear as `(1,234)`. Parse them.

---

## 7. The recommendation — read this before designing it

**There is no ground-truth dataset for "should I invest in this company."** This is the point where the project can quietly become dishonest, so the framing is fixed:

**Do not build a model that predicts stock returns.** Labelling filings with subsequent price movement and training on it produces a return predictor that will not generalize, and claiming otherwise is the fastest way to lose credibility with anyone who works in finance.

**Do build a transparent, rubric-based assessment.** The recommendation is a deterministic aggregation over signals the system has actually extracted and verified:

- Profitability (margins, trend)
- Growth (YoY revenue, earnings)
- Leverage and liquidity (D/E, current ratio, interest coverage)
- Cash generation (FCF, OCF vs. net income)
- Risk-factor severity (from the risk-factor extraction branch)
- Management sentiment/tone (§5.3)

Requirements:

- The rubric is **explicit, versioned, and in the repo** — weights and thresholds visible and adjustable, not hidden in a prompt.
- Output a **graded assessment with confidence**, not a bare buy/sell: e.g. `favorable | mixed | unfavorable | insufficient_evidence`, plus a confidence score and the per-factor scores that produced it.
- **`insufficient_evidence` must be a real, reachable outcome.** A system that always produces a verdict is not analyzing anything. Test that it triggers on a document lacking the necessary figures.
- Every factor in the rationale cites the chunk and page supporting it. The `verify` node has already checked these; unsupported factors cannot enter the recommendation.
- The LLM writes the *narrative explanation* of the rubric's output. It does not decide the verdict. This keeps the recommendation reproducible and auditable, and keeps the LLM in the job it is good at.
- Every report carries a not-financial-advice disclaimer.

---

## 8. Report schema

One JSON document, validated against a schema (Pydantic), plus a rendered Markdown/HTML view.

```jsonc
{
  "metadata": {
    "company": "...", "ticker": "...", "doc_type": "10-K",
    "fiscal_period": "FY2023", "source_file": "...",
    "generated_at": "...", "pipeline_version": "...",
    "models": { "embedder": "...", "xbrl_tagger": "...",
                "sentiment": "...", "reasoner": "..." }
  },
  "financial_metrics": {
    "extracted": [ { "name": "revenue", "value": 383285000000, "unit": "USD",
                     "period": "FY2023", "source": {"chunk_id": "...", "page": 31},
                     "method": "xbrl_model", "confidence": 0.94 } ],
    "derived":   [ { "name": "gross_margin_pct", "value": 44.1,
                     "formula": "gross_profit / revenue * 100",
                     "operands": {"gross_profit": 169148000000, "revenue": 383285000000} } ],
    "yoy_changes": { "revenue_pct": -2.8 },
    "disagreements": [ { "name": "ebitda", "xbrl_model": 1.2e11, "rule_based": 1.19e11 } ]
  },
  "sentiment": { "overall": "neutral", "score": 0.12, "by_section": {...} },
  "risk_factors": [ { "risk": "...", "severity": "high",
                      "source": {"chunk_id": "...", "page": 18} } ],
  "evidence": [ { "chunk_id": "...", "page": 31, "text": "...", "similarity": 0.71 } ],
  "reasoning": {
    "findings": [ { "claim": "...", "supporting_chunks": ["..."],
                    "verification": "supported" } ],
    "chain_of_thought": "..."
  },
  "recommendation": {
    "assessment": "mixed",
    "confidence": 0.62,
    "rubric_version": "1.0",
    "factor_scores": { "profitability": 0.8, "growth": -0.3, "leverage": 0.5,
                       "cash_generation": 0.7, "risk": -0.4, "tone": 0.1 },
    "rationale": [ { "factor": "growth", "statement": "...",
                     "citations": [{"chunk_id": "...", "page": 31}] } ],
    "disclaimer": "Research and educational use only. Not investment advice."
  },
  "retrieval_diagnostics": { "chunks_retrieved": 5, "mean_similarity": 0.68,
                             "retries": 1, "reformulations": ["..."] },
  "unsupported_claims_dropped": 2
}
```

---

## 9. Evaluation

A CLI harness per component, all sharing one interface, each with `--test-set`, `--output`, `--limit`, `--run-agent`, `--baseline`.

| Component | Dataset | Metrics | Baseline to beat |
|---|---|---|---|
| XBRL tagger | `nlpaueb/finer-139` test | seqeval micro-F1, per-concept P/R/F1 | paper 89.2% (full split); note your subset size |
| Retrieval | `BeIR/fiqa` test, `PatronusAI/financebench` | NDCG@10, Hit@1/5, MRR | stock `bge-base/large`, same corpus + seed |
| Sentiment | `takala/financial_phrasebank` held-out | accuracy, macro-F1, confusion matrix | `ProsusAI/finbert` on the same held-out split |
| KPI extraction | hand-labelled set from 10–20 real filings + `ibm/finqa` | value accuracy (tolerance-aware), extraction recall, unit-error rate | rule-based extractor |
| Numerical reasoning | `ibm/finqa` test | execution accuracy, per-operator breakdown | base model zero-shot; hosted model |
| RAG faithfulness | your labelled set | **citation coverage**, claim-support precision, hallucination rate | ungrounded generation |
| Recommendation | 20–30 filings you label against the rubric | rubric agreement, `insufficient_evidence` rate, factor-level agreement | — |

Notes:

- Retrieval evaluation needs **no LLM calls**, so before/after comparisons are free. Do those often.
- If you sample the corpus for speed, absolute numbers are not comparable to published figures — only your own before/after delta is valid. State which you are quoting.
- Faithfulness is the metric that matters most for this project and is the least standard. Define it precisely: a claim is supported if its numeric values and entities appear in, or are directly computable from, its cited chunks. Automate it, and hand-check a sample to report agreement.
- Build a small **hand-labelled KPI set from real filings**. It is the most valuable dataset in the project because it measures the actual product, and nobody else has it.

---

## 10. Engineering requirements

- **Python 3.10+**, `pyproject.toml`, `pip install -e ".[dev,eval,train]"` works.
- **Tests** — pytest, and every bug fixed gets a regression test. Cover: financial-notation cleaning, unit/scale conversion, negative-number parsing, chunker termination, threshold reachability, delta-returning graph nodes, schema validation, and the API contract with models mocked.
- **CI** — GitHub Actions running the suite on 3.10 and 3.11, with a cached HF model directory keyed on the modules holding model names. Do not add a lint job you have not made pass.
- **Config** — YAML that is *actually loaded*. If a config file exists, it must drive behavior; documentation-only config files are a trap.
- **API** — FastAPI upload endpoint returning the report, plus a thin UI (Streamlit or Gradio) with the filing on one side and cited evidence on the other. A visible demo is worth more than another agent.
- **Notebooks** — one Colab notebook per fine-tune (§5), each self-contained: mount Drive, install, data, train **with resumable checkpointing (§5.5)**, evaluate against baseline, push to Hub. Each must check that the cloned repo is current before running, and each must be safe to re-run from the top after a disconnect — landing back where it left off rather than starting over.
- **Docs** — README with an architecture diagram, real sample output, measured results separated from literature references, and a roadmap that describes what is *not* done. RUNNING.md with a clean-checkout walkthrough.
- **Model cards** on every pushed model, carrying the real evaluation numbers and the training subset size.

---

## 11. Anti-patterns — do not reproduce these

Each of these was a real, measured failure in a previous project:

1. A conditional threshold that cannot fire because an upstream filter already guarantees the condition is false. Assert threshold relationships at import.
2. Two independent retry mechanisms with different trigger conditions, fighting each other.
3. A retry loop whose retries do not change the query.
4. Documentation stating `OR` where the code does `AND`, or naming a model that was swapped out.
5. A CLI flag documented in the README that does not exist in `argparse`.
6. A config file the docs treat as live that is never loaded.
7. Reporting a fine-tune's score on the split that selected its checkpoint.
8. Fine-tuning a model on data it was already trained on, then reporting the score as evidence of quality.
9. A benchmark table of published numbers for models the repo does not use, placed where it reads as the repo's own results.
10. Parallel graph nodes returning the entire state.
11. Assuming concurrency is a speedup without measuring it — CPU thread oversubscription can make parallel *slower*, and GPU branches share one device.
12. Rescaling ground truth so a metric improves.
13. Writing training checkpoints to ephemeral storage, or adding resume logic and never testing it. Both look like working checkpointing right up until the run you needed it for.
14. Resuming a run after changing the seed, the subset size, or the data order — the global step then indexes into different data and the resumed run is quietly meaningless.

---

## 12. Milestones

1. **Ingestion + report skeleton** — parse/clean/chunk/embed, schema, tests. No models yet.
2. **Retrieval fine-tune** (§5.2) with before/after on FiQA and FinanceBench.
3. **XBRL tagger** (§5.1) plus KPI extraction and the hand-labelled KPI set.
4. **Sentiment** (§5.3) with a fair `finbert` comparison.
5. **Agentic graph** — fan-out, bounded re-query, `verify` node, faithfulness metric.
6. **Rubric recommendation** (§7) with `insufficient_evidence` tested.
7. **Numerical-reasoning LLM** (§5.4) as three-way comparison.
8. **Demo UI, CI, docs, model cards.**

Ship each milestone working, with its evaluation, before starting the next. A finished milestone 5 beats a half-built milestone 8.

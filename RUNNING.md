# Running this project from a clean checkout

Every command below was run on a clean checkout. Where something needs a GPU, a
network download or data that does not exist yet, that is stated rather than
implied.

---

## 1. Install

Python 3.10 or 3.11.

```bash
git clone <this repo>
cd agentic-financial-filing-analysis
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The core install is deliberately light — Pydantic, PyYAML, NumPy. Schema,
config, cleaning, chunking, unit normalisation, the rubric and the routing logic
all import without PyTorch, so the test suite runs in seconds.

Extras, added as you need them:

| Extra | Brings | Needed for |
|---|---|---|
| `ingest` | pdfplumber, BeautifulSoup, spaCy, ChromaDB, sentence-transformers | PDF/HTML parsing, real embeddings, persistent vector store |
| `agent` | langgraph, fastapi, uvicorn, streamlit | the graph executor, the API, the UI |
| `eval` | datasets (`<4.0`), seqeval, scikit-learn, pandas | the evaluation harnesses |
| `train` | torch, transformers, peft, bitsandbytes, accelerate | fine-tuning |
| `hosted` | anthropic, openai | the hosted-model comparison |

```bash
pip install -e ".[dev,ingest,agent,eval]"
```

### The `datasets` pin is load-bearing

`nlpaueb/finer-139`, `takala/financial_phrasebank` and `ibm/finqa` are
**loading-script datasets**, and `datasets>=4.0` removed script execution
entirely — they do not load at all. The `eval` and `train` extras pin
`datasets>=2.19,<4.0`.

If you already have `datasets` 4.x in your environment:

```bash
pip install "datasets>=2.19,<4.0"
```

Do **not** substitute a third-party parquet mirror. At least one is
deduplicated, which changes the splits and makes any number you get
incomparable to the published figures. `training/common.py` checks the version
and fails loudly rather than letting you find out later.

---

## 2. Run the tests

```bash
pytest
```

Expect 194 passing in under 20 seconds. No network, no GPU, no model downloads —
if a run tries to reach the network, something regressed.

```bash
pytest --cov=affa --cov-report=term-missing    # with coverage
pytest tests/test_verify.py -v                 # one module
ruff check . && ruff format --check .           # lint, as CI runs it
```

---

## 3. Analyze the bundled sample

```bash
python -m affa.cli data/samples/demo_10k.json --format markdown --in-memory
```

The first run downloads `BAAI/bge-base-en-v1.5` (~440MB). Without
`sentence-transformers` installed, the pipeline falls back to a hashing stub
embedder, warns loudly, and every report it produces carries that warning —
similarity scores from it are lexical, not semantic.

`--in-memory` keeps the run out of the persistent Chroma collection. Use it for
one-off analysis; drop it to build a durable index.

Other formats:

```bash
python -m affa.cli data/samples/demo_10k.json --format json -o report.json --in-memory
python -m affa.cli data/samples/demo_10k.json --format html -o report.html --in-memory
python -m affa.cli my_filing.pdf --ticker AAPL --market-price 190.50
```

`--market-price` is optional and only feeds P/E. A filing does not contain a
share price, so it is an explicit input rather than something inferred.

**The bundled sample is synthetic.** Northwind Systems is fictional. It
exercises the pipeline end to end; it is not evidence about real filings.

---

## 4. Confirm `insufficient_evidence` works

A system that always produces a verdict is not analyzing anything. To see the
abstention path:

```bash
cat > /tmp/thin.json <<'EOF'
{"blocks": [
  {"kind": "narrative", "page": 1,
   "text": "FORM 10-K Annual Report. Thin Filings Corp. Trading Symbol: THIN. For the fiscal year ended December 31, 2024."},
  {"kind": "narrative", "page": 5,
   "text": "The Company continued to operate during the period. Competition could adversely affect us."}
]}
EOF

python -m affa.cli /tmp/thin.json --format markdown --in-memory
```

Expected: `Insufficient Evidence`, no aggregate score, and a list of the factors
that could not be scored.

---

## 5. Serve the API and the UI

```bash
uvicorn affa.api.main:app --reload
```

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | multipart upload → structured report (`response_format`: `json`/`markdown`/`html`) |
| `GET /health` | which models are actually loaded, not which are configured |
| `GET /config` | thresholds in force |
| `GET /docs` | OpenAPI UI |

```bash
curl -F "file=@data/samples/demo_10k.json" \
     -F "response_format=markdown" \
     http://localhost:8000/analyze
```

```bash
streamlit run src/affa/ui/app.py
```

The UI puts retrieved passages on the left and verified findings on the right,
so "why did it say that" is one click.

---

## 6. Run the evaluation harnesses

All seven share the same flags:

```
affa-eval <component> [--test-set ...] [--output ...] [--limit N] [--run-agent] [--baseline NAME]
```

Works today, no GPU or network needed:

```bash
affa-eval kpi --output eval_results/kpi.json
```

This measures against `data/kpi_gold/`, which currently holds only the synthetic
sample — so it verifies the harness, not the product. See
[`data/kpi_gold/README.md`](data/kpi_gold/README.md) for how to add real filings;
section 9 asks for 10–20 of them.

Needs `[eval]` and a network download:

```bash
affa-eval retrieval --test-set BeIR/fiqa --limit 200 --output eval_results/fiqa.json
affa-eval sentiment --output eval_results/sentiment.json
affa-eval xbrl --limit 2000 --output eval_results/xbrl.json
```

Needs a configured reasoner (`models.reasoner.backend` = `local` or `hosted`):

```bash
affa-eval finqa --limit 100 --output eval_results/finqa.json
affa-eval faithfulness --hand-check 30 --output eval_results/faithfulness.json
```

Two things every harness does:

- it **refuses to report a metric without a baseline** measured on the same data
  with the same protocol;
- `--limit` records the subset size, and the output warns that sampled numbers
  are comparable to the baseline row but **not** to published figures.

---

## 7. Fine-tune a model

Nothing here has been run yet — the notebooks and scripts exist, the guards are
tested, the training is not done.

### On Colab (intended path)

Open a notebook from `notebooks/` in Colab, set **Runtime → Change runtime type
→ T4 GPU**, and run from the top. Each notebook mounts Drive, checks the repo is
current, pins `datasets<4.0`, trains with resumable checkpointing, evaluates
against a measured baseline, and pushes a model card.

### Locally

```bash
python training/train_sentiment.py --output-dir /path/on/durable/disk/sentiment
python training/train_retrieval.py --output-dir /path/on/durable/disk/embedder
python training/train_xbrl_tagger.py --output-dir /path/.../xbrl --train-samples 200000
python training/train_finqa_qlora.py --output-dir /path/.../finqa
```

`--output-dir` is **rejected** if it looks ephemeral (`/content`, `/tmp`,
`./results`). That is the point: writing checkpoints to storage that dies with
the VM looks like working checkpointing right up until the run you needed it
for. Pass `--allow-ephemeral` only for a deliberate throwaway job.

### Test the resume path before the real run

Do not assume it works.

1. Start training and let it write two checkpoints.
2. Kill it (Ctrl-C, or let the Colab runtime die).
3. Re-run the **identical** command.

You should see `resuming from .../checkpoint-N` and the loss continuing from
where it stopped. If the step counter restarts at 0, resume is broken — fix it
before spending 12 hours.

Changing `--seed`, `--train-samples` or `--base-model` and then resuming is
**refused**, with a message explaining why: the global step indexes into a
specific data order, so the resumed run would be silently meaningless.

---

## 8. Use a fine-tuned model

Point the config at it — no code change:

```yaml
# configs/default.yaml
models:
  embedder:
    name: "YOUR_USERNAME/affa-retrieval-embedder"
  xbrl_tagger:
    finetuned: "YOUR_USERNAME/affa-xbrl-tagger"
    enabled: true
  sentiment:
    finetuned: "YOUR_USERNAME/affa-sentiment"
    enabled: true
  reasoner:
    backend: "local"
    local_adapter: "YOUR_USERNAME/affa-finqa-qlora"
```

**Changing the embedder requires re-indexing.** Vectors written by one model are
meaningless when queried by another, and a vector store answers such a query
confidently instead of erroring. The collection name is derived from the
embedder id, so a new model lands in a new collection automatically; the old one
stays on disk until you delete `.affa/chroma`.

---

## 9. Regenerate the notebooks

The notebooks are generated so the shared cells stay identical across all four:

```bash
python scripts/make_notebooks.py
```

Edit `scripts/make_notebooks.py`, not the `.ipynb` files.

---

## Troubleshooting

**`ConfigError: unreachable routing threshold`** — you set
`routing.retry_below_mean_similarity` at or below `retrieval.min_similarity`.
Retrieval discards everything below the floor, so the retry could never fire.
Raise the retry threshold above the floor.

**`ResumeMismatchError`** — you changed the seed, subset size or base model since
the checkpoint was written. Restore the original settings, or start fresh in a
new checkpoint directory.

**`RuntimeError: datasets 4.x is installed`** — see the pin note in §1.

**"falling back to HashingEmbedder"** — `sentence-transformers` is not installed
or the model could not be downloaded. Install `.[ingest]`. The stub is lexical,
so any retrieval number from it is meaningless.

**`EmbedderMismatchError`** — a Chroma collection was written by a different
embedding model. Delete `.affa/chroma` and re-index, or change
`vector_store.collection_prefix`.

**Chunking hangs** — it should not; the packing loop raises `ChunkerError` rather
than spinning. If you see one, the sentence splitter returned something
unexpected: file it with the document.

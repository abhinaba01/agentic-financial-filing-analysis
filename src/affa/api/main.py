"""FastAPI service: upload a filing, get the structured report.

Run it with::

    uvicorn affa.api.main:app --reload

The response body is the same :class:`~affa.schema.AnalysisReport` the CLI
emits - one contract, so the UI, the CLI and any downstream consumer all see
identical output and cannot drift apart.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from affa import DISCLAIMER, __version__
from affa.config import get_config
from affa.pipeline import DEFAULT_QUESTION, analyze_filing
from affa.report.render import to_html, to_markdown

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError('the API needs fastapi. Install: pip install -e ".[agent]"') from exc

SUPPORTED_SUFFIXES = {".pdf", ".htm", ".html", ".txt", ".text", ".md", ".json"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

app = FastAPI(
    title="Agentic Financial Filing Analyst",
    version=__version__,
    description=(
        "Upload a 10-K/10-Q and receive a structured, cited investment assessment. " + DISCLAIMER
    ),
)


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = get_config()
    return {
        "status": "ok",
        "version": __version__,
        "pipeline_version": cfg.pipeline_version,
        "models": {
            "embedder": cfg.models.embedder.name,
            "xbrl_tagger": cfg.models.xbrl_tagger.active_name,
            "xbrl_tagger_enabled": cfg.models.xbrl_tagger.enabled,
            "sentiment": cfg.models.sentiment.active_name,
            "sentiment_enabled": cfg.models.sentiment.enabled,
            "reasoner_backend": cfg.models.reasoner.backend,
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/config")
def config_summary() -> dict[str, Any]:
    """The thresholds actually in force, so the UI never has to guess them."""
    cfg = get_config()
    return {
        "retrieval": {
            "top_k": cfg.retrieval.top_k,
            "min_similarity": cfg.retrieval.min_similarity,
        },
        "routing": {
            "retry_below_mean_similarity": cfg.routing.retry_below_mean_similarity,
            "max_retrieval_attempts": cfg.routing.max_retrieval_attempts,
            "min_chunks_for_sufficiency": cfg.routing.min_chunks_for_sufficiency,
        },
        "verification": {
            "numeric_tolerance_pct": cfg.verification.numeric_tolerance_pct,
            "min_entity_overlap": cfg.verification.min_entity_overlap,
            "drop_unsupported_claims": cfg.verification.drop_unsupported_claims,
        },
        "rubric_version": _rubric_version(),
    }


def _rubric_version() -> str:
    from affa.config import load_rubric

    return str(load_rubric()["version"])


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="10-K/10-Q as PDF, HTML, TXT or JSON"),
    question: str = Form(DEFAULT_QUESTION),
    ticker: str | None = Form(None),
    company: str | None = Form(None),
    fiscal_period: str | None = Form(None),
    market_price: float | None = Form(None),
    response_format: str = Form("json"),
):
    """Analyze an uploaded filing.

    ``market_price`` is optional and only used for P/E. A filing does not contain
    a share price, so it is an explicit input rather than something inferred.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (file.filename or f"upload{suffix}")
        path.write_bytes(payload)
        try:
            result = analyze_filing(
                path,
                question=question,
                ticker=ticker or None,
                company=company or None,
                fiscal_period=fiscal_period or None,
                market_price_per_share=market_price,
                # Uploads are transient, so they must not accumulate in the
                # persistent collection.
                in_memory=True,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"a parser for this format is not installed: {exc}",
            ) from exc

    if response_format == "markdown":
        return PlainTextResponse(to_markdown(result.report))
    if response_format == "html":
        return HTMLResponse(to_html(result.report))
    return JSONResponse(result.report.model_dump(mode="json"))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!doctype html><meta charset="utf-8"><title>Agentic Financial Filing Analyst</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:44rem;margin:3rem auto;padding:0 1rem}}
code{{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}}
blockquote{{border-left:3px solid #c00;padding-left:1rem;color:#600}}</style>
<h1>Agentic Financial Filing Analyst</h1>
<blockquote>{DISCLAIMER}</blockquote>
<p>Version {__version__}. Endpoints:</p>
<ul>
  <li><code>POST /analyze</code> - multipart upload, returns the structured report</li>
  <li><code>GET /health</code> - which models are actually loaded</li>
  <li><code>GET /config</code> - thresholds in force</li>
  <li><code>GET /docs</code> - OpenAPI UI</li>
</ul>
<p>For the side-by-side evidence view, run the Streamlit UI:
<code>streamlit run src/affa/ui/app.py</code></p>
"""

"""Filing parsers: PDF, HTML, plain text and JSON.

Page numbers and table structure are preserved because the report cites them.
Heavy parser dependencies are imported inside the functions that need them, so
``affa.ingestion.parse`` imports cleanly in an environment with only the core
dependencies installed and fails with a precise message only when a PDF is
actually handed to it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from affa.config import ParseConfig
from affa.ingestion.clean import clean_text
from affa.ingestion.types import Block, ParsedDocument

# --- document-level metadata sniffing -------------------------------------

_TICKER_RE = re.compile(
    r"(?:trading\s+symbol|ticker\s+symbol|symbol)\s*[:\)]?\s*[\"'(]?\s*([A-Z]{1,5})\b",
    re.IGNORECASE,
)
_FY_RE = re.compile(
    r"(?:fiscal\s+year\s+(?:ended|ending)[^.\n]{0,40}?|for\s+the\s+fiscal\s+year\s+\w+\s+)"
    r"(?:\w+\s+\d{1,2},\s*)?(\d{4})",
    re.IGNORECASE,
)
_FY_LABEL_RE = re.compile(r"\bFY\s?(\d{4})\b", re.IGNORECASE)
_DOC_TYPE_RE = re.compile(r"\bFORM\s+(10-K/A|10-Q/A|10-K|10-Q|8-K|20-F|40-F)\b", re.IGNORECASE)
_COMPANY_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9&.,'\- ]{2,60}?(?:,?\s+(?:Inc|Corp|Corporation|Company|Co|Ltd|LLC|PLC|N\.V|S\.A)\.?))\s*$",
    re.MULTILINE,
)


def sniff_metadata(text: str) -> dict[str, str | None]:
    """Best-effort company / ticker / period / doc-type from the cover page.

    Deliberately shallow: these are hints for filtered retrieval and the report
    header, and any of them may legitimately be ``None``. The API lets a caller
    override every one, because a wrong ticker silently filters retrieval down
    to nothing.
    """
    head = text[:20000]
    doc_type = None
    if m := _DOC_TYPE_RE.search(head):
        doc_type = m.group(1).upper()

    fiscal_period = None
    if m := _FY_LABEL_RE.search(head):
        fiscal_period = f"FY{m.group(1)}"
    elif m := _FY_RE.search(head):
        fiscal_period = f"FY{m.group(1)}"

    ticker = None
    if m := _TICKER_RE.search(head):
        ticker = m.group(1).upper()

    company = None
    if m := _COMPANY_RE.search(head):
        company = " ".join(m.group(1).split())

    return {
        "company": company,
        "ticker": ticker,
        "doc_type": doc_type,
        "fiscal_period": fiscal_period,
    }


def make_doc_id(path: Path | str, content: str) -> str:
    """Content-addressed document id, so re-ingesting the same file is a no-op."""
    stem = Path(path).stem[:32].replace(" ", "_")
    digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{stem}-{digest}"


# --- table rendering ------------------------------------------------------


def render_table(rows: list[list[str | None]]) -> str:
    """Flatten an extracted table to pipe-delimited text.

    Column separators are kept because the KPI extractor needs to associate a
    label with the figure on its row; a table flattened to prose loses that and
    the extractor starts matching a label to a number from a different line.
    """
    out: list[str] = []
    for row in rows:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        if not any(cells):
            continue
        out.append(" | ".join(cells))
    return "\n".join(out)


# --- format-specific parsers ----------------------------------------------


def parse_pdf(path: Path, cfg: ParseConfig) -> tuple[list[Block], int]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            'PDF parsing needs pdfplumber. Install the ingest extra: pip install -e ".[ingest]"'
        ) from exc

    blocks: list[Block] = []
    with pdfplumber.open(str(path)) as pdf:
        pages = pdf.pages if cfg.max_pages is None else pdf.pages[: cfg.max_pages]
        n_pages = len(pdf.pages)
        for page_no, page in enumerate(pages, start=1):
            tables = page.extract_tables() if cfg.extract_tables else []
            for t_idx, rows in enumerate(tables):
                rendered = render_table(rows)
                if rendered.strip():
                    blocks.append(
                        Block(
                            # Layout preserved: column alignment is the signal.
                            text=clean_text(rendered, preserve_layout=True, strip_leaders=False),
                            page=page_no,
                            kind="table",
                            table_index=t_idx,
                        )
                    )
            text = page.extract_text() or ""
            if text.strip():
                blocks.append(Block(text=clean_text(text), page=page_no, kind="narrative"))
    return blocks, n_pages


def parse_html(path: Path, cfg: ParseConfig) -> tuple[list[Block], int]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            'HTML parsing needs beautifulsoup4. Install: pip install -e ".[ingest]"'
        ) from exc

    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    blocks: list[Block] = []
    if cfg.extract_tables:
        for t_idx, table in enumerate(soup.find_all("table")):
            rows = [
                [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
                for tr in table.find_all("tr")
            ]
            rendered = render_table(rows)
            if rendered.strip():
                blocks.append(
                    Block(
                        text=clean_text(rendered, preserve_layout=True, strip_leaders=False),
                        page=None,
                        kind="table",
                        table_index=t_idx,
                    )
                )
            # Extracted separately above; leaving it in would duplicate the text.
            table.decompose()

    body = soup.body or soup
    text = clean_text(body.get_text("\n", strip=True))
    if text:
        # HTML filings have no pages. Paragraph grouping keeps blocks addressable.
        for para in re.split(r"\n{2,}", text):
            if para.strip():
                blocks.append(Block(text=para.strip(), page=None, kind="narrative"))
    return blocks, 0


def parse_txt(path: Path, cfg: ParseConfig) -> tuple[list[Block], int]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    blocks = [
        Block(text=p.strip(), page=None, kind="narrative")
        for p in re.split(r"\n{2,}", text)
        if p.strip()
    ]
    return blocks, 0


def parse_json(path: Path, cfg: ParseConfig) -> tuple[list[Block], int]:
    """Parse a pre-extracted filing.

    Accepts either ``{"blocks": [{"text":..., "page":..., "kind":...}, ...]}``
    or a bare list of such objects. This is the format the hand-labelled KPI set
    in ``data/kpi_gold/`` uses, so gold documents replay through the identical
    pipeline as live uploads.
    """
    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    items = payload.get("blocks", []) if isinstance(payload, dict) else payload
    blocks: list[Block] = []
    for item in items:
        if isinstance(item, str):
            blocks.append(Block(text=clean_text(item)))
            continue
        kind = item.get("kind", "narrative")
        text = clean_text(
            item.get("text", ""),
            preserve_layout=(kind == "table"),
            strip_leaders=(kind != "table"),
        )
        if text:
            blocks.append(
                Block(
                    text=text,
                    page=item.get("page"),
                    kind=kind,
                    table_index=item.get("table_index"),
                    section=item.get("section"),
                )
            )
    n_pages = max((b.page or 0) for b in blocks) if blocks else 0
    return blocks, n_pages


PARSERS = {
    ".pdf": parse_pdf,
    ".htm": parse_html,
    ".html": parse_html,
    ".txt": parse_txt,
    ".text": parse_txt,
    ".md": parse_txt,
    ".json": parse_json,
}


def parse_document(
    path: str | Path,
    cfg: ParseConfig | None = None,
    *,
    company: str | None = None,
    ticker: str | None = None,
    doc_type: str | None = None,
    fiscal_period: str | None = None,
) -> ParsedDocument:
    """Parse a filing into ordered blocks with page numbers and tables intact.

    Explicit metadata arguments always win over sniffed values: a caller who
    knows the ticker should never be overridden by a cover-page regex.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"filing not found: {path}")
    cfg = cfg or ParseConfig(extract_tables=True, max_pages=None)

    suffix = path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"unsupported filing format {suffix!r}; supported: {sorted(PARSERS)}")

    blocks, n_pages = parser(path, cfg)
    full_text = "\n\n".join(b.text for b in blocks)
    sniffed = sniff_metadata(full_text)

    return ParsedDocument(
        blocks=blocks,
        source_file=str(path),
        doc_id=make_doc_id(path, full_text),
        company=company or sniffed["company"],
        ticker=(ticker or sniffed["ticker"] or None),
        doc_type=doc_type or sniffed["doc_type"],
        fiscal_period=fiscal_period or sniffed["fiscal_period"],
        n_pages=n_pages or None,
        metadata={"n_blocks": len(blocks), "n_tables": sum(b.kind == "table" for b in blocks)},
    )

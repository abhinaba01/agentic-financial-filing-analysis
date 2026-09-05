"""Shared ingestion data types.

Kept separate from ``parse`` and ``chunk`` so the chunker can be imported and
tested without pulling in pdfplumber or BeautifulSoup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BlockKind = Literal["narrative", "table", "heading"]


@dataclass
class Block:
    """A contiguous region of a parsed document, with its page number.

    Tables stay whole through parsing so the chunker can keep them whole:
    a financial statement split down the middle loses the row/column
    relationship the KPI extractor reads values from.
    """

    text: str
    page: int | None = None
    kind: BlockKind = "narrative"
    table_index: int | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "table" and self.table_index is None:
            self.table_index = 0


@dataclass
class ParsedDocument:
    """Output of the parse stage: ordered blocks plus document-level metadata."""

    blocks: list[Block]
    source_file: str
    doc_id: str
    company: str | None = None
    ticker: str | None = None
    doc_type: str | None = None
    fiscal_period: str | None = None
    n_pages: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def tables(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "table"]


@dataclass
class Chunk:
    """An embedding-sized unit of text with full provenance back to the filing."""

    chunk_id: str
    text: str
    doc_id: str
    token_count: int
    page: int | None = None
    chunk_type: BlockKind = "narrative"
    ticker: str | None = None
    fiscal_period: str | None = None
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    table_index: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Chroma metadata. Values must be scalar; None is dropped, not stored.

        The filtered-retrieval keys required by section 4 are doc_id, ticker,
        fiscal_period, page_number and chunk_type.
        """
        raw = {
            "doc_id": self.doc_id,
            "ticker": self.ticker,
            "fiscal_period": self.fiscal_period,
            "page_number": self.page,
            "chunk_type": self.chunk_type,
            "section": self.section,
            "token_count": self.token_count,
            "table_index": self.table_index,
        }
        return {k: v for k, v in raw.items() if v is not None}

"""Ingestion: parse -> clean -> chunk -> embed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from affa.config import AffaConfig, get_config
from affa.ingestion.chunk import chunk_document
from affa.ingestion.clean import clean_text
from affa.ingestion.embed import Embedder, build_embedder, build_vector_store
from affa.ingestion.parse import parse_document
from affa.ingestion.types import Block, Chunk, ParsedDocument

__all__ = [
    "Block",
    "Chunk",
    "Embedder",
    "IngestResult",
    "ParsedDocument",
    "build_embedder",
    "build_vector_store",
    "chunk_document",
    "clean_text",
    "ingest_filing",
    "parse_document",
]


@dataclass
class IngestResult:
    document: ParsedDocument
    chunks: list[Chunk]
    store: object
    embedder: Embedder
    n_indexed: int

    @property
    def used_stub_embedder(self) -> bool:
        return bool(getattr(self.embedder, "is_stub", False))


def ingest_filing(
    path: str | Path,
    cfg: AffaConfig | None = None,
    *,
    ticker: str | None = None,
    company: str | None = None,
    fiscal_period: str | None = None,
    doc_type: str | None = None,
    store: object | None = None,
    embedder: Embedder | None = None,
    in_memory: bool = False,
) -> IngestResult:
    """Run the full ingestion pipeline for one filing.

    The tokenizer used for chunk budgeting is the embedder's own, so a "512
    token" chunk is 512 tokens to the model that will actually embed it rather
    than to some unrelated tokenizer.
    """
    cfg = cfg or get_config()
    doc = parse_document(
        path,
        cfg.ingestion.parse,
        company=company,
        ticker=ticker,
        doc_type=doc_type,
        fiscal_period=fiscal_period,
    )
    chunks = chunk_document(doc, cfg.ingestion.chunk, tokenizer_name=cfg.models.embedder.name)
    emb = embedder or build_embedder(cfg)
    vstore = store or build_vector_store(cfg, emb, in_memory=in_memory)
    n = vstore.add_chunks(chunks)  # type: ignore[attr-defined]
    return IngestResult(document=doc, chunks=chunks, store=vstore, embedder=emb, n_indexed=n)

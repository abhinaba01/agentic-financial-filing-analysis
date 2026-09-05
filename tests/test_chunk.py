"""Chunker termination and boundary behaviour (section 4, anti-pattern list).

The infinite loop this guards against only appears on documents longer than one
chunk, so it survives every small-input smoke test and hangs on a real 10-K.
"""

from __future__ import annotations

import pytest

from affa.config import ChunkConfig, ConfigError
from affa.ingestion.chunk import _next_start, chunk_document, split_sentences
from affa.ingestion.types import Block, ParsedDocument


def make_cfg(**overrides) -> ChunkConfig:
    base = dict(
        target_tokens=100,
        overlap_tokens=20,
        min_tokens=5,
        keep_tables_whole=True,
        max_table_tokens=400,
        sentence_splitter="regex",
    )
    base.update(overrides)
    return ChunkConfig(**base)


def doc_with(text: str, kind: str = "narrative") -> ParsedDocument:
    return ParsedDocument(
        blocks=[Block(text=text, page=1, kind=kind)],
        source_file="t.txt",
        doc_id="DOC",
        ticker="TEST",
        fiscal_period="FY2024",
    )


def test_config_rejects_non_advancing_window() -> None:
    """overlap >= target can never advance. Caught at config load, not at runtime."""
    with pytest.raises(ConfigError, match="never advances"):
        make_cfg(target_tokens=100, overlap_tokens=100)
    with pytest.raises(ConfigError):
        make_cfg(target_tokens=100, overlap_tokens=150)


def test_next_start_always_advances() -> None:
    """The guard that makes termination structural rather than incidental."""
    # A trailing sentence larger than the whole overlap budget would otherwise
    # rewind the window to its own start.
    assert _next_start(current_start=0, current_end=1, sent_tokens=[500], overlap_tokens=64) == 1
    assert (
        _next_start(current_start=3, current_end=4, sent_tokens=[9] * 10, overlap_tokens=1000) > 3
    )
    for start in range(0, 9):
        nxt = _next_start(
            current_start=start, current_end=start + 1, sent_tokens=[50] * 10, overlap_tokens=49
        )
        assert nxt > start


def test_long_document_terminates_and_covers_everything() -> None:
    text = " ".join(
        f"Sentence {i} reports revenue of $1,{i:03d}.5 million for FY2023." for i in range(300)
    )
    chunks = chunk_document(doc_with(text), make_cfg())
    assert len(chunks) > 1
    assert all(c.token_count > 0 for c in chunks)
    # Nothing lost: the first and last sentences both appear somewhere.
    joined = " ".join(c.text for c in chunks)
    assert "Sentence 0 " in joined
    assert "Sentence 299 " in joined


def test_single_sentence_longer_than_window_is_split() -> None:
    """No sentence boundaries at all is the pathological case for a sentence-aware chunker."""
    chunks = chunk_document(doc_with("word " * 3000), make_cfg())
    assert len(chunks) > 1
    assert all(c.token_count <= 200 for c in chunks)


def test_chunk_ids_are_unique_and_stable() -> None:
    text = " ".join(f"Sentence {i} about revenue." for i in range(120))
    first = chunk_document(doc_with(text), make_cfg())
    second = chunk_document(doc_with(text), make_cfg())
    assert len({c.chunk_id for c in first}) == len(first)
    # Content-addressed, so re-ingesting an unchanged document keeps citations valid.
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_tables_stay_whole() -> None:
    table = "Revenue | 383,285 | 394,328\nCost | 214,137 | 223,546\nProfit | 169,148 | 170,782"
    chunks = chunk_document(doc_with(table, kind="table"), make_cfg())
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
    assert "169,148" in chunks[0].text


def test_oversized_table_is_split_rather_than_dropped() -> None:
    big = "\n".join(f"Line item {i} | {i},234 | {i},567" for i in range(400))
    chunks = chunk_document(doc_with(big, kind="table"), make_cfg(max_table_tokens=50))
    assert len(chunks) > 1


def test_metadata_carries_filter_keys() -> None:
    """Section 4 requires these keys for filtered retrieval."""
    chunks = chunk_document(doc_with("Revenue rose. Costs fell. Profit grew."), make_cfg())
    meta = chunks[0].to_metadata()
    assert meta["doc_id"] == "DOC"
    assert meta["ticker"] == "TEST"
    assert meta["fiscal_period"] == "FY2024"
    assert meta["page_number"] == 1
    assert meta["chunk_type"] == "narrative"
    # Chroma rejects None values, so absent keys must be dropped, not stored.
    assert all(v is not None for v in meta.values())


def test_abbreviations_do_not_split_sentences() -> None:
    sents = split_sentences("Apple Inc. reported record sales. Revenue grew 8%.", "regex")
    assert len(sents) == 2
    assert sents[0].startswith("Apple Inc.")


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document(doc_with("   \n\n  "), make_cfg()) == []

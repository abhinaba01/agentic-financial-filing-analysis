"""Shared fixtures.

Everything here runs without network access, model downloads or a GPU. The
suite's value depends on it running in seconds on every push, so the heavy paths
are exercised through the stub embedder and the ``heavy`` marker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from affa.config import AffaConfig, load_config
from affa.ingestion.embed import HashingEmbedder, InMemoryVectorStore
from affa.ingestion.types import Block, Chunk, ParsedDocument

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILING = REPO_ROOT / "data" / "samples" / "demo_10k.json"


@pytest.fixture(scope="session", autouse=True)
def _force_stub_embedder() -> None:
    """Keep the suite offline and fast.

    The real embedder is a 440MB download and ~5s per analysis. CI must not
    depend on the network, so every test runs against the deterministic hashing
    embedder. Its similarity scores are lexical, which is fine here: these tests
    assert pipeline behaviour and schema contracts, never retrieval quality.
    Retrieval quality is measured by `affa-eval retrieval`, which refuses to run
    with the stub.
    """
    os.environ["AFFA_FORCE_STUB_EMBEDDER"] = "1"


@pytest.fixture(scope="session")
def cfg() -> AffaConfig:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def sample_filing_path() -> Path:
    return SAMPLE_FILING


@pytest.fixture
def income_statement_chunk() -> Chunk:
    """A realistic statement table, stated in millions, with two periods."""
    text = (
        "CONSOLIDATED STATEMENTS OF OPERATIONS (In millions, except per share data)\n"
        " | 2024 | 2023\n"
        "Total net sales | 4,812.6 | 4,301.9\n"
        "Cost of sales | 1,645.8 | 1,532.4\n"
        "Gross profit | 3,166.8 | 2,769.5\n"
        "Operating income | 962.5 | 774.3\n"
        "Interest expense | (48.6) | (52.1)\n"
        "Net income | 731.4 | 588.2\n"
        "Diluted | 3.64 | 2.93"
    )
    return Chunk(
        chunk_id="chunk-income",
        text=text,
        doc_id="TESTDOC",
        token_count=120,
        page=31,
        chunk_type="table",
        ticker="NWSY",
        fiscal_period="FY2024",
    )


@pytest.fixture
def balance_sheet_chunk() -> Chunk:
    text = (
        "CONSOLIDATED BALANCE SHEETS (In millions)\n"
        " | 2024 | 2023\n"
        "Total current assets | 2,914.5 | 2,510.8\n"
        "Total assets | 5,982.3 | 5,344.1\n"
        "Total current liabilities | 1,806.2 | 1,702.9\n"
        "Total debt | 1,180.0 | 1,265.0\n"
        "Total shareholders equity | 2,455.7 | 1,972.7"
    )
    return Chunk(
        chunk_id="chunk-balance",
        text=text,
        doc_id="TESTDOC",
        token_count=90,
        page=33,
        chunk_type="table",
        ticker="NWSY",
        fiscal_period="FY2024",
    )


@pytest.fixture
def cash_flow_chunk() -> Chunk:
    text = (
        "CONSOLIDATED STATEMENTS OF CASH FLOWS (In millions)\n"
        " | 2024 | 2023\n"
        "Depreciation and amortization | 214.7 | 198.3\n"
        "Net cash provided by operating activities | 1,104.2 | 918.6\n"
        "Purchases of property and equipment | (268.4) | (241.9)"
    )
    return Chunk(
        chunk_id="chunk-cash",
        text=text,
        doc_id="TESTDOC",
        token_count=70,
        page=36,
        chunk_type="table",
        ticker="NWSY",
        fiscal_period="FY2024",
    )


@pytest.fixture
def financial_chunks(income_statement_chunk, balance_sheet_chunk, cash_flow_chunk):
    return [income_statement_chunk, balance_sheet_chunk, cash_flow_chunk]


@pytest.fixture
def parsed_document(financial_chunks) -> ParsedDocument:
    return ParsedDocument(
        blocks=[Block(text=c.text, page=c.page, kind="table") for c in financial_chunks],
        source_file="test.json",
        doc_id="TESTDOC",
        company="Northwind Systems, Inc.",
        ticker="NWSY",
        doc_type="10-K",
        fiscal_period="FY2024",
    )


@pytest.fixture
def stub_store(financial_chunks) -> InMemoryVectorStore:
    """In-memory store over the fixture chunks, using the deterministic embedder."""
    store = InMemoryVectorStore(HashingEmbedder(), "test-collection")
    store.add_chunks(financial_chunks)
    return store


@pytest.fixture
def thin_filing(tmp_path: Path) -> Path:
    """A filing with prose but no financial statements.

    Exists to prove ``insufficient_evidence`` is reachable (section 7).
    """
    doc = {
        "blocks": [
            {
                "kind": "narrative",
                "page": 1,
                "text": (
                    "FORM 10-K Annual Report. Thin Filings Corp. Trading Symbol: THIN. "
                    "For the fiscal year ended December 31, 2024."
                ),
            },
            {
                "kind": "narrative",
                "page": 5,
                "text": (
                    "The Company continued to operate its business during the period. "
                    "Competition in our markets could adversely affect us. We may not "
                    "be able to execute our strategic plans as intended."
                ),
            },
        ]
    }
    path = tmp_path / "thin_filing.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path

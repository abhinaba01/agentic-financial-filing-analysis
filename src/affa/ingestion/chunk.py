"""Token-bounded, sentence-aware chunking with a hard termination guarantee.

Section 4 calls out the sliding-window infinite loop as an easy bug on documents
longer than one chunk, so this module is built around a single invariant:

    every iteration of the packing loop starts strictly further into the
    sentence list than the previous one.

That is enforced in three places - the config rejects ``overlap >= target`` at
load time, :func:`_next_start` refuses to return a non-advancing index, and the
loop itself carries an iteration ceiling that raises rather than spinning. A
chunker that hangs on a 300-page 10-K looks identical to a slow one until you
attach a debugger, so the failure is made loud instead.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from functools import lru_cache

from affa.config import ChunkConfig
from affa.ingestion.types import Block, Chunk, ParsedDocument

# --- tokenisation ---------------------------------------------------------

# Approximation used when transformers is not installed. Financial prose runs
# around 1.3 sub-word tokens per whitespace token; numbers and tickers push it
# higher, so this over-counts slightly, which is the safe direction for a budget.
_APPROX_TOKENS_PER_WORD = 1.35

TokenCounter = Callable[[str], int]


def approximate_token_count(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, int(len(text.split()) * _APPROX_TOKENS_PER_WORD))


@lru_cache(maxsize=4)
def _hf_tokenizer(name: str):  # pragma: no cover - requires transformers + network
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def make_token_counter(model_name: str | None = None) -> TokenCounter:
    """Return a token counter, preferring the real tokenizer when available.

    Falls back to the word approximation rather than failing: chunking must work
    in a bare CI environment with no model cache and no network.
    """
    if model_name:
        try:  # pragma: no cover - exercised only when transformers is installed
            tok = _hf_tokenizer(model_name)
            return lambda text: len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return approximate_token_count


# --- sentence segmentation ------------------------------------------------

# Abbreviations that end in a period but not a sentence. Splitting on "Inc." or
# "U.S." fragments company names across chunk boundaries.
_ABBREVIATIONS = (
    "inc",
    "corp",
    "co",
    "ltd",
    "llc",
    "lp",
    "plc",
    "no",
    "vs",
    "approx",
    "est",
    "fig",
    "sec",
    "mr",
    "mrs",
    "ms",
    "dr",
    "jr",
    "sr",
    "st",
    "u.s",
    "u.k",
    "e.g",
    "i.e",
    "etc",
    "al",
)
_ABBREV_SET = {a.lower() for a in _ABBREVIATIONS}

_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9$])")
_LAST_WORD_RE = re.compile(r"([A-Za-z.]+)\.[\"')\]]*\s*$")


def regex_sentences(text: str) -> list[str]:
    """Dependency-free sentence splitter that respects filing abbreviations."""
    if not text.strip():
        return []
    pieces = _SENT_BOUNDARY_RE.split(text)
    merged: list[str] = []
    for piece in pieces:
        if merged:
            m = _LAST_WORD_RE.search(merged[-1])
            # "... Apple Inc." + "The company ..." must stay one sentence.
            if m and m.group(1).lower().strip(".") in _ABBREV_SET:
                merged[-1] = f"{merged[-1]} {piece}"
                continue
        merged.append(piece)
    return [s.strip() for s in merged if s.strip()]


@lru_cache(maxsize=2)
def _spacy_pipe(model: str = "en_core_web_sm"):  # pragma: no cover - optional dep
    import spacy

    try:
        return spacy.load(model, disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    except OSError:
        # Model not downloaded. The blank pipeline plus the rule-based sentencizer
        # needs no model weights and still beats the regex on clause boundaries.
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def spacy_sentences(text: str) -> list[str]:  # pragma: no cover - optional dep
    nlp = _spacy_pipe()
    nlp.max_length = max(nlp.max_length, len(text) + 1000)
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


def split_sentences(text: str, splitter: str = "auto") -> list[str]:
    """Segment ``text``. ``auto`` uses spaCy when importable, regex otherwise."""
    if splitter == "regex":
        return regex_sentences(text)
    if splitter in {"spacy", "auto"}:
        try:
            return spacy_sentences(text)
        except Exception:
            if splitter == "spacy":
                raise
            return regex_sentences(text)
    raise ValueError(f"unknown splitter {splitter!r}")


# --- the packing loop -----------------------------------------------------


class ChunkerError(RuntimeError):
    """Raised when the packing loop fails to make progress."""


def _next_start(
    *,
    current_start: int,
    current_end: int,
    sent_tokens: Sequence[int],
    overlap_tokens: int,
) -> int:
    """Index of the first sentence of the next window.

    Walks back from ``current_end`` while the accumulated overlap fits, then
    clamps so the result is strictly greater than ``current_start``. The clamp is
    the guard: without it, a window whose trailing sentence alone exceeds the
    overlap budget rewinds to its own start and the loop never terminates.
    """
    if current_end >= len(sent_tokens):
        return len(sent_tokens)

    acc = 0
    start = current_end
    while start > current_start + 1:
        cost = sent_tokens[start - 1]
        if acc + cost > overlap_tokens:
            break
        acc += cost
        start -= 1

    # Non-negotiable: always advance by at least one sentence.
    return max(start, current_start + 1)


def _hard_split_long_sentence(sentence: str, target_tokens: int, count: TokenCounter) -> list[str]:
    """Split a single oversized sentence on word boundaries.

    Some filing sentences are a whole paragraph of semicolon-joined clauses and
    exceed the window on their own. Splitting by words guarantees progress
    because each piece consumes at least one word.
    """
    words = sentence.split()
    if not words:
        return []
    out: list[str] = []
    buf: list[str] = []
    for word in words:
        buf.append(word)
        if count(" ".join(buf)) >= target_tokens:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    """Stable, content-addressed id.

    Content-addressed rather than positional so re-ingesting an unchanged
    document produces the same ids, and citations in an old report still resolve.
    """
    digest = hashlib.sha1(f"{doc_id}:{index}:{text}".encode()).hexdigest()[:12]
    return f"{doc_id}:{index:04d}:{digest}"


def chunk_block(
    block: Block,
    *,
    doc: ParsedDocument,
    cfg: ChunkConfig,
    count: TokenCounter,
    start_index: int,
) -> list[Chunk]:
    """Chunk one block. Tables are emitted whole when they fit."""
    text = block.text.strip()
    if not text:
        return []

    n_tokens = count(text)

    if block.kind == "table" and cfg.keep_tables_whole and n_tokens <= cfg.max_table_tokens:
        return [
            Chunk(
                chunk_id=_chunk_id(doc.doc_id, start_index, text),
                text=text,
                doc_id=doc.doc_id,
                token_count=n_tokens,
                page=block.page,
                chunk_type="table",
                ticker=doc.ticker,
                fiscal_period=doc.fiscal_period,
                section=block.section,
                table_index=block.table_index,
            )
        ]

    sentences = split_sentences(text, cfg.sentence_splitter)
    if not sentences:
        sentences = [text]

    # Pre-split anything that cannot fit, so every unit is <= the window.
    expanded: list[str] = []
    for s in sentences:
        if count(s) > cfg.target_tokens:
            expanded.extend(_hard_split_long_sentence(s, cfg.target_tokens, count))
        else:
            expanded.append(s)
    sentences = [s for s in expanded if s.strip()]
    if not sentences:
        return []

    sent_tokens = [count(s) for s in sentences]

    chunks: list[Chunk] = []
    start = 0
    index = start_index
    # One iteration must consume at least one sentence, so the loop cannot run
    # more times than there are sentences. Exceeding that means the invariant
    # broke and we raise instead of hanging.
    max_iterations = len(sentences) + 1
    iterations = 0

    while start < len(sentences):
        iterations += 1
        if iterations > max_iterations:
            raise ChunkerError(
                f"chunker failed to advance on doc {doc.doc_id!r} at sentence {start} "
                f"({len(sentences)} sentences, target={cfg.target_tokens}, "
                f"overlap={cfg.overlap_tokens})"
            )

        end = start
        acc = 0
        while end < len(sentences) and acc + sent_tokens[end] <= cfg.target_tokens:
            acc += sent_tokens[end]
            end += 1
        if end == start:  # first sentence alone exceeds the budget
            end = start + 1
            acc = sent_tokens[start]

        piece = " ".join(sentences[start:end]).strip()
        is_last = end >= len(sentences)
        # Drop runt tails, but never drop the only chunk of a block.
        if piece and (acc >= cfg.min_tokens or not is_last or not chunks):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.doc_id, index, piece),
                    text=piece,
                    doc_id=doc.doc_id,
                    token_count=acc,
                    page=block.page,
                    chunk_type=block.kind if block.kind != "heading" else "narrative",
                    ticker=doc.ticker,
                    fiscal_period=doc.fiscal_period,
                    section=block.section,
                    table_index=block.table_index,
                )
            )
            index += 1

        next_start = _next_start(
            current_start=start,
            current_end=end,
            sent_tokens=sent_tokens,
            overlap_tokens=cfg.overlap_tokens,
        )
        assert next_start > start, "chunker window did not advance"  # noqa: S101
        start = next_start

    return chunks


def chunk_document(
    doc: ParsedDocument,
    cfg: ChunkConfig,
    *,
    tokenizer_name: str | None = None,
) -> list[Chunk]:
    """Chunk every block of a parsed document, preserving order and provenance."""
    count = make_token_counter(tokenizer_name)
    out: list[Chunk] = []
    for block in doc.blocks:
        out.extend(chunk_block(block, doc=doc, cfg=cfg, count=count, start_index=len(out)))
    return out

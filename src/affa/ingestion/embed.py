"""Embedding and vector storage.

Section 4 warning, implemented here rather than documented: vectors written by
one embedding model are meaningless when queried by another, and a vector store
answers such a query confidently instead of erroring. Two defences:

1. the collection name is derived from the embedder id, so a different model
   lands in a different collection;
2. the embedder id is written into collection metadata and re-checked on open,
   so an existing collection cannot be silently reused by the wrong model even
   if someone hard-codes the name.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from affa.config import AffaConfig, EmbedderConfig, VectorStoreConfig
from affa.ingestion.types import Chunk


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used by the store and the retriever."""

    name: str

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (stock or fine-tuned from section 5.2)."""

    def __init__(self, cfg: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                'embedding needs sentence-transformers. Install: pip install -e ".[ingest]"'
            ) from exc
        self.cfg = cfg
        self.name = cfg.name
        self._model = SentenceTransformer(cfg.name)

    def _prefix(self, text: str) -> str:
        # BGE ships a query instruction. Section 5.2: use it in training and
        # inference, or in neither - a mismatch is worse than skipping it.
        # configs/default.yaml sets this to null to match training/train_retrieval.py.
        instr = self.cfg.query_instruction
        return f"{instr}{text}" if instr else text

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize_embeddings,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [self._prefix(text)],
            normalize_embeddings=self.cfg.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        return vec.tolist()


class HashingEmbedder:
    """Deterministic hashing embedder. NOT a semantic model.

    Exists so the whole pipeline - ingest, retrieve, verify, report - runs in CI
    and on a laptop with no model download. Its "similarity" is lexical overlap,
    which is enough to exercise the graph and the schema but tells you nothing
    about retrieval quality. Every report produced with it carries a warning, and
    the evaluation harness refuses to benchmark retrieval with it.
    """

    is_stub = True

    def __init__(self, dim: int = 256, name: str = "hashing-stub") -> None:
        self.dim = dim
        self.name = name

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in text.lower().split() if t]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(cfg: AffaConfig, *, allow_stub: bool = True) -> Embedder:
    """Real embedder when sentence-transformers can load it, stub otherwise."""
    # Explicit test/CI switch. Named so it cannot be mistaken for a performance
    # option: reports produced under it carry the stub warning like any other
    # stub run, and the retrieval harness refuses to benchmark with it.
    if os.environ.get("AFFA_FORCE_STUB_EMBEDDER") == "1":
        return HashingEmbedder(name=f"hashing-stub::{cfg.models.embedder.slug}")
    try:
        return SentenceTransformerEmbedder(cfg.models.embedder)
    except Exception as exc:
        if not allow_stub:
            raise
        import warnings

        warnings.warn(
            f"falling back to HashingEmbedder (lexical, not semantic): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return HashingEmbedder(name=f"hashing-stub::{cfg.models.embedder.slug}")


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    similarity: float
    metadata: dict[str, Any]

    @property
    def page(self) -> int | None:
        p = self.metadata.get("page_number")
        return int(p) if p is not None else None

    @property
    def chunk_type(self) -> str:
        return str(self.metadata.get("chunk_type", "narrative"))


class EmbedderMismatchError(RuntimeError):
    """Raised when a collection was written by a different embedding model."""


class InMemoryVectorStore:
    """Dependency-free store with the same contract as the Chroma one."""

    def __init__(self, embedder: Embedder, collection: str) -> None:
        self.embedder = embedder
        self.collection = collection
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._vecs: list[list[float]] = []

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        fresh = [c for c in chunks if c.chunk_id not in set(self._ids)]
        if not fresh:
            return 0
        vecs = self.embedder.encode_documents([c.text for c in fresh])
        for c, v in zip(fresh, vecs, strict=True):
            self._ids.append(c.chunk_id)
            self._texts.append(c.text)
            self._meta.append(c.to_metadata())
            self._vecs.append(v)
        return len(fresh)

    def count(self) -> int:
        return len(self._ids)

    def query(
        self, text: str, *, top_k: int = 8, where: dict[str, Any] | None = None
    ) -> list[Retrieved]:
        if not self._ids:
            return []
        q = self.embedder.encode_query(text)
        scored: list[Retrieved] = []
        for i, vec in enumerate(self._vecs):
            meta = self._meta[i]
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            dot = sum(a * b for a, b in zip(q, vec, strict=True))
            na = math.sqrt(sum(a * a for a in q)) or 1.0
            nb = math.sqrt(sum(b * b for b in vec)) or 1.0
            scored.append(
                Retrieved(
                    chunk_id=self._ids[i],
                    text=self._texts[i],
                    similarity=dot / (na * nb),
                    metadata=meta,
                )
            )
        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:top_k]

    def delete_document(self, doc_id: str) -> None:
        keep = [i for i, m in enumerate(self._meta) if m.get("doc_id") != doc_id]
        self._ids = [self._ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]
        self._vecs = [self._vecs[i] for i in keep]


class ChromaVectorStore:
    """Persistent ChromaDB store, namespaced by embedder."""

    def __init__(self, embedder: Embedder, cfg: VectorStoreConfig, collection: str) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                'ChromaDB not installed. Install: pip install -e ".[ingest]"'
            ) from exc
        self.embedder = embedder
        self.collection_name = collection
        self._client = chromadb.PersistentClient(path=cfg.persist_dir)
        expected = getattr(embedder, "name", "unknown")
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": cfg.distance, "affa_embedder": expected},
        )
        # Defence #2: even a hard-coded collection name cannot cross models.
        found = (self._col.metadata or {}).get("affa_embedder")
        if found and found != expected:
            raise EmbedderMismatchError(
                f"collection {collection!r} was written by embedder {found!r} but the "
                f"active embedder is {expected!r}. Cosine similarity across models is "
                "meaningless and the store will not error on its own. Re-index with "
                "scripts/reindex.py, or point vector_store.collection_prefix elsewhere."
            )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        existing = set()
        try:
            got = self._col.get(ids=[c.chunk_id for c in chunks], include=[])
            existing = set(got.get("ids") or [])
        except Exception:
            pass
        fresh = [c for c in chunks if c.chunk_id not in existing]
        if not fresh:
            return 0
        self._col.add(
            ids=[c.chunk_id for c in fresh],
            documents=[c.text for c in fresh],
            embeddings=self.embedder.encode_documents([c.text for c in fresh]),
            metadatas=[c.to_metadata() for c in fresh],
        )
        return len(fresh)

    def count(self) -> int:
        return int(self._col.count())

    def query(
        self, text: str, *, top_k: int = 8, where: dict[str, Any] | None = None
    ) -> list[Retrieved]:
        if self.count() == 0:
            return []
        # Chroma needs an explicit $and for multi-key filters.
        flt: dict[str, Any] | None = None
        if where:
            clauses = [{k: {"$eq": v}} for k, v in where.items()]
            flt = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._col.query(
            query_embeddings=[self.embedder.encode_query(text)],
            n_results=min(top_k, self.count()),
            where=flt,
            include=["documents", "metadatas", "distances"],
        )
        out: list[Retrieved] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            out.append(
                Retrieved(
                    chunk_id=cid,
                    text=doc,
                    # Chroma reports cosine *distance*; similarity is 1 - d.
                    similarity=1.0 - float(dist),
                    metadata=dict(meta or {}),
                )
            )
        return out

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})


VectorStore = InMemoryVectorStore | ChromaVectorStore


def build_vector_store(
    cfg: AffaConfig, embedder: Embedder, *, in_memory: bool = False
) -> VectorStore:
    """Construct the configured store, falling back to memory if Chroma is absent."""
    collection = cfg.vector_store.collection_name(cfg.models.embedder)
    if getattr(embedder, "is_stub", False):
        # A stub embedder must never contaminate the persistent collection that a
        # real model wrote, so it is confined to memory.
        return InMemoryVectorStore(embedder, collection + "__stub")
    if in_memory or cfg.vector_store.provider != "chroma":
        return InMemoryVectorStore(embedder, collection)
    try:
        return ChromaVectorStore(embedder, cfg.vector_store, collection)
    except ImportError:
        return InMemoryVectorStore(embedder, collection)

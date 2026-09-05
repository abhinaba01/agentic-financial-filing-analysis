"""Configuration loading.

The YAML in ``configs/`` is live: every value here is read by running code.
Anti-pattern #6 in the build spec is a config file the docs treat as authoritative
that nothing ever loads, so ``tests/test_config.py`` asserts that each key in
``configs/default.yaml`` is reachable through this module's dataclasses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Directory containing ``configs/``.

    Resolved from this file rather than the CWD so the API, the Streamlit UI and
    pytest all find the same configuration regardless of where they are launched.
    """
    return Path(__file__).resolve().parents[2]


DEFAULT_CONFIG_PATH = repo_root() / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or self-contradictory."""


def _require(mapping: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required config key {ctx}.{key}")
    return mapping[key]


@dataclass(frozen=True)
class EmbedderConfig:
    name: str
    query_instruction: str | None
    normalize_embeddings: bool
    batch_size: int

    @property
    def slug(self) -> str:
        """Filesystem/collection-safe identifier for this exact embedder.

        Used to namespace the vector collection. Two different embedders must
        never share a collection: cosine similarity between vectors written by
        different models is meaningless, and Chroma returns it without erroring.
        """
        return self.name.replace("/", "__").replace(".", "_")


@dataclass(frozen=True)
class TaggerConfig:
    name: str
    finetuned: str | None
    max_length: int
    enabled: bool

    @property
    def active_name(self) -> str:
        return self.finetuned or self.name


@dataclass(frozen=True)
class ReasonerConfig:
    backend: str
    local_name: str
    local_adapter: str | None
    hosted_provider: str
    hosted_name: str
    max_new_tokens: int
    temperature: float

    def __post_init__(self) -> None:
        allowed = {"stub", "local", "hosted"}
        if self.backend not in allowed:
            raise ConfigError(
                f"models.reasoner.backend must be one of {sorted(allowed)}, got {self.backend!r}"
            )

    @property
    def active_name(self) -> str:
        if self.backend == "hosted":
            return self.hosted_name
        if self.backend == "local":
            return self.local_adapter or self.local_name
        return "stub"


@dataclass(frozen=True)
class ModelsConfig:
    embedder: EmbedderConfig
    xbrl_tagger: TaggerConfig
    sentiment: TaggerConfig
    reasoner: ReasonerConfig


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int
    overlap_tokens: int
    min_tokens: int
    keep_tables_whole: bool
    max_table_tokens: int
    sentence_splitter: str

    def __post_init__(self) -> None:
        # A chunker whose overlap is not strictly smaller than its window cannot
        # advance: each step would rewind to at or before its own start. This is
        # the infinite-loop guard from section 4, enforced before any document is read.
        if self.overlap_tokens >= self.target_tokens:
            raise ConfigError(
                "ingestion.chunk.overlap_tokens must be < target_tokens "
                f"(got overlap={self.overlap_tokens}, target={self.target_tokens}); "
                "otherwise the sliding window never advances"
            )
        if self.target_tokens <= 0:
            raise ConfigError("ingestion.chunk.target_tokens must be positive")
        if self.sentence_splitter not in {"auto", "spacy", "regex"}:
            raise ConfigError(
                "ingestion.chunk.sentence_splitter must be auto|spacy|regex, "
                f"got {self.sentence_splitter!r}"
            )


@dataclass(frozen=True)
class CleanConfig:
    normalize_unicode: bool
    dehyphenate: bool


@dataclass(frozen=True)
class ParseConfig:
    extract_tables: bool
    max_pages: int | None


@dataclass(frozen=True)
class IngestionConfig:
    chunk: ChunkConfig
    clean: CleanConfig
    parse: ParseConfig


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str
    persist_dir: str
    collection_prefix: str
    distance: str

    def collection_name(self, embedder: EmbedderConfig) -> str:
        """Collection name bound to the embedder that wrote it (section 4)."""
        return f"{self.collection_prefix}__{embedder.slug}"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    min_similarity: float
    mmr_lambda: float
    filter_by_ticker: bool


@dataclass(frozen=True)
class RoutingConfig:
    retry_below_mean_similarity: float
    min_chunks_for_sufficiency: int
    max_retrieval_attempts: int


@dataclass(frozen=True)
class VerificationConfig:
    numeric_tolerance_pct: float
    min_entity_overlap: float
    drop_unsupported_claims: bool
    keep_contradicted_as_flagged: bool


@dataclass(frozen=True)
class KpiConfig:
    percent_canonical: str
    percent_ambiguity_band: tuple[float, float]
    default_scale: str
    tolerance_pct: float
    require_provenance: bool


@dataclass(frozen=True)
class RecommendationConfig:
    rubric_file: str

    def rubric_path(self) -> Path:
        p = Path(self.rubric_file)
        return p if p.is_absolute() else repo_root() / p


@dataclass(frozen=True)
class ReportConfig:
    disclaimer: str
    include_chain_of_thought: bool


@dataclass(frozen=True)
class AffaConfig:
    pipeline_version: str
    seed: int
    models: ModelsConfig
    ingestion: IngestionConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    routing: RoutingConfig
    verification: VerificationConfig
    kpi: KpiConfig
    recommendation: RecommendationConfig
    report: ReportConfig
    source_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        validate_threshold_reachability(
            min_similarity=self.retrieval.min_similarity,
            retry_below=self.routing.retry_below_mean_similarity,
        )
        if self.routing.max_retrieval_attempts < 1:
            raise ConfigError("routing.max_retrieval_attempts must be >= 1")


def validate_threshold_reachability(*, min_similarity: float, retry_below: float) -> None:
    """Enforce the reachability invariant from section 3 / anti-pattern #1.

    Retrieval discards every chunk scoring below ``min_similarity`` (f). The mean
    similarity of what survives is therefore always >= f. A routing rule of
    "retry when mean similarity < t" can only ever fire when ``t > f``; with
    ``t <= f`` the retry branch is dead code that still looks live in the diagram.

    Called from :class:`AffaConfig` construction and, at import time, from
    :mod:`affa.agent.routing` against the shipped defaults.
    """
    if not 0.0 <= min_similarity <= 1.0:
        raise ConfigError(f"retrieval.min_similarity must be in [0,1], got {min_similarity}")
    if not 0.0 <= retry_below <= 1.0:
        raise ConfigError(
            f"routing.retry_below_mean_similarity must be in [0,1], got {retry_below}"
        )
    if retry_below <= min_similarity:
        raise ConfigError(
            "unreachable routing threshold: routing.retry_below_mean_similarity "
            f"({retry_below}) must be strictly greater than retrieval.min_similarity "
            f"({min_similarity}). Retrieval already discards everything below the "
            "floor, so the retry branch could never fire."
        )


def _build(raw: dict[str, Any], source: Path | None) -> AffaConfig:
    models_raw = _require(raw, "models", "root")
    emb = _require(models_raw, "embedder", "models")
    tag = _require(models_raw, "xbrl_tagger", "models")
    sen = _require(models_raw, "sentiment", "models")
    rea = _require(models_raw, "reasoner", "models")
    ing = _require(raw, "ingestion", "root")
    kpi_raw = _require(raw, "kpi", "root")
    band = tuple(_require(kpi_raw, "percent_ambiguity_band", "kpi"))
    if len(band) != 2:
        raise ConfigError("kpi.percent_ambiguity_band must be a [low, high] pair")

    return AffaConfig(
        pipeline_version=str(_require(raw, "pipeline_version", "root")),
        seed=int(_require(raw, "seed", "root")),
        models=ModelsConfig(
            embedder=EmbedderConfig(
                name=emb["name"],
                query_instruction=emb.get("query_instruction"),
                normalize_embeddings=bool(emb["normalize_embeddings"]),
                batch_size=int(emb["batch_size"]),
            ),
            xbrl_tagger=TaggerConfig(
                name=tag["name"],
                finetuned=tag.get("finetuned"),
                max_length=int(tag["max_length"]),
                enabled=bool(tag["enabled"]),
            ),
            sentiment=TaggerConfig(
                name=sen["name"],
                finetuned=sen.get("finetuned"),
                max_length=int(sen["max_length"]),
                enabled=bool(sen["enabled"]),
            ),
            reasoner=ReasonerConfig(
                backend=rea["backend"],
                local_name=rea["local_name"],
                local_adapter=rea.get("local_adapter"),
                hosted_provider=rea["hosted_provider"],
                hosted_name=rea["hosted_name"],
                max_new_tokens=int(rea["max_new_tokens"]),
                temperature=float(rea["temperature"]),
            ),
        ),
        ingestion=IngestionConfig(
            chunk=ChunkConfig(**ing["chunk"]),
            clean=CleanConfig(**ing["clean"]),
            parse=ParseConfig(**ing["parse"]),
        ),
        vector_store=VectorStoreConfig(**_require(raw, "vector_store", "root")),
        retrieval=RetrievalConfig(**_require(raw, "retrieval", "root")),
        routing=RoutingConfig(**_require(raw, "routing", "root")),
        verification=VerificationConfig(**_require(raw, "verification", "root")),
        kpi=KpiConfig(
            percent_canonical=kpi_raw["percent_canonical"],
            percent_ambiguity_band=(float(band[0]), float(band[1])),
            default_scale=kpi_raw["default_scale"],
            tolerance_pct=float(kpi_raw["tolerance_pct"]),
            require_provenance=bool(kpi_raw["require_provenance"]),
        ),
        recommendation=RecommendationConfig(**_require(raw, "recommendation", "root")),
        report=ReportConfig(**_require(raw, "report", "root")),
        source_path=source,
    )


def _resolve_path(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("AFFA_CONFIG", DEFAULT_CONFIG_PATH))


def load_config(path: str | os.PathLike[str] | None = None) -> AffaConfig:
    """Load and validate configuration.

    Resolution order: explicit ``path`` argument, then ``$AFFA_CONFIG``, then
    ``configs/default.yaml``.
    """
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    try:
        return _build(raw, resolved)
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"malformed config {resolved}: {exc}") from exc


@lru_cache(maxsize=4)
def _cached(path_str: str) -> AffaConfig:
    return load_config(path_str)


def get_config(path: str | os.PathLike[str] | None = None) -> AffaConfig:
    """Cached :func:`load_config` for hot paths (graph nodes, request handlers)."""
    return _cached(str(_resolve_path(path)))


def load_rubric(cfg: AffaConfig | None = None) -> dict[str, Any]:
    """Load the versioned rubric YAML referenced by the active config."""
    cfg = cfg or get_config()
    path = cfg.recommendation.rubric_path()
    if not path.is_file():
        raise ConfigError(f"rubric file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        rubric = yaml.safe_load(fh)
    if not isinstance(rubric, dict) or "version" not in rubric:
        raise ConfigError(f"rubric {path} must be a mapping with a version key")
    return rubric

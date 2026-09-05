"""XBRL numeric tagging: the model side of KPI extraction (section 5.1).

Wraps the fine-tuned FiNER-139 token classifier so it answers one question the
rule-based extractor cannot: *is this particular number a ``Revenues`` or a
``NetIncomeLoss``?* Regex matches a label to a nearby figure; the tagger reads
the number in context.

The class is import-safe without torch. ``load()`` returns ``False`` when the
model is unavailable and the pipeline continues rule-based only, recording that
fact in the report rather than pretending a model ran.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from affa.config import TaggerConfig
from affa.ingestion.types import Chunk
from affa.kpi.catalog import CONCEPT_TO_METRIC
from affa.kpi.units import detect_scale, parse_financial_number
from affa.schema import ExtractionMethod, Scale

log = logging.getLogger(__name__)

# FiNER-139 tags numbers, so only numeric tokens can carry a concept.
_NUMERIC_TOKEN_RE = re.compile(r"\d")


@dataclass
class TagHit:
    """A number the tagger assigned an XBRL concept to."""

    metric: str
    concept: str
    value: float
    scale: Scale
    chunk_id: str
    page: int | None
    raw_text: str
    confidence: float
    method: ExtractionMethod = ExtractionMethod.XBRL_MODEL


class XBRLTagger:
    """Token-classification wrapper over ``nlpaueb/sec-bert-base`` fine-tuned on FiNER-139."""

    def __init__(self, cfg: TaggerConfig) -> None:
        self.cfg = cfg
        self._pipe = None
        self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self.cfg.active_name if self._loaded else "rule_based_only"

    def load(self) -> bool:
        """Load the model. Returns False (never raises) if it cannot be loaded."""
        if self._loaded:
            return True
        if not self.cfg.enabled:
            log.info("XBRL tagger disabled in config; using rule-based extraction only")
            return False
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            name = self.cfg.active_name
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForTokenClassification.from_pretrained(name)
            self._pipe = pipeline(
                "token-classification",
                model=model,
                tokenizer=tok,
                aggregation_strategy="simple",
            )
            self._loaded = True
            log.info("XBRL tagger loaded: %s", name)
            return True
        except Exception as exc:
            log.warning("XBRL tagger unavailable (%s); falling back to rule-based extraction", exc)
            self._loaded = False
            return False

    def tag_chunk(self, chunk: Chunk) -> list[TagHit]:
        """Tag numbers in one chunk. Empty list when the model is not loaded."""
        if not self._loaded or self._pipe is None:
            return []
        text = chunk.text[: self.cfg.max_length * 6]  # rough char budget for max_length
        try:
            spans = self._pipe(text)
        except Exception as exc:  # pragma: no cover - runtime robustness
            log.warning("tagging failed on chunk %s: %s", chunk.chunk_id, exc)
            return []

        scale = detect_scale(chunk.text, Scale.UNITS)
        hits: list[TagHit] = []
        for span in spans:
            group = str(span.get("entity_group") or span.get("entity") or "")
            concept = group.split("-", 1)[-1] if "-" in group else group
            metric = CONCEPT_TO_METRIC.get(concept)
            if metric is None:
                continue
            word = str(span.get("word", "")).strip()
            if not _NUMERIC_TOKEN_RE.search(word):
                continue
            parsed = parse_financial_number(word, default_scale=scale)
            if parsed is None:
                continue
            # Re-read the sign from the surrounding text: aggregation strips the
            # parentheses that make a filing figure negative.
            start, end = int(span.get("start", 0)), int(span.get("end", 0))
            context = text[max(0, start - 2) : min(len(text), end + 2)]
            recontext = parse_financial_number(context, default_scale=scale)
            value = recontext.value if recontext is not None else parsed.value
            hits.append(
                TagHit(
                    metric=metric,
                    concept=concept,
                    value=value,
                    scale=parsed.scale_hint or scale,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    raw_text=text[max(0, start - 60) : min(len(text), end + 60)],
                    confidence=float(span.get("score", 0.5)),
                )
            )
        return hits

    def tag_chunks(self, chunks: list[Chunk]) -> list[TagHit]:
        out: list[TagHit] = []
        for chunk in chunks:
            out.extend(self.tag_chunk(chunk))
        return out

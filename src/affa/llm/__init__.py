"""LLM backend registry. Backend choice is a config flag, never a code change."""

from __future__ import annotations

import logging

from affa.config import AffaConfig, ReasonerConfig
from affa.llm.backends import HostedLLM, LocalLLM, StubLLM
from affa.llm.base import LLMClient, LLMResponse, extract_json

log = logging.getLogger(__name__)

__all__ = [
    "HostedLLM",
    "LLMClient",
    "LLMResponse",
    "LocalLLM",
    "StubLLM",
    "build_llm",
    "extract_json",
]

BACKENDS = {"stub": StubLLM, "local": LocalLLM, "hosted": HostedLLM}


def build_llm(cfg: AffaConfig | ReasonerConfig, *, allow_stub_fallback: bool = True) -> LLMClient:
    """Construct the configured LLM backend.

    Falls back to :class:`StubLLM` when the requested backend cannot be built
    (no GPU, no API key, no adapter). The fallback is loud - the caller can check
    ``is_stub`` and the report records which model actually ran, so a run that
    silently degraded is never reported as a run of the model you asked for.
    """
    rcfg = cfg.models.reasoner if isinstance(cfg, AffaConfig) else cfg
    if rcfg.backend == "stub":
        return StubLLM()
    try:
        return BACKENDS[rcfg.backend](rcfg)  # type: ignore[operator]
    except Exception as exc:
        if not allow_stub_fallback:
            raise
        log.warning(
            "reasoner backend %r unavailable (%s); falling back to the stub. "
            "Narrative text and LLM-authored findings will be omitted.",
            rcfg.backend,
            exc,
        )
        return StubLLM(name=f"stub::{rcfg.backend}-unavailable")

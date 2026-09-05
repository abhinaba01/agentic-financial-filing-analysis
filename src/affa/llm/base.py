"""One LLM interface, several backends.

Section 5.4 requires that a local fine-tuned model and a hosted API model sit
behind the same interface, selectable by config, so the three-way comparison
(base zero-shot / QLoRA / hosted) is a flag rather than a rewrite. Everything
downstream - the reasoning node, the narrative writer, the FinQA harness -
depends only on :class:`LLMClient`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: Any = field(default=None, repr=False)


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_SPAN_RE = re.compile(r"[\[{].*[\]}]", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Pull the first JSON value out of a model response.

    Models wrap JSON in prose and fences no matter how the prompt is worded, so
    parsing is best-effort by design. Returning ``None`` on failure is
    load-bearing: the caller degrades to a deterministic path instead of
    inventing findings, which keeps an unparseable response from becoming an
    uncited claim in the report.
    """
    if not text:
        return None
    for candidate in (m.group(1) for m in _JSON_BLOCK_RE.finditer(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    if m := _JSON_SPAN_RE.search(text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None

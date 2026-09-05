"""Concrete LLM backends: stub, local (QLoRA-capable) and hosted."""

from __future__ import annotations

import logging
import os

from affa.config import ReasonerConfig
from affa.llm.base import LLMResponse

log = logging.getLogger(__name__)


class StubLLM:
    """Deterministic, offline backend.

    Not a language model and does not pretend to be one. It exists so the graph,
    the schema and the tests run with no weights and no API key. It returns an
    empty JSON array for structured requests, which makes callers fall back to
    their deterministic path rather than receive invented content - a stub that
    fabricated plausible findings would quietly corrupt the faithfulness metric.
    """

    is_stub = True

    def __init__(self, name: str = "stub") -> None:
        self.name = name

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        lowered = prompt.lower()
        if "json" in lowered:
            return LLMResponse(text="[]", model=self.name)
        return LLMResponse(text="", model=self.name)


class LocalLLM:
    """Local HF causal LM, optionally with a QLoRA adapter from section 5.4."""

    def __init__(self, cfg: ReasonerConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                'local reasoner needs torch + transformers. Install: pip install -e ".[train]"'
            ) from exc

        self.cfg = cfg
        self.name = cfg.active_name
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(cfg.local_name)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.local_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if cfg.local_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, cfg.local_adapter)
            log.info("loaded QLoRA adapter %s", cfg.local_adapter)
        self._model = model.eval()

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = (f"{system}\n\n" if system else "") + prompt

        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                # temperature=0 means greedy; passing do_sample=True with it is a
                # transformers warning and a source of irreproducible eval runs.
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        completion = self._tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return LLMResponse(
            text=completion,
            model=self.name,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(out.shape[1] - inputs["input_ids"].shape[1]),
        )


class HostedLLM:
    """Hosted API backend (Anthropic or OpenAI), for the frontier baseline."""

    def __init__(self, cfg: ReasonerConfig) -> None:
        self.cfg = cfg
        self.name = cfg.hosted_name
        self.provider = cfg.hosted_provider
        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    'hosted reasoner needs the anthropic SDK. Install: pip install -e ".[hosted]"'
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic()
        elif self.provider == "openai":
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    'hosted reasoner needs the openai SDK. Install: pip install -e ".[hosted]"'
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = openai.OpenAI()
        else:
            raise ValueError(f"unknown hosted provider {self.provider!r}")

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.cfg.hosted_name,
                max_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return LLMResponse(
                text=text,
                model=self.cfg.hosted_name,
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
                raw=resp,
            )

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = self._client.chat.completions.create(
            model=self.cfg.hosted_name,
            messages=messages,
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.cfg.hosted_name,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
            completion_tokens=resp.usage.completion_tokens if resp.usage else None,
            raw=resp,
        )

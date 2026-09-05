"""Numerical-reasoning evaluation on FinQA (sections 5.4 and 9).

The interesting result here is a **three-way** comparison on identical prompts:
base model zero-shot, the QLoRA fine-tune, and a hosted frontier model. "Reached
X% of the hosted model's execution accuracy at zero marginal cost" is a strong
and honest claim; "our model scored X%" on its own is not.

Execution accuracy compares the *executed* answer, not the program string: there
are many correct programs for one answer, and string matching would understate
every model equally but unpredictably.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict

from affa.config import get_config
from affa.eval.harness import EvaluationResult, Evaluator
from affa.llm import build_llm
from affa.llm.backends import HostedLLM, LocalLLM, StubLLM

log = logging.getLogger(__name__)

PROMPT = """Answer the question using the financial data below. Show only the final numeric answer.

{context}

Question: {question}
Answer:"""

_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


class FinQAEvaluator(Evaluator):
    """Execution accuracy on FinQA, three ways."""

    name = "finqa"
    default_dataset = "ibm/finqa"
    default_baseline = "base_zero_shot"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--tolerance", type=float, default=1.0, help="Relative tolerance %%")
        parser.add_argument(
            "--hosted",
            action="store_true",
            help="Also measure the hosted model (costs API calls)",
        )

    def run(self, args: argparse.Namespace) -> EvaluationResult:
        cfg = get_config(args.config) if args.config else get_config()
        examples, total = _load_finqa(args.test_set, args.limit, args.seed)

        model = build_llm(cfg, allow_stub_fallback=True)
        notes: list[str] = ["prompts are identical across all systems compared"]

        if isinstance(model, StubLLM):
            raise SystemExit(
                "the reasoner is the stub backend, which answers nothing. Set "
                "models.reasoner.backend to 'local' or 'hosted' in the config "
                "before running the FinQA harness."
            )

        metrics, by_operator = _score(model, examples, args.tolerance)

        # Baseline: the SAME base model without the adapter, so the delta is the
        # fine-tune and nothing else.
        import dataclasses

        base_cfg = dataclasses.replace(cfg.models.reasoner, backend="local", local_adapter=None)
        try:
            base_model = LocalLLM(base_cfg)
            baseline_metrics, _ = _score(base_model, examples, args.tolerance)
            baseline_name = f"{base_cfg.local_name} (zero-shot, no adapter)"
            baseline_absent = None
        except Exception as exc:
            baseline_metrics = {}
            baseline_name = "unavailable"
            baseline_absent = f"base model could not be loaded for the zero-shot baseline: {exc}"

        if args.hosted:
            hosted_cfg = dataclasses.replace(cfg.models.reasoner, backend="hosted")
            try:
                hosted = HostedLLM(hosted_cfg)
                hosted_metrics, _ = _score(hosted, examples, args.tolerance)
                notes.append(
                    f"hosted {hosted_cfg.hosted_name}: "
                    + ", ".join(f"{k}={v}" for k, v in hosted_metrics.items())
                )
                exec_acc = metrics.get("execution_accuracy", 0.0)
                hosted_acc = hosted_metrics.get("execution_accuracy", 0.0)
                if hosted_acc:
                    notes.append(
                        f"local model reached {exec_acc / hosted_acc:.1%} of the hosted "
                        "model's execution accuracy on identical prompts"
                    )
            except Exception as exc:
                notes.append(f"hosted comparison skipped: {exc}")

        for operator, stats in sorted(by_operator.items()):
            if stats["n"]:
                notes.append(
                    f"per-operator {operator}: {stats['correct']}/{stats['n']} "
                    f"= {stats['correct'] / stats['n']:.3f}"
                )

        if args.limit:
            notes.append(
                f"SUBSET: {len(examples)} of {total} test examples; only the "
                "model-vs-baseline delta is comparable, not the absolute number"
            )

        return EvaluationResult(
            component="finqa",
            dataset=args.test_set or self.default_dataset,
            split="test",
            n_examples=len(examples),
            metrics=metrics,
            baseline_name=args.baseline or baseline_name,
            baseline_metrics=baseline_metrics,
            model_name=model.name,
            notes=notes,
            subset_of=total,
            seed=args.seed,
            baseline_absent_reason=baseline_absent,
        )


def _load_finqa(test_set: str | None, limit: int | None, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError('needs `datasets`. Install: pip install -e ".[eval]"') from exc

    name = test_set or "ibm/finqa"
    ds = load_dataset(name, trust_remote_code=True)
    split = ds["test"] if "test" in ds else ds["validation"]
    total = len(split)
    if limit:
        split = split.shuffle(seed=seed).select(range(min(limit, total)))
    return list(split), total


def _context_of(row: dict) -> str:
    parts = [
        " ".join(row.get("pre_text", []) or []),
        _render_table(row.get("table", [])),
        " ".join(row.get("post_text", []) or []),
    ]
    return "\n".join(p for p in parts if p)[:6000]


def _render_table(table) -> str:
    if not table:
        return ""
    return "\n".join(" | ".join(str(c) for c in row) for row in table)


def _gold_answer(row: dict) -> str:
    return str(row.get("answer") or row.get("final_result") or "").strip()


def _operators(row: dict) -> list[str]:
    """Operations the gold program uses, for the per-operator breakdown."""
    program = str(row.get("program") or row.get("program_re") or "")
    return sorted(
        set(re.findall(r"\b(add|subtract|multiply|divide|exp|greater|table_\w+)\(", program))
    ) or ["unknown"]


def _to_float(text: str) -> float | None:
    match = _NUM_RE.search(text.replace(" ", ""))
    if not match:
        return None
    raw = match.group(0)
    is_pct = raw.endswith("%")
    raw = raw.replace("$", "").replace(",", "").replace("%", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    # FinQA answers appear as both "14.1%" and "0.141"; normalise to the
    # fraction so the two conventions do not count as disagreement.
    return value / 100.0 if is_pct else value


def _score(model, examples: list[dict], tolerance_pct: float):
    correct = 0
    by_operator: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})

    for row in examples:
        prompt = PROMPT.format(context=_context_of(row), question=row.get("question", ""))
        try:
            response = model.complete(prompt)
            predicted = _to_float(response.text)
        except Exception as exc:  # pragma: no cover - runtime robustness
            log.warning("generation failed: %s", exc)
            predicted = None

        gold = _to_float(_gold_answer(row))
        # Both conventions accepted, because the dataset mixes them; this is
        # convention handling, not ground-truth rescaling.
        hit = False
        if predicted is not None and gold is not None:
            for candidate in (predicted, predicted / 100.0, predicted * 100.0):
                if gold == 0:
                    hit = hit or abs(candidate) < 1e-9
                elif abs(candidate - gold) / abs(gold) * 100.0 <= tolerance_pct:
                    hit = True
        correct += int(hit)
        for operator in _operators(row):
            by_operator[operator]["n"] += 1
            by_operator[operator]["correct"] += int(hit)

    n = max(len(examples), 1)
    return {"execution_accuracy": round(correct / n, 4)}, dict(by_operator)

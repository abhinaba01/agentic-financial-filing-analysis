"""Shared evaluation harness (section 9).

One interface for every component, and one rule enforced structurally rather
than by discipline: **no metric is emitted without a baseline measured on the
same data with the same protocol.** :class:`EvaluationResult` cannot be
serialised with an empty ``baseline`` unless the evaluator explicitly declares
why one does not exist, and published figures from papers live in a separate
``literature`` field that the renderer prints under its own heading.

Every harness accepts the same flags::

    --test-set   dataset id or path
    --output     where to write the JSON result
    --limit      cap the number of examples (records the subset size)
    --run-agent  evaluate through the full agent graph rather than the component
    --baseline   which baseline to measure against

``--limit`` is not cosmetic. Sampling makes absolute numbers incomparable to
published figures, so the result records ``subset_of`` and the renderer prints a
warning next to any sampled number.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from affa import __version__


@dataclass
class LiteratureReference:
    """A number from a paper. Never mixed with measured results (anti-pattern #9)."""

    source: str
    metric: str
    value: float
    conditions: str


@dataclass
class EvaluationResult:
    component: str
    dataset: str
    split: str
    n_examples: int
    metrics: dict[str, float]
    baseline_name: str
    baseline_metrics: dict[str, float]
    model_name: str
    notes: list[str] = field(default_factory=list)
    literature: list[LiteratureReference] = field(default_factory=list)
    subset_of: int | None = None
    seed: int = 42
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    affa_version: str = __version__
    python_version: str = field(default_factory=lambda: platform.python_version())
    baseline_absent_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.baseline_metrics and not self.baseline_absent_reason:
            raise ValueError(
                f"{self.component}: a metric without a baseline measured the same way is "
                "not a result. Provide baseline_metrics, or set baseline_absent_reason "
                "to state explicitly why no baseline exists."
            )

    @property
    def is_sampled(self) -> bool:
        return self.subset_of is not None and self.n_examples < self.subset_of

    def deltas(self) -> dict[str, float]:
        """Model minus baseline, for every metric both of them report."""
        return {
            k: round(v - self.baseline_metrics[k], 6)
            for k, v in self.metrics.items()
            if k in self.baseline_metrics
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["deltas"] = self.deltas()
        payload["is_sampled"] = self.is_sampled
        return payload

    def render(self) -> str:
        lines = [
            f"=== {self.component} on {self.dataset} [{self.split}] ===",
            f"model:    {self.model_name}",
            f"baseline: {self.baseline_name}",
            f"examples: {self.n_examples}"
            + (f" (sampled from {self.subset_of})" if self.is_sampled else ""),
            "",
        ]
        width = max((len(k) for k in self.metrics), default=10)
        lines.append(f"{'metric'.ljust(width)}  {'model':>10}  {'baseline':>10}  {'delta':>10}")
        for key, value in self.metrics.items():
            base = self.baseline_metrics.get(key)
            delta = f"{value - base:+.4f}" if base is not None else "-"
            base_str = f"{base:.4f}" if base is not None else "-"
            lines.append(f"{key.ljust(width)}  {value:>10.4f}  {base_str:>10}  {delta:>10}")

        if self.baseline_absent_reason:
            lines += ["", f"no baseline: {self.baseline_absent_reason}"]

        if self.is_sampled:
            lines += [
                "",
                "NOTE: these are subset numbers. They are comparable to the baseline row "
                "above (same data, same protocol) and NOT to published figures.",
            ]

        if self.notes:
            lines += ["", "notes:"] + [f"  - {n}" for n in self.notes]

        if self.literature:
            lines += [
                "",
                "--- published figures (NOT produced by this repo, different conditions) ---",
            ]
            for ref in self.literature:
                lines.append(f"  {ref.source}: {ref.metric} = {ref.value} ({ref.conditions})")
        return "\n".join(lines)


class Evaluator(ABC):
    """Base class for every component evaluator."""

    name: str = "evaluator"
    default_dataset: str = ""
    default_baseline: str = ""

    @abstractmethod
    def run(self, args: argparse.Namespace) -> EvaluationResult: ...

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:  # noqa: B027
        """Optional hook for component-specific flags.

        Deliberately concrete and empty rather than abstract: most evaluators
        need no extra flags, and forcing every one to define a no-op override
        would be noise.
        """


def base_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """The flag set every harness shares."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--test-set", default=None, help="Dataset id or path to evaluate on")
    parser.add_argument("--output", default=None, help="Write the JSON result here")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of examples. Sampled numbers are not comparable to published figures.",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Evaluate through the full agent graph instead of the component in isolation",
    )
    parser.add_argument("--baseline", default=None, help="Baseline to measure against")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser


def emit(result: EvaluationResult, output: str | None) -> None:
    print(result.render())
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {path}", file=sys.stderr)


# --- registry -------------------------------------------------------------

COMPONENTS = (
    "xbrl",
    "retrieval",
    "sentiment",
    "kpi",
    "finqa",
    "faithfulness",
    "recommendation",
)


def _load(component: str) -> Evaluator:
    """Import evaluators lazily; several need optional heavy dependencies."""
    if component == "xbrl":
        from affa.eval.xbrl_eval import XBRLEvaluator

        return XBRLEvaluator()
    if component == "retrieval":
        from affa.eval.retrieval_eval import RetrievalEvaluator

        return RetrievalEvaluator()
    if component == "sentiment":
        from affa.eval.sentiment_eval import SentimentEvaluator

        return SentimentEvaluator()
    if component == "kpi":
        from affa.eval.kpi_eval import KPIEvaluator

        return KPIEvaluator()
    if component == "finqa":
        from affa.eval.finqa_eval import FinQAEvaluator

        return FinQAEvaluator()
    if component == "faithfulness":
        from affa.eval.faithfulness_eval import FaithfulnessEvaluator

        return FaithfulnessEvaluator()
    if component == "recommendation":
        from affa.eval.recommendation_eval import RecommendationEvaluator

        return RecommendationEvaluator()
    raise ValueError(f"unknown component {component!r}; choose from {list(COMPONENTS)}")


def main(argv: list[str] | None = None) -> int:
    """``affa-eval <component> [flags]``"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: affa-eval {{{','.join(COMPONENTS)}}} [--test-set ...] [--output ...]")
        print("       [--limit N] [--run-agent] [--baseline NAME] [--config PATH] [--seed N]")
        return 0

    component, rest = argv[0], argv[1:]
    evaluator = _load(component)
    parser = base_parser(f"affa-eval {component}", evaluator.__doc__ or component)
    evaluator.add_arguments(parser)
    args = parser.parse_args(rest)

    result = evaluator.run(args)
    emit(result, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

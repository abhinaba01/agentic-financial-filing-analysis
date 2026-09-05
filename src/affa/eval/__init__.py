"""Evaluation harnesses. One interface, a measured baseline for every metric."""

from affa.eval.harness import (
    COMPONENTS,
    EvaluationResult,
    Evaluator,
    LiteratureReference,
    base_parser,
    emit,
    main,
)

__all__ = [
    "COMPONENTS",
    "EvaluationResult",
    "Evaluator",
    "LiteratureReference",
    "base_parser",
    "emit",
    "main",
]

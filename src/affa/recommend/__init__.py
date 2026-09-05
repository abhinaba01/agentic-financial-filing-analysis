"""Rubric-based recommendation (section 7)."""

from affa.recommend.rubric import (
    FactorScore,
    RubricOutcome,
    assessment_from_score,
    build_values,
    evaluate,
    score_factor,
    score_from_bands,
)

__all__ = [
    "FactorScore",
    "RubricOutcome",
    "assessment_from_score",
    "build_values",
    "evaluate",
    "score_factor",
    "score_from_bands",
]

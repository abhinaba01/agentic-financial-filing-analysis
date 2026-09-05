"""Metric implementations shared by the harnesses.

Written out rather than pulled from a library so the definitions are inspectable:
the exact NDCG discount and the exact tie-handling matter when comparing a
fine-tune against its baseline, and a silent library change would move both
numbers without anyone noticing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def hit_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    # log2(i + 1) with 1-based i: the standard discount.
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """Graded NDCG@k. Returns 0.0 when the query has no relevant document."""
    gains = [relevance.get(doc_id, 0.0) for doc_id in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    scores = []
    for cls in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
        scores.append(precision_recall_f1(tp, fp, fn)[2])
    return sum(scores) / len(scores) if scores else 0.0


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> list[list[int]]:
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t][p] += 1
    return matrix


def class_distribution(labels: Sequence[int]) -> dict[int, int]:
    """Class counts. Reported alongside accuracy because a skewed split makes it meaningless."""
    return dict(Counter(labels))


def overlap_count(
    train_texts: Sequence[str], eval_texts: Sequence[str], *, normalize: bool = True
) -> tuple[int, list[str]]:
    """Exact-match overlap between a training set and an evaluation set.

    Section 2 requires this check on every dataset used for training, and
    requires the count to be reported *even when it is zero* - a stated zero is
    evidence the check ran, whereas silence is not.
    """

    def norm(text: str) -> str:
        return " ".join(text.lower().split()) if normalize else text

    train_set = {norm(t) for t in train_texts}
    hits = [t for t in eval_texts if norm(t) in train_set]
    return len(hits), hits[:20]

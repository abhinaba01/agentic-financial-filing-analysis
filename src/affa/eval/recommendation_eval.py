"""Recommendation evaluation (sections 7 and 9).

There is no ground truth for "should I invest in this company", so this harness
does not pretend to measure predictive accuracy. What it measures is
**reproducibility and agreement with a human applying the same rubric**:

* **rubric agreement** - does the system's verdict match the labeller's, where
  the labeller applied ``configs/rubric_v1.yaml`` by hand to the same filing;
* **insufficient_evidence rate** - how often the system declines to score, which
  must be non-zero on a realistic corpus or the sufficiency gate is decorative;
* **factor-level agreement** - per-factor sign agreement, which localises a
  disagreement to the factor that caused it.

A high rubric agreement means the rubric is implemented as written. It says
nothing about whether the rubric is a good one - that is a judgement the weights
in the YAML make available for argument, which is the point of putting them
there.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from affa.config import get_config, repo_root
from affa.eval.harness import EvaluationResult, Evaluator
from affa.pipeline import analyze_filing
from affa.schema import Assessment

log = logging.getLogger(__name__)

LABEL_DIR = repo_root() / "data" / "recommendation_labels"


class RecommendationEvaluator(Evaluator):
    """Agreement with hand-applied rubric labels, plus the abstention rate."""

    name = "recommendation"
    default_dataset = "data/recommendation_labels"
    default_baseline = "always_mixed"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--market-prices",
            default=None,
            help="JSON mapping ticker -> share price, for P/E where labels used one",
        )

    def run(self, args: argparse.Namespace) -> EvaluationResult:
        cfg = get_config(args.config) if args.config else get_config()
        label_dir = Path(args.test_set) if args.test_set else LABEL_DIR
        labels = _load_labels(label_dir)
        if args.limit:
            labels = labels[: args.limit]
        if not labels:
            raise SystemExit(
                f"no labelled filings in {label_dir}. See its README for the format: "
                "section 9 asks for 20-30 filings labelled against the rubric by hand."
            )

        prices = {}
        if args.market_prices:
            prices = json.loads(Path(args.market_prices).read_text(encoding="utf-8"))

        agree = 0
        insufficient = 0
        factor_total = 0
        factor_agree = 0
        notes: list[str] = []

        for label in labels:
            result = analyze_filing(
                label["source_file"],
                cfg=cfg,
                ticker=label.get("ticker"),
                market_price_per_share=prices.get(label.get("ticker")),
                in_memory=True,
            )
            rec = result.report.recommendation
            predicted = rec.assessment.value
            expected = label["assessment"]

            if predicted == expected:
                agree += 1
            else:
                notes.append(
                    f"{Path(label['source_file']).name}: predicted {predicted}, labelled {expected}"
                )
            if rec.assessment is Assessment.INSUFFICIENT_EVIDENCE:
                insufficient += 1

            for factor, expected_score in (label.get("factor_scores") or {}).items():
                if factor not in rec.factor_scores:
                    continue
                factor_total += 1
                # Sign agreement: exact score matching would measure band
                # boundaries rather than judgement.
                if _sign(rec.factor_scores[factor]) == _sign(float(expected_score)):
                    factor_agree += 1

        n = len(labels)
        metrics = {
            "rubric_agreement": round(agree / n, 4),
            "insufficient_evidence_rate": round(insufficient / n, 4),
            "factor_sign_agreement": round(factor_agree / factor_total, 4) if factor_total else 0.0,
        }

        # Majority-class baseline: always answer "mixed". A system that cannot
        # beat this is not analysing anything.
        baseline_agree = sum(1 for label in labels if label["assessment"] == "mixed")
        baseline_metrics = {
            "rubric_agreement": round(baseline_agree / n, 4),
            "insufficient_evidence_rate": 0.0,
            "factor_sign_agreement": 0.0,
        }

        notes.insert(0, f"labelled filings: {n}")
        notes.append(
            "rubric agreement measures whether the rubric is implemented as written. "
            "It is not evidence that the rubric predicts anything about returns, and "
            "no such claim is made."
        )
        if insufficient == 0:
            notes.append(
                "WARNING: insufficient_evidence never fired on this set. Either every "
                "filing was rich enough, or the sufficiency gate is not reachable in "
                "practice - check with a filing that lacks financial statements."
            )

        return EvaluationResult(
            component="recommendation",
            dataset=str(label_dir),
            split="hand-labelled",
            n_examples=n,
            metrics=metrics,
            baseline_name=args.baseline or self.default_baseline,
            baseline_metrics=baseline_metrics,
            model_name=f"rubric v{_rubric_version(cfg)}",
            notes=notes,
            seed=args.seed,
        )


def _sign(value: float) -> int:
    if value > 0.15:
        return 1
    if value < -0.15:
        return -1
    return 0


def _rubric_version(cfg) -> str:
    from affa.config import load_rubric

    return str(load_rubric(cfg)["version"])


def _load_labels(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    labels: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "assessment" not in payload or "source_file" not in payload:
            continue
        source = Path(payload["source_file"])
        payload["source_file"] = str(source if source.is_absolute() else repo_root() / source)
        labels.append(payload)
    return labels

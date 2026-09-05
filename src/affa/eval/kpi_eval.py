"""KPI extraction evaluation against a hand-labelled set (sections 6 and 9).

Section 9 calls the hand-labelled set the most valuable dataset in the project,
because it measures the actual product rather than a proxy benchmark. The
metrics are chosen to keep the two failure modes apart:

* **value accuracy** - tolerance-aware agreement with the gold figure;
* **extraction recall** - how many gold metrics were found at all;
* **unit-error rate** - values that are right but at the wrong scale, or with a
  flipped sign from a parenthesised negative.

The third is reported separately on purpose. A unit error and a wrong answer
have completely different fixes, and folding them together hides which one you
have. Ground truth is never rescaled to improve a score (anti-pattern #12) - the
mismatch is reported as a unit error and the *converter* gets fixed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from affa.config import get_config, repo_root
from affa.eval.harness import EvaluationResult, Evaluator
from affa.ingestion import ingest_filing
from affa.kpi.extract import extract_kpis
from affa.kpi.rules import extract_rule_based
from affa.kpi.units import MatchOutcome, compare_values
from affa.kpi.xbrl import XBRLTagger
from affa.schema import SCALE_MULTIPLIER

log = logging.getLogger(__name__)

GOLD_DIR = repo_root() / "data" / "kpi_gold"


@dataclass
class GoldDocument:
    source_file: str
    metrics: dict[str, float]
    company: str | None = None
    ticker: str | None = None
    fiscal_period: str | None = None
    label_notes: str = ""


def load_gold(directory: Path | str = GOLD_DIR) -> list[GoldDocument]:
    """Load every hand-labelled filing in ``directory``."""
    directory = Path(directory)
    docs: list[GoldDocument] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in payload:
            continue
        source = payload["source_file"]
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = repo_root() / source
        docs.append(
            GoldDocument(
                source_file=str(source_path),
                metrics={
                    k: float(v["value"] if isinstance(v, dict) else v)
                    for k, v in payload["metrics"].items()
                },
                company=payload.get("company"),
                ticker=payload.get("ticker"),
                fiscal_period=payload.get("fiscal_period"),
                label_notes=payload.get("notes", ""),
            )
        )
    return docs


class KPIEvaluator(Evaluator):
    """Tolerance-aware KPI accuracy against hand-labelled filings."""

    name = "kpi"
    default_dataset = "data/kpi_gold"
    default_baseline = "rule_based"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--tolerance",
            type=float,
            default=None,
            help="Relative tolerance %% (default: kpi.tolerance_pct)",
        )

    def run(self, args: argparse.Namespace) -> EvaluationResult:
        cfg = get_config(args.config) if args.config else get_config()
        tolerance = args.tolerance if args.tolerance is not None else cfg.kpi.tolerance_pct
        gold_dir = Path(args.test_set) if args.test_set else GOLD_DIR
        docs = load_gold(gold_dir)
        if args.limit:
            docs = docs[: args.limit]

        if not docs:
            raise SystemExit(
                f"no hand-labelled documents in {gold_dir}. See data/kpi_gold/README.md "
                "for the format - this set has to be built by hand, and it is the only "
                "measurement of the actual product."
            )

        tagger = XBRLTagger(cfg.models.xbrl_tagger)
        tagger_available = tagger.load()

        model_stats = _Stats()
        baseline_stats = _Stats()
        notes: list[str] = []

        for doc in docs:
            ingested = ingest_filing(
                doc.source_file,
                cfg,
                ticker=doc.ticker,
                company=doc.company,
                fiscal_period=doc.fiscal_period,
                in_memory=True,
            )
            combined = extract_kpis(ingested.chunks, cfg, tagger=tagger)
            model_values = {m.name: m.value_in_units for m in combined.metrics.extracted}
            baseline_values = _rule_only_values(ingested.chunks)

            model_stats.score(doc.metrics, model_values, tolerance)
            baseline_stats.score(doc.metrics, baseline_values, tolerance)

        if not tagger_available:
            notes.append(
                "XBRL tagger not loaded, so the evaluated system IS the rule-based "
                "baseline. The two columns below are identical by construction and "
                "this run does not measure the tagger. Enable models.xbrl_tagger in "
                "the config and point it at a fine-tuned checkpoint."
            )

        notes.append(f"documents: {len(docs)}, gold metric instances: {model_stats.total}")
        notes.append(f"tolerance: {tolerance}% relative")
        if model_stats.unit_errors:
            notes.append(
                f"{model_stats.unit_errors} unit errors detected "
                f"({', '.join(sorted(model_stats.unit_error_metrics)[:6])}). These are "
                "scale or sign mismatches, reported separately from wrong values and "
                "fixed in the converter, never by rescaling ground truth."
            )
        for name, detail in list(model_stats.mismatch_detail.items())[:10]:
            notes.append(f"mismatch {name}: {detail}")

        return EvaluationResult(
            component="kpi",
            dataset=str(gold_dir.relative_to(repo_root()) if gold_dir.is_absolute() else gold_dir),
            split="hand-labelled",
            n_examples=model_stats.total,
            metrics=model_stats.metrics(),
            baseline_name=args.baseline or self.default_baseline,
            baseline_metrics=baseline_stats.metrics(),
            model_name=(
                cfg.models.xbrl_tagger.active_name if tagger_available else "rule_based_only"
            ),
            notes=notes,
            seed=args.seed,
        )


def _rule_only_values(chunks) -> dict[str, float]:
    """Best rule-based value per metric, in absolute units."""
    from affa.kpi.extract import _best_rule_hit

    by_metric: dict[str, list] = {}
    for hit in extract_rule_based(chunks):
        by_metric.setdefault(hit.metric, []).append(hit)
    out: dict[str, float] = {}
    for name, hits in by_metric.items():
        best = _best_rule_hit(hits)
        if best is not None:
            out[name] = best.value * SCALE_MULTIPLIER[best.scale]
    return out


class _Stats:
    def __init__(self) -> None:
        self.total = 0
        self.found = 0
        self.correct = 0
        self.unit_errors = 0
        self.mismatches = 0
        self.unit_error_metrics: set[str] = set()
        self.mismatch_detail: dict[str, str] = {}

    def score(self, gold: dict[str, float], predicted: dict[str, float], tolerance: float) -> None:
        for name, gold_value in gold.items():
            self.total += 1
            pred = predicted.get(name)
            if pred is None:
                continue
            self.found += 1
            comparison = compare_values(pred, gold_value, tolerance_pct=tolerance)
            if comparison.outcome is MatchOutcome.MATCH:
                self.correct += 1
            elif comparison.outcome is MatchOutcome.UNIT_ERROR:
                self.unit_errors += 1
                self.unit_error_metrics.add(name)
                self.mismatch_detail.setdefault(name, comparison.detail)
            else:
                self.mismatches += 1
                self.mismatch_detail.setdefault(
                    name, f"predicted {pred:,.4g} vs gold {gold_value:,.4g}"
                )

    def metrics(self) -> dict[str, float]:
        total = max(self.total, 1)
        found = max(self.found, 1)
        return {
            # Over all gold metrics: what fraction did we get right end to end.
            "value_accuracy": round(self.correct / total, 4),
            # Of the ones we found: how often the value was right.
            "precision_when_found": round(self.correct / found, 4),
            "extraction_recall": round(self.found / total, 4),
            "unit_error_rate": round(self.unit_errors / total, 4),
            "mismatch_rate": round(self.mismatches / total, 4),
        }

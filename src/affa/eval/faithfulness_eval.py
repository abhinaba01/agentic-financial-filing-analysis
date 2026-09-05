"""Faithfulness evaluation (section 9).

The metric that matters most for this project, and the least standard, so the
definition is stated here and implemented exactly once, in
:mod:`affa.agent.verify`:

    A claim is **supported** when every number in it appears in one of its cited
    chunks - allowing a reporting-scale factor and a stated tolerance - or is
    directly computable from numbers that do, *and* the financial subjects of the
    claim are discussed in the cited text.

Three numbers come out of it:

* **citation coverage** - fraction of claims that cite anything at all;
* **claim-support precision** - fraction of claims their citations actually support;
* **hallucination rate** - fraction unsupported or contradicted.

The baseline is **ungrounded generation**: the same claims with their citations
stripped. That is the honest comparison - it shows what the verification step
buys - and it is measured on the same documents in the same run.

``--hand-check`` samples claims for manual review and writes them out, because
section 9 asks for the automated metric to be validated against human agreement
rather than trusted on its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from affa.agent.graph import build_graph
from affa.agent.state import new_state
from affa.agent.verify import verify_findings
from affa.config import get_config
from affa.eval.harness import EvaluationResult, Evaluator
from affa.eval.kpi_eval import GOLD_DIR, load_gold
from affa.ingestion import ingest_filing
from affa.schema import Finding, Verification

log = logging.getLogger(__name__)


class FaithfulnessEvaluator(Evaluator):
    """Citation coverage, claim-support precision and hallucination rate."""

    name = "faithfulness"
    default_dataset = "data/kpi_gold"
    default_baseline = "ungrounded_generation"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hand-check",
            type=int,
            default=0,
            help="Sample N claims to a JSON file for manual agreement checking",
        )
        parser.add_argument(
            "--hand-check-output",
            default="eval_results/faithfulness_hand_check.json",
            help="Where to write the hand-check sample",
        )
        parser.add_argument(
            "--documents",
            nargs="*",
            default=None,
            help="Filing paths to evaluate (default: the hand-labelled set)",
        )

    def run(self, args: argparse.Namespace) -> EvaluationResult:
        cfg = get_config(args.config) if args.config else get_config()

        if args.documents:
            paths = [Path(p) for p in args.documents]
        else:
            gold_dir = Path(args.test_set) if args.test_set else GOLD_DIR
            paths = [Path(d.source_file) for d in load_gold(gold_dir)]
        if args.limit:
            paths = paths[: args.limit]
        if not paths:
            raise SystemExit(
                "no documents to evaluate. Pass --documents, or populate "
                "data/kpi_gold/ (see its README)."
            )

        all_checks = []
        baseline_checks = []
        notes: list[str] = []

        for path in paths:
            ingested = ingest_filing(path, cfg, in_memory=True)
            bundle = build_graph(cfg, store=ingested.store, document=ingested.document)
            state = new_state(
                doc_id=ingested.document.doc_id,
                source_file=str(path),
                chunks=ingested.chunks,
                question="What were revenue, margins, leverage, cash generation and risks?",
            )
            final = bundle.app.invoke(state, {"recursion_limit": 25})

            findings = final.get("findings", [])
            evidence = final.get("evidence", [])

            grounded = verify_findings(findings, evidence, cfg.verification)
            all_checks.extend(grounded.checks)

            # Baseline: the same claims with citations removed. Nothing else
            # changes, so the delta isolates what grounding contributes.
            stripped = [
                Finding(
                    claim=f.claim,
                    supporting_chunks=[],
                    verification=Verification.UNSUPPORTED,
                )
                for f in findings
            ]
            baseline_checks.extend(verify_findings(stripped, evidence, cfg.verification).checks)

            if ingested.used_stub_embedder:
                notes.append(
                    f"{path.name}: retrieval used the hashing stub embedder; the "
                    "evidence set is lexical, not semantic"
                )

        metrics = _summarise(all_checks)
        baseline_metrics = _summarise(baseline_checks)

        notes.insert(0, f"documents: {len(paths)}, claims verified: {len(all_checks)}")
        notes.append(
            "baseline strips citations from the identical claims, so the delta is "
            "exactly what grounding contributes"
        )
        notes.append(
            f"tolerance: {cfg.verification.numeric_tolerance_pct}% numeric, "
            f"{cfg.verification.min_entity_overlap} minimum subject overlap"
        )
        contradicted = sum(1 for c in all_checks if c.verdict is Verification.CONTRADICTED)
        notes.append(f"contradicted claims (evidence found and disagreeing): {contradicted}")

        if args.hand_check and all_checks:
            sample = random.Random(args.seed).sample(
                all_checks, min(args.hand_check, len(all_checks))
            )
            out = Path(args.hand_check_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    [
                        {
                            "claim": c.claim,
                            "automated_verdict": c.verdict.value,
                            "detail": c.detail,
                            "cited_chunks": c.cited_chunks,
                            "human_verdict": None,
                        }
                        for c in sample
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            notes.append(
                f"wrote {len(sample)} claims to {out} for manual review. Fill in "
                "human_verdict and report agreement alongside the automated number - "
                "an unvalidated automated faithfulness metric is not evidence."
            )

        return EvaluationResult(
            component="faithfulness",
            dataset=str(args.test_set or self.default_dataset),
            split="documents",
            n_examples=len(all_checks),
            metrics=metrics,
            baseline_name=args.baseline or self.default_baseline,
            baseline_metrics=baseline_metrics,
            model_name=cfg.models.reasoner.active_name,
            notes=notes,
            seed=args.seed,
        )


def _summarise(checks) -> dict[str, float]:
    if not checks:
        return {
            "citation_coverage": 0.0,
            "claim_support_precision": 0.0,
            "hallucination_rate": 0.0,
            "contradiction_rate": 0.0,
        }
    n = len(checks)
    supported = sum(1 for c in checks if c.verdict is Verification.SUPPORTED)
    contradicted = sum(1 for c in checks if c.verdict is Verification.CONTRADICTED)
    cited = sum(1 for c in checks if c.cited_chunks)
    return {
        "citation_coverage": round(cited / n, 4),
        "claim_support_precision": round(supported / n, 4),
        "hallucination_rate": round((n - supported) / n, 4),
        "contradiction_rate": round(contradicted / n, 4),
    }

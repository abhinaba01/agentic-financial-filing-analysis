"""``affa-analyze`` - run the pipeline over a filing from the command line.

Every flag documented in README.md exists here, and ``tests/test_docs_contract.py``
enforces that in both directions. Anti-pattern #5 is a README flag that argparse
has never heard of.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from affa.config import get_config
from affa.pipeline import DEFAULT_QUESTION, analyze_filing
from affa.report.render import to_html, to_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="affa-analyze",
        description=(
            "Analyze a financial filing and emit a cited, rubric-based assessment. "
            "Research and educational use only. Not investment advice."
        ),
    )
    parser.add_argument("filing", help="Path to a 10-K/10-Q as PDF, HTML, TXT or JSON")
    parser.add_argument(
        "--config", default=None, help="Path to a config YAML (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Write the JSON report here (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "html"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION, help="Analysis question driving retrieval"
    )
    parser.add_argument(
        "--ticker", default=None, help="Override the ticker sniffed from the filing"
    )
    parser.add_argument("--company", default=None, help="Override the company name")
    parser.add_argument(
        "--fiscal-period", default=None, help="Override the fiscal period, e.g. FY2024"
    )
    parser.add_argument(
        "--market-price",
        type=float,
        default=None,
        help="Share price for P/E. A filing does not contain one, so it is an explicit input.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use an in-memory vector store instead of the persistent Chroma collection",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = get_config(args.config) if args.config else get_config()
    result = analyze_filing(
        args.filing,
        cfg=cfg,
        question=args.question,
        ticker=args.ticker,
        company=args.company,
        fiscal_period=args.fiscal_period,
        market_price_per_share=args.market_price,
        in_memory=args.in_memory,
    )

    if args.format == "json":
        rendered = result.report.to_json()
    elif args.format == "markdown":
        rendered = to_markdown(result.report)
    else:
        rendered = to_html(result.report)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(
            f"wrote {args.format} report to {args.output} "
            f"({result.n_chunks} chunks, graph backend: {result.backend})",
            file=sys.stderr,
        )
    else:
        print(rendered)

    rec = result.report.recommendation
    print(
        f"assessment={rec.assessment.value} confidence={rec.confidence:.2f} "
        f"rubric=v{rec.rubric_version}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

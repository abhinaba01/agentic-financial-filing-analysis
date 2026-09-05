"""Render an :class:`AnalysisReport` as Markdown or HTML.

The rendered view is a projection of the JSON, never a second source of truth:
everything printed here is read off the validated report object, so the document
a reader sees and the document a program consumes cannot drift apart.
"""

from __future__ import annotations

import html

from affa.kpi.catalog import metric_label
from affa.schema import AnalysisReport, Assessment, Verification

_ASSESSMENT_BLURB = {
    Assessment.FAVORABLE: "the rubric's signals lean positive",
    Assessment.MIXED: "the rubric's signals are mixed",
    Assessment.UNFAVORABLE: "the rubric's signals lean negative",
    Assessment.INSUFFICIENT_EVIDENCE: "the filing did not yield enough verified evidence to score",
}


def _fmt_money(value: float, unit: str) -> str:
    if unit not in {"USD", "shares"}:
        return f"{value:,.2f}"
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= cutoff:
            return f"{value / cutoff:,.2f}{suffix}"
    return f"{value:,.2f}"


def to_markdown(report: AnalysisReport) -> str:
    m = report.metadata
    rec = report.recommendation
    lines: list[str] = []

    title = m.company or m.ticker or "Filing analysis"
    lines.append(f"# {title} - {m.doc_type or 'filing'} {m.fiscal_period or ''}".strip())
    lines.append("")
    lines.append(f"> **{rec.disclaimer}**")
    lines.append("")

    lines.append("## Assessment")
    lines.append("")
    lines.append(
        f"**{rec.assessment.value.replace('_', ' ').title()}** "
        f"(confidence {rec.confidence:.2f}, rubric v{rec.rubric_version}) - "
        f"{_ASSESSMENT_BLURB[rec.assessment]}."
    )
    lines.append("")
    if rec.aggregate_score is not None:
        lines.append(
            f"Aggregate score `{rec.aggregate_score:+.3f}` over "
            f"{len(rec.factors_scored)} factors covering "
            f"{rec.weight_covered:.0%} of rubric weight."
        )
        lines.append("")
    if rec.factors_missing:
        lines.append(f"Factors not scored: {', '.join(rec.factors_missing)}.")
        lines.append("")

    if rec.factor_scores:
        lines.append("| Factor | Score | Rationale |")
        lines.append("|---|---:|---|")
        by_factor = {r.factor: r for r in rec.rationale}
        for name, score in rec.factor_scores.items():
            item = by_factor.get(name)
            cites = ""
            if item and item.citations:
                cites = " " + " ".join(
                    f"[p.{c.page}]" if c.page else "[cited]" for c in item.citations
                )
            lines.append(
                f"| {name.replace('_', ' ')} | {score:+.2f} | "
                f"{(item.statement if item else '')}{cites} |"
            )
        lines.append("")

    if rec.narrative:
        lines.append("### Narrative")
        lines.append("")
        lines.append(rec.narrative)
        lines.append("")
        lines.append(
            "_The narrative above explains the rubric's output. It did not determine the verdict._"
        )
        lines.append("")

    fm = report.financial_metrics
    if fm.extracted:
        lines.append("## Extracted metrics")
        lines.append("")
        lines.append("| Metric | Value | Method | Confidence | Source |")
        lines.append("|---|---:|---|---:|---|")
        for e in fm.extracted:
            page = f"p.{e.source.page}" if e.source.page else e.source.chunk_id[:14]
            lines.append(
                f"| {metric_label(e.name)} | {_fmt_money(e.value_in_units, e.unit)} | "
                f"{e.method.value} | {e.confidence:.2f} | {page} |"
            )
        lines.append("")

    if fm.derived:
        lines.append("## Derived metrics")
        lines.append("")
        lines.append("| Metric | Value | Formula | Operands |")
        lines.append("|---|---:|---|---|")
        for d in fm.derived:
            operands = ", ".join(f"{k}={v:,.4g}" for k, v in d.operands.items())
            lines.append(
                f"| {metric_label(d.name)} | {d.value:,.4g} | `{d.formula}` | {operands} |"
            )
        lines.append("")

    if fm.yoy_changes:
        lines.append("## Year-over-year change")
        lines.append("")
        for name, value in fm.yoy_changes.items():
            lines.append(f"- {metric_label(name)}: **{value:+.2f}%**")
        lines.append("")

    if fm.disagreements:
        lines.append("## Extractor disagreements")
        lines.append("")
        lines.append(
            "The XBRL model and the rule-based extractor produced different "
            "values. Both are shown; neither is hidden."
        )
        lines.append("")
        lines.append("| Metric | XBRL model | Rule-based | Difference | Resolved to |")
        lines.append("|---|---:|---:|---:|---|")
        for d in fm.disagreements:
            lines.append(
                f"| {metric_label(d.name)} | {d.xbrl_model:,.4g} | {d.rule_based:,.4g} | "
                f"{(d.relative_difference_pct or 0):.2f}% | "
                f"{d.resolved_to.value if d.resolved_to else '-'} |"
            )
        lines.append("")

    lines.append("## Sentiment")
    lines.append("")
    s = report.sentiment
    lines.append(
        f"Overall **{s.overall}** (score {s.score:+.3f}), source: `{s.model_name}`"
        + ("" if s.available else " - fallback, not the fine-tuned classifier")
        + "."
    )
    lines.append("")

    if report.risk_factors:
        lines.append("## Risk factors")
        lines.append("")
        for r in report.risk_factors[:15]:
            page = f" (p.{r.source.page})" if r.source.page else ""
            lines.append(f"- **{r.severity.value}**{page}: {r.risk[:220]}")
        lines.append("")

    lines.append("## Findings")
    lines.append("")
    if report.reasoning.findings:
        for f in report.reasoning.findings:
            mark = {
                Verification.SUPPORTED: "supported",
                Verification.CONTRADICTED: "CONTRADICTED",
                Verification.UNSUPPORTED: "unsupported",
            }[f.verification]
            cites = ", ".join(c[:14] for c in f.supporting_chunks)
            lines.append(f"- [{mark}] {f.claim} _(chunks: {cites})_")
    else:
        lines.append("_No findings survived verification._")
    lines.append("")
    lines.append(
        f"{report.unsupported_claims_dropped} claim(s) were dropped for lacking "
        "support in their cited passages."
    )
    lines.append("")

    d = report.retrieval_diagnostics
    lines.append("## Retrieval diagnostics")
    lines.append("")
    lines.append(f"- Chunks retrieved: {d.chunks_retrieved}")
    lines.append(f"- Discarded below similarity floor: {d.chunks_discarded_below_floor}")
    lines.append(f"- Mean similarity: {d.mean_similarity:.3f}")
    lines.append(f"- Retries: {d.retries}")
    lines.append(f"- Sufficient: {d.sufficient} ({d.stop_reason})")
    if d.reformulations:
        lines.append("- Queries tried:")
        for q in d.reformulations:
            lines.append(f"  - `{q}`")
    lines.append("")

    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"Pipeline `{m.pipeline_version}` - embedder `{m.models.embedder}`, "
        f"XBRL tagger `{m.models.xbrl_tagger}`, sentiment `{m.models.sentiment}`, "
        f"reasoner `{m.models.reasoner}` - generated {m.generated_at}."
    )
    lines.append("")
    lines.append(rec.disclaimer)
    return "\n".join(lines)


def to_html(report: AnalysisReport) -> str:
    """Minimal standalone HTML wrapper around the Markdown projection."""
    md = to_markdown(report)
    body: list[str] = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells if c):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
            tag = "th" if not body[-1].startswith("<tr") else "td"
            body.append(
                "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"
            )
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        elif line.startswith("- "):
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip() == "---":
            body.append("<hr>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table>")

    title = report.metadata.company or report.metadata.ticker or "Filing analysis"
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:15px/1.6 system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;color:#1a1a1a}table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "td,th{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}blockquote{border-left:3px solid #c00;padding-left:1rem;"
        "color:#600}code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}</style>"
        + "\n".join(body)
    )

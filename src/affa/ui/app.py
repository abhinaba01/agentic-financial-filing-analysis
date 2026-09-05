"""Streamlit UI: the filing on one side, cited evidence on the other.

    streamlit run src/affa/ui/app.py

The layout is the point. Every figure and every claim in the right-hand pane is
clickable back to the passage it came from in the left-hand pane, which is the
whole argument of this project - "why did it say that" should be one click, not
an investigation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from affa import DISCLAIMER, __version__
from affa.config import get_config
from affa.pipeline import DEFAULT_QUESTION, analyze_filing
from affa.report.render import to_markdown
from affa.schema import Verification

st.set_page_config(page_title="Financial Filing Analyst", layout="wide")

VERDICT_COLOR = {
    "favorable": "#1a7f37",
    "mixed": "#9a6700",
    "unfavorable": "#b42318",
    "insufficient_evidence": "#57606a",
}
VERIFICATION_BADGE = {
    Verification.SUPPORTED: ("supported", "#1a7f37"),
    Verification.CONTRADICTED: ("contradicted", "#b42318"),
    Verification.UNSUPPORTED: ("unsupported", "#9a6700"),
}


@st.cache_data(show_spinner=False)
def _run(path_str: str, question: str, ticker: str, price: float | None) -> str:
    """Analyze and return the report as JSON.

    Cached on the arguments so tweaking the view does not re-run the pipeline.
    Returns JSON rather than the model object because Streamlit's cache needs a
    picklable, hashable-by-value result.
    """
    result = analyze_filing(
        path_str,
        question=question,
        ticker=ticker or None,
        market_price_per_share=price,
        in_memory=True,
    )
    return result.report.to_json()


def main() -> None:
    st.title("Agentic Financial Filing Analyst")
    st.caption(f"v{__version__} - {DISCLAIMER}")

    cfg = get_config()

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Filing", type=["pdf", "html", "htm", "txt", "md", "json"])
        sample = Path(__file__).resolve().parents[3] / "data" / "samples" / "demo_10k.json"
        use_sample = st.checkbox(
            "Use the bundled sample (synthetic)", value=not uploaded and sample.is_file()
        )
        question = st.text_area("Analysis question", DEFAULT_QUESTION, height=100)
        ticker = st.text_input("Ticker override", "")
        price_raw = st.text_input(
            "Share price (optional, for P/E)",
            "",
            help="A filing contains no share price. Supply one only if you want P/E.",
        )
        price = float(price_raw) if price_raw.strip() else None

        st.divider()
        st.subheader("Active configuration")
        st.caption(
            f"embedder: `{cfg.models.embedder.name}`\n\n"
            f"XBRL tagger: {'on' if cfg.models.xbrl_tagger.enabled else 'off (rule-based)'}\n\n"
            f"sentiment: {'on' if cfg.models.sentiment.enabled else 'off (lexicon)'}\n\n"
            f"reasoner: `{cfg.models.reasoner.backend}`\n\n"
            f"similarity floor: {cfg.retrieval.min_similarity} - "
            f"retry below: {cfg.routing.retry_below_mean_similarity}"
        )
        run = st.button("Analyze", type="primary", use_container_width=True)

    if not run:
        st.info("Upload a filing or tick the sample box, then press Analyze.")
        return

    if uploaded is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / uploaded.name
            path.write_bytes(uploaded.getvalue())
            payload = _run(str(path), question, ticker, price)
    elif use_sample and sample.is_file():
        payload = _run(str(sample), question, ticker, price)
    else:
        st.error("No filing provided.")
        return

    report = json.loads(payload)
    _render(report)


def _render(report: dict) -> None:
    rec = report["recommendation"]
    colour = VERDICT_COLOR.get(rec["assessment"], "#57606a")

    st.markdown(
        f"<h2 style='color:{colour};margin-bottom:0'>"
        f"{rec['assessment'].replace('_', ' ').title()}</h2>"
        f"<p style='color:#57606a;margin-top:.2rem'>confidence {rec['confidence']:.2f} - "
        f"rubric v{rec['rubric_version']} - "
        f"{len(rec['factors_scored'])} factors scored, "
        f"{rec['weight_covered']:.0%} of rubric weight</p>",
        unsafe_allow_html=True,
    )
    st.warning(rec["disclaimer"])

    if rec["assessment"] == "insufficient_evidence":
        st.info(
            "The rubric declined to score this filing. Factors missing: "
            + ", ".join(rec["factors_missing"])
        )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Evidence")
        st.caption(
            "Passages retrieved for the question, plus every chunk a published "
            "figure was read from."
        )
        for chunk in report["evidence"]:
            page = f"p.{chunk['page']}" if chunk.get("page") else "no page"
            label = (
                f"{page} - {chunk['chunk_type']} - similarity {chunk['similarity']:.3f}"
                if chunk["similarity"] > 0
                else f"{page} - {chunk['chunk_type']} - cited as provenance"
            )
            with st.expander(label):
                st.code(chunk["text"][:4000], language=None)
                st.caption(f"chunk id: `{chunk['chunk_id']}`")

    with right:
        st.subheader("Findings")
        st.caption("Every claim re-checked against the passages it cites.")
        for finding in report["reasoning"]["findings"]:
            verdict = Verification(finding["verification"])
            text, badge_colour = VERIFICATION_BADGE[verdict]
            st.markdown(
                f"<span style='background:{badge_colour};color:white;padding:.1rem .4rem;"
                f"border-radius:3px;font-size:.75rem'>{text}</span> {finding['claim']}",
                unsafe_allow_html=True,
            )
            if finding.get("verification_detail"):
                st.caption(finding["verification_detail"])
        dropped = report["unsupported_claims_dropped"]
        if dropped:
            st.caption(f"{dropped} claim(s) dropped as unsupported by their citations.")

        st.subheader("Factor scores")
        for factor, score in rec["factor_scores"].items():
            st.progress((score + 1) / 2, text=f"{factor.replace('_', ' ')}: {score:+.2f}")

    st.divider()
    tabs = st.tabs(["Metrics", "Risks", "Retrieval", "Full report", "Raw JSON"])

    with tabs[0]:
        fm = report["financial_metrics"]
        if fm["extracted"]:
            st.dataframe(
                [
                    {
                        "metric": m["name"],
                        "value": m["value"],
                        "scale": m["scale"],
                        "method": m["method"],
                        "confidence": m["confidence"],
                        "page": (m["source"] or {}).get("page"),
                    }
                    for m in fm["extracted"]
                ],
                use_container_width=True,
            )
        if fm["derived"]:
            st.caption("Derived - each with the formula and operands behind it.")
            st.dataframe(
                [
                    {"metric": d["name"], "value": d["value"], "formula": d["formula"]}
                    for d in fm["derived"]
                ],
                use_container_width=True,
            )
        if fm["disagreements"]:
            st.warning("The two extractors disagreed. Both values are shown.")
            st.dataframe(fm["disagreements"], use_container_width=True)

    with tabs[1]:
        for risk in report["risk_factors"]:
            page = f" (p.{risk['source']['page']})" if risk["source"].get("page") else ""
            st.markdown(f"**{risk['severity']}**{page} - {risk['risk']}")

    with tabs[2]:
        diagnostics = report["retrieval_diagnostics"]
        cols = st.columns(4)
        cols[0].metric("Chunks retrieved", diagnostics["chunks_retrieved"])
        cols[1].metric("Mean similarity", f"{diagnostics['mean_similarity']:.3f}")
        cols[2].metric("Retries", diagnostics["retries"])
        cols[3].metric("Sufficient", str(diagnostics["sufficient"]))
        st.caption(diagnostics["stop_reason"] or "")
        st.write("Queries tried:")
        for query in diagnostics["reformulations"]:
            st.code(query, language=None)

    with tabs[3]:
        from affa.schema import AnalysisReport

        st.markdown(to_markdown(AnalysisReport.model_validate(report)))

    with tabs[4]:
        st.json(report)

    if report.get("warnings"):
        st.divider()
        st.subheader("Warnings")
        for warning in report["warnings"]:
            st.caption(f"- {warning}")


if __name__ == "__main__":
    main()

"""The verify node - the project's differentiator (sections 3 and 9).

A verifier that never rejects anything is worse than no verifier, because it
produces a faithfulness number that looks good and means nothing. These tests
pin down that all three verdicts are reachable and that each fires for the right
reason.
"""

from __future__ import annotations

import pytest

from affa.agent.verify import subject_overlap, subjects, verify_claim, verify_findings
from affa.schema import EvidenceChunk, Finding, Verification


@pytest.fixture
def evidence() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="c1",
            page=31,
            text=(
                "CONSOLIDATED STATEMENTS OF OPERATIONS (In millions)\n"
                "Total net sales | 4,812.6 | 4,301.9\n"
                "Gross profit | 3,166.8 | 2,769.5\n"
                "Operating income | 962.5 | 774.3\n"
                "Net income | 731.4 | 588.2"
            ),
            similarity=0.71,
            chunk_type="table",
        ),
        EvidenceChunk(
            chunk_id="c2",
            page=36,
            text="Depreciation and amortization | 214.7\nNet cash provided by operating activities | 1,104.2",
            similarity=0.62,
            chunk_type="table",
        ),
    ]


def test_true_claim_is_supported(evidence, cfg) -> None:
    check = verify_claim(
        "Total net sales were 4,812.6 million.", ["c1"], evidence, cfg.verification
    )
    assert check.verdict is Verification.SUPPORTED
    assert check.numbers_unmatched == []


def test_fabricated_figure_is_caught(evidence, cfg) -> None:
    check = verify_claim("Revenue was 9,999.9 million.", ["c1"], evidence, cfg.verification)
    assert check.verdict in (Verification.UNSUPPORTED, Verification.CONTRADICTED)


def test_contradiction_is_distinguished_from_absence(evidence, cfg) -> None:
    """Evidence found and disagreeing is a stronger signal than evidence missing."""
    check = verify_claim("Net income was 500.0 million.", ["c1"], evidence, cfg.verification)
    assert check.verdict is Verification.CONTRADICTED
    assert "states" in check.detail


def test_uncited_claim_is_unsupported(evidence, cfg) -> None:
    check = verify_claim("Revenue grew strongly.", [], evidence, cfg.verification)
    assert check.verdict is Verification.UNSUPPORTED
    assert "cites no chunk" in check.detail


def test_citation_to_a_missing_chunk_is_unsupported(evidence, cfg) -> None:
    check = verify_claim("Revenue was 4,812.6 million.", ["nope"], evidence, cfg.verification)
    assert check.verdict is Verification.UNSUPPORTED


def test_offtopic_evidence_is_rejected_even_with_matching_numbers(evidence, cfg) -> None:
    check = verify_claim(
        "The workforce numbered 4,812.6 thousand people.", ["c1"], evidence, cfg.verification
    )
    assert check.verdict is Verification.UNSUPPORTED
    assert "overlap" in check.detail


def test_derived_value_is_supported_when_computable(evidence, cfg) -> None:
    """A margin is grounded in the two figures that produce it."""
    check = verify_claim("Net margin on net sales was 15.20%.", ["c1"], evidence, cfg.verification)
    assert check.verdict is Verification.SUPPORTED


def test_computability_is_scale_aware(evidence, cfg) -> None:
    """Regression: a correct derivation was marked CONTRADICTED.

    The statement is printed in millions (962.5 and 214.7) while the claim states
    the absolute EBITDA those imply (1,177,200,000). Checking only raw magnitudes
    called correct arithmetic a fabrication.
    """
    check = verify_claim(
        "EBITDA computes to 1,177,200,000 from operating income and depreciation and amortization.",
        ["c1", "c2"],
        evidence,
        cfg.verification,
    )
    assert check.verdict is Verification.SUPPORTED


def test_subject_matching_bridges_filing_vocabulary(evidence, cfg) -> None:
    """The filing says "net sales"; the claim says "revenue". Same subject.

    Regression: raw word overlap rejected correct, well-cited claims purely for
    using the extractor's normalised vocabulary.
    """
    assert "revenue" in subjects("Revenue is reported as 4,812.6 million.")
    assert "revenue" in subjects("Total net sales | 4,812.6")
    overlap, basis = subject_overlap("Revenue was 4,812.6 million.", evidence[0].text)
    assert basis == "subject"
    assert overlap == pytest.approx(1.0)


def test_unsupported_claims_are_dropped_and_counted(evidence, cfg) -> None:
    findings = [
        Finding(
            claim="Total net sales were 4,812.6 million.",
            supporting_chunks=["c1"],
            verification=Verification.UNSUPPORTED,
        ),
        Finding(
            claim="Revenue was 9,999.9 million.",
            supporting_chunks=["c1"],
            verification=Verification.UNSUPPORTED,
        ),
        Finding(
            claim="Something with no citation at all.",
            supporting_chunks=[],
            verification=Verification.UNSUPPORTED,
        ),
    ]
    outcome = verify_findings(findings, evidence, cfg.verification)
    assert outcome.dropped >= 1
    assert all(
        f.verification in (Verification.SUPPORTED, Verification.CONTRADICTED) for f in outcome.kept
    )


def test_faithfulness_metrics_are_computed(evidence, cfg) -> None:
    findings = [
        Finding(
            claim="Total net sales were 4,812.6 million.",
            supporting_chunks=["c1"],
            verification=Verification.UNSUPPORTED,
        ),
        Finding(
            claim="Profit was 12,345.6 million.",
            supporting_chunks=["c1"],
            verification=Verification.UNSUPPORTED,
        ),
    ]
    outcome = verify_findings(findings, evidence, cfg.verification)
    assert 0.0 <= outcome.citation_coverage <= 1.0
    assert 0.0 <= outcome.support_precision <= 1.0
    assert outcome.hallucination_rate == pytest.approx(1.0 - outcome.support_precision)


def test_empty_findings_do_not_divide_by_zero(cfg) -> None:
    outcome = verify_findings([], [], cfg.verification)
    assert outcome.citation_coverage == 0.0
    assert outcome.hallucination_rate == 0.0

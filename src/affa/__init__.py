"""Agentic Financial Filing Analyst (affa).

An agentic RAG system that ingests SEC filings, extracts KPIs with provenance,
reasons over retrieved evidence with a verification step, and emits a
rubric-based investment assessment in which every claim cites a source passage.

Research and educational use only. Not investment advice.
"""

__version__ = "0.1.0"
PIPELINE_VERSION = __version__

DISCLAIMER = "Research and educational use only. Not investment advice."

__all__ = ["__version__", "PIPELINE_VERSION", "DISCLAIMER"]

"""Documentation-vs-code contract (anti-patterns #4, #5, #6, #9).

Documentation drifts silently. These tests make three specific kinds of drift
fail the build:

* a CLI flag documented in the README that ``argparse`` has never heard of, or a
  flag that exists and is undocumented;
* a model named in the docs that the code does not use;
* published figures presented where they read as this repo's own results.
"""

from __future__ import annotations

import re

import pytest
import yaml

from affa.cli import build_parser
from affa.config import repo_root
from affa.eval.harness import COMPONENTS, base_parser

README = repo_root() / "README.md"
RUNNING = repo_root() / "RUNNING.md"


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def documented_cli_flags() -> set[str]:
    """Long flags from the README's CLI table."""
    flags: set[str] = set()
    for line in readme_text().splitlines():
        if not line.strip().startswith("| `--"):
            continue
        cell = line.split("|")[1]
        flags |= {m.group(0) for m in re.finditer(r"--[a-z][a-z0-9-]*", cell)}
    return flags


def argparse_flags() -> set[str]:
    parser = build_parser()
    flags: set[str] = set()
    for action in parser._actions:
        flags |= {opt for opt in action.option_strings if opt.startswith("--")}
    return flags - {"--help"}


def test_every_documented_flag_exists() -> None:
    """Anti-pattern #5: a README flag that argparse does not have."""
    documented = documented_cli_flags()
    actual = argparse_flags()
    assert documented, "no CLI flags found in the README table - has it moved?"
    missing = documented - actual
    assert not missing, f"README documents flags that do not exist: {sorted(missing)}"


def test_every_real_flag_is_documented() -> None:
    """The other direction: an undocumented flag is invisible to users."""
    undocumented = argparse_flags() - documented_cli_flags()
    assert not undocumented, f"CLI flags missing from the README: {sorted(undocumented)}"


def test_eval_harness_exposes_the_documented_shared_flags() -> None:
    """Section 9 requires all harnesses to share this interface."""
    parser = base_parser("test", "test")
    flags = {opt for action in parser._actions for opt in action.option_strings}
    for required in ("--test-set", "--output", "--limit", "--run-agent", "--baseline"):
        assert required in flags, f"shared eval flag {required} is missing"


def test_readme_lists_every_eval_component() -> None:
    text = readme_text()
    for component in COMPONENTS:
        assert f"`{component}`" in text, f"eval component {component!r} is undocumented"


@pytest.mark.parametrize(
    "model",
    [
        "nlpaueb/sec-bert-base",
        "BAAI/bge-base-en-v1.5",
        "takala/financial_phrasebank",
        "nlpaueb/finer-139",
        "virattt/financial-qa-10K",
        "ibm/finqa",
        "Qwen2.5-3B-Instruct",
    ],
)
def test_documented_models_appear_in_code(model: str) -> None:
    """Anti-pattern #4: docs naming a model that was swapped out."""
    assert model in readme_text(), f"{model} vanished from the README"
    sources = list((repo_root() / "src").rglob("*.py")) + list(
        (repo_root() / "training").rglob("*.py")
    )
    blob = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    assert model in blob, f"README names {model} but no source file references it"


def test_config_model_names_match_the_readme() -> None:
    raw = yaml.safe_load((repo_root() / "configs" / "default.yaml").read_text(encoding="utf-8"))
    text = readme_text()
    assert raw["models"]["embedder"]["name"] in text
    assert raw["models"]["xbrl_tagger"]["name"] in text


def test_published_figures_are_separated_from_measured_results() -> None:
    """Anti-pattern #9: a table of published numbers placed where it reads as ours."""
    text = readme_text()
    assert "Published figures — not produced by this repo" in text
    assert "Measured in this repo" in text
    # The paper's headline number must sit after the separation heading.
    measured = text.index("### Measured in this repo")
    published = text.index("### Published figures")
    assert measured < published
    assert text.index("89.2% micro-F1") > published, (
        "the FiNER paper's number appears before the published-figures heading, "
        "where it reads as this repo's result"
    )


def test_readme_states_what_is_not_done() -> None:
    """Section 10 requires a roadmap describing what is not done."""
    text = readme_text()
    assert "## What is not done" in text
    assert "No model has been fine-tuned" in text


def test_disclaimer_is_present_everywhere_it_must_be() -> None:
    from affa import DISCLAIMER
    from affa.config import get_config

    assert "Not investment advice" in readme_text()
    assert get_config().report.disclaimer == DISCLAIMER


def test_running_md_exists_and_covers_a_clean_checkout() -> None:
    assert RUNNING.is_file(), "RUNNING.md is required by section 10"
    text = RUNNING.read_text(encoding="utf-8")
    assert "pip install" in text
    assert "pytest" in text

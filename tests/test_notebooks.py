"""Static checks on the generated notebooks (``scripts/make_notebooks.py``).

CI has no GPU, so a bug that only manifests on real CUDA hardware - like a
wrong attribute name on ``torch.cuda.get_device_properties()`` - cannot be
caught by executing the notebooks. These tests catch what a CPU-only pipeline
*can* catch: the notebooks are in sync with their generator, every Python cell
at least parses, and known-bad attribute names that have actually shipped once
do not ship again.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

from affa.config import repo_root

NOTEBOOK_DIR = repo_root() / "notebooks"
NOTEBOOKS = [
    "01_xbrl_tagger.ipynb",
    "02_retrieval_embedder.ipynb",
    "03_sentiment.ipynb",
    "04_finqa_qlora.ipynb",
]

# Attribute names that have actually shipped in a generated notebook and only
# fail on real GPU hardware, so no CPU-only check catches them except a named
# regression test. One entry per incident, please - this is a graveyard, not a
# general lint rule.
KNOWN_BAD_SNIPPETS: tuple[tuple[str, str], ...] = (
    (
        ".total_mem ",
        "torch.cuda.get_device_properties(0) has no `total_mem` attribute; "
        "the real one is `total_memory`. Reproduced on a real T4 in Colab - "
        "'AttributeError: _CudaDeviceProperties object has no attribute "
        "total_mem'. CPU-only CI cannot execute this cell to catch it.",
    ),
    (
        "agentic-financial-filing-analyst",
        "typo repo name - 'analyst' where the real repo is 'analysis'. "
        "REPO_DIR was a separately hand-typed constant that drifted from "
        "REPO_URL; the generator now derives REPO_DIR from REPO_URL so they "
        "cannot diverge again. A notebook with this typo `cd`s into a "
        "directory `git clone` never created and fails on its very first "
        "cell - only on a real clone, never under a static check that doesn't "
        "diff the two constants against each other.",
    ),
)


def load_notebook(name: str) -> dict:
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def cell_sources(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def strip_notebook_magics(source: str) -> str:
    """Drop shell (`!`) and IPython magic (`%`) lines, continuations included,
    so the rest can be parsed as plain Python. They are valid in a notebook
    cell and invalid everywhere else, and are not what these tests check.

    A `!command \` that wraps across lines is one shell statement; only its
    first line starts with `!`, so a naive per-line filter leaves the
    continuation lines behind as orphaned Python and fails to parse them.
    """
    kept: list[str] = []
    in_shell_continuation = False
    for line in source.splitlines():
        if in_shell_continuation:
            in_shell_continuation = line.rstrip().endswith("\\")
            continue
        if line.lstrip().startswith(("!", "%")):
            in_shell_continuation = line.rstrip().endswith("\\")
            continue
        kept.append(line)
    return "\n".join(kept)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_in_sync_with_its_generator(name: str, tmp_path) -> None:
    """The .ipynb files are generated output. Hand-edits or a stale generator
    run both show up here, the same way CI's notebook-sync step catches them.
    """
    committed = (NOTEBOOK_DIR / name).read_bytes()

    result = subprocess.run(
        [sys.executable, "scripts/make_notebooks.py"],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    regenerated = (NOTEBOOK_DIR / name).read_bytes()

    assert committed == regenerated, (
        f"{name} does not match its generator's output. Edit "
        "scripts/make_notebooks.py, not the .ipynb file, then re-run it."
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_code_cells_parse_as_python(name: str) -> None:
    """Every code cell, magics stripped, must at least be syntactically valid.

    Does not prove the cell runs correctly - it needs the actual GPU/Colab
    environment for that - but a SyntaxError is worth catching before someone
    hits it interactively three cells into a training run.
    """
    notebook = load_notebook(name)
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = strip_notebook_magics("".join(cell["source"]))
        if not source.strip():
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{name} cell {i} does not parse: {exc}\n---\n{source}")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_known_bad_snippets(name: str) -> None:
    """Regression test for bugs that only surface on real GPU hardware."""
    source = "".join(cell_sources(load_notebook(name)))
    for snippet, explanation in KNOWN_BAD_SNIPPETS:
        assert snippet not in source, f"{name}: {explanation}"


def test_gpu_check_cell_uses_the_real_torch_attribute() -> None:
    """Pins the specific fix: `total_memory`, not `total_mem`."""
    for name in NOTEBOOKS:
        source = "".join(cell_sources(load_notebook(name)))
        if "get_device_properties" in source:
            assert "get_device_properties(0).total_memory" in source, (
                f"{name} calls get_device_properties but not with .total_memory"
            )


def test_repo_dir_cannot_drift_from_repo_url() -> None:
    """REPO_DIR must be derived from REPO_URL, never a second hand-typed constant.

    Regression: REPO_URL said '...-analyst.git' and REPO_DIR said '-analyst'
    while the real repo is '...-analysis' - the two constants agreed with each
    other and disagreed with reality, so nothing that only compares them to
    each other would have caught it. This checks the generator's *source code*
    computes REPO_DIR from REPO_URL rather than assigning it a literal, which
    is what makes the two impossible to desynchronize again.
    """
    from scripts import make_notebooks

    generator_source = (repo_root() / "scripts" / "make_notebooks.py").read_text(encoding="utf-8")
    assert 'REPO_DIR = REPO_URL.rsplit("/", 1)[-1].removesuffix(".git")' in generator_source, (
        "REPO_DIR is no longer derived from REPO_URL in scripts/make_notebooks.py - "
        "a hand-typed REPO_DIR constant is exactly what drifted into a typo before"
    )
    assert make_notebooks.REPO_URL.rsplit("/", 1)[-1].removesuffix(".git") == (
        make_notebooks.REPO_DIR
    )


def test_notebooks_clone_the_actual_project_repo() -> None:
    """The clone target must be a real, reachable repo - not a template
    placeholder left over from generation.
    """
    for name in NOTEBOOKS:
        source = "".join(cell_sources(load_notebook(name)))
        assert "YOUR_USERNAME" not in source.split("HUB_ID")[0], (
            f"{name}: the repo-clone cell still has a placeholder URL"
        )

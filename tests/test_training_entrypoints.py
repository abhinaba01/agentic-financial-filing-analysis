"""Every training script must be runnable the way the docs say to run it.

Regression: all four scripts did ``sys.path.insert(0, .../"src")`` and then
``from training.common import ...``. Running ``python training/train_x.py`` puts
only ``training/`` on ``sys.path``, so the repo root was never importable and
every script died at import with ``ModuleNotFoundError: No module named
'training'``.

The suite missed it because pytest injects both paths via ``pythonpath`` in
pyproject.toml, so the modules imported fine under test and only failed for a
real user. These tests therefore run the scripts as *subprocesses*, with the
same entry point RUNNING.md and the notebooks use.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from affa.config import repo_root

SCRIPTS = [
    "train_xbrl_tagger.py",
    "train_retrieval.py",
    "train_sentiment.py",
    "train_finqa_qlora.py",
]


def run_script(script: str, *args: str, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo_root() / "training" / script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=180,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_imports_when_run_directly(script: str) -> None:
    """``python training/<script> --help`` must reach argparse, not die importing.

    ``--help`` is the cheapest possible probe: it exercises every module-level
    import and exits before touching a dataset or a GPU.
    """
    result = run_script(script, "--help", cwd=repo_root())
    assert "ModuleNotFoundError" not in result.stderr, (
        f"{script} cannot be run directly:\n{result.stderr}"
    )
    assert result.returncode == 0, f"{script} --help failed:\n{result.stderr}"
    assert "--output-dir" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_from_any_working_directory(script: str, tmp_path) -> None:
    """Path resolution must be anchored to the file, not to the caller's cwd.

    Colab runs these after ``os.chdir(REPO_DIR)``, but a user may not.
    """
    result = run_script(script, "--help", cwd=tmp_path)
    assert "ModuleNotFoundError" not in result.stderr, (
        f"{script} breaks when run from another directory:\n{result.stderr}"
    )
    assert result.returncode == 0


@pytest.mark.parametrize("script", SCRIPTS)
def test_output_dir_is_required(script: str) -> None:
    """Omitting --output-dir must fail fast, before any download starts."""
    result = run_script(script, cwd=repo_root())
    assert result.returncode != 0
    assert "--output-dir" in result.stderr


def test_documented_training_commands_use_the_real_flags() -> None:
    """RUNNING.md's commands must match the scripts' actual argparse."""
    running = (repo_root() / "RUNNING.md").read_text(encoding="utf-8")
    for script in SCRIPTS:
        assert f"training/{script}" in running, f"{script} is undocumented in RUNNING.md"

    help_text = run_script("train_xbrl_tagger.py", "--help", cwd=repo_root()).stdout
    for flag in ("--output-dir", "--train-samples", "--allow-ephemeral"):
        assert flag in help_text, f"{flag} is documented but missing from argparse"


def test_common_imports_torch_before_anything_else_can() -> None:
    """Windows DLL-load ordering regression.

    On Windows, ``pyarrow`` (pulled in transitively by ``datasets``) bundles its
    own copies of the MSVC runtime DLLs. If those load into the process before
    torch's, torch's ``c10.dll`` can fail to initialize - reproduced running
    this project's own training scripts, where ``require_datasets_below_4()``
    imported ``datasets`` before anything imported ``torch``. Fixed by importing
    torch at the top of ``training/common.py``, so it is first regardless of
    which function is called first.

    Run as a fresh subprocess rather than by popping ``sys.modules`` and
    reimporting in-process: torch 2.x registers native ``TORCH_LIBRARY``
    namespaces on import and correctly refuses to do so twice in one process,
    so an in-process reimport crashes for a reason that has nothing to do with
    the defence being tested.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import training.common, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=str(repo_root()),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True", (
        "training.common no longer imports torch eagerly; the Windows "
        "DLL-ordering defense (pyarrow-before-torch) has regressed"
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_training_common_is_imported_before_datasets_in_source(script: str) -> None:
    """The eager torch import only helps if it runs before `datasets` does.

    Guards the actual invariant the fix depends on: each script's first
    project-level import must be `training.common` (or come before any
    `datasets` import), not just that `training.common` imports torch in
    isolation.
    """
    source = (repo_root() / "training" / script).read_text(encoding="utf-8")
    common_pos = source.index("from training.common import")
    dataset_positions = [
        source.index(needle)
        for needle in ("import datasets", "from datasets import")
        if needle in source
    ]
    assert dataset_positions, f"{script} does not import datasets at all"
    assert common_pos < min(dataset_positions), (
        f"{script} imports datasets before training.common; the Windows "
        "DLL-ordering defense only works if torch loads first"
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_script_guards_accelerate_before_training(script: str) -> None:
    """Regression: a missing `accelerate` surfaced as a transformers internals
    traceback four frames deep inside `TrainingArguments.__post_init__`,
    reproduced running this project's own training scripts. Every script uses
    `Trainer` or its `SentenceTransformerTrainer` subclass, so every script
    must call the guard before constructing training arguments.
    """
    source = (repo_root() / "training" / script).read_text(encoding="utf-8")
    assert "require_accelerate" in source, f"{script} does not guard for accelerate"

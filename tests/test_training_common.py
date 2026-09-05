"""Checkpointing and resume guards (section 5.5, anti-patterns #13 and #14).

Untested resume logic is usually broken resume logic, and the moment you find
out is the moment you have already lost the run. These tests exercise the guards
without needing a GPU or a training run.
"""

from __future__ import annotations

import json

import pytest

from training.common import (
    RUN_CONFIG_FILENAME,
    ResumeMismatchError,
    RunConfig,
    assert_checkpoint_selection_is_valid,
    resolve_checkpoint_dir,
    resume_checkpoint,
)


def make_config(**overrides) -> RunConfig:
    base = dict(
        task="xbrl_tagger",
        base_model="nlpaueb/sec-bert-base",
        dataset="nlpaueb/finer-139",
        seed=42,
        train_samples=200_000,
        eval_samples=10_000,
        max_length=256,
        learning_rate=3e-5,
        per_device_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=2.0,
    )
    base.update(overrides)
    return RunConfig(**base)


def test_ephemeral_checkpoint_dir_is_refused() -> None:
    """Anti-pattern #13: /content vanishes with the VM, which is the whole failure."""
    with pytest.raises(ValueError, match="ephemeral"):
        resolve_checkpoint_dir("/content/outputs")
    with pytest.raises(ValueError, match="ephemeral"):
        resolve_checkpoint_dir("./results")


def test_drive_path_is_accepted(tmp_path) -> None:
    drive = tmp_path / "drive" / "MyDrive" / "affa"
    assert resolve_checkpoint_dir(drive).is_dir()


def test_ephemeral_allowed_when_explicit(tmp_path) -> None:
    assert resolve_checkpoint_dir(tmp_path / "scratch", allow_ephemeral=True).is_dir()


def test_first_run_writes_the_run_config(tmp_path) -> None:
    directory = resolve_checkpoint_dir(tmp_path / "drive" / "ckpt")
    assert resume_checkpoint(directory, make_config()) is None
    written = json.loads((directory / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert written["seed"] == 42
    assert written["train_samples"] == 200_000


def test_resume_refuses_a_changed_seed(tmp_path) -> None:
    """Anti-pattern #14: the global step then indexes into different data."""
    directory = resolve_checkpoint_dir(tmp_path / "drive" / "ckpt")
    make_config().save(directory)
    (directory / "checkpoint-500").mkdir()

    with pytest.raises(ResumeMismatchError, match="seed"):
        resume_checkpoint(directory, make_config(seed=1234))


def test_resume_refuses_a_changed_subset_size(tmp_path) -> None:
    directory = resolve_checkpoint_dir(tmp_path / "drive" / "ckpt")
    make_config().save(directory)
    (directory / "checkpoint-500").mkdir()

    with pytest.raises(ResumeMismatchError, match="train_samples"):
        resume_checkpoint(directory, make_config(train_samples=50_000))


def test_resume_refuses_a_changed_base_model(tmp_path) -> None:
    directory = resolve_checkpoint_dir(tmp_path / "drive" / "ckpt")
    make_config().save(directory)
    (directory / "checkpoint-500").mkdir()

    with pytest.raises(ResumeMismatchError, match="base_model"):
        resume_checkpoint(directory, make_config(base_model="bert-base-uncased"))


def test_resume_refuses_a_checkpoint_with_no_run_config(tmp_path) -> None:
    """Unknown settings mean an unverifiable resume, so it is refused."""
    directory = resolve_checkpoint_dir(tmp_path / "drive" / "ckpt")
    (directory / "checkpoint-500").mkdir()
    with pytest.raises(ResumeMismatchError, match="no affa_run_config"):
        resume_checkpoint(directory, make_config())


def test_matching_settings_resume_cleanly(tmp_path) -> None:
    directory = resolve_checkpoint_dir(tmp_path / "drive" / "ckpt")
    make_config().save(directory)
    (directory / "checkpoint-500").mkdir()

    resumed = resume_checkpoint(directory, make_config())
    assert resumed is not None
    assert "checkpoint-500" in str(resumed)


def test_resume_is_idempotent(tmp_path) -> None:
    """Re-running the cell after a crash resumes, with no code edit."""
    directory = resolve_checkpoint_dir(tmp_path / "drive" / "ckpt")
    config = make_config()
    assert resume_checkpoint(directory, config) is None  # fresh
    (directory / "checkpoint-500").mkdir()
    first = resume_checkpoint(directory, config)
    second = resume_checkpoint(directory, config)
    assert first == second


def test_run_config_differences_are_reported() -> None:
    diffs = make_config(seed=7).differences(make_config())
    assert "seed" in diffs
    assert diffs["seed"] == (42, 7)


def test_reporting_split_cannot_be_the_selection_split() -> None:
    """Anti-pattern #7: reporting a fine-tune's score on the split that chose it."""
    with pytest.raises(ValueError, match="Select on validation"):
        assert_checkpoint_selection_is_valid(selection_split="test", reporting_split="test")
    assert_checkpoint_selection_is_valid(selection_split="validation", reporting_split="test")

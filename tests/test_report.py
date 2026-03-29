"""Tests for golem.report."""

import math

from golem.report import _compute_summary


def _row(*, epoch: int, val_loss: float) -> dict:
    return {
        "epoch": epoch,
        "train_loss": 1.5 + epoch,
        "val_loss": val_loss,
        "train_descriptor_loss": 1.0 + epoch,
        "val_descriptor_loss": val_loss,
        "val_rmse": 2.0 + epoch,
        "learning_rate": 1e-4,
        "elapsed_seconds": 10.0 * (epoch + 1),
        "train_alignment_loss": math.nan,
        "val_alignment_loss": math.nan,
        "val_alignment_spearman": math.nan,
        "val_alignment_kendall": math.nan,
    }


def test_compute_summary_ignores_nonfinite_val_loss_when_selecting_best():
    summary = _compute_summary(
        [
            _row(epoch=0, val_loss=math.nan),
            _row(epoch=1, val_loss=0.8),
            _row(epoch=2, val_loss=1.1),
        ],
        {},
    )

    assert summary["best_epoch"] == 1
    assert summary["best_epoch_display"] == 2
    assert summary["best_val_objective"] == 0.8


def test_compute_summary_falls_back_to_last_row_when_all_val_losses_are_nonfinite():
    summary = _compute_summary(
        [
            _row(epoch=0, val_loss=math.nan),
            _row(epoch=1, val_loss=math.nan),
        ],
        {},
    )

    assert summary["best_epoch"] == 1
    assert summary["best_epoch_display"] == 2
    assert math.isnan(summary["best_val_objective"])

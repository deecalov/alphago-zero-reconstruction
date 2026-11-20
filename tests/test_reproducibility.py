from __future__ import annotations

"""Compare two CPU pipeline runs initialized from the same seed."""

import copy
from pathlib import Path

import pytest

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.pipeline import run_training_pipeline


@pytest.mark.slow
def test_same_seed_gives_same_structural_metrics(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    cfg_a = tiny_config
    cfg_a.runtime.work_dir = tmp_path / "run_a"

    # Use the same settings and seed in a separate run directory.
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b.runtime.work_dir = tmp_path / "run_b"

    summary_a = run_training_pipeline(cfg_a)
    summary_b = run_training_pipeline(cfg_b)

    assert summary_a["global_step"] == summary_b["global_step"]
    assert summary_a["best_iteration"] == summary_b["best_iteration"]
    assert summary_a["replay_buffer_size"] == summary_b["replay_buffer_size"]

from __future__ import annotations

"""Validate explicit device selection and CUDA fallback behavior."""

import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.pipeline import run_training_pipeline


def test_invalid_device_raises(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.runtime.device = "definitely_not_a_device"
    with pytest.raises(ValueError):
        run_training_pipeline(tiny_config)


@pytest.mark.slow
def test_cuda_falls_back_to_cpu_when_unavailable(tiny_config: AlphaGoZeroConfig) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, so the CPU fallback path cannot be exercised")
    tiny_config.runtime.device = "cuda"
    summary = run_training_pipeline(tiny_config)
    assert summary["device"] == "cpu"

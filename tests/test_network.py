from __future__ import annotations

"""Policy/value network output shapes, ranges and inference-mode tests."""

import numpy as np
import pytest
import torch

from alphago_zero.config import NetworkConfig
from alphago_zero.network import AlphaGoZeroNet

ACTION_SIZE = 5 * 5 + 1


def test_forward_output_shapes(tiny_model: AlphaGoZeroNet, tiny_network_config: NetworkConfig) -> None:
    tiny_model.eval()
    x = torch.zeros(3, 17, 5, 5)
    policy_logits, value = tiny_model(x)
    assert policy_logits.shape == (3, tiny_network_config.action_size)
    assert value.shape == (3, 1)


def test_value_is_bounded(tiny_model: AlphaGoZeroNet) -> None:
    tiny_model.eval()
    x = torch.randn(4, 17, 5, 5)
    _, value = tiny_model(x)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_predict_single_shapes(tiny_model: AlphaGoZeroNet) -> None:
    encoded = np.zeros((17, 5, 5), dtype=np.float32)
    logits, value = tiny_model.predict_single(encoded)
    assert logits.shape == (ACTION_SIZE,)
    assert isinstance(value, float)


def test_predict_single_wrong_ndim_raises(tiny_model: AlphaGoZeroNet) -> None:
    with pytest.raises(ValueError):
        tiny_model.predict_single(np.zeros((5, 5), dtype=np.float32))


def test_predict_single_restores_training_mode(tiny_model: AlphaGoZeroNet) -> None:
    encoded = np.zeros((17, 5, 5), dtype=np.float32)

    # Restore the original training mode after predict_single.
    tiny_model.train()
    tiny_model.predict_single(encoded)
    assert tiny_model.training is True

    # A model already in evaluation mode stays there.
    tiny_model.eval()
    tiny_model.predict_single(encoded)
    assert tiny_model.training is False


def test_eval_mode_is_deterministic(tiny_model: AlphaGoZeroNet) -> None:
    tiny_model.eval()
    x = torch.randn(2, 17, 5, 5)
    with torch.no_grad():
        p1, v1 = tiny_model(x)
        p2, v2 = tiny_model(x)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(v1, v2)


def test_summary_dict_keys(tiny_model: AlphaGoZeroNet) -> None:
    summary = tiny_model.summary_dict()
    for key in (
        "board_size",
        "action_size",
        "channels",
        "num_res_blocks",
        "total_parameters",
        "trainable_parameters",
    ):
        assert key in summary
    assert summary["total_parameters"] > 0

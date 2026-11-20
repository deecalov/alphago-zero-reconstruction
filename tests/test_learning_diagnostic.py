"""Validate the measurement used by the long GPU learning experiment."""

import math

import numpy as np
import pytest
import torch

from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.replay_buffer import TrainingExample
from examples.verify_learning import prediction_errors


def test_holdout_metrics_weight_partial_batch_and_do_not_update_model(
    tiny_model: AlphaGoZeroNet,
) -> None:
    # Uniform predictions give analytic CE/KL. A final one-position batch
    # catches accidentally averaging batch means rather than all positions.
    with torch.no_grad():
        for parameter in tiny_model.parameters():
            parameter.zero_()
    tiny_model.train()
    before = {key: tensor.clone() for key, tensor in tiny_model.state_dict().items()}
    state = np.zeros((17, 5, 5), dtype=np.float32)
    uniform = np.full(26, 1 / 26, dtype=np.float32)
    point_mass = np.zeros(26, dtype=np.float32)
    point_mass[0] = 1
    examples = [TrainingExample(state, uniform, 0.0) for _ in range(256)]
    examples.append(TrainingExample(state, point_mass, -1.0))

    result = prediction_errors(tiny_model, examples)

    assert result["policy_ce"] == pytest.approx(math.log(26), abs=1e-6)
    assert result["policy_kl"] == pytest.approx(math.log(26) / 257, abs=1e-6)
    assert result["value_mse"] == pytest.approx(1 / 257)
    assert result["total_loss"] == pytest.approx(math.log(26) + 1 / 257, abs=1e-6)
    assert not tiny_model.training
    assert all(torch.equal(tensor, tiny_model.state_dict()[key]) for key, tensor in before.items())

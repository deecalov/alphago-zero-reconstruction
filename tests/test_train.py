from __future__ import annotations

"""Check training losses, optimizer parameter groups and learning-rate schedules."""

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig, make_demo_config, paper_learning_rate
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.replay_buffer import ReplayBuffer, TrainingExample
from alphago_zero.train import (
    _compute_losses,
    _get_learning_rate,
    create_optimizer,
    train_for_steps,
)

BOARD_SIZE = 5
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1


def _make_example(seed: int) -> TrainingExample:
    rng = np.random.default_rng(seed)
    state = rng.random((17, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    policy = rng.random(ACTION_SIZE).astype(np.float32)
    policy /= policy.sum()
    value = 1.0 if seed % 2 == 0 else -1.0
    return TrainingExample(state=state, policy=policy, value=value)


def test_compute_losses_known_values() -> None:
    # value_pred == target_value -> value loss = 0.
    # Uniform logits and a uniform target give cross-entropy log(num_actions).
    policy_logits = torch.zeros(2, 4)
    value_pred = torch.tensor([[0.5], [-0.5]])
    target_policy = torch.full((2, 4), 0.25)
    target_value = torch.tensor([0.5, -0.5])

    total, policy_loss, value_loss = _compute_losses(
        policy_logits, value_pred, target_policy, target_value
    )
    assert value_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert policy_loss.item() == pytest.approx(float(np.log(4)), abs=1e-5)
    assert total.item() == pytest.approx(policy_loss.item() + value_loss.item(), abs=1e-6)


def test_zero_training_steps_is_noop_with_nonempty_buffer(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator,
) -> None:
    tiny_config.training.gradient_steps_per_iteration = 0
    tiny_config.validate()
    buffer = ReplayBuffer(8, BOARD_SIZE, rng)
    buffer.add(_make_example(0))
    before = {key: tensor.clone() for key, tensor in tiny_model.state_dict().items()}
    rng_before = rng.bit_generator.state
    optimizer = create_optimizer(tiny_model, tiny_config)
    metrics, step = train_for_steps(tiny_model, optimizer, buffer, tiny_config, rng, global_step=17)
    assert metrics.steps == 0 and metrics.total_loss == 0.0 and step == 17
    assert not optimizer.state
    assert rng.bit_generator.state == rng_before
    assert all(torch.equal(tensor, tiny_model.state_dict()[key]) for key, tensor in before.items())


def test_train_reduces_loss_on_fixed_data(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    tiny_config.training.gradient_steps_per_iteration = 60
    tiny_config.training.batch_size = 4

    buffer = ReplayBuffer(capacity=64, board_size=BOARD_SIZE, rng=rng)
    for i in range(8):
        buffer.add(_make_example(seed=i))

    optimizer = create_optimizer(tiny_model, tiny_config)
    metrics1, step1 = train_for_steps(tiny_model, optimizer, buffer, tiny_config, rng, global_step=0)
    metrics2, step2 = train_for_steps(
        tiny_model, optimizer, buffer, tiny_config, rng, global_step=step1
    )

    # A fixed training set should be small enough to overfit and reduce mean loss.
    assert metrics2.total_loss < metrics1.total_loss
    assert step1 == 60
    assert step2 == 120


def test_default_optimizer_applies_weight_decay_to_every_parameter(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    # The default optimizer decays all parameters in one group.
    optimizer = create_optimizer(tiny_model, tiny_config)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["weight_decay"] == tiny_config.training.weight_decay
    assert {id(p) for p in group["params"]} == {id(p) for p in tiny_model.parameters()}


def test_exclude_norm_bias_splits_parameters_by_rank(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    tiny_config.training.weight_decay_exclude_norm_bias = True
    optimizer = create_optimizer(tiny_model, tiny_config)
    assert len(optimizer.param_groups) == 2
    decayed, plain = optimizer.param_groups
    assert decayed["weight_decay"] == tiny_config.training.weight_decay
    assert plain["weight_decay"] == 0.0
    assert decayed["params"] and plain["params"]
    # Apply optional decay to conv/linear weights and exclude one-dimensional biases and BatchNorm
    # parameters.
    assert all(p.ndim >= 2 for p in decayed["params"])
    assert all(p.ndim < 2 for p in plain["params"])
    # Each parameter appears in exactly one group.
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in tiny_model.parameters())
    plain_ids = {id(p) for p in plain["params"]}
    batch_norms = [m for m in tiny_model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert batch_norms
    for module in batch_norms:
        assert id(module.weight) in plain_ids and id(module.bias) in plain_ids


@pytest.mark.parametrize("exclude", [False, True])
def test_weight_decay_step_touches_only_decayed_parameters(
    tiny_config: AlphaGoZeroConfig, exclude: bool
) -> None:
    # With zero gradients, an SGD step isolates weight decay: p <- p * (1 - lr * wd).
    tiny_config.training.learning_rate = 0.1
    tiny_config.training.weight_decay = 0.1
    tiny_config.training.weight_decay_exclude_norm_bias = exclude
    torch.manual_seed(0)
    model = AlphaGoZeroNet(tiny_config.network)
    optimizer = create_optimizer(model, tiny_config)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()

    shrink = 1.0 - 0.1 * 0.1
    for name, param in model.named_parameters():
        decayed = param.ndim >= 2 or not exclude
        expected = before[name] * shrink if decayed else before[name]
        assert torch.allclose(param.detach(), expected, atol=1e-7), name

    # BatchNorm gamma starts at one.
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert bn.weight.detach().mean().item() == pytest.approx(1.0 if exclude else shrink)


def test_train_for_steps_sets_lr_on_every_param_group(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    tiny_config.training.weight_decay_exclude_norm_bias = True
    tiny_config.training.gradient_steps_per_iteration = 3
    tiny_config.training.batch_size = 4
    buffer = ReplayBuffer(capacity=64, board_size=BOARD_SIZE, rng=rng)
    for i in range(8):
        buffer.add(_make_example(seed=i))

    optimizer = create_optimizer(tiny_model, tiny_config)
    metrics, step = train_for_steps(tiny_model, optimizer, buffer, tiny_config, rng, global_step=0)

    assert step == 3 and np.isfinite(metrics.total_loss)
    assert [g["lr"] for g in optimizer.param_groups] == [tiny_config.training.learning_rate] * 2


def test_learning_rate_schedule_modes() -> None:
    cfg = make_demo_config()
    cfg.training.learning_rate = 0.01

    cfg.training.lr_schedule = "constant"
    assert _get_learning_rate(0, cfg) == pytest.approx(0.01)
    assert _get_learning_rate(123456, cfg) == pytest.approx(0.01)

    cfg.training.lr_schedule = "paper"
    # The paper schedule also starts at 1e-2.
    assert _get_learning_rate(0, cfg) == pytest.approx(1e-2)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 1e-2),
        (199_999, 1e-2),
        (200_000, 1e-2),
        (399_999, 1e-2),
        (400_000, 1e-3),  # First step from 1e-2 to 1e-3.
        (599_999, 1e-3),
        (600_000, 1e-4),  # Second step from 1e-3 to 1e-4.
        (699_999, 1e-4),
        (800_000, 1e-4),
        (10_000_000, 1e-4),
    ],
)
def test_paper_learning_rate_breakpoints(step: int, expected: float) -> None:
    # Extended Data Table 3 uses 1e-2 up to 400k steps, then 1e-3 up to 600k and 1e-4 thereafter.
    assert paper_learning_rate(step) == pytest.approx(expected, rel=1e-12)


def test_paper_learning_rate_never_increases() -> None:
    steps = range(0, 1_000_001, 10_000)
    rates = [paper_learning_rate(step) for step in steps]
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))


def test_paper_schedule_overrides_constant_learning_rate() -> None:
    # The paper schedule overrides the fixed learning_rate field.
    cfg = make_demo_config()
    cfg.training.lr_schedule = "paper"
    cfg.training.learning_rate = 0.123
    assert _get_learning_rate(450_000, cfg) == pytest.approx(1e-3)
    assert _get_learning_rate(650_000, cfg) == pytest.approx(1e-4)


def test_train_on_empty_buffer_does_nothing(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    buffer = ReplayBuffer(capacity=10, board_size=BOARD_SIZE, rng=rng)
    optimizer = create_optimizer(tiny_model, tiny_config)
    metrics, step = train_for_steps(tiny_model, optimizer, buffer, tiny_config, rng, global_step=0)
    assert metrics.steps == 0
    assert step == 0

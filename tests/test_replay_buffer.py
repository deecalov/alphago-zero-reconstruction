from __future__ import annotations

"""Replay ring eviction, sampling and augmentation tests."""

import numpy as np
import pytest

from alphago_zero.replay_buffer import ReplayBuffer, TrainingExample

BOARD_SIZE = 5
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1


def _make_example(value: float = 1.0, seed: int = 0) -> TrainingExample:
    rng = np.random.default_rng(seed)
    state = rng.random((17, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    policy = rng.random(ACTION_SIZE).astype(np.float32)
    policy /= policy.sum()
    return TrainingExample(state=state, policy=policy, value=value)


def test_add_and_len() -> None:
    buf = ReplayBuffer(capacity=10, board_size=BOARD_SIZE)
    assert len(buf) == 0
    buf.add(_make_example())
    assert len(buf) == 1


def test_capacity_eviction_keeps_latest() -> None:
    buf = ReplayBuffer(capacity=3, board_size=BOARD_SIZE)
    for i in range(5):
        buf.add(_make_example(value=float(i), seed=i))
    assert len(buf) == 3
    # Retain the newest three examples with values 2, 3 and 4.
    assert {ex.value for ex in buf._buffer} == {2.0, 3.0, 4.0}


def test_sample_batch_shapes() -> None:
    buf = ReplayBuffer(capacity=100, board_size=BOARD_SIZE, rng=np.random.default_rng(0))
    for i in range(20):
        buf.add(_make_example(seed=i))
    states, policies, values = buf.sample_batch(batch_size=8, augment=False)
    assert states.shape == (8, 17, BOARD_SIZE, BOARD_SIZE)
    assert policies.shape == (8, ACTION_SIZE)
    assert values.shape == (8,)


def test_sample_with_replacement_when_underfilled() -> None:
    buf = ReplayBuffer(capacity=100, board_size=BOARD_SIZE, rng=np.random.default_rng(0))
    buf.add(_make_example(seed=1))  # The buffer contains one example.
    states, _, _ = buf.sample_batch(batch_size=8, augment=False)
    assert states.shape == (8, 17, BOARD_SIZE, BOARD_SIZE)  # Sampling with replacement can duplicate it.


def test_empty_buffer_raises() -> None:
    buf = ReplayBuffer(capacity=10, board_size=BOARD_SIZE)
    with pytest.raises(ValueError):
        buf.sample_batch(batch_size=4)


def test_augment_keeps_value_and_policy_mass() -> None:
    buf = ReplayBuffer(capacity=10, board_size=BOARD_SIZE, rng=np.random.default_rng(0))
    buf.add(_make_example(value=1.0, seed=5))
    _, policies, values = buf.sample_batch(batch_size=4, augment=True)
    # Augmentation preserves the scalar value.
    assert np.all(values == 1.0)
    # The policy remains a probability distribution.
    np.testing.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-5)

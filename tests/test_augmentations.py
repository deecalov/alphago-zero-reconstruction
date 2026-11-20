from __future__ import annotations

"""Properties of the eight square-board symmetries."""

import numpy as np
import pytest

from alphago_zero.augmentations import (
    INVERSE_DIHDERAL,
    apply_dihedral_to_planes,
    apply_dihedral_to_policy,
    augment_state_and_policy,
    invert_dihedral_on_policy,
)

BOARD_SIZE = 5
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1


def test_inverse_table_is_permutation() -> None:
    assert sorted(INVERSE_DIHDERAL.keys()) == list(range(8))
    assert sorted(INVERSE_DIHDERAL.values()) == list(range(8))


@pytest.mark.parametrize("transform_id", range(8))
def test_planes_roundtrip(transform_id: int) -> None:
    rng = np.random.default_rng(transform_id)
    planes = rng.random((17, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    transformed = apply_dihedral_to_planes(planes, transform_id)
    restored = apply_dihedral_to_planes(transformed, INVERSE_DIHDERAL[transform_id])
    np.testing.assert_allclose(restored, planes, atol=1e-6)


@pytest.mark.parametrize("transform_id", range(8))
def test_policy_roundtrip(transform_id: int) -> None:
    rng = np.random.default_rng(100 + transform_id)
    policy = rng.random(ACTION_SIZE).astype(np.float32)
    transformed = apply_dihedral_to_policy(policy, BOARD_SIZE, transform_id)
    restored = invert_dihedral_on_policy(transformed, BOARD_SIZE, transform_id)
    np.testing.assert_allclose(restored, policy, atol=1e-6)


def test_identity_transform_is_noop() -> None:
    planes = np.random.default_rng(7).random((17, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    np.testing.assert_array_equal(apply_dihedral_to_planes(planes, 0), planes)


@pytest.mark.parametrize("transform_id", range(8))
def test_pass_element_unchanged(transform_id: int) -> None:
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[-1] = 0.7  # Pass probability.
    transformed = apply_dihedral_to_policy(policy, BOARD_SIZE, transform_id)
    assert transformed[-1] == pytest.approx(0.7)


def test_augment_preserves_shape_and_mass() -> None:
    rng = np.random.default_rng(3)
    state = rng.random((17, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    policy = rng.random(ACTION_SIZE).astype(np.float32)
    policy /= policy.sum()

    aug_state, aug_policy = augment_state_and_policy(state, policy, BOARD_SIZE, rng)
    assert aug_state.shape == state.shape
    assert aug_policy.shape == policy.shape
    # A permutation preserves total probability.
    assert aug_policy.sum() == pytest.approx(policy.sum(), abs=1e-5)
    assert aug_policy[-1] == pytest.approx(policy[-1])  # Pass stays fixed.

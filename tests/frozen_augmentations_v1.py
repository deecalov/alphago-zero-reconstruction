# Frozen augmentation implementation from before the search optimizations.
#
# Use np.rot90 and np.flip as the independent symmetry reference.
# (tests/test_search_fast_paths.py).
from __future__ import annotations

"""The eight rotations and reflections of a square Go board. Apply the same transformation to the input
planes and the policy. After inference, map the predicted moves back into the original orientation.
"""

import numpy as np


def apply_dihedral_to_planes(planes: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply one of the eight symmetries to a tensor of shape [C, H, W]."""
    if transform_id < 4:
        # Rotate by 0, 90, 180 or 270 degrees.
        return np.rot90(planes, k=transform_id, axes=(-2, -1)).copy()

    # Reflect left to right before rotating.
    flipped = np.flip(planes, axis=-1)
    return np.rot90(flipped, k=transform_id - 4, axes=(-2, -1)).copy()


def apply_dihedral_to_board(board_2d: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply the same symmetry to a single board of shape [H, W]."""
    if transform_id < 4:
        return np.rot90(board_2d, k=transform_id, axes=(-2, -1)).copy()
    flipped = np.flip(board_2d, axis=-1)
    return np.rot90(flipped, k=transform_id - 4, axes=(-2, -1)).copy()


def _compute_inverse_table() -> dict[int, int]:
    """Find each inverse by composing transformations on a numbered board."""
    template = np.arange(9, dtype=np.int32).reshape(3, 3)
    inverse: dict[int, int] = {}
    for a in range(8):
        ta = apply_dihedral_to_board(template, a)
        for b in range(8):
            if np.array_equal(apply_dihedral_to_board(ta, b), template):
                inverse[a] = b
                break
    return inverse


INVERSE_DIHDERAL = _compute_inverse_table()


def apply_dihedral_to_policy(policy: np.ndarray, board_size: int, transform_id: int) -> np.ndarray:
    """Transform a policy vector in the same way as a board. Preserve its final entry for pass."""
    board_part = policy[:-1].reshape(board_size, board_size)
    transformed = apply_dihedral_to_board(board_part, transform_id).reshape(-1)
    return np.concatenate([transformed, policy[-1:]], axis=0)


def invert_dihedral_on_policy(policy: np.ndarray, board_size: int, transform_id: int) -> np.ndarray:
    """Map a policy from the transformed orientation back to the original board."""
    inverse_id = INVERSE_DIHDERAL[transform_id]
    return apply_dihedral_to_policy(policy, board_size, inverse_id)


def random_dihedral_index(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 8))


def augment_state_and_policy(
    state_planes: np.ndarray,
    policy: np.ndarray,
    board_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a random symmetry for training."""
    t = random_dihedral_index(rng)
    return (
        apply_dihedral_to_planes(state_planes, t),
        apply_dihedral_to_policy(policy, board_size, t),
    )

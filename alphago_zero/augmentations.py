from __future__ import annotations

"""The eight rotations and reflections of a square Go board. Apply the same transformation to the input
planes and the policy. After inference, map the predicted moves back into the original orientation.
"""

import math
from functools import lru_cache

import numpy as np

_TRANSFORMS = 8
# Cache gather indices for common shapes such as [17, N, N] and [N, N]. Use the reference
# implementation for larger batches to bound cache growth.
_MAX_CACHED_PLANES = 64


def _rotate_or_flip(array: np.ndarray, transform_id: int) -> np.ndarray:
    """Reference geometry for all eight symmetries on the last two axes. Use it to build permutation tables
    and handle inputs outside the fast path.
    """
    if transform_id < 4:
        # Rotate by 0, 90, 180 or 270 degrees.
        return np.rot90(array, k=transform_id, axes=(-2, -1)).copy()

    # Reflect left to right before rotating.
    flipped = np.flip(array, axis=-1)
    return np.rot90(flipped, k=transform_id - 4, axes=(-2, -1)).copy()


def _readonly(array: np.ndarray) -> np.ndarray:
    """Keep shared permutation tables read-only so that one caller cannot corrupt later transformations."""
    array.flags.writeable = False
    return array


@lru_cache(maxsize=None)
def _cell_permutations(board_size: int) -> tuple[np.ndarray, ...]:
    """Return cell permutations for which transformed.ravel() equals original.ravel()[perm]. Derive the
    tables by applying the reference geometry to a numbered board.
    """
    template = np.arange(board_size * board_size, dtype=np.intp).reshape(board_size, board_size)
    return tuple(
        _readonly(_rotate_or_flip(template, transform_id).reshape(-1)) for transform_id in range(_TRANSFORMS)
    )


@lru_cache(maxsize=None)
def _policy_permutations(board_size: int) -> tuple[np.ndarray, ...]:
    """Extend the board permutation to the N*N+1 policy entries. The pass action stays in place."""
    pass_index = board_size * board_size
    return tuple(_readonly(np.append(permutation, pass_index)) for permutation in _cell_permutations(board_size))


@lru_cache(maxsize=32)
def _gather_indices(board_size: int, planes_count: int) -> tuple[np.ndarray, ...]:
    """Build flat gather indices for all planes. A single gather produces a C-contiguous copy with the same
    layout as the reference rotation.
    """
    offsets = np.arange(planes_count, dtype=np.intp)[:, None] * (board_size * board_size)
    return tuple(
        _readonly((offsets + permutation[None, :]).reshape(-1)) for permutation in _cell_permutations(board_size)
    )


def _is_standard_transform(transform_id: object) -> bool:
    """Accept integer symmetry indices from 0 through 7, including NumPy integers. Booleans and
    floating-point values follow the reference path.
    """
    if type(transform_id) is int:
        return 0 <= transform_id < _TRANSFORMS
    return isinstance(transform_id, np.integer) and bool(0 <= transform_id < _TRANSFORMS)


def _apply_by_cells(array: np.ndarray, transform_id: int) -> np.ndarray | None:
    """Gather a square board or stack of boards into a new array. Return None for unsupported shapes, types
    or indices so the reference implementation handles them.
    """
    if type(array) is not np.ndarray or array.ndim < 2 or not _is_standard_transform(transform_id):
        return None
    size = array.shape[-1]
    if size < 1 or array.shape[-2] != size:
        return None
    planes_count = math.prod(array.shape[:-2])
    if not 1 <= planes_count <= _MAX_CACHED_PLANES:
        return None
    indices = _gather_indices(size, planes_count)[transform_id]
    return array.reshape(-1)[indices].reshape(array.shape)


def apply_dihedral_to_planes(planes: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply one of the eight symmetries to a tensor of shape [C, H, W]."""
    transformed = _apply_by_cells(planes, transform_id)
    if transformed is not None:
        return transformed
    return _rotate_or_flip(planes, transform_id)


def apply_dihedral_to_board(board_2d: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply the same symmetry to a single board of shape [H, W]."""
    transformed = _apply_by_cells(board_2d, transform_id)
    if transformed is not None:
        return transformed
    return _rotate_or_flip(board_2d, transform_id)


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
    if (
        type(policy) is np.ndarray
        and policy.ndim == 1
        and type(board_size) is int
        and board_size >= 1
        and policy.shape[0] == board_size * board_size + 1
        and _is_standard_transform(transform_id)
    ):
        # Gather the entire policy, including the fixed pass entry.
        return policy[_policy_permutations(board_size)[transform_id]]

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

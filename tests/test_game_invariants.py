from __future__ import annotations

"""Randomized rule invariants with fixed seeds."""

import numpy as np
import pytest

from alphago_zero.game import BLACK, EMPTY, GoState


def _random_playthrough(board_size: int, seed: int):
    """Yield states and transitions from a random game."""
    rng = np.random.default_rng(seed)
    state = GoState.new_game(board_size=board_size, komi=7.5)
    while not state.is_terminal():
        legal = state.legal_actions()
        action = int(rng.choice(legal))
        next_state = state.play(action)
        yield state, action, next_state
        state = next_state


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_is_legal_consistent_with_legal_actions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    state = GoState.new_game(board_size=5, komi=7.5)
    for _ in range(30):
        if state.is_terminal():
            break
        legal = set(state.legal_actions())
        for action in range(state.action_size):
            assert state.is_legal(action) == (action in legal)
        state = state.play(int(rng.choice(sorted(legal))))


@pytest.mark.parametrize("seed", [10, 11])
def test_stone_move_board_delta_invariants(seed: int) -> None:
    for state, action, next_state in _random_playthrough(board_size=7, seed=seed):
        if action == state.pass_action:
            assert np.array_equal(next_state.board, state.board)
            continue

        row, col = state.action_to_coord(action)
        mover = state.to_play

        # Place exactly one stone of the moving color at the chosen point.
        assert state.board[row, col] == EMPTY
        assert next_state.board[row, col] == mover

        # Suicide is forbidden, so earlier friendly stones remain on the board.
        own_before = state.board == mover
        own_after = next_state.board == mover
        assert np.all(own_after[own_before])

        # Opponent stones may disappear through capture but cannot appear.
        opp_before = state.board == -mover
        opp_after = next_state.board == -mover
        assert np.all(opp_before[opp_after])


@pytest.mark.parametrize("seed", [20, 21])
def test_positional_superko_no_repeats(seed: int) -> None:
    seen: set[bytes] = set()
    first = GoState.new_game(board_size=5, komi=7.5)
    seen.add(first.board.tobytes())
    for state, action, next_state in _random_playthrough(board_size=5, seed=seed):
        if action == state.pass_action:
            continue  # Pass preserves the board and is exempt from superko.
        key = next_state.board.tobytes()
        assert key not in seen, "stone placement repeated an earlier board"
        seen.add(key)


def test_pass_always_legal_until_terminal() -> None:
    state = GoState.new_game(board_size=5, komi=7.5)
    rng = np.random.default_rng(33)
    for _ in range(20):
        if state.is_terminal():
            break
        assert state.is_legal(state.pass_action)
        state = state.play(int(rng.choice(state.legal_actions())))
    if state.is_terminal():
        assert not state.is_legal(state.pass_action)


@pytest.mark.parametrize("seed", [40])
def test_encode_planes_are_binary_and_canonical(seed: int) -> None:
    for state, _action, next_state in _random_playthrough(board_size=5, seed=seed):
        encoded = next_state.encode()
        assert encoded.shape == (17, 5, 5)
        # All stone planes contain only zero or one.
        assert set(np.unique(encoded[:16])).issubset({0.0, 1.0})
        # Plane zero holds the current player's stones on the current board.
        assert np.array_equal(encoded[0], (next_state.board == next_state.to_play).astype(np.float32))
        # The final plane is one for Black to move and zero for White.
        expected = 1.0 if next_state.to_play == BLACK else 0.0
        assert np.all(encoded[16] == expected)


def test_from_board_records_position_for_superko() -> None:
    board = np.zeros((5, 5), dtype=np.int8)
    board[2, 2] = BLACK
    state = GoState.from_board(board, to_play=BLACK, komi=7.5)
    # A repeated position is detected through the current hash in seen_hashes.
    assert state.position_hash in state.seen_hashes
    assert len(state.board_history) == 1

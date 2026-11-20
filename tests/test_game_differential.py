from __future__ import annotations

"""Differential tests against the simulation-based reference Go engine."""

import numpy as np
import pytest

from alphago_zero.game import GoState
from reference_game import ReferenceGoState


def _assert_states_match(fast: GoState, ref: ReferenceGoState) -> None:
    assert fast.to_play == ref.to_play
    assert fast.consecutive_passes == ref.consecutive_passes
    assert fast.move_number == ref.move_number
    assert fast.is_terminal() == ref.is_terminal()
    assert np.array_equal(fast.board, ref.board)


def _play_random_game_in_lockstep(board_size: int, seed: int, max_steps: int | None = None) -> int:
    """Play the same random game in both engines and compare each move."""
    rng = np.random.default_rng(seed)
    fast = GoState.new_game(board_size=board_size, komi=7.5)
    ref = ReferenceGoState.new_game(board_size=board_size, komi=7.5)

    steps = 0
    while not ref.is_terminal():
        if max_steps is not None and steps >= max_steps:
            break

        fast_legal = fast.legal_actions()
        ref_legal = ref.legal_actions()
        assert fast_legal == ref_legal, (
            f"legal_actions mismatch at move {steps} (seed={seed}): "
            f"only in fast: {sorted(set(fast_legal) - set(ref_legal))}, "
            f"only in reference: {sorted(set(ref_legal) - set(fast_legal))}"
        )

        action = int(rng.choice(fast_legal))
        fast = fast.play(action)
        ref = ref.play(action)
        _assert_states_match(fast, ref)
        steps += 1

    _assert_states_match(fast, ref)
    if ref.is_terminal():
        assert fast.area_score() == ref.area_score()
        assert fast.winner_color() == ref.winner_color()
    return steps


@pytest.mark.parametrize("seed", range(25))
def test_differential_random_games_5x5(seed: int) -> None:
    steps = _play_random_game_in_lockstep(board_size=5, seed=seed)
    assert steps > 0


@pytest.mark.parametrize("seed", [100, 101, 102, 103])
def test_differential_random_games_9x9(seed: int) -> None:
    steps = _play_random_game_in_lockstep(board_size=9, seed=seed)
    assert steps > 0


@pytest.mark.slow
def test_differential_19x19_game_prefix() -> None:
    # Compare a long prefix on 19x19 to bound the slow reference's runtime.
    steps = _play_random_game_in_lockstep(board_size=19, seed=7, max_steps=120)
    assert steps == 120


def test_differential_capture_heavy_position() -> None:
    """Compare legal actions and successor states in a dense capture position."""
    board = np.array(
        [
            [0, 1, -1, -1, 0],
            [1, 1, -1, 0, -1],
            [0, 1, -1, -1, -1],
            [1, -1, 0, 1, 1],
            [-1, -1, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    for to_play in (1, -1):
        fast = GoState.from_board(board, to_play=to_play, komi=7.5)
        ref = ReferenceGoState.from_board(board, to_play=to_play, komi=7.5)
        assert fast.legal_actions() == ref.legal_actions()
        for action in fast.legal_actions():
            next_fast = fast.play(action)
            next_ref = ref.play(action)
            assert np.array_equal(next_fast.board, next_ref.board), f"action={action}, to_play={to_play}"


def _reference_encode(state: GoState, history_length: int = 8) -> np.ndarray:
    """Frozen plane-by-plane encoding from before vectorization."""
    planes: list[np.ndarray] = []
    for t in range(history_length):
        if t < len(state.board_history):
            board_t = state.board_history[t]
            planes.append((board_t == state.to_play).astype(np.float32))
            planes.append((board_t == -state.to_play).astype(np.float32))
        else:
            zeros = np.zeros((state.board_size, state.board_size), dtype=np.float32)
            planes.append(zeros)
            planes.append(zeros.copy())
    planes.append(
        np.full(
            (state.board_size, state.board_size),
            1.0 if state.to_play == 1 else 0.0,
            dtype=np.float32,
        )
    )
    return np.stack(planes, axis=0)


@pytest.mark.parametrize(("board_size", "seed"), [(5, 0), (5, 1), (7, 2), (9, 3)])
def test_encode_matches_reference(board_size: int, seed: int) -> None:
    """Check exact equality with the reference throughout a game, including short histories."""
    rng = np.random.default_rng(seed)
    state = GoState.new_game(board_size=board_size, komi=7.5)
    while not state.is_terminal():
        for history_length in (8, 1, 3):
            fast = state.encode(history_length=history_length)
            ref = _reference_encode(state, history_length=history_length)
            assert fast.dtype == ref.dtype == np.float32
            assert fast.shape == ref.shape
            assert np.array_equal(fast, ref), (state.move_number, history_length)
        legal = state.legal_actions()
        # Pass frequently to include duplicate boards in the history.
        action = state.pass_action if rng.random() < 0.1 else int(rng.choice(legal))
        state = state.play(action)

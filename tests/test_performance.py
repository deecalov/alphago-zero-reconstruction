from __future__ import annotations

"""Performance regression budgets for board operations and random games."""

import sys
import time

import numpy as np
import pytest

from alphago_zero.game import GoState

# Coverage tracing changes hot-path costs, invalidating wall-clock budgets.
_UNDER_COVERAGE = "coverage" in sys.modules
_skip_under_coverage = pytest.mark.skipif(
    _UNDER_COVERAGE, reason="performance budgets are skipped under coverage tracing"
)


def _mid_game_state(board_size: int, num_moves: int, seed: int) -> GoState:
    rng = np.random.default_rng(seed)
    state = GoState.new_game(board_size=board_size, komi=7.5)
    for _ in range(num_moves):
        if state.is_terminal():
            break
        state = state.play(int(rng.choice(state.legal_actions())))
    return state


@pytest.mark.slow
@_skip_under_coverage
def test_legal_actions_19x19_budget() -> None:
    state = _mid_game_state(board_size=19, num_moves=120, seed=3)
    assert not state.is_terminal()

    # Copy the position to discard cached analysis before each measurement.
    repeats = 30
    t0 = time.perf_counter()
    for _ in range(repeats):
        state.copy().legal_actions()
    mean_ms = (time.perf_counter() - t0) / repeats * 1000.0

    # The pre-optimization reference took about 3.6 ms. The optimized path commonly took 0.4–1.6 ms.
    assert mean_ms < 2.0, f"legal_actions on 19x19 is too slow: {mean_ms:.2f} ms"


@pytest.mark.slow
@_skip_under_coverage
def test_random_playout_9x9_budget() -> None:
    t0 = time.perf_counter()
    state = _mid_game_state(board_size=9, num_moves=10_000, seed=5)  # The game may finish earlier.
    elapsed = time.perf_counter() - t0
    assert state.is_terminal()
    # A full random 9x9 game of at most about 162 moves should finish within seconds.
    assert elapsed < 2.0, f"random 9x9 game is too slow: {elapsed:.2f} s"

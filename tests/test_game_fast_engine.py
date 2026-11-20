"""Check the optimized rules engine against the frozen predecessor."""

from __future__ import annotations

import numpy as np
import pytest
from frozen_game_v1 import GoState as FrozenGoState
from frozen_game_v1 import _analyze_board as frozen_analyze_board

from alphago_zero.game import (
    BLACK,
    WHITE,
    GoState,
    _analyze_moves,
    _board_zobrist_hash,
    _geometry,
    _zobrist_table,
)

RULES = ("positional", "situational")


# ---------------------------------------------------------------------------
# State and position comparisons
# ---------------------------------------------------------------------------
def _assert_same_state(new: GoState, old: FrozenGoState, where: object) -> None:
    """Compare all position fields between engines."""
    assert new.board_size == old.board_size, where
    assert new.komi == old.komi, where
    assert new.max_moves == old.max_moves, where
    assert new.board.dtype == old.board.dtype, where
    assert np.array_equal(new.board, old.board), where
    assert new.to_play == old.to_play, where
    assert new.consecutive_passes == old.consecutive_passes, where
    assert new.move_number == old.move_number, where
    assert new.position_hash == old.position_hash, where
    assert new.seen_hashes == old.seen_hashes, where
    assert new.superko_rule == old.superko_rule, where
    assert len(new.board_history) == len(old.board_history), where
    for new_board, old_board in zip(new.board_history, old.board_history):
        assert np.array_equal(new_board, old_board), where


def _check_position(new: GoState, old: FrozenGoState, where: object) -> None:
    """Compare action lists, legality and successors for every move in one position."""
    new_legal = new.legal_actions()
    old_legal = old.legal_actions()
    assert new_legal == old_legal, (where, "legal_actions", sorted(set(new_legal) ^ set(old_legal)))

    legal = set(old_legal)
    for action in range(-1, new.pass_action + 2):  # Include indices outside both ends of the action range.
        assert new.is_legal(action) == old.is_legal(action) == (action in legal), (where, "is_legal", action)

    for action in old_legal:
        _assert_same_state(new.play(action), old.play(action), (where, "play", action))

    if not old.is_terminal():
        for action in range(new.pass_action + 1):
            if action in legal:
                continue
            with pytest.raises(ValueError):
                new.play(action)
            with pytest.raises(ValueError):
                old.play(action)


def _lockstep_game(
    board_size: int,
    seed: int,
    rule: str,
    pass_probability: float = 0.05,
    max_steps: int | None = None,
    check_every: int = 1,
) -> int:
    """Play matching random moves in both engines and return game length."""
    rng = np.random.default_rng(seed * 100 + board_size)
    komi = 0.5 + board_size % 3
    new = GoState.new_game(board_size=board_size, komi=komi, superko_rule=rule)
    old = FrozenGoState.new_game(board_size=board_size, komi=komi, superko_rule=rule)

    step = 0
    while not old.is_terminal() and (max_steps is None or step < max_steps):
        if step % check_every == 0:
            _check_position(new, old, (board_size, seed, rule, step))
        legal = old.legal_actions()
        non_pass = [action for action in legal if action != old.pass_action]
        if non_pass and rng.random() >= pass_probability:
            action = int(rng.choice(non_pass))
        else:
            action = old.pass_action
        new, old = new.play(action), old.play(action)
        _assert_same_state(new, old, (board_size, seed, rule, step, "after move"))
        step += 1

    _assert_same_state(new, old, (board_size, seed, rule, "final"))
    assert new.is_terminal() == old.is_terminal()
    if old.is_terminal():
        assert new.legal_actions() == [] == old.legal_actions()
        assert new.area_score() == old.area_score()
    return step


def _board(rows: list[str]) -> np.ndarray:
    """Parse a board diagram using dot for empty, X for Black and O for White."""
    symbols = {".": 0, "X": BLACK, "O": WHITE}
    return np.array([[symbols[char] for char in row] for row in rows], dtype=np.int8)


def _index(board_size: int, row: int, col: int) -> int:
    return row * board_size + col


# ---------------------------------------------------------------------------
# Games in lockstep with the earlier engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("board_size", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lockstep_random_games_small_boards(board_size: int, seed: int, rule: str) -> None:
    assert _lockstep_game(board_size, seed, rule) > 0


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("seed", [0, 1])
def test_lockstep_random_games_9x9(seed: int, rule: str) -> None:
    assert _lockstep_game(9, seed, rule, pass_probability=0.03) > 0


@pytest.mark.slow
@pytest.mark.parametrize(("board_size", "steps"), [(13, 80), (19, 60)])
def test_lockstep_large_board_prefix(board_size: int, steps: int) -> None:
    # Use a long 19x19 prefix with captures and group contact to keep runtime bounded.
    assert _lockstep_game(board_size, seed=3, rule="positional", max_steps=steps, check_every=3) == steps


@pytest.mark.parametrize("rule", RULES)
def test_lockstep_games_with_many_passes(rule: str) -> None:
    """Frequent passes produce repeated boards and exercise two-pass termination."""
    for seed in range(4):
        assert _lockstep_game(5, seed, rule, pass_probability=0.4) > 0


# ---------------------------------------------------------------------------
# Arbitrary boards including unreachable positions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("board_size", [2, 3, 5, 7, 9])
@pytest.mark.parametrize("density", [0.3, 0.6, 0.9])
def test_random_dense_boards(board_size: int, density: float, rule: str) -> None:
    """Generate boards over a range of densities, including artificial positions unreachable in play."""
    rng = np.random.default_rng(board_size * 1000 + int(density * 100))
    probabilities = [1 - density, density / 2, density / 2]
    for index in range(16):
        board = rng.choice([0, BLACK, WHITE], size=(board_size, board_size), p=probabilities).astype(np.int8)
        for to_play in (BLACK, WHITE):
            new = GoState.from_board(board, to_play=to_play, komi=6.5, superko_rule=rule)
            old = FrozenGoState.from_board(board, to_play=to_play, komi=6.5, superko_rule=rule)
            _check_position(new, old, (board_size, density, rule, index, to_play))


# ---------------------------------------------------------------------------
# Hand-built positions with known outcomes
# ---------------------------------------------------------------------------
KO_POSITION = [
    ".XO..",
    "XO.O.",
    ".XO..",
    ".....",
    ".....",
]


@pytest.mark.parametrize("rule", RULES)
def test_ko_recapture_is_forbidden_until_the_position_changes(rule: str) -> None:
    """Prohibit immediate ko recapture, then permit recapture after an exchange elsewhere."""
    state = GoState.from_board(_board(KO_POSITION), to_play=BLACK, komi=0.5, superko_rule=rule)
    recapture = _index(5, 1, 1)  # Point of the captured white stone.
    black_capture = _index(5, 1, 2)
    assert black_capture in state.legal_actions()

    after_capture = state.play(black_capture)
    assert after_capture.board[1, 1] == 0  # White was captured.
    assert recapture not in after_capture.legal_actions()
    assert not after_capture.is_legal(recapture)
    with pytest.raises(ValueError):
        after_capture.play(recapture)

    # An exchange elsewhere changes the board and permits recapture.
    after_threat = after_capture.play(_index(5, 4, 4)).play(_index(5, 4, 0))
    assert recapture in after_threat.legal_actions()

    # Verify the same ko sequence against the earlier engine.
    old = FrozenGoState.from_board(_board(KO_POSITION), to_play=BLACK, komi=0.5, superko_rule=rule)
    old_after_capture = old.play(black_capture)
    assert recapture not in old_after_capture.legal_actions()
    _check_position(after_capture, old_after_capture, ("ko", rule))


def test_one_move_captures_two_groups_sharing_a_single_liberty() -> None:
    """Capture two White groups that share their last liberty with a single move."""
    board = _board(
        [
            "..X..",
            ".XOX.",
            ".X.X.",
            ".XOX.",
            "..X..",
        ]
    )
    state = GoState.from_board(board, to_play=BLACK, komi=0.5)
    center = _index(5, 2, 2)
    assert center in state.legal_actions()

    after = state.play(center)
    assert after.board[1, 2] == 0 and after.board[3, 2] == 0  # Both groups were captured.
    assert after.board[2, 2] == BLACK
    assert after.position_hash == _board_zobrist_hash(after.board, 5)  # Check the incremental hash.

    old_after = FrozenGoState.from_board(board, to_play=BLACK, komi=0.5).play(center)
    _assert_same_state(after, old_after, "double capture")


def test_enclosed_point_suicide_is_illegal_unless_it_captures_or_keeps_a_liberty() -> None:
    """Check suicide, capture and spare friendly liberties at fully surrounded points."""
    # At (1,1), White's neighbors have spare liberties. Black would commit suicide.
    suicide = _board(
        [
            ".O...",
            "O.O..",
            ".O...",
            ".....",
            ".....",
        ]
    )
    point = _index(5, 1, 1)
    black = GoState.from_board(suicide, to_play=BLACK, komi=0.5)
    white = GoState.from_board(suicide, to_play=WHITE, komi=0.5)
    assert point not in black.legal_actions()
    assert not black.is_legal(point)
    assert point in white.legal_actions()

    # A friendly neighboring group with another liberty can connect at (1,1).
    spare = _board(
        [
            ".X...",
            "X.O..",
            ".O...",
            ".....",
            ".....",
        ]
    )
    assert point in GoState.from_board(spare, to_play=BLACK, komi=0.5).legal_actions()

    # The surrounded corner move survives only because it captures the adjacent Black pair.
    capture = _board(
        [
            ".OX..",
            "XXO..",
            "OO...",
            ".....",
            ".....",
        ]
    )
    corner = _index(5, 0, 0)
    black_capture = GoState.from_board(capture, to_play=BLACK, komi=0.5)
    white_capture = GoState.from_board(capture, to_play=WHITE, komi=0.5)
    assert corner in black_capture.legal_actions()
    assert black_capture.play(corner).board[0, 1] == 0  # White was captured.
    assert corner in white_capture.legal_actions()
    white_after = white_capture.play(corner)
    assert white_after.board[1, 0] == 0 and white_after.board[1, 1] == 0  # Black pair removed.

    for state in (black, white, GoState.from_board(spare, to_play=BLACK, komi=0.5), black_capture, white_capture):
        old = FrozenGoState.from_board(state.board, to_play=state.to_play, komi=0.5)
        _check_position(state, old, "surrounded point")


# ---------------------------------------------------------------------------
# Internal move analysis versus the earlier group algorithm
# ---------------------------------------------------------------------------
def _expected_captures(state: GoState) -> dict[int, tuple[int, list[int]]]:
    """Map each capturing liberty to its XOR hash change and sorted captured stones."""
    size = state.board_size
    analysis = frozen_analyze_board(state.board, size)
    table = _zobrist_table(size)
    opponent = -state.to_play
    opponent_keys = table[0 if opponent == BLACK else 1]
    expected: dict[int, tuple[int, list[int]]] = {}
    for gid, color in enumerate(analysis.colors):
        if color != opponent or analysis.liberty_counts[gid] != 1:
            continue
        stones = analysis.stones[gid]
        liberty = None
        for row, col in stones:
            for nr, nc in state.neighbors(row, col):
                if state.board[nr, nc] == 0:
                    liberty = nr * size + nc
        assert liberty is not None
        flat = sorted(row * size + col for row, col in stones)
        xor = 0
        for row, col in stones:
            xor ^= opponent_keys[row][col]
        previous = expected.get(liberty)
        if previous is not None:
            xor ^= previous[0]
            flat = sorted(previous[1] + flat)
        expected[liberty] = (xor, flat)
    return expected


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_analysis_captures_match_group_analysis_of_previous_engine(board_size: int) -> None:
    rng = np.random.default_rng(board_size)
    checked_with_captures = 0
    for game in range(6):
        state = GoState.new_game(board_size=board_size, komi=0.5)
        while not state.is_terminal():
            analysis = state._get_analysis()
            expected = _expected_captures(state)
            actual = analysis.captures or {}
            assert {liberty: (xor, sorted(stones)) for liberty, (xor, stones) in actual.items()} == expected, (
                game,
                state.move_number,
            )
            checked_with_captures += bool(expected)

            # Candidates are empty, sorted and consistent with the byte mask.
            assert analysis.cells == sorted(set(analysis.cells))
            assert all(state.board.flat[cell] == 0 for cell in analysis.cells)
            assert [i for i, flag in enumerate(analysis.candidates) if flag] == analysis.cells
            assert all(liberty in analysis.cells for liberty in expected)  # Capturing is never suicide.

            legal = [a for a in state.legal_actions() if a != state.pass_action]
            if not legal or rng.random() < 0.03:
                state = state.play(state.pass_action)
            else:
                state = state.play(int(rng.choice(legal)))
    assert checked_with_captures > 0, "the games contained no group in atari, so the test did not exercise captures"


@pytest.mark.parametrize("to_play", [BLACK, WHITE])
def test_analysis_is_stable_for_any_memory_layout_and_dtype(to_play: int) -> None:
    """Board dtype and memory layout must not change analysis."""
    rng = np.random.default_rng(7)
    board = rng.choice([0, BLACK, WHITE], size=(7, 7), p=[0.4, 0.3, 0.3]).astype(np.int8)
    reference = _analyze_moves(board, 7, to_play)
    for variant in (np.asfortranarray(board), board.astype(np.int64), board.T.copy().T):
        analysis = _analyze_moves(variant, 7, to_play)
        assert analysis.candidates == reference.candidates
        assert analysis.cells == reference.cells
        assert analysis.captures == reference.captures


# ---------------------------------------------------------------------------
# Geometry and cache behavior
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("board_size", [1, 2, 3, 5, 9, 13, 19])
def test_geometry_constants(board_size: int) -> None:
    geometry = _geometry(board_size)
    n = board_size
    assert geometry.size == n and geometry.cell_count == n * n and geometry.row_shift == 8 * n
    assert geometry.ones.to_bytes(n * n, "little") == b"\x01" * (n * n)
    left = geometry.left_mask.to_bytes(n * n, "little")
    right = geometry.right_mask.to_bytes(n * n, "little")
    state = GoState.new_game(board_size=n, komi=0.5)
    table = _zobrist_table(n)
    for row in range(n):
        for col in range(n):
            flat = row * n + col
            assert left[flat] == (1 if col != 0 else 0)
            assert right[flat] == (1 if col != n - 1 else 0)
            assert sorted(geometry.neighbors[flat]) == sorted(nr * n + nc for nr, nc in state.neighbors(row, col))
            assert geometry.zobrist_black[flat] == table[0][row][col]
            assert geometry.zobrist_white[flat] == table[1][row][col]
    assert list(geometry.indices) == list(range(n * n))
    assert _geometry(n) is geometry  # Geometry is cached.


def test_legal_actions_returns_a_fresh_list_each_call() -> None:
    state = GoState.new_game(board_size=5, komi=0.5)
    first = state.legal_actions()
    first.clear()
    first.append(-123)
    second = state.legal_actions()
    assert second == list(range(26))  # Twenty-five board points plus pass.
    assert second is not state.legal_actions()


def test_superko_is_read_live_from_seen_hashes_not_from_the_cached_analysis() -> None:
    """Cache board analysis per state but check repetition history for each query."""
    state = GoState.new_game(board_size=5, komi=0.5)
    assert 12 in state.legal_actions()  # The move analysis is now cached.
    cached = state._analysis
    assert cached is not None

    after_key = state.position_hash ^ _zobrist_table(5)[0][2][2]  # Position after Black takes the center.
    state.seen_hashes.add(after_key)
    assert 12 not in state.legal_actions()
    assert not state.is_legal(12)
    assert state._analysis is cached  # The board and player have not changed, so reuse the cached analysis.

    state.seen_hashes.discard(after_key)
    assert 12 in state.legal_actions()
    assert state.is_legal(12)


def test_copy_does_not_share_the_cached_analysis() -> None:
    state = GoState.new_game(board_size=5, komi=0.5)
    state.legal_actions()
    clone = state.copy()
    assert state._analysis is not None and clone._analysis is None
    assert clone.legal_actions() == state.legal_actions()


def test_terminal_state_has_no_moves() -> None:
    state = GoState.new_game(board_size=5, komi=0.5).play(25).play(25)
    assert state.is_terminal()
    assert state.legal_actions() == []
    assert not state.is_legal(0) and not state.is_legal(25)

from __future__ import annotations

"""Go engine tests for coordinates, captures, suicide, superko, scoring and history."""

import numpy as np
import pytest

from alphago_zero.game import BLACK, EMPTY, WHITE, GoState


# ---------------------------------------------------------------------------
# Geometry and conversions
# ---------------------------------------------------------------------------
def test_new_game_defaults() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    assert state.board.shape == (5, 5)
    assert np.all(state.board == EMPTY)
    assert state.to_play == BLACK
    assert state.action_size == 5 * 5 + 1
    assert state.pass_action == 25
    assert not state.is_terminal()


def test_action_coord_roundtrip() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    # Pass has no board coordinate.
    assert state.action_to_coord(state.pass_action) is None
    # Round-trip every board action.
    for action in range(state.pass_action):
        coord = state.action_to_coord(action)
        assert coord is not None
        row, col = coord
        assert state.coord_to_action(row, col) == action


def test_action_to_coord_out_of_range_raises() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    with pytest.raises(ValueError):
        state.action_to_coord(-1)
    with pytest.raises(ValueError):
        state.action_to_coord(state.pass_action + 1)


def test_is_legal_out_of_range_is_false() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    assert not state.is_legal(-1)
    assert not state.is_legal(state.action_size + 5)


def test_neighbors_counts() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    assert len(list(state.neighbors(0, 0))) == 2  # Corner.
    assert len(list(state.neighbors(0, 2))) == 3  # Edge.
    assert len(list(state.neighbors(2, 2))) == 4  # Center.


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------
def test_capture_single_stone() -> None:
    # Black fills the central white stone's fourth and last liberty.
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, 1] = BLACK
    board[1, 0] = BLACK
    board[1, 2] = BLACK
    board[1, 1] = WHITE
    state = GoState.from_board(board, to_play=BLACK, komi=0.5)

    action = state.coord_to_action(2, 1)
    assert state.is_legal(action)
    new_state = state.play(action)
    assert new_state.board[1, 1] == EMPTY  # White stone removed.
    assert new_state.board[2, 1] == BLACK


def test_capture_group() -> None:
    # The two white stones share a single liberty at (0,2).
    board = np.zeros((5, 5), dtype=np.int8)
    board[0, 0] = WHITE
    board[0, 1] = WHITE
    board[1, 0] = BLACK
    board[1, 1] = BLACK
    state = GoState.from_board(board, to_play=BLACK, komi=0.5)

    new_state = state.play(state.coord_to_action(0, 2))
    assert new_state.board[0, 0] == EMPTY
    assert new_state.board[0, 1] == EMPTY


# ---------------------------------------------------------------------------
# Suicide
# ---------------------------------------------------------------------------
def test_suicide_is_illegal() -> None:
    # A white stone played inside the Black surround would have no liberties.
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, 1] = BLACK
    board[1, 0] = BLACK
    board[1, 2] = BLACK
    board[2, 1] = BLACK
    state = GoState.from_board(board, to_play=WHITE, komi=0.5)

    action = state.coord_to_action(1, 1)
    assert not state.is_legal(action)
    with pytest.raises(ValueError):
        state.play(action)


def test_capture_takes_priority_over_suicide() -> None:
    # The corner move gains liberties by capturing the adjacent enemy groups.
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, 1] = BLACK
    board[0, 2] = WHITE
    board[1, 0] = BLACK
    board[1, 1] = WHITE
    board[2, 0] = WHITE
    state = GoState.from_board(board, to_play=WHITE, komi=0.5)

    action = state.coord_to_action(0, 0)
    assert state.is_legal(action)
    new_state = state.play(action)
    # Black stones at (0,1) and (1,0) disappear. White survives in the corner.
    assert new_state.board[0, 1] == EMPTY
    assert new_state.board[1, 0] == EMPTY
    assert new_state.board[0, 0] == WHITE


# ---------------------------------------------------------------------------
# Superko
# ---------------------------------------------------------------------------
def test_superko_forbids_positional_repetition() -> None:
    board = np.zeros((5, 5), dtype=np.int8)
    board[0, 1] = BLACK
    board[1, 0] = BLACK
    board[2, 1] = BLACK
    board[0, 2] = WHITE
    board[1, 3] = WHITE
    board[2, 2] = WHITE
    board[1, 1] = WHITE  # The white stone in atari has its last liberty at (1,2).
    # from_board includes the initial position in the repetition history.
    state0 = GoState.from_board(board, to_play=BLACK, komi=2.5)

    # Black plays at (1,2) to capture.
    capture = state0.coord_to_action(1, 2)
    assert state0.is_legal(capture)
    state1 = state0.play(capture)
    assert state1.board[1, 1] == EMPTY

    # Immediate recapture would restore state0 and violate superko.
    recapture = state1.coord_to_action(1, 1)
    assert not state1.is_legal(recapture)


def test_from_board_has_no_past_before_its_position() -> None:
    # from_board cannot infer the history before the supplied board.
    board = np.zeros((5, 5), dtype=np.int8)
    for row, col in ((0, 1), (1, 0), (2, 1)):
        board[row, col] = BLACK
    for row, col in ((0, 2), (1, 3), (2, 2), (1, 1)):
        board[row, col] = WHITE  # The last liberty of the white stone at (1,1) is (1,2).

    played = GoState.from_board(board, to_play=BLACK, komi=2.5)
    after_capture = played.play(played.coord_to_action(1, 2))
    recapture = after_capture.coord_to_action(1, 1)
    # A played game retains enough history to prohibit the ko recapture.
    assert not after_capture.is_legal(recapture)
    assert after_capture.encode()[2:16].any(), "the game has previous plies in its history"

    # Reconstructing the same board from scratch loses the earlier ko position.
    restarted = GoState.from_board(after_capture.board, to_play=after_capture.to_play, komi=2.5)
    assert restarted.is_legal(recapture)
    assert len(restarted.seen_hashes) == 1
    assert not restarted.encode()[2:16].any()


# ---------------------------------------------------------------------------
# Game termination
# ---------------------------------------------------------------------------
def test_two_passes_terminate() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    after_one = state.play(state.pass_action)
    assert not after_one.is_terminal()
    after_two = after_one.play(after_one.pass_action)
    assert after_two.is_terminal()


def test_max_moves_terminates_game() -> None:
    state = GoState.new_game(board_size=5, komi=2.5, max_moves=4)
    current = state
    for row, col in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        assert not current.is_terminal()
        current = current.play(current.coord_to_action(row, col))
    assert current.is_terminal()  # Reached the four-move limit.


def test_legal_actions_includes_pass_excludes_occupied() -> None:
    state = GoState.new_game(board_size=3, komi=0.5)
    after = state.play(state.coord_to_action(1, 1))
    legal = after.legal_actions()
    assert after.pass_action in legal
    assert after.coord_to_action(1, 1) not in legal  # Occupied point.


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_area_score_empty_board() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    black, white = state.area_score()
    assert black == 0.0
    assert white == 2.5  # Komi only.
    assert state.winner_color() == WHITE


def test_area_score_black_controls_territory() -> None:
    # Black surrounds the center and owns it as territory.
    board = np.zeros((3, 3), dtype=np.int8)
    for row, col in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]:
        board[row, col] = BLACK
    state = GoState.from_board(board, to_play=WHITE, komi=0.5)
    black, white = state.area_score()
    assert black == 9.0  # Eight stones plus the central point of territory.
    assert white == 0.5  # komi
    assert state.winner_color() == BLACK
    assert state.outcome_for_player(BLACK) == 1.0
    assert state.outcome_for_player(WHITE) == -1.0


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------
def test_encode_shape_and_initial_planes() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    encoded = state.encode()
    assert encoded.shape == (17, 5, 5)
    # On an empty board the first 16 stone planes are zero.
    assert np.all(encoded[:16] == 0.0)
    # Black's side-to-play plane is one.
    assert np.all(encoded[16] == 1.0)


def test_encode_canonicalizes_to_current_player() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    after = state.play(state.coord_to_action(2, 2))  # After Black plays, White is to move.
    encoded = after.encode()
    # Plane zero holds White's current stones and is empty.
    assert np.all(encoded[0] == 0.0)
    # Plane one contains the opponent's stone at (2,2).
    assert encoded[1][2, 2] == 1.0
    # White's side-to-play plane is zero.
    assert np.all(encoded[16] == 0.0)


# ---------------------------------------------------------------------------
# Copying and immutability
# ---------------------------------------------------------------------------
def test_copy_is_deep() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    clone = state.copy()
    clone.board[0, 0] = BLACK
    clone.seen_hashes.add(123456789)
    # The original state must stay unchanged.
    assert state.board[0, 0] == EMPTY
    assert 123456789 not in state.seen_hashes


def test_from_board_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        GoState.from_board(np.zeros((3, 5), dtype=np.int8), to_play=BLACK, komi=0.5)


def test_play_on_occupied_point_raises() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    after = state.play(state.coord_to_action(2, 2))
    with pytest.raises(ValueError):
        after.play(after.coord_to_action(2, 2))  # Occupied point.


# ---------------------------------------------------------------------------
# Draws and public constructor validation
# ---------------------------------------------------------------------------
def test_equal_score_is_a_draw_not_a_white_win() -> None:
    # An empty board with zero komi scores as a draw.
    state = GoState.new_game(board_size=5, komi=0.0)
    assert state.area_score() == (0.0, 0.0)
    assert state.winner_color() is None
    assert state.outcome_for_player(BLACK) == 0.0
    assert state.outcome_for_player(WHITE) == 0.0


def test_draw_after_two_passes_with_integer_komi() -> None:
    state = GoState.new_game(board_size=5, komi=0.0)
    state = state.play(state.pass_action).play(state.pass_action)
    assert state.is_terminal()
    assert state.winner_color() is None


@pytest.mark.parametrize("board_size", [0, -3])
def test_new_game_rejects_bad_board_size(board_size: int) -> None:
    with pytest.raises(ValueError, match="board_size"):
        GoState.new_game(board_size=board_size, komi=2.5)


@pytest.mark.parametrize("komi", [float("nan"), float("inf"), float("-inf")])
def test_new_game_rejects_non_finite_komi(komi: float) -> None:
    with pytest.raises(ValueError, match="komi"):
        GoState.new_game(board_size=5, komi=komi)


def test_new_game_rejects_unknown_superko_rule() -> None:
    with pytest.raises(ValueError, match="superko_rule"):
        GoState.new_game(board_size=5, komi=2.5, superko_rule="simple_ko")  # type: ignore[arg-type]


def test_new_game_rejects_bad_max_moves() -> None:
    with pytest.raises(ValueError, match="max_moves"):
        GoState.new_game(board_size=5, komi=2.5, max_moves=0)


def test_from_board_rejects_values_outside_colors() -> None:
    board = np.zeros((3, 3), dtype=np.int8)
    board[1, 1] = 2
    with pytest.raises(ValueError, match="board entries"):
        GoState.from_board(board, to_play=BLACK, komi=0.5)
    # Reject fractional values before they can round to empty under an int8 cast.
    with pytest.raises(ValueError, match="board entries"):
        GoState.from_board(np.full((3, 3), 0.5), to_play=BLACK, komi=0.5)


def test_from_board_rejects_bad_to_play_and_superko() -> None:
    board = np.zeros((3, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="to_play"):
        GoState.from_board(board, to_play=0, komi=0.5)
    with pytest.raises(ValueError, match="superko_rule"):
        GoState.from_board(board, to_play=BLACK, komi=0.5, superko_rule="ko")  # type: ignore[arg-type]

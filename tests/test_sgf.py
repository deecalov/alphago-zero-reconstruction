from __future__ import annotations

"""Test SGF encoding, import, replay and self-play move records."""

from pathlib import Path

import numpy as np
import pytest

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.selfplay import ResignationController, play_self_play_game
from alphago_zero.sgf import (
    SGFParseError,
    _action_to_sgf,
    format_result,
    game_to_sgf,
    parse_sgf,
    replay_game,
    replay_states,
    save_sgf,
)


def test_action_to_sgf_coordinates() -> None:
    # Top left (row 0, col 0) is aa.
    assert _action_to_sgf(0, 5) == "aa"
    # Bottom right of a 5x5 board is ee.
    assert _action_to_sgf(24, 5) == "ee"
    # Column two, row one is cb. SGF writes column first.
    assert _action_to_sgf(1 * 5 + 2, 5) == "cb"
    # Pass uses an empty coordinate.
    assert _action_to_sgf(25, 5) == ""
    # The last 19x19 point is ss.
    assert _action_to_sgf(19 * 19 - 1, 19) == "ss"


def test_format_result_variants() -> None:
    assert format_result(BLACK, resigned=True, score_margin=None) == "B+R"
    assert format_result(WHITE, resigned=False, score_margin=3.5) == "W+3.5"
    assert format_result(BLACK, resigned=False, score_margin=10.0) == "B+10"
    assert format_result(WHITE, resigned=False, score_margin=None) == "W+"


def test_game_to_sgf_structure() -> None:
    sgf = game_to_sgf(
        board_size=5,
        komi=2.5,
        moves=[12, 7, 25],  # B (row2,col2)='cc', W (row1,col2)='cb', B pass
        winner=BLACK,
        resigned=False,
        score_margin=5.0,
        black_name="netA",
        white_name="netB",
        comment="test game",
    )
    assert sgf.startswith("(;") and sgf.endswith(")")
    for fragment in ("GM[1]", "FF[4]", "SZ[5]", "KM[2.5]", "PB[netA]", "PW[netB]", "RE[B+5]"):
        assert fragment in sgf
    assert "C[test game]" in sgf
    # Alternate from Black and write pass as empty brackets.
    assert ";B[cc]" in sgf
    assert ";W[cb]" in sgf
    assert ";B[]" in sgf


def _header_of(**changes: object) -> str:
    """Build a three-move 5x5 SGF with optional game_to_sgf overrides."""
    arguments: dict[str, object] = {"board_size": 5, "komi": 2.5, "moves": [12, 7, 25], "winner": BLACK}
    arguments.update(changes)
    return game_to_sgf(**arguments)  # type: ignore[arg-type]


def test_explicit_result_replaces_the_winner_fields() -> None:
    """Set RE explicitly for a draw or void game with no winner."""
    assert "RE[0]" in _header_of(winner=None, result="0")
    assert "RE[Void]" in _header_of(winner=None, result="Void")
    # An explicit result overrides the inferred winner, resignation and margin.
    assert "RE[W+3.5]" in _header_of(winner=BLACK, resigned=True, result="W+3.5")


def test_game_without_winner_and_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="winner or result"):
        _header_of(winner=None)


def test_text_properties_are_escaped() -> None:
    """Escape closing brackets and backslashes from external engine names."""
    sgf = _header_of(black_name="a]b", white_name="c\\d", comment="x]y\\z", result="B+1]")
    assert "PB[a\\]b]" in sgf
    assert "PW[c\\\\d]" in sgf
    assert "C[x\\]y\\\\z]" in sgf
    assert "RE[B+1\\]]" in sgf
    assert sgf.count(";") == 4  # The root node and three moves must remain parseable.


def test_sgf_round_trip_replays_the_same_states() -> None:
    """An exported game must replay through exactly the same positions after import."""
    moves = [12, 6, 18, 8, 0, 24, 25, 4, 20]  # 5x5 game ending with a Black pass.
    original = GoState.new_game(board_size=5, komi=2.5)
    original_hashes = [original.position_hash]
    for action in moves:
        original = original.play(action)
        original_hashes.append(original.position_hash)

    sgf = game_to_sgf(board_size=5, komi=2.5, moves=moves, winner=BLACK, score_margin=5.0)
    parsed = parse_sgf(sgf)
    assert parsed.board_size == 5
    assert parsed.komi == 2.5
    assert parsed.result == "B+5"
    assert parsed.moves == tuple(
        (BLACK if i % 2 == 0 else WHITE, action) for i, action in enumerate(moves)
    )

    states = replay_states(parsed)
    assert [state.position_hash for state in states] == original_hashes
    final = states[-1]
    assert np.array_equal(final.board, original.board)
    assert final.area_score() == original.area_score()
    # Replaying moves reconstructs actual history, which a standalone from_board position lacks.
    assert len(final.board_history) == len(original.board_history) > 1


def test_parse_external_sgf_with_variations_and_unknown_props() -> None:
    """Read main-line GUI records and warn about skipped variations or unsupported properties."""
    text = (
        "(;GM[1]FF[4]AP[CGoban:3]SZ[9]KM[7.5]PB[na\\]me]PW[white]"
        "C[root comment];B[dd];W[tt](;B[ee])(;B[ff]))"
    )
    parsed = parse_sgf(text)
    assert parsed.black_name == "na]me"  # Decode the escaped closing bracket.
    assert parsed.root_comment == "root comment"
    # On 9x9, tt denotes pass. Follow B[ee] and report the skipped second variation.
    assert parsed.moves == ((BLACK, 3 * 9 + 3), (WHITE, 81), (BLACK, 4 * 9 + 4))
    assert any("AP" in warning for warning in parsed.warnings)
    assert any("variation" in warning for warning in parsed.warnings)
    final = replay_game(parsed)
    assert final.move_number == 3


def test_replay_setup_stones_and_handicap_turn() -> None:
    """Black handicap setup without an explicit PL gives White the first move."""
    parsed = parse_sgf("(;GM[1]FF[4]SZ[9]KM[7.5]HA[2]AB[cc][gc];W[ee];B[ce])")
    assert parsed.handicap == 2
    state = replay_game(parsed)
    assert int(state.board[2, 2]) == BLACK
    assert int(state.board[2, 6]) == BLACK
    states = replay_states(parsed)
    assert states[0].to_play == WHITE
    assert state.move_number == 2


def test_replay_enforces_ko_via_superko() -> None:
    """Reject immediate ko recapture after setting up an initial position."""
    # White at bb captures the black stone cb surrounded by ca, cc and db.
    setup = "(;GM[1]FF[4]SZ[5]KM[0]AB[ba][ab][bc][cb]AW[ca][cc][db]PL[W];W[bb]"
    state = replay_game(parse_sgf(setup + ")"))
    assert int(state.board[1, 1]) == WHITE
    assert int(state.board[1, 2]) == 0  # Captured black stone.
    with pytest.raises(SGFParseError, match="move 2"):
        replay_game(parse_sgf(setup + ";B[cb])"))


def test_replay_rejects_broken_games() -> None:
    with pytest.raises(SGFParseError):
        parse_sgf("this is not SGF")
    with pytest.raises(SGFParseError):
        parse_sgf("(;SZ[5];B[aa]")  # Unclosed tree.
    with pytest.raises(SGFParseError):
        parse_sgf("(;SZ[5];B[zz])")  # Coordinate outside the board.
    with pytest.raises(SGFParseError, match="move 2"):
        replay_game(parse_sgf("(;SZ[5];B[aa];W[aa])"))  # Occupied point.
    with pytest.raises(SGFParseError, match="move 2"):
        replay_game(parse_sgf("(;SZ[5];B[aa];B[bb])"))  # Wrong color to play.
    with pytest.raises(SGFParseError, match="setup"):
        parse_sgf("(;SZ[5];B[aa]AB[bb])")  # Setup after the first move.


def test_save_sgf_creates_directories(tmp_path: Path) -> None:
    target = tmp_path / "sgf" / "iter_001" / "game_000.sgf"
    save_sgf(target, "(;GM[1]FF[4]SZ[5])")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "(;GM[1]FF[4]SZ[5])"


def test_selfplay_records_moves_and_margin(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    controller = ResignationController(
        threshold=tiny_config.search.resign_threshold,
        target_false_positive_rate=tiny_config.search.target_false_positive_rate,
        adjustment=tiny_config.search.resignation_adjustment,
        window=tiny_config.search.resignation_window,
    )
    result = play_self_play_game(
        model=tiny_model, config=tiny_config, rng=rng, resignation_controller=controller
    )

    # Record actions alongside their pre-move training positions.
    assert len(result.moves) == result.num_moves
    board_size = tiny_config.game.board_size
    assert all(0 <= a <= board_size * board_size for a in result.moves)

    # Score margin agrees with the way the game ended.
    if result.resigned:
        assert result.score_margin is None
    else:
        assert result.score_margin is not None
        assert result.score_margin >= 0.0

    # Serialize the result as valid SGF.
    sgf = game_to_sgf(
        board_size=board_size,
        komi=tiny_config.game.komi,
        moves=result.moves,
        winner=result.winner,
        resigned=result.resigned,
        score_margin=result.score_margin,
    )
    assert sgf.startswith("(;GM[1]")
    assert sgf.count(";") == len(result.moves) + 1  # One root node plus a node per move.

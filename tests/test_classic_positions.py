"""Classical Go positions for testing legality, tactical sequences and terminal scoring."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.gtp_client import GTPClient, split_command
from alphago_zero.sgf import parse_sgf, replay_game

_SYMBOLS = {"X": BLACK, "O": WHITE, ".": 0}


def _board(rows: list[str]) -> np.ndarray:
    return np.array([[_SYMBOLS[ch] for ch in row] for row in rows], dtype=np.int8)


def _state(
    rows: list[str],
    *,
    to_play: int = BLACK,
    komi: float = 0.0,
    superko_rule: str = "positional",
) -> GoState:
    return GoState.from_board(_board(rows), to_play=to_play, komi=komi, superko_rule=superko_rule)


def _play(state: GoState, *actions: int) -> GoState:
    for action in actions:
        state = state.play(action)
    return state


def _stones(state: GoState, color: int) -> int:
    return int((state.board == color).sum())


def _point(state: GoState, row: int, col: int) -> int:
    return row * state.board_size + col


# ----------------------------------------------------------------------
# Seki with shared liberties and neutral territory
# ----------------------------------------------------------------------

SEKI = [
    "OXXO",
    "O..O",
    "OXXO",
    "OOOO",
]


def test_seki_shared_region_is_neutral_at_double_pass() -> None:
    """Shared liberties in a terminal seki score for neither side."""
    state = _state(SEKI, to_play=BLACK)
    # Only the two shared points (1,1), (1,2) and pass are legal.
    assert set(state.legal_actions()) == {_point(state, 1, 1), _point(state, 1, 2), state.pass_action}

    terminal = _play(state, state.pass_action, state.pass_action)
    assert terminal.is_terminal()
    # Both groups live (4 versus 10 stones) and neither owns the shared points.
    assert terminal.area_score() == (4.0, 10.0)


def test_seki_first_to_play_into_shared_liberties_loses_the_group() -> None:
    """Playing first in either shared liberty loses the group."""
    state = _state(SEKI, to_play=BLACK)
    after_black = state.play(_point(state, 1, 1))
    after_white = after_black.play(_point(state, 1, 2))
    assert _stones(after_white, BLACK) == 0


# ----------------------------------------------------------------------
# Snapback after a single-stone capture
# ----------------------------------------------------------------------

SNAPBACK = [
    ".......",
    ".XXXXX.",
    ".XOOOX.",
    ".XOX.X.",
    ".XOOXX.",
    ".XXX...",
    ".......",
]


def test_snapback_capture_then_recapture_takes_the_whole_group() -> None:
    """White captures (3,3) by playing (3,4) and then loses seven stones."""
    state = _state(SNAPBACK, to_play=WHITE)
    assert _stones(state, BLACK) == 16
    assert _stones(state, WHITE) == 6

    after_capture = state.play(_point(state, 3, 4))
    assert int(after_capture.board[3, 3]) == 0  # The single stone was captured.
    assert _stones(after_capture, WHITE) == 7

    after_recapture = after_capture.play(_point(state, 3, 3))
    assert _stones(after_recapture, WHITE) == 0  # The entire group is caught in the snapback.


# ----------------------------------------------------------------------
# Nakade: the vital point in a straight three
# ----------------------------------------------------------------------

NAKADE = [
    ".XXXXX.",
    "XOOOOOX",
    "XO...OX",
    "XOOOOOX",
    ".XXXXX.",
    ".......",
    ".......",
]


def test_nakade_attacker_takes_the_vital_point_and_captures() -> None:
    """Black at (2,3) can capture the ring after any White reply."""
    state = _state(NAKADE, to_play=BLACK)
    attacked = state.play(_point(state, 2, 3))
    for reply in (_point(state, 2, 2), _point(state, 2, 4)):
        line = _play(attacked, reply, _point(state, 2, 4) if reply == _point(state, 2, 2) else _point(state, 2, 2))
        assert _stones(line, WHITE) == 0


def test_nakade_defender_takes_the_vital_point_and_lives() -> None:
    """White at the central vital point makes two eyes and Black invasions are suicide."""
    state = _state(NAKADE, to_play=WHITE)
    defended = state.play(_point(state, 2, 3))
    assert not defended.is_legal(_point(state, 2, 2))
    assert not defended.is_legal(_point(state, 2, 4))


# ----------------------------------------------------------------------
# Bent four in the corner under play-out rules
# ----------------------------------------------------------------------

BENT_FOUR = [
    ".XXXO",
    ".XXXO",
    "..XXO",
    "OOOOO",
    ".OOO.",
]


def test_bent_four_counts_alive_at_double_pass() -> None:
    """The terminal board scores eight corner stones and no territory before a capturing sequence is played."""
    state = _state(BENT_FOUR, to_play=WHITE)
    terminal = _play(state, state.pass_action, state.pass_action)
    assert terminal.area_score() == (8.0, 13.0)


def test_bent_four_is_captured_when_played_out() -> None:
    """White can force capture by invading the bend at (2,1)."""
    state = _state(BENT_FOUR, to_play=WHITE)
    invasion = state.play(_point(state, 2, 1))
    p = state.pass_action
    # Enumerate all Black defenses and the winning White replies found by searching the engine.
    lines = [
        (_point(state, 1, 0), [_point(state, 2, 0), p, _point(state, 0, 0)]),
        (_point(state, 0, 0), [_point(state, 1, 0)]),
        (_point(state, 2, 0), [_point(state, 0, 0), _point(state, 1, 0), _point(state, 0, 0)]),
        (p, [_point(state, 0, 0), _point(state, 1, 0), _point(state, 2, 0), p, _point(state, 0, 0)]),
    ]
    for defense, continuation in lines:
        line = _play(invasion, defense, *continuation)
        assert _stones(line, BLACK) == 0, f"defense {defense} survived:\n{line.to_ascii()}"


# ----------------------------------------------------------------------
# Double ko under positional and situational superko
# ----------------------------------------------------------------------

DOUBLE_KO = [
    ".XO......",
    "X.XO.....",
    ".XO......",
    ".........",
    ".....XO..",
    "....XO.O.",
    ".....XO..",
    ".........",
    ".........",
]


@pytest.mark.parametrize("superko_rule", ["positional", "situational"])
def test_double_ko_immediate_recapture_is_banned(superko_rule: str) -> None:
    """Both rules prohibit an immediate recapture in either ko."""
    state = _state(DOUBLE_KO, to_play=BLACK, superko_rule=superko_rule)
    after_capture = state.play(_point(state, 5, 6))
    assert not after_capture.is_legal(_point(state, 5, 5))


@pytest.mark.parametrize(
    ("superko_rule", "closing_capture_legal"),
    [("positional", False), ("situational", True)],
)
def test_double_ko_superko_rules_close_the_cycle_at_different_moments(
    superko_rule: str, closing_capture_legal: bool
) -> None:
    """A sequence with a pass recreates the board with a different player to move. Positional superko rejects it."""
    state = _state(DOUBLE_KO, to_play=BLACK, superko_rule=superko_rule)
    state = _play(
        state,
        _point(state, 5, 6),  # Black captures in the second ko.
        _point(state, 1, 1),  # White captures in the first ko.
        state.pass_action,  # Black passes.
        _point(state, 5, 5),  # White recaptures in the second ko.
    )
    closing = _point(state, 1, 2)
    assert state.is_legal(closing) is closing_capture_legal
    if closing_capture_legal:
        closed = state.play(closing)
        assert np.array_equal(closed.board, _board(DOUBLE_KO))
        assert closed.to_play == WHITE  # The original board recurs with the other player to move.


# ----------------------------------------------------------------------
# Triple ko and a six-capture cycle
# ----------------------------------------------------------------------

TRIPLE_KO = [
    ".XO...XO.",
    "XO.O.X.XO",
    ".XO...XO.",
    ".........",
    ".........",
    ".........",
    ".....XO..",
    "....XO.O.",
    ".....XO..",
]


@pytest.mark.parametrize("superko_rule", ["positional", "situational"])
def test_triple_ko_cycle_is_banned_by_both_rules(superko_rule: str) -> None:
    """The sixth capture would recreate the starting situation. Both rules reject it."""
    state = _state(TRIPLE_KO, to_play=BLACK, superko_rule=superko_rule)
    cycle = [
        _point(state, 1, 2),  # Black captures at A.
        _point(state, 1, 6),  # White captures at B.
        _point(state, 7, 6),  # Black captures at C.
        _point(state, 1, 1),  # White recaptures at A.
        _point(state, 1, 7),  # Black recaptures at B.
    ]
    for action in cycle:
        assert state.is_legal(action)
        state = state.play(action)
    assert not state.is_legal(_point(state, 7, 5))  # White would close the cycle at C.


# ----------------------------------------------------------------------
# A ladder running into the board edge
# ----------------------------------------------------------------------

LADDER = [
    ".........",
    ".........",
    ".........",
    "....XXXXX",
    "...XOO...",
    "....X....",
    ".........",
    ".........",
    ".........",
]


def test_ladder_runs_into_the_edge_and_is_captured() -> None:
    """Successive ataris leave White a single escape until the edge ends the ladder."""
    state = _state(LADDER, to_play=BLACK)
    chase = [
        (5, 5), (4, 6),  # Atari and forced extension.
        (5, 6), (4, 7),  # Atari and forced extension.
        (5, 7), (4, 8),  # The extension reaches the edge.
        (5, 8),  # Capture five stones.
    ]
    state = _play(state, *[_point(state, r, c) for r, c in chase])
    assert _stones(state, WHITE) == 0


def test_ladder_without_extension_is_captured_at_once() -> None:
    """White loses the group on the next move after leaving the ladder."""
    state = _state(LADDER, to_play=BLACK)
    state = _play(state, _point(state, 5, 5), state.pass_action, _point(state, 4, 6))
    assert _stones(state, WHITE) == 0


# ----------------------------------------------------------------------
# Two false eyes allow legal throw-ins
# ----------------------------------------------------------------------

FALSE_EYES = [
    ".........",
    ".XXO.XXO.",
    ".X.XOX.XO",
    ".XXO.XXO.",
    ".XXXXXX..",
    ".........",
    ".........",
    ".........",
    ".........",
]


def test_two_false_eyes_can_be_thrown_into() -> None:
    """White can throw into both apparent eyes and capture part of the wall."""
    state = _state(FALSE_EYES, to_play=WHITE)
    for eye in (_point(state, 2, 2), _point(state, 2, 6)):
        assert state.is_legal(eye)  # A true eye would make this move suicide.

    invaded = state.play(_point(state, 2, 2))
    assert int(invaded.board[2, 3]) == 0  # The throw-in captures the wall stone at (2,3).
    # Immediate recapture is illegal, so Black needs a ko threat.
    assert not invaded.is_legal(_point(state, 2, 3))
    # White connects after the ko sequence and removes the apparent eye.
    connected = _play(invaded, invaded.pass_action, _point(state, 2, 3))
    assert int(connected.board[2, 2]) == WHITE
    assert int(connected.board[2, 3]) == WHITE


# ----------------------------------------------------------------------
# Optional external scoring of the classical terminal positions
# ----------------------------------------------------------------------

# Known positions where dead-stone adjudication differs from raw area scoring.
EXPECTED_DIVERGENCES = {"bent_four"}
_SCORE_RE = re.compile(r"^([BW])\+([0-9]+(?:\.[0-9]+)?)$")


def _gnugo_command() -> list[str] | None:
    explicit = os.environ.get("AGZ_GNUGO_COMMAND", "").strip()
    if explicit:
        return split_command(explicit)
    program = shutil.which("gnugo")
    if program is None:
        return None
    return [program, "--mode", "gtp", "--level", "1", "--chinese-rules", "--positional-superko"]


def _terminal_sgf(state: GoState) -> str:
    """Build an SGF with setup stones and two final passes."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    size = state.board_size

    def setup(ident: str, color: int) -> str:
        values = "".join(
            f"[{letters[col]}{letters[row]}]"
            for row in range(size)
            for col in range(size)
            if int(state.board[row, col]) == color
        )
        return f"{ident}{values}" if values else ""  # An SGF property requires at least one value.

    # Order the passes consistently with the first player inferred from the setup.
    has_black = bool((state.board == BLACK).any())
    has_white = bool((state.board == WHITE).any())
    first, second = ("W", "B") if has_black and not has_white else ("B", "W")
    return (
        f"(;GM[1]FF[4]SZ[{size}]KM[{state.komi:g}]"
        f"{setup('AB', BLACK)}{setup('AW', WHITE)};{first}[];{second}[])"
    )


def _our_result(state: GoState) -> tuple[str, float]:
    black_score, white_score = state.area_score()
    if black_score > white_score:
        return "B", black_score - white_score
    if white_score > black_score:
        return "W", white_score - black_score
    return "0", 0.0


def _terminal_cases() -> dict[str, GoState]:
    seki = _state(SEKI, to_play=BLACK)
    bent = _state(BENT_FOUR, to_play=WHITE)
    nakade_start = _state(NAKADE, to_play=BLACK)
    nakade = _play(
        nakade_start,
        _point(nakade_start, 2, 3),
        _point(nakade_start, 2, 2),
        _point(nakade_start, 2, 4),
    )
    ladder_state = _state(LADDER, to_play=BLACK)
    ladder = _play(
        ladder_state,
        *[_point(ladder_state, r, c) for r, c in [(5, 5), (4, 6), (5, 6), (4, 7), (5, 7), (4, 8), (5, 8)]],
    )
    return {
        "seki": _play(seki, seki.pass_action, seki.pass_action),
        "bent_four": _play(bent, bent.pass_action, bent.pass_action),
        "nakade_captured": _play(nakade, nakade.pass_action, nakade.pass_action),
        "ladder_captured": _play(ladder, ladder.pass_action, ladder.pass_action),
    }


@pytest.mark.parametrize("case", ["seki", "bent_four", "nakade_captured", "ladder_captured"])
def test_terminal_sgf_round_trips_through_our_parser(case: str) -> None:
    """Round-trip the setup SGF and compare the final board."""
    state = _terminal_cases()[case]
    replayed = replay_game(parse_sgf(_terminal_sgf(state)))
    assert np.array_equal(replayed.board, state.board)


@pytest.mark.parametrize("case", ["seki", "bent_four", "nakade_captured", "ladder_captured"])
def test_terminal_scores_against_gnugo(case: str, tmp_path: Path) -> None:
    """Compare terminal scores with GNU Go and allow only the documented disagreements."""
    command = _gnugo_command()
    if command is None:
        pytest.skip("GNU Go required: put gnugo on PATH or set AGZ_GNUGO_COMMAND (see docs/testing.md)")
    state = _terminal_cases()[case]
    sgf_path = tmp_path / f"{case}.sgf"
    sgf_path.write_text(_terminal_sgf(state), encoding="utf-8")

    with GTPClient(command, name="gnugo", default_timeout=60, startup_timeout=180) as client:
        client.command(f"loadsgf {sgf_path}")
        verdict = client.command("final_score").strip()

    if verdict == "0":
        judge = ("0", 0.0)
    else:
        match = _SCORE_RE.match(verdict)
        assert match, f"unrecognized GNU Go final_score: {verdict!r}"
        judge = (match.group(1), float(match.group(2)))

    ours = _our_result(state)
    agrees = judge[0] == ours[0] and abs(judge[1] - ours[1]) < 0.01
    assert agrees is (case not in EXPECTED_DIVERGENCES), (
        f"{case}: our score {ours}, GNU Go {judge}; "
        "investigate new discrepancies before changing EXPECTED_DIVERGENCES"
    )

"""Optional integration tests using GNU Go as an independent rules engine and opponent."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from alphago_zero.game import BLACK, EMPTY, GoState
from alphago_zero.gtp import action_to_vertex
from alphago_zero.gtp_client import GTPClient, GTPCommandError, split_command
from examples import external_match

STANDARD_FLAGS = ("--mode", "gtp", "--level", "1", "--chinese-rules", "--positional-superko")
RESULT = re.compile(r"^(?:[BW]\+(?:R|\d+(?:\.\d)?)|0)$")
CAPTURE_BIAS = 0.35
PASS_PROBABILITY = 0.03


def engine_command_text() -> str | None:
    """Use AGZ_GNUGO_COMMAND or launch gnugo from PATH with the configured rules."""
    explicit = os.environ.get("AGZ_GNUGO_COMMAND", "").strip()
    if explicit:
        return explicit
    program = shutil.which("gnugo")
    if program is None:
        return None
    return " ".join(f'"{part}"' for part in (program, *STANDARD_FLAGS))


@pytest.fixture(scope="module")
def gnugo(request: pytest.FixtureRequest) -> str:
    """Skip when GNU Go is unavailable unless --require-gnugo requires it."""
    text = engine_command_text()
    if text is None:
        message = "GNU Go required: put gnugo on PATH or set AGZ_GNUGO_COMMAND (see docs/testing.md)"
        if request.config.getoption("--require-gnugo"):
            pytest.fail(message)
        pytest.skip(message)
    return text


# ----------------------------------------------------------------------
# Classify illegal points independently of game.py
# ----------------------------------------------------------------------
def _neighbors(size: int, row: int, col: int) -> list[tuple[int, int]]:
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    return [(row + dr, col + dc) for dr, dc in steps if 0 <= row + dr < size and 0 <= col + dc < size]


def _group(board: np.ndarray, row: int, col: int) -> tuple[list[tuple[int, int]], int]:
    """Collect a group and count its liberties using breadth-first search."""
    size = board.shape[0]
    color = board[row, col]
    stones = {(row, col)}
    stack = [(row, col)]
    liberties: set[tuple[int, int]] = set()
    while stack:
        current = stack.pop()
        for neighbor in _neighbors(size, *current):
            if board[neighbor] == EMPTY:
                liberties.add(neighbor)
            elif board[neighbor] == color and neighbor not in stones:
                stones.add(neighbor)
                stack.append(neighbor)
    return sorted(stones), len(liberties)


def is_suicide(board: np.ndarray, row: int, col: int, color: int) -> bool:
    """A suicide placement leaves the friendly group without liberties after captures."""
    trial = board.copy()
    trial[row, col] = color
    for neighbor in _neighbors(board.shape[0], row, col):
        if trial[neighbor] == -color:
            stones, liberties = _group(trial, *neighbor)
            if liberties == 0:
                for stone in stones:
                    trial[stone] = EMPTY
    return _group(trial, row, col)[1] == 0


def captures(state: GoState, action: int) -> bool:
    """A capture leaves fewer stones than the old stone count plus one."""
    after = state.play(action)
    return int((after.board != EMPTY).sum()) < int((state.board != EMPTY).sum()) + 1


def choose_move(state: GoState, legal: set[int], rng: random.Random) -> int:
    """Favor captures in random play to exercise more ko and repetition cases."""
    candidates = sorted(action for action in legal if action != state.pass_action)
    if not candidates or rng.random() < PASS_PROBABILITY:
        return state.pass_action
    capturing = [action for action in candidates if captures(state, action)]
    if capturing and rng.random() < CAPTURE_BIAS:
        return rng.choice(capturing)
    return rng.choice(candidates)


def compare_with_gnugo(
    command: list[str], *, size: int, komi: float, games: int, seed: int
) -> tuple[Counter[str], list[str]]:
    """Check every point against GNU Go's is_legal and play identical moves in both engines."""
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    problems: list[str] = []
    with GTPClient(command, name="gnugo", default_timeout=60, startup_timeout=180) as client:
        for game in range(games):
            for line in (f"boardsize {size}", "clear_board", f"komi {komi}"):
                client.command(line)
            state = GoState.new_game(board_size=size, komi=komi)
            while not state.is_terminal():
                color = "b" if state.to_play == BLACK else "w"
                legal = set(state.legal_actions())
                for action in range(size * size):
                    vertex = action_to_vertex(action, size)
                    theirs = client.command(f"is_legal {color} {vertex}").strip() == "1"
                    mine = action in legal
                    row, col = divmod(action, size)
                    if state.board[row, col] == EMPTY and not mine:
                        suicide = is_suicide(state.board, row, col, state.to_play)
                        stats["illegal_suicide" if suicide else "illegal_ko_or_superko"] += 1
                    if mine != theirs:
                        problems.append(
                            f"game {game}, move {state.move_number}, {color} {vertex}: "
                            f"local {mine}, GNU Go {theirs}\n{state.to_ascii()}"
                        )
                stats["positions"] += 1
                action = choose_move(state, legal, rng)
                try:
                    client.command(f"play {color} {action_to_vertex(action, size)}")
                except GTPCommandError as error:
                    problems.append(f"game {game}, move {state.move_number}: GNU Go rejected our legal move: {error}")
                    break
                state = state.play(action)
    return stats, problems


def test_our_rules_agree_with_gnugo_on_every_point_of_random_games(gnugo: str) -> None:
    stats, problems = compare_with_gnugo(split_command(gnugo), size=5, komi=0.5, games=25, seed=1)
    assert not problems, (
        "GNU Go disagreements (use --positional-superko when comparing repetition rules):\n"
        + "\n".join(problems[:3])
    )
    # Require suicide and superko examples as well as occupied points.
    assert stats["positions"] >= 500
    assert stats["illegal_suicide"] > 0
    assert stats["illegal_ko_or_superko"] > 0


@pytest.mark.parametrize("separate_referee", [False, True], ids=["opponent-scores", "separate-referee"])
def test_a_match_against_real_gnugo_has_no_void_games(
    gnugo: str,
    checkpoints: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    separate_referee: bool,
) -> None:
    """Run the actual 5x5 checkpoint frontend against GNU Go."""
    out = tmp_path / "gnugo.json"
    sgf_dir = tmp_path / "sgf"
    argv = [
        "--checkpoint", str(checkpoints[0]), "--simulations", "4", "--model-name", "tiny",
        "--opponent-command", gnugo, "--opponent-name", "gnugo",
        "--games", "4", "--board-size", "5", "--komi", "2.5", "--opening-moves", "2", "--seed", "3",
        "--json-out", str(out), "--sgf-dir", str(sgf_dir),
        "--startup-timeout", "180", "--move-timeout", "60", "--engine-timeout", "60",
    ]  # fmt: skip
    if separate_referee:
        argv += ["--referee-command", gnugo, "--referee-name", "gnugo-referee"]
    code = external_match.main(argv)
    capsys.readouterr()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and data["aborted"] is None
    assert data["games_played"] == 4 and data["voids"] == 0
    assert data["engines"]["opponent"]["name"] == "GNU Go"
    assert data["scoring"] == ("referee" if separate_referee else "opponent")
    results = [game["result"] for game in data["games_detail"]]
    assert all(RESULT.match(result) for result in results), results
    assert len(list(sgf_dir.glob("*.sgf"))) == 4

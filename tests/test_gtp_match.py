"""Use in-memory GTP participants to test match play, scoring and failure handling."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.gtp import vertex_to_action
from alphago_zero.gtp_client import GTPCommandError, GTPConnectionError
from alphago_zero.gtp_match import (
    ExternalGame,
    MatchError,
    MatchResult,
    MatchSettings,
    choose_opening,
    parse_final_score,
    run_match,
)

Handler = Callable[[list[str]], str]
SETTINGS = MatchSettings(games=2, board_size=5, komi=7.5, opening_moves=0)
SETUP = ["boardsize 5", "komi 7.5", "clear_board"]


class ScriptedPlayer:
    """Script genmove responses by game and emulate other GTP commands."""

    def __init__(
        self, name: str, scripts: Sequence[Sequence[str]] = ((),), *, final_score: str = "B+1.5"
    ) -> None:
        self.name = name
        self.scripts = [list(script) for script in scripts]
        self.final_score = final_score
        self.log: list[str] = []
        self.timeouts: list[tuple[str, float | None]] = []  # Command and supplied timeout.
        self.games_started = 0  # Number of clear_board calls.
        self.restarts = 0
        self.restart_error: Exception | None = None
        self.overrides: dict[str, Handler] = {}
        self._own_moves = 0

    def command(self, line: str, *, timeout: float | None = None) -> str:
        self.log.append(line)
        name, *args = line.split()
        self.timeouts.append((name, timeout))
        handler = self.overrides.get(name)
        return handler(args) if handler is not None else self.respond(name, args)

    def respond(self, name: str, args: list[str]) -> str:
        """Default handler for overrides that delegate ordinary commands."""
        if name == "clear_board":
            self.games_started += 1
            self._own_moves = 0
        elif name == "genmove":
            script = self.scripts[min(self.games_started - 1, len(self.scripts) - 1)]
            reply = script[self._own_moves] if self._own_moves < len(script) else "pass"
            self._own_moves += 1
            return reply
        elif name == "final_score":
            return self.final_score
        return ""

    def restart(self) -> None:
        if self.restart_error is not None:
            raise self.restart_error
        self.restarts += 1

    def genmoves(self) -> list[str]:
        return [line for line in self.log if line.startswith("genmove")]


def rejecting(player: ScriptedPlayer, command: str, message: str = "illegal move") -> Handler:
    def handler(_args: list[str]) -> str:
        raise GTPCommandError(player.name, command, message)

    return handler


def failing_once(player: ScriptedPlayer, command: str, error: Exception) -> Handler:
    """Raise on the first call and resume ordinary handling afterward."""
    calls = {"count": 0}

    def handler(args: list[str]) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise error
        return player.respond(command, args)

    return handler


def slow(player: ScriptedPlayer, seconds: float) -> Handler:
    """Delay genmove so the test can measure game duration."""

    def handler(args: list[str]) -> str:
        time.sleep(seconds)
        return player.respond("genmove", args)

    return handler


def play_one(model: ScriptedPlayer, opponent: ScriptedPlayer, **changes: object) -> ExternalGame:
    settings = replace(SETTINGS, games=1, **changes)  # type: ignore[arg-type]
    return run_match(model, opponent, settings).games[0]


# ----------------------------------------------------------------------
# parse_final_score
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B+3.5", (BLACK, 3.5)),
        ("W+10", (WHITE, 10.0)),
        ("w+0.5", (WHITE, 0.5)),
        ("b+7.", (BLACK, 7.0)),
        ("  B+7.5  ", (BLACK, 7.5)),
        ("0", (0, 0.0)),
        ("0.0", (0, 0.0)),
        ("jigo", (0, 0.0)),
        ("B+0", (0, 0.0)),
        ("W+0.0", (0, 0.0)),
    ],
)
def test_parse_final_score_accepts_the_gtp_forms(text: str, expected: tuple[int, float]) -> None:
    assert parse_final_score(text) == expected


@pytest.mark.parametrize(
    "text", ["", "B+", "+3", "B+abc", "B+R", "B+Resign", "X+3", "B+inf", "B+nan", "B+-3", "B+3.5 extra", "B 3.5", "3.5"]
)
def test_parse_final_score_rejects_everything_else(text: str) -> None:
    with pytest.raises(ValueError, match="final_score"):
        parse_final_score(text)


# ----------------------------------------------------------------------
# MatchSettings
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"games": 0}, "games"),
        ({"board_size": 2}, "board_size"),
        ({"board_size": 26}, "board_size"),
        ({"komi": float("nan")}, "komi"),
        ({"komi": float("inf")}, "komi"),
        ({"opening_moves": -1}, "opening_moves"),
        ({"opening_moves": 7}, "opening_moves"),  # On 5x5, the maximum opening is 25 // 4 = 6.
        ({"move_timeout": 0.0}, "move_timeout"),
        ({"move_timeout": float("nan")}, "move_timeout"),
        ({"move_timeout": float("inf")}, "move_timeout"),
        ({"max_moves": 0}, "max_moves"),
        ({"opening_moves": 2, "max_moves": 2}, "max_moves"),
        ({"max_consecutive_voids": 0}, "max_consecutive_voids"),
    ],
)
def test_settings_validate_their_fields(changes: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(SETTINGS, **changes)  # type: ignore[arg-type]


def test_settings_accept_the_boundaries() -> None:
    MatchSettings(games=1, board_size=3, opening_moves=2)
    MatchSettings(games=1, board_size=25, opening_moves=0)
    MatchSettings(games=1, board_size=5, opening_moves=6, max_moves=7)


# ----------------------------------------------------------------------
# Shared openings
# ----------------------------------------------------------------------
def test_opening_is_deterministic_per_seed_and_pair() -> None:
    settings = MatchSettings(games=2, board_size=9, opening_moves=4, seed=7)
    assert choose_opening(settings, 3) == choose_opening(settings, 3)
    assert choose_opening(settings, 3) != choose_opening(settings, 4)
    assert choose_opening(settings, 3) != choose_opening(replace(settings, seed=8), 3)


def test_opening_consists_of_legal_interior_moves() -> None:
    settings = MatchSettings(games=2, board_size=9, opening_moves=6)
    for pair in range(20):
        opening = choose_opening(settings, pair)
        assert len(opening) == 6
        state = GoState.new_game(board_size=9, komi=7.5)
        for action in opening:
            row, col = divmod(action, 9)
            assert action != state.pass_action
            assert 1 <= row <= 7 and 1 <= col <= 7  # Interior points only.
            assert state.is_legal(action)
            state = state.play(action)


def test_opening_varies_across_pairs() -> None:
    settings = MatchSettings(games=2, board_size=9, opening_moves=4)
    assert len({choose_opening(settings, pair) for pair in range(10)}) >= 8


def test_zero_opening_moves_give_an_empty_opening() -> None:
    assert choose_opening(replace(SETTINGS, opening_moves=0), 0) == ()


def test_opening_stops_when_no_interior_move_is_left() -> None:
    """The only interior point on a 3x3 board cannot provide two opening placements."""
    assert len(choose_opening(MatchSettings(games=1, board_size=3, opening_moves=2), 0)) == 1


# ----------------------------------------------------------------------
# Game progression
# ----------------------------------------------------------------------
def test_two_passes_end_the_game_and_the_opponent_scores_it() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["pass"]], final_score="W+3.5")
    result = run_match(model, opponent, replace(SETTINGS, games=1))
    game = result.games[0]
    assert game.valid and game.winner == WHITE
    assert game.result == "W+3.5" and game.margin == 3.5 and not game.resigned
    assert game.moves == (25, 25)  # Both passes use action N*N.
    assert game.model_is_black and game.model_outcome == 0.0
    assert (result.wins, result.losses, result.draws, result.voids) == (0, 1, 0, 0)


def test_every_move_is_relayed_to_the_other_engine_and_the_model_is_never_asked_to_score() -> None:
    model = ScriptedPlayer("model", [["C3", "pass"]])
    opponent = ScriptedPlayer("pachi", [["B2", "pass"]], final_score="B+9.5")
    game = play_one(model, opponent)
    assert model.log == [*SETUP, "genmove B", "play W B2", "genmove B", "play W pass"]
    assert opponent.log == [*SETUP, "play B C3", "genmove W", "play B pass", "genmove W", "final_score"]
    assert game.winner == BLACK and game.result == "B+9.5" and game.model_outcome == 1.0
    assert game.moves == tuple(vertex_to_action(v, 5) for v in ("C3", "B2", "pass", "pass"))


def test_only_thinking_commands_get_the_move_timeout() -> None:
    """Allow long timeouts for genmove and final_score. Other commands use the client default."""
    model = ScriptedPlayer("model", [["C3", "pass"]])
    opponent = ScriptedPlayer("pachi", [["B2", "pass"]])
    play_one(model, opponent, move_timeout=33.0)
    for player in (model, opponent):
        for name, timeout in player.timeouts:
            expected = 33.0 if name in {"genmove", "final_score"} else None
            assert timeout == expected, f"{player.name}: {name}"
    assert {name for name, _ in opponent.timeouts} == {"boardsize", "komi", "clear_board", "play", "genmove", "final_score"}


@pytest.mark.parametrize(("score", "result"), [("W+10.0", "W+10"), ("b+7.5", "B+7.5"), ("B+7.", "B+7")])
def test_the_result_string_is_normalised(score: str, result: str) -> None:
    game = play_one(ScriptedPlayer("model", [["pass"]]), ScriptedPlayer("pachi", [["pass"]], final_score=score))
    assert game.result == result


def test_the_game_duration_is_measured() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["pass"]])
    opponent.overrides["genmove"] = slow(opponent, 0.05)
    assert play_one(model, opponent).seconds >= 0.04


def test_the_duration_of_a_void_game_is_measured_too() -> None:
    """Measure elapsed time for void games too, so failed matches account for their cost."""
    opponent = ScriptedPlayer("pachi", [["banana"]])
    opponent.overrides["genmove"] = slow(opponent, 0.05)  # The following response, banana, is invalid.
    game = play_one(ScriptedPlayer("model", [["pass"]]), opponent)
    assert not game.valid and game.seconds >= 0.04


@pytest.mark.parametrize("reply", ["resign", "Resign", "RESIGN"])
def test_resignation_ends_the_game_without_scoring(reply: str) -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [[reply]], final_score="W+99")
    game = play_one(model, opponent)
    assert game.winner == BLACK and game.result == "B+R" and game.resigned and game.margin is None
    assert "final_score" not in opponent.log and "final_score" not in model.log


def test_resignation_by_the_model_is_a_loss() -> None:
    model = ScriptedPlayer("model", [["resign"]])
    game = play_one(model, ScriptedPlayer("pachi"))
    assert game.winner == WHITE and game.result == "W+R" and game.model_outcome == 0.0


def test_vertices_are_accepted_in_any_case() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["b2", "PASS"]])
    game = play_one(model, opponent)
    assert game.valid
    assert "play W B2" in model.log  # Forward our normalized vertex spelling to the other player.


def test_the_game_stops_at_the_move_limit_and_is_scored() -> None:
    model = ScriptedPlayer("model", [["C3", "E5", "E1"]])
    opponent = ScriptedPlayer("pachi", [["A1", "A5", "C1"]], final_score="B+9.5")
    game = play_one(model, opponent, max_moves=6)
    assert len(game.moves) == 6 and game.valid and game.winner == BLACK
    assert len(model.genmoves()) == 3 and len(opponent.genmoves()) == 3


def test_a_draw_is_recorded_as_such() -> None:
    result = run_match(
        ScriptedPlayer("model", [["pass"]]),
        ScriptedPlayer("pachi", [["pass"]], final_score="0"),
        replace(SETTINGS, games=1),
    )
    game = result.games[0]
    assert game.winner == 0 and game.result == "0" and game.margin is None
    assert game.model_outcome == 0.5 and result.draws == 1
    assert result.points == 0.5 and result.win_rate == 0.5


def test_colors_alternate_between_games() -> None:
    model = ScriptedPlayer("model")
    opponent = ScriptedPlayer("pachi")
    result = run_match(model, opponent, replace(SETTINGS, games=4))
    assert [g.model_is_black for g in result.games] == [True, False, True, False]
    assert model.genmoves() == ["genmove B", "genmove W", "genmove B", "genmove W"]
    assert opponent.genmoves() == ["genmove W", "genmove B", "genmove W", "genmove B"]


def test_the_opening_is_shared_inside_a_pair_and_sent_to_both_engines() -> None:
    settings = MatchSettings(games=4, board_size=9, opening_moves=2, seed=5)
    model, opponent = ScriptedPlayer("model"), ScriptedPlayer("pachi")
    games = run_match(model, opponent, settings).games
    assert games[0].opening == games[1].opening and games[2].opening == games[3].opening
    assert games[0].opening != games[2].opening
    assert all(len(g.opening) == 2 and g.moves[:2] == g.opening for g in games)
    expected = [f"play B {_vertex(games[0].opening[0])}", f"play W {_vertex(games[0].opening[1])}"]
    for player in (model, opponent):
        start = player.log.index("clear_board") + 1
        assert player.log[start : start + 2] == expected


def _vertex(action: int) -> str:
    from alphago_zero.gtp import action_to_vertex

    return action_to_vertex(action, 9)


# ----------------------------------------------------------------------
# Void games
# ----------------------------------------------------------------------
def test_an_illegal_move_makes_the_game_void_naming_the_culprit() -> None:
    model = ScriptedPlayer("model", [["C3"]])
    opponent = ScriptedPlayer("pachi", [["C3"]])  # Occupied point.
    game = play_one(model, opponent)
    assert not game.valid and game.winner is None and game.result == "Void" and game.model_outcome is None
    assert game.void_reason is not None
    assert "disputed move" in game.void_reason and "pachi" in game.void_reason and "C3" in game.void_reason
    assert game.moves == (vertex_to_action("C3", 5),)
    assert "play W C3" not in model.log  # Do not forward an illegal move.


@pytest.mark.parametrize("reply", ["banana", "", "Z99", "A0"])
def test_a_reply_that_is_not_a_move_makes_the_game_void(reply: str) -> None:
    game = play_one(ScriptedPlayer("model", [["pass"]]), ScriptedPlayer("pachi", [[reply]]))
    assert not game.valid and game.void_reason is not None and "invalid move" in game.void_reason


def test_a_rejected_relay_makes_the_game_void_naming_the_engine_and_the_command() -> None:
    model = ScriptedPlayer("model", [["C3"]])
    opponent = ScriptedPlayer("pachi")
    opponent.overrides["play"] = rejecting(opponent, "play B C3")
    game = play_one(model, opponent)
    assert not game.valid and game.void_reason is not None
    assert "pachi" in game.void_reason and "play B C3" in game.void_reason
    assert opponent.restarts == 0 and model.restarts == 0  # A rejected command leaves the engine alive and requires no restart.


def test_a_rejected_genmove_makes_the_game_void() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi")
    opponent.overrides["genmove"] = rejecting(opponent, "genmove W", "cannot think")
    game = play_one(model, opponent)
    assert not game.valid and game.void_reason is not None and "cannot think" in game.void_reason


def test_a_crash_voids_the_game_and_restarts_only_the_broken_engine() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["pass"]], final_score="B+2.5")
    opponent.overrides["genmove"] = failing_once(opponent, "genmove", GTPConnectionError("pachi: engine exited"))
    result = run_match(model, opponent, replace(SETTINGS, games=2))
    first, second = result.games
    assert not first.valid and first.void_reason == "pachi: engine exited"
    assert (opponent.restarts, model.restarts) == (1, 0)
    assert second.valid and second.winner == BLACK and not second.model_is_black  # The series resumed.
    assert (result.voids, result.losses, result.valid_games) == (1, 1, 1)


def test_a_crash_during_setup_voids_the_game_and_restarts_the_engine() -> None:
    model = ScriptedPlayer("model")
    opponent = ScriptedPlayer("pachi")
    model.overrides["boardsize"] = failing_once(model, "boardsize", GTPConnectionError("model: no response"))
    result = run_match(model, opponent, replace(SETTINGS, games=2))
    assert [g.valid for g in result.games] == [False, True]
    assert model.restarts == 1 and opponent.restarts == 0


def test_a_restart_failure_aborts_the_match() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi")
    opponent.overrides["genmove"] = failing_once(opponent, "genmove", GTPConnectionError("pachi: crashed"))
    opponent.restart_error = GTPConnectionError("pachi: failed to start")
    with pytest.raises(MatchError, match="restart") as excinfo:
        run_match(model, opponent, replace(SETTINGS, games=2))
    assert "pachi" in str(excinfo.value)


@pytest.mark.parametrize("command", ["boardsize", "komi", "clear_board"])
def test_a_rejected_setup_command_aborts_the_match_instead_of_voiding_games(command: str) -> None:
    model = ScriptedPlayer("model")
    model.overrides[command] = rejecting(model, command, "unacceptable size")
    with pytest.raises(MatchError, match=command) as excinfo:
        run_match(model, ScriptedPlayer("pachi"), replace(SETTINGS, games=3))
    assert "model" in str(excinfo.value) and excinfo.value.games == ()


def test_a_setup_rejection_midway_keeps_the_games_already_played() -> None:
    """An aborted match must retain completed game records."""
    model = ScriptedPlayer("model")
    opponent = ScriptedPlayer("pachi")

    def reject_from_the_second_game(args: list[str]) -> str:
        if model.games_started >= 1:
            raise GTPCommandError("model", "boardsize 5", "unacceptable size")
        return ""

    model.overrides["boardsize"] = reject_from_the_second_game
    with pytest.raises(MatchError, match="boardsize") as excinfo:
        run_match(model, opponent, replace(SETTINGS, games=4))
    assert [g.index for g in excinfo.value.games] == [0] and excinfo.value.games[0].valid


def test_unreadable_final_score_voids_the_game() -> None:
    game = play_one(
        ScriptedPlayer("model", [["pass"]]), ScriptedPlayer("pachi", [["pass"]], final_score="B+lots")
    )
    assert not game.valid and game.void_reason is not None
    assert "final_score" in game.void_reason and "pachi" in game.void_reason


def test_a_series_of_voids_aborts_the_match_with_the_games_played() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["banana"]])
    seen: list[int] = []
    with pytest.raises(MatchError, match="3 consecutive void games") as excinfo:
        run_match(model, opponent, replace(SETTINGS, games=6), on_game=lambda g: seen.append(g.index))
    assert len(excinfo.value.games) == 3 and all(not g.valid for g in excinfo.value.games)
    assert "banana" in str(excinfo.value) and seen == [0, 1, 2]


def test_the_abort_message_lists_only_the_void_streak() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["pass"], ["banana"], ["banana"], ["banana"]])
    with pytest.raises(MatchError) as excinfo:
        run_match(model, opponent, replace(SETTINGS, games=6))
    message = str(excinfo.value)
    assert "game 2" in message and "game 4" in message
    assert "game 1" not in message and "None" not in message  # The first game was valid.


def test_a_valid_game_resets_the_void_streak() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    scripts = [["banana"], ["banana"], ["pass"], ["banana"], ["banana"]]
    opponent = ScriptedPlayer("pachi", scripts)
    result = run_match(model, opponent, replace(SETTINGS, games=5))
    assert [g.valid for g in result.games] == [False, False, True, False, False]
    assert result.voids == 4


def test_the_void_limit_is_configurable() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["banana"]])
    with pytest.raises(MatchError, match="2 consecutive void") as excinfo:
        run_match(model, opponent, replace(SETTINGS, games=6, max_consecutive_voids=2))
    assert len(excinfo.value.games) == 2


# ----------------------------------------------------------------------
# Independent referee
# ----------------------------------------------------------------------
def test_a_dedicated_referee_replays_the_game_and_decides_the_result() -> None:
    model = ScriptedPlayer("model", [["C3", "pass"]])
    opponent = ScriptedPlayer("pachi", [["B2", "pass"]], final_score="B+99")
    referee = ScriptedPlayer("gnugo", final_score="W+8.5")
    result = run_match(model, opponent, replace(SETTINGS, games=1), referee=referee)
    assert referee.log == [*SETUP, "play B C3", "play W B2", "play B pass", "play W pass", "final_score"]
    assert "final_score" not in opponent.log  # Only the referee supplies the score.
    game = result.games[0]
    assert game.winner == WHITE and game.result == "W+8.5"


def test_the_referee_is_not_consulted_after_a_resignation() -> None:
    referee = ScriptedPlayer("gnugo")
    game = play_one_with_referee(ScriptedPlayer("model", [["resign"]]), ScriptedPlayer("pachi"), referee)
    assert game.resigned and referee.log == []


def play_one_with_referee(model: ScriptedPlayer, opponent: ScriptedPlayer, referee: ScriptedPlayer) -> ExternalGame:
    return run_match(model, opponent, replace(SETTINGS, games=1), referee=referee).games[0]


def test_a_referee_that_rejects_a_replayed_move_voids_the_game() -> None:
    referee = ScriptedPlayer("gnugo")
    referee.overrides["play"] = rejecting(referee, "play B C3", "illegal move")
    game = play_one_with_referee(ScriptedPlayer("model", [["C3"]]), ScriptedPlayer("pachi", [["B2"]]), referee)
    assert not game.valid and game.void_reason is not None and "gnugo" in game.void_reason


def test_a_referee_crash_restarts_the_referee_only() -> None:
    model, opponent, referee = ScriptedPlayer("model"), ScriptedPlayer("pachi"), ScriptedPlayer("gnugo")
    referee.overrides["final_score"] = failing_once(referee, "final_score", GTPConnectionError("gnugo: crashed"))
    result = run_match(model, opponent, replace(SETTINGS, games=2), referee=referee)
    assert [g.valid for g in result.games] == [False, True]
    assert (referee.restarts, model.restarts, opponent.restarts) == (1, 0, 0)


def test_a_referee_that_cannot_set_up_the_board_aborts_the_match() -> None:
    referee = ScriptedPlayer("gnugo")
    referee.overrides["boardsize"] = rejecting(referee, "boardsize 5", "unacceptable size")
    with pytest.raises(MatchError, match="gnugo"):
        play_one_with_referee(ScriptedPlayer("model"), ScriptedPlayer("pachi"), referee)


# ----------------------------------------------------------------------
# Counters and callbacks
# ----------------------------------------------------------------------
def _game(index: int, model_is_black: bool, winner: int | None) -> ExternalGame:
    return ExternalGame(index, model_is_black, (), (), winner, "x", None, False, None if winner is not None else "why", 0.0)


def test_match_result_counts_outcomes_and_per_color_records() -> None:
    games = (
        _game(0, True, BLACK),  # Model plays Black and wins.
        _game(1, False, BLACK),  # Model plays White and loses.
        _game(2, True, WHITE),  # Model plays Black and loses.
        _game(3, False, WHITE),  # Model plays White and wins.
        _game(4, True, 0),  # Draw.
        _game(5, False, None),  # Void game.
    )
    result = MatchResult(games)
    assert (result.wins, result.losses, result.draws, result.voids) == (2, 2, 1, 1)
    assert result.valid_games == 5 and result.points == 2.5 and result.win_rate == 0.5
    assert result.side_record(as_black=True) == (3, 1.5)
    assert result.side_record(as_black=False) == (2, 1.0)


def test_an_empty_result_has_a_zero_win_rate() -> None:
    assert MatchResult(()).win_rate == 0.0


def test_on_game_sees_every_game_in_order_including_voids() -> None:
    model = ScriptedPlayer("model", [["pass"]])
    opponent = ScriptedPlayer("pachi", [["banana"], ["pass"]])
    seen: list[tuple[int, bool]] = []
    run_match(model, opponent, replace(SETTINGS, games=2), on_game=lambda g: seen.append((g.index, g.valid)))
    assert seen == [(0, False), (1, True)]


# ----------------------------------------------------------------------
# End-to-end synchronization
# ----------------------------------------------------------------------
class StatefulPlayer:
    """Miniature engine that plays legal random moves and enforces color order using GoState."""

    def __init__(self, name: str, seed: int, max_moves: int) -> None:
        self.name = name
        self.rng = random.Random(seed)
        self.board_size = 5
        self.komi = 7.5
        self.max_moves = max_moves  # Match the controller's move limit so the engine cannot terminate early.
        self.state = GoState.new_game(board_size=5, komi=7.5, max_moves=max_moves)

    def command(self, line: str, *, timeout: float | None = None) -> str:
        name, *args = line.split()
        if name == "boardsize":
            self.board_size = int(args[0])
        elif name == "komi":
            self.komi = float(args[0])
        elif name == "clear_board":
            self.state = GoState.new_game(board_size=self.board_size, komi=self.komi, max_moves=self.max_moves)
        elif name == "play":
            self._apply(line, args[0], vertex_to_action(args[1], self.board_size))
        elif name == "genmove":
            return self._genmove(line, args[0])
        elif name == "final_score":
            black, white = self.state.area_score()
            return "0" if black == white else f"{'B' if black > white else 'W'}+{abs(black - white):g}"
        else:
            raise GTPCommandError(self.name, line, "unknown command")
        return ""

    def _apply(self, line: str, color: str, action: int) -> None:
        expected = "B" if self.state.to_play == BLACK else "W"
        if color != expected or not self.state.is_legal(action):
            raise GTPCommandError(self.name, line, "illegal move")
        self.state = self.state.play(action)

    def _genmove(self, line: str, color: str) -> str:
        from alphago_zero.gtp import action_to_vertex

        stones = [a for a in self.state.legal_actions() if a != self.state.pass_action]
        action = self.rng.choice(stones) if stones and self.rng.random() < 0.93 else self.state.pass_action
        self._apply(line, color, action)
        return action_to_vertex(action, self.board_size)

    def restart(self) -> None:
        raise AssertionError("normal games should not require a restart")


def test_the_controller_and_both_engines_stay_in_sync_over_random_games() -> None:
    model, opponent = StatefulPlayer("model", 1, 60), StatefulPlayer("opponent", 2, 60)
    settings = MatchSettings(games=8, board_size=5, komi=7.5, opening_moves=2, seed=3, max_moves=60)
    result = run_match(model, opponent, settings)
    assert result.voids == 0 and result.valid_games == 8
    for game in result.games:
        replay = GoState.new_game(board_size=5, komi=7.5, max_moves=60)
        for action in game.moves:
            assert replay.is_legal(action)
            replay = replay.play(action)
        assert replay.is_terminal()
        black, white = replay.area_score()
        expected = BLACK if black > white else WHITE
        assert game.winner == expected and game.margin == pytest.approx(abs(black - white))
    # Both engines must end on the board obtained by replaying the recorded actions.
    last = result.games[-1]
    final = GoState.new_game(board_size=5, komi=7.5, max_moves=60)
    for action in last.moves:
        final = final.play(action)
    assert (model.state.board == final.board).all() and (opponent.state.board == final.board).all()

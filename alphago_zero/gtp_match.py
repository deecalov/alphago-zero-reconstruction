"""Play and adjudicate a series against an external GTP engine. Keep void games separate from valid results."""

from __future__ import annotations

import math
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from .game import BLACK, WHITE, GoState
from .gtp import action_to_vertex, vertex_to_action
from .gtp_client import GTPCommandError, GTPConnectionError, GTPError

_COLOR_NAME: Final = {BLACK: "B", WHITE: "W"}
_SCORE: Final = re.compile(r"([BW])\+(\d+(?:\.\d*)?)", re.IGNORECASE)
_DRAW_TEXTS: Final = frozenset({"0", "0.0", "jigo"})
_SUPERKO: Final = "positional"


class GTPPlayer(Protocol):
    """Protocol for a match participant, implemented by GTPClient."""

    name: str

    def command(self, line: str, *, timeout: float | None = None) -> str: ...

    def restart(self) -> None: ...


class MatchError(Exception):
    """Fatal match setup, restart or repeated-void error. Retain completed game records."""

    def __init__(self, message: str, games: Sequence[ExternalGame] = ()) -> None:
        super().__init__(message)
        self.games: tuple[ExternalGame, ...] = tuple(games)


class _VoidGame(Exception):
    """Internal signal for a void game. The broken participant may need a fresh process."""

    def __init__(self, reason: str, broken: GTPPlayer | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.broken = broken


@dataclass(frozen=True, slots=True)
class MatchSettings:
    """Series settings. The default game length limit is twice the number of board points."""

    games: int
    board_size: int = 9
    komi: float = 7.5
    opening_moves: int = 4
    move_timeout: float = 120.0
    max_moves: int | None = None
    seed: int = 0
    max_consecutive_voids: int = 3

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError("games must be at least 1")
        if not 3 <= self.board_size <= 25:
            raise ValueError("board_size must be between 3 and 25")
        if not math.isfinite(self.komi):
            raise ValueError("komi must be finite")
        if not 0 <= self.opening_moves <= self.board_size**2 // 4:
            raise ValueError(f"opening_moves must be between 0 and {self.board_size**2 // 4} for board size {self.board_size}")
        if not (math.isfinite(self.move_timeout) and self.move_timeout > 0):
            raise ValueError("move_timeout must be a positive finite number")
        if self.max_moves is not None and self.max_moves <= self.opening_moves:
            raise ValueError("max_moves must exceed opening_moves")
        if self.max_consecutive_voids < 1:
            raise ValueError("max_consecutive_voids must be at least 1")


@dataclass(frozen=True, slots=True)
class ExternalGame:
    """Game result. The winner is BLACK, WHITE, zero for a draw or None for a void game."""

    index: int
    model_is_black: bool
    opening: tuple[int, ...]
    moves: tuple[int, ...]
    winner: int | None
    result: str
    margin: float | None
    resigned: bool
    void_reason: str | None
    seconds: float

    @property
    def valid(self) -> bool:
        return self.winner is not None

    @property
    def model_outcome(self) -> float | None:
        """Return model points (1, 0.5 or 0), or None for a void game."""
        if self.winner is None:
            return None
        if self.winner == 0:
            return 0.5
        model_color = BLACK if self.model_is_black else WHITE
        return 1.0 if self.winner == model_color else 0.0


@dataclass(frozen=True, slots=True)
class MatchResult:
    """All game records, including voids, together with series totals."""

    games: tuple[ExternalGame, ...]

    def _count(self, outcome: float) -> int:
        return sum(1 for game in self.games if game.model_outcome == outcome)

    @property
    def wins(self) -> int:
        return self._count(1.0)

    @property
    def losses(self) -> int:
        return self._count(0.0)

    @property
    def draws(self) -> int:
        return self._count(0.5)

    @property
    def voids(self) -> int:
        return sum(1 for game in self.games if not game.valid)

    @property
    def valid_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def points(self) -> float:
        """Model points for win rate and Wilson interval calculations."""
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.points / self.valid_games if self.valid_games else 0.0

    def side_record(self, *, as_black: bool) -> tuple[int, float]:
        """Return valid game count and model points for the specified color."""
        played = [g for g in self.games if g.valid and g.model_is_black == as_black]
        return len(played), sum(g.model_outcome or 0.0 for g in played)


def parse_final_score(text: str) -> tuple[int, float]:
    """Parse a numeric GTP score and return its winning color and margin. Treat zero as a draw."""
    token = text.strip()
    if token.lower() in _DRAW_TEXTS:
        return 0, 0.0
    matched = _SCORE.fullmatch(token)
    if matched is None:
        raise ValueError(f"could not parse final_score: {text!r}")
    margin = float(matched.group(2))
    if margin == 0.0:
        return 0, 0.0
    return (BLACK if matched.group(1).upper() == "B" else WHITE), margin


def _new_state(settings: MatchSettings) -> GoState:
    return GoState.new_game(
        board_size=settings.board_size, komi=settings.komi, max_moves=settings.max_moves, superko_rule=_SUPERKO
    )


def _is_interior(action: int, board_size: int) -> bool:
    """Test whether an action is an interior placement. Pass is outside the board."""
    row, col = divmod(action, board_size)
    return 1 <= row <= board_size - 2 and 1 <= col <= board_size - 2


def choose_opening(settings: MatchSettings, pair_index: int) -> tuple[int, ...]:
    """Generate legal interior opening moves deterministically from the seed and pair number."""
    rng = random.Random(f"external-opening:{settings.seed}:{pair_index}")
    state = _new_state(settings)
    moves: list[int] = []
    for _ in range(settings.opening_moves):
        candidates = [action for action in state.legal_actions() if _is_interior(action, settings.board_size)]
        if not candidates:
            break
        action = rng.choice(candidates)
        state = state.play(action)
        moves.append(action)
    return tuple(moves)


@dataclass(slots=True)
class _GameRun:
    """Track the local rules board and moves accepted by both players."""

    black: GTPPlayer
    white: GTPPlayer
    settings: MatchSettings
    state: GoState
    moves: list[int] = field(default_factory=list)
    resigned_by: int | None = None

    def player(self, color: int) -> GTPPlayer:
        return self.black if color == BLACK else self.white


@dataclass(frozen=True, slots=True)
class _Verdict:
    winner: int
    result: str
    margin: float | None = None
    resigned: bool = False


def _play_line(color: int, action: int, settings: MatchSettings) -> str:
    return f"play {_COLOR_NAME[color]} {action_to_vertex(action, settings.board_size)}"


def _send(player: GTPPlayer, line: str, *, timeout: float | None = None) -> str:
    """Send a player command. Keep engine errors as void-game diagnostics."""
    try:
        return player.command(line, timeout=timeout)
    except GTPCommandError as error:
        raise _VoidGame(f"{player.name} rejected command {line!r}: {error.message}") from error
    except GTPConnectionError as error:
        raise _VoidGame(str(error), broken=player) from error


def _setup_player(player: GTPPlayer, settings: MatchSettings) -> None:
    """Configure board size and komi and clear the board. Setup failures abort the series."""
    for line in (f"boardsize {settings.board_size}", f"komi {settings.komi:g}", "clear_board"):
        try:
            player.command(line)
        except GTPCommandError as error:
            raise MatchError(
                f"{player.name}: setup command {line!r} was rejected ({error.message}); "
                "the engine cannot play under these match settings"
            ) from error
        except GTPConnectionError as error:
            raise _VoidGame(str(error), broken=player) from error


def _advance(run: _GameRun, action: int) -> None:
    run.state = run.state.play(action)
    run.moves.append(action)


def _play_opening(run: _GameRun, opening: Sequence[int]) -> None:
    for action in opening:
        color = run.state.to_play
        for player in (run.black, run.white):
            _send(player, _play_line(color, action, run.settings))
        _advance(run, action)


def _parse_move(run: _GameRun, mover: GTPPlayer, reply: str) -> int | None:
    """Parse a generated move or resignation. Treat disputed moves as void games."""
    token = reply.strip().lower()
    if token == "resign":
        return None
    try:
        action = vertex_to_action(token, run.settings.board_size)
    except ValueError as error:
        raise _VoidGame(f"{mover.name} returned an invalid move: {reply!r} ({error})") from error
    if not run.state.is_legal(action):
        raise _VoidGame(
            f"disputed move: {mover.name} played {reply.strip()} at move {run.state.move_number + 1}, "
            "which is illegal under the match rules"
        )
    return action


def _play_out(run: _GameRun) -> None:
    while not run.state.is_terminal():
        color = run.state.to_play
        mover = run.player(color)
        reply = _send(mover, f"genmove {_COLOR_NAME[color]}", timeout=run.settings.move_timeout)
        action = _parse_move(run, mover, reply)
        if action is None:
            run.resigned_by = color
            return
        _send(run.player(WHITE if color == BLACK else BLACK), _play_line(color, action, run.settings))
        _advance(run, action)


def _replay(run: _GameRun, referee: GTPPlayer) -> None:
    """Replay all moves on an independent referee, starting with Black."""
    _setup_player(referee, run.settings)
    for index, action in enumerate(run.moves):
        _send(referee, _play_line(BLACK if index % 2 == 0 else WHITE, action, run.settings))


def _verdict(run: _GameRun, scorer: GTPPlayer) -> _Verdict:
    if run.resigned_by is not None:
        winner = WHITE if run.resigned_by == BLACK else BLACK
        return _Verdict(winner, f"{_COLOR_NAME[winner]}+R", resigned=True)
    if scorer is not run.black and scorer is not run.white:
        _replay(run, scorer)  # A participating referee already has the final position.
    text = _send(scorer, "final_score", timeout=run.settings.move_timeout)
    try:
        winner, margin = parse_final_score(text)
    except ValueError as error:
        raise _VoidGame(f"{scorer.name}: {error}") from error
    if winner == 0:
        return _Verdict(0, "0")
    return _Verdict(winner, f"{_COLOR_NAME[winner]}+{margin:g}", margin)


def _revive(player: GTPPlayer | None) -> None:
    if player is None:
        return
    try:
        player.restart()
    except GTPError as error:
        raise MatchError(f"{player.name}: failed to restart engine after failure: {error}") from error


def _play_one(
    index: int,
    model: GTPPlayer,
    opponent: GTPPlayer,
    scorer: GTPPlayer,
    opening: tuple[int, ...],
    settings: MatchSettings,
) -> ExternalGame:
    model_is_black = index % 2 == 0
    black, white = (model, opponent) if model_is_black else (opponent, model)
    run = _GameRun(black=black, white=white, settings=settings, state=_new_state(settings))
    started = time.monotonic()
    try:
        for player in (black, white):
            _setup_player(player, settings)
        _play_opening(run, opening)
        _play_out(run)
        verdict = _verdict(run, scorer)
    except _VoidGame as void:
        _revive(void.broken)
        elapsed = time.monotonic() - started
        return ExternalGame(
            index, model_is_black, opening, tuple(run.moves), None, "Void", None, False, void.reason, elapsed
        )
    return ExternalGame(
        index,
        model_is_black,
        opening,
        tuple(run.moves),
        verdict.winner,
        verdict.result,
        verdict.margin,
        verdict.resigned,
        None,
        time.monotonic() - started,
    )


def _abort_message(games: Sequence[ExternalGame], limit: int) -> str:
    reasons = "; ".join(f"game {g.index + 1}: {g.void_reason}" for g in games[-limit:])
    return (
        f"{limit} consecutive void games. Check rules and engine arguments. "
        f"Errors: {reasons}"
    )


def run_match(
    model: GTPPlayer,
    opponent: GTPPlayer,
    settings: MatchSettings,
    *,
    referee: GTPPlayer | None = None,
    on_game: Callable[[ExternalGame], None] | None = None,
) -> MatchResult:
    """Play alternating colors with a shared opening for each pair. Use the referee or designated engine to
    score each completed game.
    """
    scorer = referee if referee is not None else opponent
    games: list[ExternalGame] = []
    voids_in_a_row = 0
    for index in range(settings.games):
        opening = choose_opening(settings, index // 2)
        try:
            game = _play_one(index, model, opponent, scorer, opening, settings)
        except MatchError as error:
            error.games = tuple(games)
            raise
        games.append(game)
        if on_game is not None:
            on_game(game)
        voids_in_a_row = 0 if game.valid else voids_in_a_row + 1
        if voids_in_a_row >= settings.max_consecutive_voids:
            raise MatchError(_abort_message(games, voids_in_a_row), games)
    return MatchResult(tuple(games))

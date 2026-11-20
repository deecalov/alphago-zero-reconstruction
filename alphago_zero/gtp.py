from __future__ import annotations

"""Go Text Protocol v2 interface for playing the model from a GUI or match controller."""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable, TextIO

import numpy as np
import torch

from .config import AlphaGoZeroConfig
from .game import BLACK, WHITE, GoState
from .mcts import AlphaGoZeroMCTS, MCTSNode, SearchOutput
from .network import PolicyValueModel
from .sgf import parse_sgf, replay_states

# GTP column labels skip I.
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

ENGINE_NAME = "alphago-zero-reconstruction"
ENGINE_VERSION = "0.1"


def action_to_vertex(action: int, board_size: int) -> str:
    """Convert an internal action to a GTP vertex such as D4 or pass."""
    if action == board_size * board_size:
        return "pass"
    row = action // board_size
    col = action % board_size
    gtp_row = board_size - row  # GTP numbers rows from the bottom.
    return f"{GTP_COLUMNS[col]}{gtp_row}"


def vertex_to_action(vertex: str, board_size: int) -> int:
    """Convert a GTP vertex to an action. Raise ValueError for invalid input."""
    token = vertex.strip().upper()
    if token == "PASS":
        return board_size * board_size

    if len(token) < 2:
        raise ValueError(f"invalid vertex: {vertex!r}")
    col_letter, row_digits = token[0], token[1:]
    if col_letter not in GTP_COLUMNS or not row_digits.isdigit():
        raise ValueError(f"invalid vertex: {vertex!r}")

    col = GTP_COLUMNS.index(col_letter)
    gtp_row = int(row_digits)
    row = board_size - gtp_row
    if col >= board_size or not 0 <= row < board_size:
        raise ValueError(f"vertex outside the board: {vertex!r}")
    return row * board_size + col


def _parse_color(token: str) -> int:
    lowered = token.strip().lower()
    if lowered in ("b", "black"):
        return BLACK
    if lowered in ("w", "white"):
        return WHITE
    raise ValueError(f"invalid color: {token!r}")


@dataclass(slots=True)
class _ParsedCommand:
    command_id: str  # Empty string when the controller omitted the command ID.
    name: str
    args: list[str]


def _parse_line(line: str) -> _ParsedCommand | None:
    """Parse an optional ID, a command and its arguments. Ignore blank lines and comments."""
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    tokens = text.split()
    command_id = ""
    if tokens[0].isdigit():
        command_id = tokens[0]
        tokens = tokens[1:]
        if not tokens:
            return None
    return _ParsedCommand(command_id=command_id, name=tokens[0].lower(), args=tokens[1:])


class GTPEngine:
    """Manage a game and its search tree in response to commands from a GTP controller."""

    def __init__(
        self,
        model: PolicyValueModel,
        config: AlphaGoZeroConfig,
        rng: np.random.Generator | None = None,
        allow_resign: bool = True,
        opening_moves: int = 0,
        tree_reuse: bool = True,
        clock: Callable[[], float] = time.monotonic,
        policy_only: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.allow_resign = allow_resign
        # Sample the configured number of opening moves to vary deterministic games.
        self.opening_moves = opening_moves
        self.tree_reuse = tree_reuse
        self.policy_only = policy_only
        self._clock = clock
        # Track main time, Canadian overtime and remaining seconds per color from controller updates.
        self._time_settings: tuple[float, float, int] | None = None
        self._time_left: dict[int, float] = {}

        self.board_size = config.game.board_size
        self.komi = config.game.komi
        self.state: GoState
        self.search: AlphaGoZeroMCTS
        self._history: list[GoState] = []  # Position stack for undo.
        # If present, the search root always represents self.state.
        self._root: MCTSNode | None = None
        self._new_game()

        self._handlers: dict[str, Callable[[list[str]], str]] = {
            "protocol_version": self._cmd_protocol_version,
            "name": self._cmd_name,
            "version": self._cmd_version,
            "known_command": self._cmd_known_command,
            "list_commands": self._cmd_list_commands,
            "quit": self._cmd_quit,
            "boardsize": self._cmd_boardsize,
            "clear_board": self._cmd_clear_board,
            "komi": self._cmd_komi,
            "play": self._cmd_play,
            "genmove": self._cmd_genmove,
            "reg_genmove": self._cmd_genmove,
            "undo": self._cmd_undo,
            "loadsgf": self._cmd_loadsgf,
            "showboard": self._cmd_showboard,
            "final_score": self._cmd_final_score,
            "gogui-analyze_commands": self._cmd_gogui_analyze_commands,
            "agz_best_moves": self._cmd_best_moves,
            # Per-move search time budget, computed by _move_time_budget.
            "time_settings": self._cmd_time_settings,
            "time_left": self._cmd_time_left,
        }

    # ------------------------------------------------------------------
    # Public protocol interface
    # ------------------------------------------------------------------
    def handle_line(self, line: str) -> tuple[str | None, bool]:
        """Handle one protocol line. Return a formatted response and a flag indicating whether to quit."""
        parsed = _parse_line(line)
        if parsed is None:
            return None, False

        handler = self._handlers.get(parsed.name)
        if handler is None:
            return self._failure(parsed.command_id, "unknown command"), False

        try:
            result = handler(parsed.args)
        except ValueError as exc:
            return self._failure(parsed.command_id, str(exc)), False

        return self._success(parsed.command_id, result), parsed.name == "quit"

    def run(self, stream_in: TextIO = sys.stdin, stream_out: TextIO = sys.stdout) -> None:
        """Read commands until quit or end of input and write their responses."""
        for line in stream_in:
            response, should_quit = self.handle_line(line)
            if response is not None:
                stream_out.write(response)
                stream_out.flush()
            if should_quit:
                break

    # ------------------------------------------------------------------
    # Protocol commands
    # ------------------------------------------------------------------
    def _cmd_protocol_version(self, _args: list[str]) -> str:
        return "2"

    def _cmd_name(self, _args: list[str]) -> str:
        return ENGINE_NAME

    def _cmd_version(self, _args: list[str]) -> str:
        return ENGINE_VERSION

    def _cmd_known_command(self, args: list[str]) -> str:
        if not args:
            raise ValueError("known_command requires an argument")
        return "true" if args[0].lower() in self._handlers else "false"

    def _cmd_list_commands(self, _args: list[str]) -> str:
        return "\n".join(sorted(self._handlers))

    def _cmd_quit(self, _args: list[str]) -> str:
        return ""

    def _cmd_boardsize(self, args: list[str]) -> str:
        if not args or not args[0].isdigit():
            raise ValueError("boardsize requires an integer")
        size = int(args[0])
        if size != self.config.game.board_size:
            # The network supports the board size used at training time.
            raise ValueError("unacceptable size")
        self.board_size = size
        self._new_game()
        return ""

    def _cmd_clear_board(self, _args: list[str]) -> str:
        self._new_game()
        return ""

    def _cmd_komi(self, args: list[str]) -> str:
        if not args:
            raise ValueError("komi requires a number")
        try:
            new_komi = float(args[0])
        except ValueError as exc:
            raise ValueError(f"invalid komi: {args[0]!r}") from exc
        # Komi is absent from the input planes. The checkpoint's value estimates were calibrated at its
        # training komi.
        if not (new_komi == self.config.game.komi):
            raise ValueError("unacceptable komi")
        self.komi = new_komi
        self._new_game()
        return ""

    def _cmd_play(self, args: list[str]) -> str:
        if len(args) < 2:
            raise ValueError("play requires a color and vertex")
        color = _parse_color(args[0])
        action = vertex_to_action(args[1], self.board_size)

        if color != self.state.to_play:
            raise ValueError("illegal move: out of turn")
        if not self.state.is_legal(action):
            raise ValueError("illegal move")
        self._apply_move(action)
        return ""

    def _cmd_genmove(self, args: list[str]) -> str:
        if not args:
            raise ValueError("genmove requires a color")
        color = _parse_color(args[0])
        if color != self.state.to_play:
            raise ValueError("illegal move: out of turn")
        if self.state.is_terminal():
            return "pass"

        if self.policy_only:
            # Casual mode plays the raw policy without search or resignation.
            action = self._policy_action()
            self._apply_move(action)
            return action_to_vertex(action, self.board_size)

        # Apply the controller's time allowance as an additional bound on the simulation budget.
        budget = self._move_time_budget(color)
        started = self._clock()
        output = self.search.run(
            root_state=self.state,
            temperature=self.config.search.deterministic_temperature,
            add_root_noise=False,
            root=self._root if self.tree_reuse else None,
            time_budget_seconds=budget,
            early_exit_unreachable=True,
            clock=self._clock,
        )
        if budget is not None and color in self._time_left:
            spent = self._clock() - started
            self._time_left[color] = max(0.0, self._time_left[color] - spent)
        if self.tree_reuse:
            # Advance this search tree along the chosen move in _apply_move.
            self._root = output.root

        if self.allow_resign and self._should_resign(output):
            return "resign"

        if self.state.move_number < self.opening_moves:
            distribution = output.visit_distribution
            total = float(distribution.sum())
            if total > 0.0:
                action = int(self.rng.choice(len(distribution), p=distribution / total))
            else:
                action = int(np.argmax(output.policy))
        else:
            action = int(np.argmax(output.policy))
        self._apply_move(action)
        return action_to_vertex(action, self.board_size)

    def _policy_action(self) -> int:
        """Select the legal action with the highest raw policy logit, as PolicyPlayer does."""
        logits, _value = self.model.predict_single(self.state.encode())
        legal = self.state.legal_actions()
        return int(legal[int(np.argmax(logits[legal]))])

    def _policy_best_moves(self, limit: int) -> str:
        """Build a GoGui analysis display from raw policy probabilities in casual mode."""
        logits, value = self.model.predict_single(self.state.encode())
        legal = self.state.legal_actions()
        legal_logits = logits[legal].astype(np.float64)
        legal_logits -= float(np.max(legal_logits))
        weights = np.exp(legal_logits)
        probs = weights / weights.sum()
        winrate = (float(value) + 1.0) / 2.0 * 100.0
        side = "black" if self.state.to_play == BLACK else "white"
        lines = [f"TEXT winrate {winrate:.1f}% for {side} (policy only, no search)"]
        shown = 0
        for index in np.argsort(-probs):
            if shown >= limit:
                break
            action = int(legal[int(index)])
            if action == self.state.pass_action:
                continue
            vertex = action_to_vertex(action, self.board_size)
            lines.append(f"LABEL {vertex} {probs[int(index)] * 100.0:.0f}%")
            shown += 1
        return "\n".join(lines)

    def _should_resign(self, output: SearchOutput) -> bool:
        """Require the value-based resignation test and any configured move number or score safeguards."""
        search = self.config.search
        if self.state.move_number < search.resign_min_move:
            return False
        if not (
            output.root_q < search.resign_threshold
            and output.best_child_q < search.resign_threshold
        ):
            return False
        if search.resign_score_margin is None:
            return True
        margin = self._predicted_score_margin()
        return margin is None or margin < -search.resign_score_margin

    def _predicted_score_margin(self) -> float | None:
        """Return the predicted score lead for the player to move, or None when the model lacks a score head."""
        forward_full = getattr(self.model, "forward_full", None)
        if forward_full is None:
            return None
        device = torch.device("cpu")
        parameters = getattr(self.model, "parameters", None)
        if parameters is not None:
            try:
                device = next(parameters()).device
            except StopIteration:
                pass
        planes = torch.from_numpy(np.ascontiguousarray(self.state.encode()))
        with torch.no_grad():
            outputs = forward_full(planes.unsqueeze(0).to(device))
        score = outputs.get("score")
        if score is None:
            return None
        return float(score.item()) * float(self.board_size * self.board_size)

    def _cmd_undo(self, _args: list[str]) -> str:
        """Undo one move and discard the search root."""
        if not self._history:
            raise ValueError("cannot undo")
        self.state = self._history.pop()
        self._root = None
        return ""

    def _cmd_loadsgf(self, args: list[str]) -> str:
        """Replay an SGF to the requested move, reconstructing history and superko along the way."""
        if not args:
            raise ValueError("loadsgf requires a filename")
        try:
            text = Path(args[0]).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"loadsgf: could not read file: {exc}") from exc
        parsed = parse_sgf(text)  # SGFParseError derives from ValueError and becomes a GTP error response.
        if parsed.board_size != self.config.game.board_size:
            raise ValueError("unacceptable size")
        if parsed.komi != self.config.game.komi:
            raise ValueError("unacceptable komi")
        move_number = len(parsed.moves)
        if len(args) > 1:
            try:
                move_number = int(args[1])
            except ValueError as exc:
                raise ValueError(f"loadsgf: invalid move number: {args[1]!r}") from exc
            if not 0 <= move_number <= len(parsed.moves):
                raise ValueError(
                    f"loadsgf: move number outside range 0..{len(parsed.moves)}: {move_number}"
                )
        states = replay_states(parsed, superko_rule=self.config.game.superko_rule)
        self._history = list(states[:move_number])
        self.state = states[move_number]
        self._root = None
        return ""

    def _cmd_gogui_analyze_commands(self, _args: list[str]) -> str:
        return "gfx/Best moves/agz_best_moves"

    def _cmd_best_moves(self, args: list[str]) -> str:
        """Report root evaluation, visit counts and win rates as GoGui gfx text and labels."""
        limit = 10
        if args:
            try:
                limit = int(args[0])
            except ValueError as exc:
                raise ValueError(f"agz_best_moves: invalid limit: {args[0]!r}") from exc
            if limit < 1:
                raise ValueError("agz_best_moves: limit must be >= 1")
        if self.state.is_terminal():
            return "TEXT terminal position: no moves"

        if self.policy_only:
            return self._policy_best_moves(limit)

        output = self.search.run(
            root_state=self.state,
            temperature=self.config.search.deterministic_temperature,
            add_root_noise=False,
            root=None,
        )
        total = int(output.visit_counts.sum())
        winrate = (output.root_q + 1.0) / 2.0 * 100.0
        side = "black" if self.state.to_play == BLACK else "white"
        summary = f"TEXT winrate {winrate:.1f}% for {side} ({total} sims)"
        pass_edge = output.root.edges.get(self.state.pass_action)
        if pass_edge is not None and pass_edge.visit_count > 0:
            summary += f"; pass {pass_edge.visit_count}v"
        lines = [summary]
        ranked = sorted(
            output.root.edges.items(), key=lambda item: (-item[1].visit_count, item[0])
        )
        shown = 0
        for action, edge in ranked:
            if shown >= limit or edge.visit_count <= 0:
                break
            if action == self.state.pass_action:
                continue
            move_winrate = (edge.q_value + 1.0) / 2.0 * 100.0
            vertex = action_to_vertex(action, self.board_size)
            lines.append(f"LABEL {vertex} {edge.visit_count}v/{move_winrate:.0f}%")
            shown += 1
        return "\n".join(lines)

    def _cmd_showboard(self, _args: list[str]) -> str:
        return "\n" + self.state.to_ascii()

    def _cmd_time_settings(self, args: list[str]) -> str:
        """Set main time and Canadian overtime (seconds per period and stones per period)."""
        if len(args) < 3:
            raise ValueError("time_settings requires three numbers: main byoyomi stones")
        try:
            main_time = float(args[0])
            byoyomi_time = float(args[1])
            byoyomi_stones = int(args[2])
        except ValueError as exc:
            raise ValueError(f"time_settings: invalid numbers: {args!r}") from exc
        if main_time < 0 or byoyomi_time < 0 or byoyomi_stones < 0:
            raise ValueError("time_settings: times must be nonnegative")
        self._time_settings = (main_time, byoyomi_time, byoyomi_stones)
        self._time_left = {BLACK: main_time, WHITE: main_time}
        return ""

    def _cmd_time_left(self, args: list[str]) -> str:
        """Accept the controller's authoritative remaining time for the given color."""
        if len(args) < 3:
            raise ValueError("time_left requires color, time and stone count")
        color = _parse_color(args[0])
        try:
            seconds = float(args[1])
            int(args[2])  # The budget calculation does not use the remaining stone count.
        except ValueError as exc:
            raise ValueError(f"time_left: invalid numbers: {args!r}") from exc
        self._time_left[color] = max(0.0, seconds)
        return ""

    def _move_time_budget(self, color: int) -> float | None:
        """Allocate seconds for a move from remaining time and expected moves left. Return None when no
        clock was set.
        """
        if self._time_settings is None:
            return None
        main_time, byoyomi_time, byoyomi_stones = self._time_settings
        per_move_byoyomi = (
            byoyomi_time / byoyomi_stones if byoyomi_stones > 0 else byoyomi_time
        )
        remaining = self._time_left.get(color, main_time)
        empty = int(np.count_nonzero(self.state.board == 0))
        moves_left = max(8, (empty + 1) // 2)
        return remaining / moves_left + per_move_byoyomi

    def _cmd_final_score(self, _args: list[str]) -> str:
        black_score, white_score = self.state.area_score()
        if black_score > white_score:
            return f"B+{black_score - white_score:g}"
        if white_score > black_score:
            return f"W+{white_score - black_score:g}"
        return "0"

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------
    def _apply_move(self, action: int) -> None:
        """Push the previous state for undo, play the move and advance or clear the cached search root."""
        self._history.append(self.state)
        self.state = self.state.play(action)
        if self._root is not None:
            self._root = self.search.advance_root(self._root, action)

    def _new_game(self) -> None:
        self.state = GoState.new_game(
            board_size=self.board_size,
            komi=self.komi,
            max_moves=self.config.game.max_moves,
            superko_rule=self.config.game.superko_rule,
        )
        self._history = []
        self._root = None
        self.model.eval()
        self.search = AlphaGoZeroMCTS(model=self.model, config=self.config.search, rng=self.rng)

    @staticmethod
    def _success(command_id: str, result: str) -> str:
        prefix = f"={command_id}" if command_id else "="
        if result:
            return f"{prefix} {result}\n\n"
        return f"{prefix}\n\n"

    @staticmethod
    def _failure(command_id: str, message: str) -> str:
        prefix = f"?{command_id}" if command_id else "?"
        return f"{prefix} {message}\n\n"

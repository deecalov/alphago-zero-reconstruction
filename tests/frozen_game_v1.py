# Frozen Go engine from before the board-level move-generation optimization.
#
# The simple reference simulates each move. This second reference uses the earlier group algorithm.
#
# Keep this reference independent of later changes to the production engine.
from __future__ import annotations

"""Historical Go engine used for differential tests of rule behavior."""

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


BLACK = 1
WHITE = -1
EMPTY = 0


# ---------------------------------------------------------------------------
# Zobrist position hashes
# ---------------------------------------------------------------------------
# A fixed seed makes hashes reproducible across processes and runs.
_ZOBRIST_CACHE: dict[int, list[list[list[int]]]] = {}


def _zobrist_table(board_size: int) -> list[list[list[int]]]:
    """Build a [color][row][column] table of 63-bit Python integers for fast scalar XOR."""
    table = _ZOBRIST_CACHE.get(board_size)
    if table is None:
        rng = np.random.default_rng(0x5EED0A60 + board_size)
        raw = rng.integers(1, np.iinfo(np.int64).max, size=(2, board_size, board_size), dtype=np.int64)
        table = [
            [[int(raw[color, row, col]) for col in range(board_size)] for row in range(board_size)]
            for color in range(2)
        ]
        _ZOBRIST_CACHE[board_size] = table
    return table


def _color_index(color: int) -> int:
    """Map black to Zobrist table index zero and white to one."""
    return 0 if color == BLACK else 1


def _board_zobrist_hash(board: np.ndarray, board_size: int) -> int:
    """Recompute a board hash when constructing a state from scratch."""
    table = _zobrist_table(board_size)
    h = 0
    rows = board.tolist()
    for row in range(board_size):
        row_values = rows[row]
        for col in range(board_size):
            value = row_values[col]
            if value != EMPTY:
                h ^= table[_color_index(value)][row][col]
    return h


# For situational superko, mix in this key when White is to play. Black contributes zero.
_WHITE_TO_PLAY_KEY = int(
    np.random.default_rng(0x5EED0A60).integers(1, np.iinfo(np.int64).max)
)


def _situation_key(position_hash: int, to_play: int, superko_rule: str) -> int:
    """Return the repetition key for positional or situational superko."""
    if superko_rule == "situational" and to_play == WHITE:
        return position_hash ^ _WHITE_TO_PLAY_KEY
    return position_hash


# ---------------------------------------------------------------------------
# Analyze groups once per board instead of simulating every candidate.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _BoardAnalysis:
    """Single-pass group analysis permits constant-time legality checks after preprocessing."""

    group_id: list[list[int]]  # Use -1 for empty points.
    colors: list[int]
    liberty_counts: list[int]
    stones: list[list[tuple[int, int]]]


def _analyze_board(board: np.ndarray, board_size: int) -> _BoardAnalysis:
    """Collect group IDs, liberties and stones in one traversal."""
    cells = board.tolist()
    group_id: list[list[int]] = [[-1] * board_size for _ in range(board_size)]
    colors: list[int] = []
    liberty_counts: list[int] = []
    stones: list[list[tuple[int, int]]] = []

    last = board_size - 1
    for row in range(board_size):
        for col in range(board_size):
            if cells[row][col] == EMPTY or group_id[row][col] != -1:
                continue

            gid = len(colors)
            color = cells[row][col]
            group_stones: list[tuple[int, int]] = []
            liberties: set[tuple[int, int]] = set()
            stack = [(row, col)]
            group_id[row][col] = gid

            while stack:
                r, c = stack.pop()
                group_stones.append((r, c))
                # Inline neighbor access avoids generator overhead in this loop.
                if r > 0:
                    v = cells[r - 1][c]
                    if v == EMPTY:
                        liberties.add((r - 1, c))
                    elif v == color and group_id[r - 1][c] == -1:
                        group_id[r - 1][c] = gid
                        stack.append((r - 1, c))
                if r < last:
                    v = cells[r + 1][c]
                    if v == EMPTY:
                        liberties.add((r + 1, c))
                    elif v == color and group_id[r + 1][c] == -1:
                        group_id[r + 1][c] = gid
                        stack.append((r + 1, c))
                if c > 0:
                    v = cells[r][c - 1]
                    if v == EMPTY:
                        liberties.add((r, c - 1))
                    elif v == color and group_id[r][c - 1] == -1:
                        group_id[r][c - 1] = gid
                        stack.append((r, c - 1))
                if c < last:
                    v = cells[r][c + 1]
                    if v == EMPTY:
                        liberties.add((r, c + 1))
                    elif v == color and group_id[r][c + 1] == -1:
                        group_id[r][c + 1] = gid
                        stack.append((r, c + 1))

            colors.append(color)
            liberty_counts.append(len(liberties))
            stones.append(group_stones)

    return _BoardAnalysis(
        group_id=group_id,
        colors=colors,
        liberty_counts=liberty_counts,
        stones=stones,
    )


@dataclass(slots=True)
class GoState:
    """Go position in absolute colors (Black +1, White -1, empty 0). Network inputs use the perspective of
    the player to move.
    """

    board_size: int
    komi: float
    max_moves: int
    board: np.ndarray
    to_play: int = BLACK
    consecutive_passes: int = 0
    move_number: int = 0

    # History starts with the current board. States share earlier arrays and treat them as read-only.
    board_history: tuple[np.ndarray, ...] = ()

    # Store all previous repetition keys. Situational keys include the player to move.
    seen_hashes: set[int] = field(default_factory=set)

    # Update the current board hash incrementally in play().
    position_hash: int = 0

    # Select positional or situational superko.
    superko_rule: str = "positional"

    # Populate the group cache lazily for each position.
    _analysis: _BoardAnalysis | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def new_game(
        cls,
        board_size: int = 19,
        komi: float = 7.5,
        max_moves: int | None = None,
        superko_rule: str = "positional",
    ) -> "GoState":
        """Start an empty game."""
        if max_moves is None:
            max_moves = board_size * board_size * 2

        board = np.zeros((board_size, board_size), dtype=np.int8)
        # An empty board with Black to move has key zero under both repetition rules.
        return cls(
            board_size=board_size,
            komi=komi,
            max_moves=max_moves,
            board=board,
            to_play=BLACK,
            consecutive_passes=0,
            move_number=0,
            board_history=(board,),
            seen_hashes={0},
            position_hash=0,
            superko_rule=superko_rule,
        )

    @classmethod
    def from_board(
        cls,
        board: np.ndarray,
        to_play: int,
        komi: float,
        max_moves: int | None = None,
        consecutive_passes: int = 0,
        move_number: int = 0,
        superko_rule: str = "positional",
    ) -> "GoState":
        """Construct an analysis position with consistent board history and repetition information."""
        board = np.asarray(board, dtype=np.int8).copy()
        if board.ndim != 2 or board.shape[0] != board.shape[1]:
            raise ValueError(f"expected a square [N, N] board, got {board.shape}")
        board_size = int(board.shape[0])
        if max_moves is None:
            max_moves = board_size * board_size * 2

        position_hash = _board_zobrist_hash(board, board_size)
        return cls(
            board_size=board_size,
            komi=komi,
            max_moves=max_moves,
            board=board,
            to_play=to_play,
            consecutive_passes=consecutive_passes,
            move_number=move_number,
            board_history=(board,),
            seen_hashes={_situation_key(position_hash, to_play, superko_rule)},
            position_hash=position_hash,
            superko_rule=superko_rule,
        )

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size + 1

    @property
    def pass_action(self) -> int:
        return self.board_size * self.board_size

    def copy(self) -> "GoState":
        """Copy the current board and game state for independent inspection."""
        board_copy = self.board.copy()
        return GoState(
            board_size=self.board_size,
            komi=self.komi,
            max_moves=self.max_moves,
            board=board_copy,
            to_play=self.to_play,
            consecutive_passes=self.consecutive_passes,
            move_number=self.move_number,
            board_history=(board_copy,) + self.board_history[1:],
            seen_hashes=set(self.seen_hashes),
            position_hash=self.position_hash,
            superko_rule=self.superko_rule,
        )

    # ---------------------------------------------------------------------
    # Action and coordinate conversion
    # ---------------------------------------------------------------------
    def action_to_coord(self, action: int) -> tuple[int, int] | None:
        # Pass maps to None. Board actions map to (row, col). Reject values outside the action space.
        if action < 0 or action > self.pass_action:
            raise ValueError(
                f"action outside range [0, {self.pass_action}]: {action}"
            )
        if action == self.pass_action:
            return None
        row = action // self.board_size
        col = action % self.board_size
        return row, col

    def coord_to_action(self, row: int, col: int) -> int:
        return row * self.board_size + col

    # ---------------------------------------------------------------------
    # Board geometry
    # ---------------------------------------------------------------------
    def neighbors(self, row: int, col: int) -> Iterator[tuple[int, int]]:
        if row > 0:
            yield row - 1, col
        if row + 1 < self.board_size:
            yield row + 1, col
        if col > 0:
            yield row, col - 1
        if col + 1 < self.board_size:
            yield row, col + 1

    # ---------------------------------------------------------------------
    # Group analysis and move evaluation
    # ---------------------------------------------------------------------
    def _get_analysis(self) -> _BoardAnalysis:
        """Compute group analysis on demand and share it across legality queries."""
        if self._analysis is None:
            self._analysis = _analyze_board(self.board, self.board_size)
        return self._analysis

    def _evaluate_stone_move(self, row: int, col: int) -> tuple[bool, list[int], int]:
        """Return legality, captured group IDs and the resulting Zobrist hash."""
        analysis = self._get_analysis()
        gid_grid = analysis.group_id
        if gid_grid[row][col] != -1:
            return False, [], 0  # Occupied point.

        liberty_counts = analysis.liberty_counts
        colors = analysis.colors
        own_color = self.to_play
        last = self.board_size - 1

        has_empty_neighbor = False
        own_spare_liberty = False
        captured: list[int] = []

        # Inline neighbor checks on the legal_actions hot path.
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nr < 0 or nr > last or nc < 0 or nc > last:
                continue
            gid = gid_grid[nr][nc]
            if gid == -1:
                has_empty_neighbor = True
            elif colors[gid] == own_color:
                if liberty_counts[gid] >= 2:
                    own_spare_liberty = True
            elif liberty_counts[gid] == 1 and gid not in captured:
                # The move occupies an opponent group's last liberty.
                captured.append(gid)

        if not has_empty_neighbor and not own_spare_liberty and not captured:
            return False, [], 0  # Suicide is illegal.

        # Update the position hash incrementally.
        table = _zobrist_table(self.board_size)
        new_hash = self.position_hash ^ table[_color_index(own_color)][row][col]
        if captured:
            opponent_index = _color_index(-own_color)
            stones = analysis.stones
            for gid in captured:
                for sr, sc in stones[gid]:
                    new_hash ^= table[opponent_index][sr][sc]

        if _situation_key(new_hash, -own_color, self.superko_rule) in self.seen_hashes:
            return False, [], 0  # Repeated position or situation.

        return True, captured, new_hash

    # ---------------------------------------------------------------------
    # Legal moves
    # ---------------------------------------------------------------------
    def is_legal(self, action: int) -> bool:
        # Treat actions outside the valid range as illegal.
        if action < 0 or action > self.pass_action:
            return False

        if self.is_terminal():
            return False

        # Pass is always legal and allows the players to end the game.
        if action == self.pass_action:
            return True

        row = action // self.board_size
        col = action % self.board_size
        legal, _captured, _new_hash = self._evaluate_stone_move(row, col)
        return legal

    def legal_actions(self) -> list[int]:
        """Return legal placements and pass using cached group analysis."""
        if self.is_terminal():
            return []

        actions: list[int] = []
        board_size = self.board_size
        for row in range(board_size):
            base = row * board_size
            for col in range(board_size):
                legal, _captured, _new_hash = self._evaluate_stone_move(row, col)
                if legal:
                    actions.append(base + col)
        actions.append(self.pass_action)
        return actions

    # ---------------------------------------------------------------------
    # Transition function
    # ---------------------------------------------------------------------
    def play(self, action: int) -> "GoState":
        """Create a successor state. Earlier positions and their histories remain available to other search
        branches.
        """
        if action == self.pass_action:
            if not self.is_legal(action):
                raise ValueError(f"Illegal move: {action}")
            # Pass leaves the board unchanged and adds that board to the history.
            return GoState(
                board_size=self.board_size,
                komi=self.komi,
                max_moves=self.max_moves,
                board=self.board,
                to_play=-self.to_play,
                consecutive_passes=self.consecutive_passes + 1,
                move_number=self.move_number + 1,
                board_history=(self.board,) + self.board_history[:7],
                seen_hashes=set(self.seen_hashes),
                position_hash=self.position_hash,
                superko_rule=self.superko_rule,
            )

        if action < 0 or action > self.pass_action or self.is_terminal():
            raise ValueError(f"Illegal move: {action}")

        row = action // self.board_size
        col = action % self.board_size
        legal, captured, new_hash = self._evaluate_stone_move(row, col)
        if not legal:
            raise ValueError(f"Illegal move: {action}")

        new_board = self.board.copy()
        new_board[row, col] = self.to_play
        if captured:
            analysis = self._get_analysis()
            for gid in captured:
                for sr, sc in analysis.stones[gid]:
                    new_board[sr, sc] = EMPTY

        new_seen = set(self.seen_hashes)
        new_seen.add(_situation_key(new_hash, -self.to_play, self.superko_rule))

        return GoState(
            board_size=self.board_size,
            komi=self.komi,
            max_moves=self.max_moves,
            board=new_board,
            to_play=-self.to_play,
            consecutive_passes=0,
            move_number=self.move_number + 1,
            board_history=(new_board,) + self.board_history[:7],
            seen_hashes=new_seen,
            position_hash=new_hash,
            superko_rule=self.superko_rule,
        )

    # ---------------------------------------------------------------------
    # Terminal and scoring
    # ---------------------------------------------------------------------
    def is_terminal(self) -> bool:
        """A game ends after two consecutive passes or at its move limit."""
        return self.consecutive_passes >= 2 or self.move_number >= self.max_moves

    def _territory_counts(self) -> tuple[int, int]:
        """Count empty regions bordered exclusively by each color. A mixed boundary gives neither side territory."""
        visited = np.zeros_like(self.board, dtype=bool)
        black_territory = 0
        white_territory = 0

        for row in range(self.board_size):
            for col in range(self.board_size):
                if visited[row, col] or self.board[row, col] != EMPTY:
                    continue

                region: list[tuple[int, int]] = []
                bordering_colors: set[int] = set()
                stack = [(row, col)]
                visited[row, col] = True

                while stack:
                    r, c = stack.pop()
                    region.append((r, c))
                    for nr, nc in self.neighbors(r, c):
                        value = int(self.board[nr, nc])
                        if value == EMPTY and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                        elif value != EMPTY:
                            bordering_colors.add(value)

                if bordering_colors == {BLACK}:
                    black_territory += len(region)
                elif bordering_colors == {WHITE}:
                    white_territory += len(region)

        return black_territory, white_territory

    def ownership_map(self) -> np.ndarray:
        """Return ownership in Black's perspective (+1 black, -1 white, 0 neutral), consistent with area_score()."""
        ownership = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        ownership[self.board == BLACK] = 1
        ownership[self.board == WHITE] = -1

        visited = np.zeros_like(self.board, dtype=bool)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if visited[row, col] or self.board[row, col] != EMPTY:
                    continue

                region: list[tuple[int, int]] = []
                bordering_colors: set[int] = set()
                stack = [(row, col)]
                visited[row, col] = True

                while stack:
                    r, c = stack.pop()
                    region.append((r, c))
                    for nr, nc in self.neighbors(r, c):
                        value = int(self.board[nr, nc])
                        if value == EMPTY and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                        elif value != EMPTY:
                            bordering_colors.add(value)

                if bordering_colors == {BLACK}:
                    for r, c in region:
                        ownership[r, c] = 1
                elif bordering_colors == {WHITE}:
                    for r, c in region:
                        ownership[r, c] = -1

        return ownership

    def area_score(self) -> tuple[float, float]:
        """Score stones and surrounded empty points for each color, adding komi to White."""
        black_stones = int(np.sum(self.board == BLACK))
        white_stones = int(np.sum(self.board == WHITE))
        black_territory, white_territory = self._territory_counts()

        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory + self.komi)
        return black_score, white_score

    def winner_color(self) -> int:
        """Historical reference assigns tied scores to White."""
        black_score, white_score = self.area_score()
        return BLACK if black_score > white_score else WHITE

    def outcome_for_player(self, player: int) -> float:
        """Return +1 or -1 from the specified player's perspective."""
        winner = self.winner_color()
        return 1.0 if winner == player else -1.0

    # ---------------------------------------------------------------------
    # Encoding for the neural network
    # ---------------------------------------------------------------------
    def encode(self, history_length: int = 8) -> np.ndarray:
        """Encode eight board histories as pairs of current-player and opponent planes, followed by the
        side-to-play plane.
        """
        # Compare the entire history once per color to avoid a Python loop on the inference path.
        size = self.board_size
        planes = np.zeros((2 * history_length + 1, size, size), dtype=np.float32)
        history = self.board_history[:history_length]
        if history:
            boards = np.stack(history, axis=0)
            depth = len(history)
            planes[0 : 2 * depth : 2] = boards == self.to_play
            planes[1 : 2 * depth : 2] = boards == -self.to_play
        if self.to_play == BLACK:
            planes[-1] = 1.0
        return planes

    # ---------------------------------------------------------------------
    # Debugging helpers
    # ---------------------------------------------------------------------
    def to_ascii(self) -> str:
        """Render the board as text."""
        symbols = {BLACK: "X", WHITE: "O", EMPTY: "."}
        rows = []
        for r in range(self.board_size):
            rows.append(" ".join(symbols[int(v)] for v in self.board[r]))
        who = "BLACK" if self.to_play == BLACK else "WHITE"
        return f"to_play={who}, move={self.move_number}\n" + "\n".join(rows)

    def __str__(self) -> str:
        return self.to_ascii()

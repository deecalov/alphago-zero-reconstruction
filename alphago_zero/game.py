from __future__ import annotations

"""Go rules, immutable game transitions, scoring and network input planes."""

from dataclasses import dataclass, field
from itertools import compress
import math
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


def _validate_game_params(board_size: int, komi: float, superko_rule: str) -> None:
    """Validate public constructors. Internal state transitions can rely on the invariants already established."""
    if board_size < 1:
        raise ValueError(f"board_size must be >= 1, got {board_size}")
    if not math.isfinite(komi):
        raise ValueError(f"komi must be finite, got {komi}")
    if superko_rule not in ("positional", "situational"):
        raise ValueError(
            "superko_rule must be 'positional' or 'situational', "
            f"got {superko_rule!r}"
        )


def _situation_key(position_hash: int, to_play: int, superko_rule: str) -> int:
    """Return the repetition key for positional or situational superko."""
    if superko_rule == "situational" and to_play == WHITE:
        return position_hash ^ _WHITE_TO_PLAY_KEY
    return position_hash


# ---------------------------------------------------------------------------
# Board geometry and flat Zobrist tables
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Geometry:
    """Precomputed geometry for a square board. Point (row, col) has flat index row * N + col."""

    size: int
    cell_count: int
    ones: int  # One byte with value 0x01 per cell.
    left_mask: int  # Cells with a neighbor to the left.
    right_mask: int  # Cells with a neighbor to the right.
    row_shift: int  # Bit offset between adjacent rows.
    neighbors: tuple[tuple[int, ...], ...]  # Flat neighbor indices for each cell.
    indices: range  # Flat cell indices for itertools.compress.
    zobrist_black: list[int]  # Zobrist keys indexed by flat cell number.
    zobrist_white: list[int]


_GEOMETRY_CACHE: dict[int, _Geometry] = {}


def _geometry(board_size: int) -> _Geometry:
    """Cache board geometry by size."""
    geometry = _GEOMETRY_CACHE.get(board_size)
    if geometry is not None:
        return geometry

    n = board_size
    cell_count = n * n
    ones = int.from_bytes(b"\x01" * cell_count, "little")
    left_mask = int.from_bytes(bytes(1 if i % n != 0 else 0 for i in range(cell_count)), "little")
    right_mask = int.from_bytes(bytes(1 if i % n != n - 1 else 0 for i in range(cell_count)), "little")

    neighbors: list[tuple[int, ...]] = []
    for row in range(n):
        for col in range(n):
            around: list[int] = []
            if row > 0:
                around.append((row - 1) * n + col)
            if row + 1 < n:
                around.append((row + 1) * n + col)
            if col > 0:
                around.append(row * n + col - 1)
            if col + 1 < n:
                around.append(row * n + col + 1)
            neighbors.append(tuple(around))

    table = _zobrist_table(n)
    geometry = _Geometry(
        size=n,
        cell_count=cell_count,
        ones=ones,
        left_mask=left_mask,
        right_mask=right_mask,
        row_shift=8 * n,
        neighbors=tuple(neighbors),
        indices=range(cell_count),
        zobrist_black=[key for row_keys in table[0] for key in row_keys],
        zobrist_white=[key for row_keys in table[1] for key in row_keys],
    )
    _GEOMETRY_CACHE[board_size] = geometry
    return geometry


# ---------------------------------------------------------------------------
# Analyze moves with byte-wise operations.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _MoveAnalysis:
    """Candidate move mask and capture data for the current player, before checking superko."""

    candidates: bytes
    cells: list[int]
    captures: dict[int, tuple[int, list[int]]] | None


def _find_captures(
    raw: bytes,
    weak_stones: bytes,
    geometry: _Geometry,
    opp_byte: int,
    opp_zobrist: list[int],
) -> dict[int, tuple[int, list[int]]] | None:
    """Find opponent groups in atari. For each capturing point, record the captured stones and the change to
    the position hash.
    """
    neighbors = geometry.neighbors
    captures: dict[int, tuple[int, list[int]]] | None = None
    # Use a traversal number per stone. A previous traversal can identify a group already known to have
    # several liberties.
    marks = [0] * geometry.cell_count
    token = 0

    for start in compress(geometry.indices, weak_stones):
        if marks[start]:
            continue
        token += 1
        marks[start] = token
        group = [start]
        liberty = -1
        safe = False
        position = 0
        while position < len(group) and not safe:
            stone = group[position]
            position += 1
            for around in neighbors[stone]:
                value = raw[around]
                if value == 0:
                    if liberty < 0:
                        liberty = around
                    elif liberty != around:
                        safe = True  # A second liberty rules out atari.
                        break
                elif value == opp_byte:
                    mark = marks[around]
                    if mark == token:
                        continue
                    if mark or not weak_stones[around]:
                        safe = True  # The group is already known to have at least two liberties.
                        break
                    marks[around] = token
                    group.append(around)

        if safe or liberty < 0:  # A group with no liberties cannot occur on a legal board.
            continue
        xor = 0
        for stone in group:
            xor ^= opp_zobrist[stone]
        if captures is None:
            captures = {}
        previous = captures.get(liberty)
        if previous is not None:  # Several groups can share a single last liberty.
            xor ^= previous[0]
            group = previous[1] + group
        captures[liberty] = (xor, group)
    return captures


def _has_spare_liberty(
    point: int,
    raw: bytes,
    neighbors: tuple[tuple[int, ...], ...],
    own_byte: int,
    two_empty_neighbors: bytes,
) -> bool:
    """Check whether an adjacent friendly group has another liberty when the proposed point has no empty
    neighbors.
    """
    seen: set[int] = set()
    for first in neighbors[point]:
        if raw[first] != own_byte or first in seen:
            continue
        if two_empty_neighbors[first]:
            return True
        seen.add(first)
        stack = [first]
        while stack:
            stone = stack.pop()
            for around in neighbors[stone]:
                value = raw[around]
                if value == 0:
                    if around != point:
                        return True
                elif value == own_byte and around not in seen:
                    if two_empty_neighbors[around]:
                        return True
                    seen.add(around)
                    stack.append(around)
    return False


def _analyze_moves(board: np.ndarray, board_size: int, own_color: int) -> _MoveAnalysis:
    """Find candidate moves and captures for own_color. An otherwise surrounded move can survive by
    capturing or connecting to a group with another liberty.
    """
    geometry = _geometry(board_size)
    cell_count = geometry.cell_count
    ones = geometry.ones

    raw = board.astype(np.int8, copy=False).tobytes()
    lanes = int.from_bytes(raw, "little")
    occupied = lanes & ones
    empty = occupied ^ ones
    white = (lanes >> 1) & ones  # Only 0xFF has its second bit set among the board encodings.
    black = occupied ^ white
    if own_color == BLACK:
        opp_stones, own_byte, opp_byte, opp_zobrist = white, 1, 255, geometry.zobrist_white
    else:
        opp_stones, own_byte, opp_byte, opp_zobrist = black, 255, 1, geometry.zobrist_black

    row_shift = geometry.row_shift
    empty_neighbors = (
        ((empty << 8) & geometry.left_mask)  # Empty neighbor to the left.
        + ((empty >> 8) & geometry.right_mask)  # Empty neighbor to the right.
        + ((empty << row_shift) & ones)  # Empty neighbor above.
        + (empty >> row_shift)  # Empty neighbor below.
    )
    at_least_one = (empty_neighbors | (empty_neighbors >> 1) | (empty_neighbors >> 2)) & ones
    at_least_two = ((empty_neighbors >> 1) | (empty_neighbors >> 2)) & ones
    enclosed = empty & ~at_least_one
    weak_opp = opp_stones & ~at_least_two

    captures = None
    if weak_opp:
        weak_bytes = weak_opp.to_bytes(cell_count, "little")
        captures = _find_captures(raw, weak_bytes, geometry, opp_byte, opp_zobrist)

    candidates = empty
    if enclosed:
        two_empty: bytes | None = None
        for point in compress(geometry.indices, enclosed.to_bytes(cell_count, "little")):
            if captures is not None and point in captures:
                continue  # A capture creates a liberty.
            if two_empty is None:
                two_empty = at_least_two.to_bytes(cell_count, "little")
            if not _has_spare_liberty(point, raw, geometry.neighbors, own_byte, two_empty):
                candidates ^= 1 << (8 * point)  # Remove the suicide move from the candidates.

    candidate_bytes = candidates.to_bytes(cell_count, "little")
    return _MoveAnalysis(
        candidates=candidate_bytes,
        cells=list(compress(geometry.indices, candidate_bytes)),
        captures=captures,
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

    # Populate the move-analysis cache on first use. Each successor has its own cache.
    _analysis: _MoveAnalysis | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def new_game(
        cls,
        board_size: int = 19,
        komi: float = 7.5,
        max_moves: int | None = None,
        superko_rule: str = "positional",
    ) -> "GoState":
        """Start an empty game."""
        _validate_game_params(board_size, komi, superko_rule)
        if max_moves is None:
            max_moves = board_size * board_size * 2
        elif max_moves < 1:
            raise ValueError(f"max_moves must be >= 1, got {max_moves}")

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
        """Construct a position for analysis with consistent history, repetition keys and hash."""
        raw_board = np.asarray(board)
        if raw_board.ndim != 2 or raw_board.shape[0] != raw_board.shape[1]:
            raise ValueError(f"expected a square [N, N] board, got {raw_board.shape}")
        board_size = int(raw_board.shape[0])
        _validate_game_params(board_size, komi, superko_rule)
        if to_play not in (BLACK, WHITE):
            raise ValueError(f"to_play must be BLACK or WHITE, got {to_play}")
        # Validate before conversion to int8 so invalid values cannot round or wrap into legal colors.
        if not bool(np.isin(raw_board, (EMPTY, BLACK, WHITE)).all()):
            raise ValueError("board entries must be -1, 0 or +1")
        board = raw_board.astype(np.int8, copy=True)
        if max_moves is None:
            max_moves = board_size * board_size * 2
        elif max_moves < 1:
            raise ValueError(f"max_moves must be >= 1, got {max_moves}")

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
    # Move analysis and stone placement
    # ---------------------------------------------------------------------
    def _get_analysis(self) -> _MoveAnalysis:
        """Compute legal-move candidates for this board on first use."""
        analysis = self._analysis
        if analysis is None:
            analysis = _analyze_moves(self.board, self.board_size, self.to_play)
            self._analysis = analysis
        return analysis

    def _evaluate_stone_move(self, point: int) -> tuple[bool, list[int] | None, int]:
        """Return legality, captured stones and the resulting position hash for a placement at the flat index."""
        analysis = self._get_analysis()
        if not analysis.candidates[point]:
            return False, None, 0  # Occupied point or suicide.

        own_color = self.to_play
        geometry = _geometry(self.board_size)
        zobrist = geometry.zobrist_black if own_color == BLACK else geometry.zobrist_white
        new_hash = self.position_hash ^ zobrist[point]
        captured: list[int] | None = None
        captures = analysis.captures
        if captures is not None:
            capture = captures.get(point)
            if capture is not None:
                new_hash ^= capture[0]
                captured = capture[1]

        if _situation_key(new_hash, -own_color, self.superko_rule) in self.seen_hashes:
            return False, None, 0  # Repeated position or situation.

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

        legal, _captured, _new_hash = self._evaluate_stone_move(action)
        return legal

    def legal_actions(self) -> list[int]:
        """Return legal placements and pass. Reuse the board analysis and check the resulting keys against
        superko history.
        """
        if self.is_terminal():
            return []

        analysis = self._get_analysis()
        own_color = self.to_play
        geometry = _geometry(self.board_size)
        zobrist = geometry.zobrist_black if own_color == BLACK else geometry.zobrist_white
        # Mix in the side-to-play key when the rules use situational superko.
        base = self.position_hash
        if own_color == BLACK and self.superko_rule == "situational":
            base ^= _WHITE_TO_PLAY_KEY
        seen = self.seen_hashes

        captures = analysis.captures
        if captures is None:
            actions = [point for point in analysis.cells if (base ^ zobrist[point]) not in seen]
        else:
            # For captures, also remove the captured stones from the hash.
            actions = [
                point
                for point in analysis.cells
                if point not in captures and (base ^ zobrist[point]) not in seen
            ]
            for point, (captured_hash, _stones) in captures.items():
                if (base ^ zobrist[point] ^ captured_hash) not in seen:
                    actions.append(point)
            actions.sort()
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

        board_size = self.board_size
        legal, captured, new_hash = self._evaluate_stone_move(action)
        if not legal:
            raise ValueError(f"Illegal move: {action}")

        new_board = self.board.copy()
        new_board[action // board_size, action % board_size] = self.to_play
        if captured:
            for stone in captured:
                new_board[stone // board_size, stone % board_size] = EMPTY

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

    def winner_color(self) -> int | None:
        """Return BLACK, WHITE or None for a draw. Integer komi permits equal scores."""
        black_score, white_score = self.area_score()
        if black_score > white_score:
            return BLACK
        if white_score > black_score:
            return WHITE
        return None

    def outcome_for_player(self, player: int) -> float:
        """Return +1, 0 or -1 from the named player's perspective."""
        winner = self.winner_color()
        if winner is None:
            return 0.0
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
            # Stack equal-shaped history arrays into a contiguous (depth, N, N) array.
            boards = np.array(history)
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

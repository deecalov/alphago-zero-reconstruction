from __future__ import annotations

"""Frozen simulation-based Go implementation for differential testing."""

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np


BLACK = 1
WHITE = -1
EMPTY = 0


@dataclass(slots=True)
class ReferenceGoState:
    """Slow reference position for independent rule comparisons."""

    board_size: int
    komi: float
    max_moves: int
    board: np.ndarray
    to_play: int = BLACK
    consecutive_passes: int = 0
    move_number: int = 0

    # Store the current board at index zero and earlier boards in descending recency.
    board_history: list[np.ndarray] = field(default_factory=list)

    # Positional superko rejects a placement that recreates an earlier board.
    seen_hashes: set[bytes] = field(default_factory=set)

    @classmethod
    def new_game(
        cls,
        board_size: int = 19,
        komi: float = 7.5,
        max_moves: int | None = None,
    ) -> "ReferenceGoState":
        """Start an empty game."""
        if max_moves is None:
            max_moves = board_size * board_size * 2

        board = np.zeros((board_size, board_size), dtype=np.int8)
        history = [board.copy()]
        seen_hashes = {board.tobytes()}
        return cls(
            board_size=board_size,
            komi=komi,
            max_moves=max_moves,
            board=board,
            to_play=BLACK,
            consecutive_passes=0,
            move_number=0,
            board_history=history,
            seen_hashes=seen_hashes,
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
    ) -> "ReferenceGoState":
        """Build an arbitrary position for comparison with GoState.from_board."""
        board = np.asarray(board, dtype=np.int8).copy()
        board_size = int(board.shape[0])
        if max_moves is None:
            max_moves = board_size * board_size * 2
        return cls(
            board_size=board_size,
            komi=komi,
            max_moves=max_moves,
            board=board,
            to_play=to_play,
            consecutive_passes=consecutive_passes,
            move_number=move_number,
            board_history=[board.copy()],
            seen_hashes={board.tobytes()},
        )

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size + 1

    @property
    def pass_action(self) -> int:
        return self.board_size * self.board_size

    def copy(self) -> "ReferenceGoState":
        return ReferenceGoState(
            board_size=self.board_size,
            komi=self.komi,
            max_moves=self.max_moves,
            board=self.board.copy(),
            to_play=self.to_play,
            consecutive_passes=self.consecutive_passes,
            move_number=self.move_number,
            board_history=[b.copy() for b in self.board_history],
            seen_hashes=set(self.seen_hashes),
        )

    # ---------------------------------------------------------------------
    # Action and coordinate conversion
    # ---------------------------------------------------------------------
    def action_to_coord(self, action: int) -> tuple[int, int] | None:
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

    def _collect_group_and_liberties(
        self,
        board: np.ndarray,
        row: int,
        col: int,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Collect a connected group and its liberties."""
        color = int(board[row, col])
        if color == EMPTY:
            return set(), set()

        group: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))

            for nr, nc in self.neighbors(r, c):
                value = int(board[nr, nc])
                if value == EMPTY:
                    liberties.add((nr, nc))
                elif value == color and (nr, nc) not in group:
                    stack.append((nr, nc))

        return group, liberties

    def _remove_group(self, board: np.ndarray, group: Iterable[tuple[int, int]]) -> None:
        for r, c in group:
            board[r, c] = EMPTY

    def _board_after_stone_move(self, action: int) -> np.ndarray | None:
        """Simulate a placement and return its board, or None if illegal."""
        coord = self.action_to_coord(action)
        if coord is None:
            raise ValueError("pass must not reach _board_after_stone_move")

        row, col = coord
        if self.board[row, col] != EMPTY:
            return None

        board = self.board.copy()
        board[row, col] = self.to_play

        # Remove neighboring opponent groups that have lost their last liberty.
        opponent = -self.to_play
        processed_enemy: set[tuple[int, int]] = set()
        for nr, nc in self.neighbors(row, col):
            if board[nr, nc] != opponent or (nr, nc) in processed_enemy:
                continue
            group, liberties = self._collect_group_and_liberties(board, nr, nc)
            processed_enemy |= group
            if not liberties:
                self._remove_group(board, group)

        # Check the new friendly group for suicide after resolving captures.
        _own_group, own_liberties = self._collect_group_and_liberties(board, row, col)
        if not own_liberties:
            return None

        return board

    # ---------------------------------------------------------------------
    # Legal moves
    # ---------------------------------------------------------------------
    def is_legal(self, action: int) -> bool:
        if action < 0 or action > self.pass_action:
            return False

        if self.is_terminal():
            return False

        if action == self.pass_action:
            return True

        new_board = self._board_after_stone_move(action)
        if new_board is None:
            return False

        # Reject repeated boards under positional superko.
        new_hash = new_board.tobytes()
        if new_hash in self.seen_hashes:
            return False

        return True

    def legal_actions(self) -> list[int]:
        """Return all legal board actions plus pass."""
        if self.is_terminal():
            return []

        actions: list[int] = []
        for action in range(self.pass_action):
            if self.is_legal(action):
                actions.append(action)
        actions.append(self.pass_action)
        return actions

    # ---------------------------------------------------------------------
    # Transition function
    # ---------------------------------------------------------------------
    def play(self, action: int) -> "ReferenceGoState":
        """Return the successor after an action."""
        if not self.is_legal(action):
            raise ValueError(f"Illegal move: {action}")

        if action == self.pass_action:
            new_board = self.board.copy()
            new_history = [new_board.copy()] + [b.copy() for b in self.board_history[:7]]
            return ReferenceGoState(
                board_size=self.board_size,
                komi=self.komi,
                max_moves=self.max_moves,
                board=new_board,
                to_play=-self.to_play,
                consecutive_passes=self.consecutive_passes + 1,
                move_number=self.move_number + 1,
                board_history=new_history,
                seen_hashes=set(self.seen_hashes),
            )

        stone_board = self._board_after_stone_move(action)
        assert stone_board is not None
        new_history = [stone_board.copy()] + [b.copy() for b in self.board_history[:7]]
        new_seen = set(self.seen_hashes)
        new_seen.add(stone_board.tobytes())

        return ReferenceGoState(
            board_size=self.board_size,
            komi=self.komi,
            max_moves=self.max_moves,
            board=stone_board,
            to_play=-self.to_play,
            consecutive_passes=0,
            move_number=self.move_number + 1,
            board_history=new_history,
            seen_hashes=new_seen,
        )

    # ---------------------------------------------------------------------
    # Terminal and scoring
    # ---------------------------------------------------------------------
    def is_terminal(self) -> bool:
        return self.consecutive_passes >= 2 or self.move_number >= self.max_moves

    def _territory_counts(self) -> tuple[int, int]:
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

    def area_score(self) -> tuple[float, float]:
        black_stones = int(np.sum(self.board == BLACK))
        white_stones = int(np.sum(self.board == WHITE))
        black_territory, white_territory = self._territory_counts()

        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory + self.komi)
        return black_score, white_score

    def winner_color(self) -> int | None:
        # Mirror the production tie convention: equal scores give no winner.
        black_score, white_score = self.area_score()
        if black_score > white_score:
            return BLACK
        if white_score > black_score:
            return WHITE
        return None

    def outcome_for_player(self, player: int) -> float:
        winner = self.winner_color()
        if winner is None:
            return 0.0
        return 1.0 if winner == player else -1.0

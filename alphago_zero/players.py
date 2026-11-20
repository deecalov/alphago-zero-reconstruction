from __future__ import annotations

"""Go players used for matches and fixed-strength benchmarks."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import SearchConfig
from .game import GoState
from .mcts import AlphaGoZeroMCTS, MCTSNode
from .network import PolicyValueModel


def _same_position(a: GoState, b: GoState) -> bool:
    """Check board and side to move before reusing a subtree in a game with fixed rules."""
    return a.to_play == b.to_play and bool(np.array_equal(a.board, b.board))


class Player(Protocol):
    """Minimum agent interface for choosing a move."""

    def select_action(self, state: GoState) -> int: ...


@dataclass(slots=True)
class RandomPlayer:
    """Uniform legal random player, the zero-rated Elo anchor."""

    rng: np.random.Generator

    def select_action(self, state: GoState) -> int:
        legal = state.legal_actions()
        return int(self.rng.choice(legal))


@dataclass(slots=True)
class PolicyPlayer:
    """Choose the highest-logit legal move without search. Frozen weights give a fixed benchmark."""

    model: PolicyValueModel

    def select_action(self, state: GoState) -> int:
        logits, _value = self.model.predict_single(state.encode())
        legal = state.legal_actions()
        legal_logits = logits[legal]
        return int(legal[int(np.argmax(legal_logits))])


class MCTSPlayer:
    """Run deterministic PUCT without noise. Subtree reuse is optional."""

    def __init__(
        self,
        model: PolicyValueModel,
        search_config: SearchConfig,
        rng: np.random.Generator,
        reuse_tree: bool = False,
    ) -> None:
        self.model = model
        self.search = AlphaGoZeroMCTS(model=model, config=search_config, rng=rng)
        self.reuse_tree = reuse_tree
        self._root: MCTSNode | None = None

    def select_action(self, state: GoState) -> int:
        root = self._reconcile_root(state) if self.reuse_tree else None
        output = self.search.run(
            root_state=state,
            temperature=1e-9,  # Select the visit-count maximum.
            add_root_noise=False,
            root=root,
        )
        action = int(np.argmax(output.policy))
        self._root = (
            self.search.advance_root(output.root, action) if self.reuse_tree else None
        )
        return action

    def _reconcile_root(self, state: GoState) -> MCTSNode | None:
        """Match the incoming position against the saved root or its child after the opponent's move."""
        root = self._root
        if root is None:
            return None
        if _same_position(root.state, state):
            return root
        for edge in root.edges.values():
            child = edge.child
            if child is not None and _same_position(child.state, state):
                return child
        return None


def play_match(
    black: Player,
    white: Player,
    board_size: int,
    komi: float,
    max_moves: int | None = None,
    superko_rule: str = "positional",
) -> int | None:
    """Play one game under the project rules and return the winner."""
    state = GoState.new_game(
        board_size=board_size, komi=komi, max_moves=max_moves, superko_rule=superko_rule
    )
    while not state.is_terminal():
        player = black if state.to_play == 1 else white
        action = player.select_action(state)
        state = state.play(action)
    return state.winner_color()

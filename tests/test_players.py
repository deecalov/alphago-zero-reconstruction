from __future__ import annotations

"""Player tests with particular attention to subtree reuse."""

from typing import Any

import numpy as np
import pytest

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.mcts import AlphaGoZeroMCTS, MCTSNode, SearchOutput
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.players import MCTSPlayer, play_match


@pytest.fixture
def captured_roots(monkeypatch: pytest.MonkeyPatch) -> list[MCTSNode | None]:
    """Collect the root supplied to every MCTS.run in the test."""
    roots: list[MCTSNode | None] = []
    real_run = AlphaGoZeroMCTS.run

    def spy(self: AlphaGoZeroMCTS, *args: Any, **kwargs: Any) -> SearchOutput:
        roots.append(kwargs.get("root"))
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(AlphaGoZeroMCTS, "run", spy)
    return roots


def _same(a: GoState, b: GoState) -> bool:
    return a.to_play == b.to_play and bool(np.array_equal(a.board, b.board))


def test_mcts_player_without_reuse_always_starts_fresh(
    tiny_model: AlphaGoZeroNet,
    tiny_config: AlphaGoZeroConfig,
    captured_roots: list[MCTSNode | None],
) -> None:
    player = MCTSPlayer(
        model=tiny_model, search_config=tiny_config.search, rng=np.random.default_rng(0)
    )
    state = GoState.new_game(board_size=5, komi=2.5)
    first = player.select_action(state)
    state = state.play(first).play(state.pass_action)
    player.select_action(state)
    assert captured_roots == [None, None]
    assert player._root is None


def test_mcts_player_reuse_follows_the_game(
    tiny_model: AlphaGoZeroNet,
    tiny_config: AlphaGoZeroConfig,
    captured_roots: list[MCTSNode | None],
) -> None:
    # Provide enough simulations to expand a continuation through the opponent's reply.
    search = tiny_config.search
    search.num_simulations = 200
    player = MCTSPlayer(
        model=tiny_model,
        search_config=search,
        rng=np.random.default_rng(0),
        reuse_tree=True,
    )
    state = GoState.new_game(board_size=5, komi=2.5)
    first = player.select_action(state)
    assert captured_roots == [None]

    after_first = state.play(first)
    stored = player._root
    assert stored is not None
    assert _same(stored.state, after_first)  # The root now represents the opponent's position.

    # Choose the opponent reply from an expanded, heavily visited branch.
    candidates = [
        (action, edge.child)
        for action, edge in stored.edges.items()
        if edge.child is not None
    ]
    assert candidates, "the subtree should be expanded after 200 simulations"
    opponent_action, expected_child = max(
        candidates, key=lambda item: stored.edges[item[0]].visit_count
    )
    assert expected_child is not None
    carried_visits = sum(edge.visit_count for edge in expected_child.edges.values())

    after_second = after_first.play(opponent_action)
    player.select_action(after_second)
    assert captured_roots[-1] is expected_child  # Reuse the same node object.
    total_visits = sum(edge.visit_count for edge in expected_child.edges.values())
    assert total_visits == carried_visits + 200  # Retain its accumulated visits.


def test_mcts_player_reuse_resets_on_foreign_state(
    tiny_model: AlphaGoZeroNet,
    tiny_config: AlphaGoZeroConfig,
    captured_roots: list[MCTSNode | None],
) -> None:
    player = MCTSPlayer(
        model=tiny_model,
        search_config=tiny_config.search,
        rng=np.random.default_rng(0),
        reuse_tree=True,
    )
    state = GoState.new_game(board_size=5, komi=2.5)
    player.select_action(state)
    # An unrelated new game must invalidate the retained tree.
    player.select_action(GoState.new_game(board_size=5, komi=2.5))
    assert captured_roots[-1] is None


def test_play_match_with_reuse_completes(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    rng = np.random.default_rng(3)
    black = MCTSPlayer(
        model=tiny_model, search_config=tiny_config.search, rng=rng, reuse_tree=True
    )
    white = MCTSPlayer(
        model=tiny_model, search_config=tiny_config.search, rng=rng, reuse_tree=True
    )
    winner = play_match(black, white, board_size=5, komi=2.5, max_moves=60)
    assert winner in (BLACK, WHITE, None)

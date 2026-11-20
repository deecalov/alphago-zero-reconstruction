from __future__ import annotations

"""Rule and search fidelity tests for situational superko, FPU and resignation."""

import numpy as np
import pytest

from alphago_zero.config import AlphaGoZeroConfig, SearchConfig
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.mcts import AlphaGoZeroMCTS, EdgeStats, MCTSNode, SearchOutput
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.selfplay import ResignationController, play_self_play_game


# ----------------------------------------------------------------------
# Situational superko
# ----------------------------------------------------------------------
def test_superko_rule_propagates_through_play_and_copy() -> None:
    state = GoState.new_game(board_size=5, komi=2.5, superko_rule="situational")
    assert state.superko_rule == "situational"
    after_move = state.play(12)
    assert after_move.superko_rule == "situational"
    after_pass = after_move.play(after_move.pass_action)
    assert after_pass.superko_rule == "situational"
    assert state.copy().superko_rule == "situational"


@pytest.mark.parametrize("rule", ["positional", "situational"])
def test_simple_ko_forbidden_under_both_rules(rule: str) -> None:
    """A classic single-ko position where immediate recapture is illegal."""
    board = np.zeros((5, 5), dtype=np.int8)
    for r, c in ((0, 1), (1, 0), (2, 1)):
        board[r, c] = BLACK
    for r, c in ((0, 2), (1, 1), (1, 3), (2, 2)):
        board[r, c] = WHITE

    state = GoState.from_board(board, to_play=BLACK, komi=2.5, superko_rule=rule)
    capture = state.coord_to_action(1, 2)
    assert state.is_legal(capture)
    after_capture = state.play(capture)

    # Capturing White at (1,1) rules out immediate recapture.
    assert after_capture.board[1, 1] == 0
    recapture = after_capture.coord_to_action(1, 1)
    assert not after_capture.is_legal(recapture)


def test_situational_is_strictly_weaker_and_differences_are_position_repeats() -> None:
    """Run positional and situational rules in lockstep on the same random moves."""
    total_differences = 0

    for seed in range(40):
        rng = np.random.default_rng(seed)
        pos = GoState.new_game(board_size=5, komi=2.5, max_moves=200, superko_rule="positional")
        sit = GoState.new_game(board_size=5, komi=2.5, max_moves=200, superko_rule="situational")

        # Record the initial situation and each stone placement. Pass is exempt.
        situations: list[tuple[bytes, int]] = [(pos.board.tobytes(), pos.to_play)]

        for _ in range(120):
            if pos.is_terminal():
                break
            legal_pos = set(pos.legal_actions())
            legal_sit = set(sit.legal_actions())
            assert legal_pos <= legal_sit

            for extra in sorted(legal_sit - legal_pos):
                total_differences += 1
                preview = sit.play(extra)
                board_bytes = preview.board.tobytes()
                prior_to_plays = [tp for b, tp in situations if b == board_bytes]
                # The board has occurred before
                assert prior_to_plays
                # but with a different player to move on every previous occurrence.
                assert all(tp != preview.to_play for tp in prior_to_plays)

            action = int(rng.choice(sorted(legal_pos)))
            pos = pos.play(action)
            sit = sit.play(action)
            if action != pos.pass_action:
                situations.append((pos.board.tobytes(), pos.to_play))

    assert total_differences > 0, "the rules should diverge at least once across 40 seeds"


def test_invalid_superko_rule_rejected_by_config(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.game.superko_rule = "tromp"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="superko_rule"):
        tiny_config.validate()


# ----------------------------------------------------------------------
# FPU (first-play urgency)
# ----------------------------------------------------------------------
def _make_search(fpu: float | None, model: AlphaGoZeroNet) -> AlphaGoZeroMCTS:
    config = SearchConfig(num_simulations=8, eval_batch_size=1, fpu_reduction=fpu)
    return AlphaGoZeroMCTS(model=model, config=config, rng=np.random.default_rng(0))


def _node_with_two_edges(node_value: float) -> MCTSNode:
    """Give equal priors to a visited edge with Q=+0.05 and an unvisited edge."""
    state = GoState.new_game(board_size=5, komi=2.5)
    node = MCTSNode(state=state)
    node.node_value = node_value
    node.is_expanded = True
    node.edges = {
        0: EdgeStats(prior=0.5, visit_count=1, value_sum=0.05),
        1: EdgeStats(prior=0.5, visit_count=0, value_sum=0.0),
    }
    return node


def test_fpu_none_treats_unvisited_as_zero(tiny_model: AlphaGoZeroNet) -> None:
    # With classic FPU, an unvisited edge's Q starts at zero independently of the parent value.
    search = _make_search(None, tiny_model)
    node = _node_with_two_edges(node_value=-0.9)
    assert search._select_action(node) == 1


def test_fpu_reduction_penalizes_unvisited_from_winning_parent(tiny_model: AlphaGoZeroNet) -> None:
    # A parent value of 0.9 and reduction 2.0 clip the unvisited Q to -1.
    search = _make_search(2.0, tiny_model)
    node = _node_with_two_edges(node_value=0.9)
    assert search._select_action(node) == 0


def test_fpu_reduction_encourages_unvisited_from_good_parent(tiny_model: AlphaGoZeroNet) -> None:
    # A parent value of 0.9 and reduction 0.1 give a promising unvisited Q of 0.8.
    search = _make_search(0.1, tiny_model)
    node = _node_with_two_edges(node_value=0.9)
    assert search._select_action(node) == 1


def test_search_with_fpu_conserves_visits(tiny_model: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=12, eval_batch_size=4, fpu_reduction=0.25)
    search = AlphaGoZeroMCTS(model=tiny_model, config=config, rng=np.random.default_rng(3))
    state = GoState.new_game(board_size=5, komi=2.5)
    output = search.run(root_state=state, temperature=1.0, add_root_noise=False, root=None)
    assert int(output.visit_counts.sum()) == config.num_simulations
    assert output.policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= output.root_q <= 1.0


# ----------------------------------------------------------------------
# Search Q and resignation
# ----------------------------------------------------------------------
def test_root_q_is_visit_weighted_edge_average(tiny_model: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=16, eval_batch_size=1)
    search = AlphaGoZeroMCTS(model=tiny_model, config=config, rng=np.random.default_rng(5))
    state = GoState.new_game(board_size=5, komi=2.5)
    output = search.run(root_state=state, temperature=1.0, add_root_noise=False, root=None)

    edges = output.root.edges.values()
    expected = sum(e.value_sum for e in edges) / sum(e.visit_count for e in edges)
    assert output.root_q == pytest.approx(expected, abs=1e-9)
    # Raw network value is independent of the backed-up root Q.
    assert -1.0 <= output.root_value <= 1.0


def test_selfplay_resigns_by_mcts_q_not_raw_value(
    monkeypatch: pytest.MonkeyPatch,
    tiny_config: AlphaGoZeroConfig,
    tiny_model: AlphaGoZeroNet,
) -> None:
    """Set network value to +1 and search Q to -1 to identify which triggers resignation."""
    tiny_config.search.disable_resign_prob = 0.0  # Always permit resignation.

    action_size = tiny_config.game.board_size ** 2 + 1

    def fake_run(
        self: AlphaGoZeroMCTS,
        root_state: GoState,
        temperature: float,
        add_root_noise: bool,
        root: MCTSNode | None = None,
        num_simulations: int | None = None,
    ) -> SearchOutput:
        visit_counts = np.zeros(action_size, dtype=np.float32)
        visit_counts[action_size - 1] = 1.0  # Assign all visits to pass.
        policy = visit_counts.copy()
        return SearchOutput(
            policy=policy,
            visit_counts=visit_counts,
            root=MCTSNode(state=root_state),
            root_value=1.0,  # Raw network predicts a win.
            best_child_q=-1.0,
            root_q=-1.0,  # Search predicts a loss.
        )

    monkeypatch.setattr(AlphaGoZeroMCTS, "run", fake_run)

    controller = ResignationController(
        threshold=tiny_config.search.resign_threshold,
        target_false_positive_rate=tiny_config.search.target_false_positive_rate,
        adjustment=tiny_config.search.resignation_adjustment,
        window=tiny_config.search.resignation_window,
    )
    result = play_self_play_game(
        model=tiny_model,
        config=tiny_config,
        rng=np.random.default_rng(0),
        resignation_controller=controller,
    )

    assert result.resigned
    assert result.num_moves == 1  # Resign in the initial position.
    assert result.winner == -BLACK  # Black to move resigned, so White wins.
    assert result.score_margin is None


def test_negative_fpu_rejected_by_config(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.search.fpu_reduction = -0.5
    with pytest.raises(ValueError, match="fpu_reduction"):
        tiny_config.validate()

from __future__ import annotations

"""PUCT selection, root policies, time budgets and value backup tests."""

import numpy as np
import pytest

from alphago_zero.config import SearchConfig
from alphago_zero.game import BLACK, GoState
from alphago_zero.mcts import (
    AlphaGoZeroMCTS,
    EdgeStats,
    MCTSNode,
    SearchOutput,
    leader_unreachable,
    prune_policy_target,
    resignation_child_q,
)
from alphago_zero.network import AlphaGoZeroNet


@pytest.fixture
def search_config() -> SearchConfig:
    return SearchConfig(
        num_simulations=16,
        c_puct=1.5,
        root_dirichlet_alpha=0.3,
        root_dirichlet_epsilon=0.25,
    )


def test_resignation_ignores_unvisited_q_but_requires_evidence() -> None:
    root = MCTSNode(GoState.new_game(5), edges={
        0: EdgeStats(prior=0.8, visit_count=8, value_sum=-8.0),
        1: EdgeStats(prior=0.2),
    })
    assert resignation_child_q(root, min_visits=8) == -1.0
    assert resignation_child_q(root, min_visits=9) == 1.0
    # Even a lightly explored promising alternative still prevents resignation.
    root.edges[1].visit_count = 1
    root.edges[1].value_sum = 0.5
    assert resignation_child_q(root, min_visits=8) == 0.5


def test_search_reports_observed_q_with_unvisited_alternatives(
    monkeypatch: pytest.MonkeyPatch, tiny_model: AlphaGoZeroNet,
) -> None:
    monkeypatch.setattr(tiny_model, "predict_single", lambda x: (np.zeros(26, np.float32), 1.0))
    config = SearchConfig(num_simulations=4, eval_batch_size=1, resign_min_visits=1)
    output = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0)).run(
        GoState.new_game(5), temperature=1.0, add_root_noise=False,
    )
    assert any(edge.visit_count == 0 for edge in output.root.edges.values())
    assert output.root_q == -1.0
    assert output.best_child_q == -1.0


def test_policy_and_visit_distribution_sum_to_one(
    tiny_model: AlphaGoZeroNet, search_config: SearchConfig, rng: np.random.Generator
) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    mcts = AlphaGoZeroMCTS(tiny_model, search_config, rng)
    out = mcts.run(state, temperature=1.0, add_root_noise=True)

    assert out.policy.shape == (state.action_size,)
    assert out.policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert out.visit_distribution.sum() == pytest.approx(1.0, abs=1e-5)


def test_terminal_root_returns_zero_policy(
    tiny_model: AlphaGoZeroNet, search_config: SearchConfig, rng: np.random.Generator
) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    state = state.play(state.pass_action).play(state.pass_action)  # Two passes end the game.
    assert state.is_terminal()

    mcts = AlphaGoZeroMCTS(tiny_model, search_config, rng)
    out = mcts.run(state, temperature=1.0, add_root_noise=False)
    assert np.all(out.policy == 0.0)


def test_priors_only_for_legal_actions(
    tiny_model: AlphaGoZeroNet, search_config: SearchConfig, rng: np.random.Generator
) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    state = state.play(state.coord_to_action(2, 2))  # Occupy the center of the tiny model's 5x5 board.

    mcts = AlphaGoZeroMCTS(tiny_model, search_config, rng)
    out = mcts.run(state, temperature=1.0, add_root_noise=False)

    occupied = state.coord_to_action(2, 2)
    assert occupied not in out.root.edges  # An illegal action has no edge.
    legal = set(state.legal_actions())
    assert all(action in legal for action in out.root.edges)


def test_dirichlet_noise_keeps_prior_distribution(
    tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    config = SearchConfig(num_simulations=1, root_dirichlet_alpha=0.3, root_dirichlet_epsilon=0.25)
    state = GoState.new_game(board_size=5, komi=2.5)
    mcts = AlphaGoZeroMCTS(tiny_model, config, rng)

    root = MCTSNode(state=state)
    mcts._expand(root)
    priors_before = sum(edge.prior for edge in root.edges.values())
    assert priors_before == pytest.approx(1.0, abs=1e-5)

    mcts._add_dirichlet_noise(root)
    priors_after = sum(edge.prior for edge in root.edges.values())
    # Mixing the prior with Dirichlet noise preserves unit mass.
    assert priors_after == pytest.approx(1.0, abs=1e-5)


def test_run_applies_dirichlet_noise_only_when_requested(
    identity_symmetry: None, tiny_model: AlphaGoZeroNet
) -> None:
    """Test the root noise mixture and the clean priors selected when noise is disabled."""
    config = SearchConfig(
        num_simulations=1, eval_batch_size=1, root_dirichlet_alpha=0.3, root_dirichlet_epsilon=0.25
    )
    state = GoState.new_game(board_size=5, komi=2.5)
    seed = 7

    clean_rng = np.random.default_rng(seed)
    clean = AlphaGoZeroMCTS(tiny_model, config, clean_rng).run(state, temperature=1.0, add_root_noise=False)
    noisy = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(seed)).run(
        state, temperature=1.0, add_root_noise=True
    )

    # With fixed symmetry, disabling noise leaves the RNG untouched.
    assert clean_rng.bit_generator.state == np.random.default_rng(seed).bit_generator.state
    actions = list(clean.root.edges)
    base = np.array([clean.root.edges[a].prior for a in actions])
    eta = np.random.default_rng(seed).dirichlet(np.full(len(actions), config.root_dirichlet_alpha))
    expected = (1.0 - config.root_dirichlet_epsilon) * base + config.root_dirichlet_epsilon * eta
    got = np.array([noisy.root.edges[a].prior for a in actions])
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert not np.allclose(got, base, atol=1e-4)


def test_advance_root(
    tiny_model: AlphaGoZeroNet, search_config: SearchConfig, rng: np.random.Generator
) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    mcts = AlphaGoZeroMCTS(tiny_model, search_config, rng)
    out = mcts.run(state, temperature=1.0, add_root_noise=True)

    most_visited = max(out.root.edges, key=lambda a: out.root.edges[a].visit_count)
    child = mcts.advance_root(out.root, most_visited)
    assert child is None or isinstance(child, MCTSNode)
    # Return None for an action with no corresponding edge.
    assert mcts.advance_root(out.root, 99999) is None


def test_reuse_tree_runs_without_error(
    tiny_model: AlphaGoZeroNet, search_config: SearchConfig, rng: np.random.Generator
) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    mcts = AlphaGoZeroMCTS(tiny_model, search_config, rng)
    out = mcts.run(state, temperature=1.0, add_root_noise=True)

    action = int(np.argmax(out.visit_counts))
    reused = mcts.advance_root(out.root, action)
    next_state = state.play(action)
    # Search can continue from a retained subtree.
    out2 = mcts.run(next_state, temperature=1.0, add_root_noise=True, root=reused)
    assert out2.policy.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Regression tests for the sign of value backup
# ---------------------------------------------------------------------------
def test_backup_sign_is_negated_for_parent(
    tiny_model: AlphaGoZeroNet, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Return uniform logits and value +0.8 for the player to move at every leaf.
    action_size = 5 * 5 + 1

    def fake_predict_single(_planes: np.ndarray) -> tuple[np.ndarray, float]:
        return np.zeros(action_size, dtype=np.float32), 0.8

    monkeypatch.setattr(tiny_model, "predict_single", fake_predict_single)

    config = SearchConfig(num_simulations=1, c_puct=1.5, root_dirichlet_epsilon=0.0)
    state = GoState.new_game(board_size=5, komi=2.5)
    mcts = AlphaGoZeroMCTS(tiny_model, config, rng)
    out = mcts.run(state, temperature=1.0, add_root_noise=False)

    visited = [edge for edge in out.root.edges.values() if edge.visit_count > 0]
    assert len(visited) == 1
    assert visited[0].q_value == pytest.approx(-0.8, abs=1e-6)


def test_mcts_prefers_immediately_winning_pass(
    tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    # Black leads on points after one pass and can finish the game by passing.
    board = np.zeros((5, 5), dtype=np.int8)
    board[:, 0:3] = BLACK  # Black controls the left three columns and leads on area.
    state = GoState.from_board(board, to_play=BLACK, komi=2.5, consecutive_passes=1)
    assert state.winner_color() == BLACK

    config = SearchConfig(num_simulations=100, c_puct=1.5, root_dirichlet_epsilon=0.0)
    mcts = AlphaGoZeroMCTS(tiny_model, config, rng)
    out = mcts.run(state, temperature=1.0, add_root_noise=False)

    assert int(np.argmax(out.visit_counts)) == state.pass_action


# ----------------------------------------------------------------------
# Time budget and early stopping
# ----------------------------------------------------------------------
def test_leader_unreachable_math() -> None:
    root = MCTSNode(
        GoState.new_game(5),
        edges={
            0: EdgeStats(prior=0.6, visit_count=50, value_sum=0.0),
            1: EdgeStats(prior=0.3, visit_count=10, value_sum=0.0),
            2: EdgeStats(prior=0.1, visit_count=5, value_sum=0.0),
        },
    )
    assert leader_unreachable(root, 39)  # 10 + 39 cannot catch the leader at 50.
    assert not leader_unreachable(root, 40)  # A tie could still change argmax.
    single = MCTSNode(
        GoState.new_game(5), edges={0: EdgeStats(prior=1.0, visit_count=10, value_sum=0.0)}
    )
    assert leader_unreachable(single, 9)  # All remaining actions have zero visits.
    assert not leader_unreachable(single, 10)
    assert not leader_unreachable(MCTSNode(GoState.new_game(5)), 100)  # No edges.


class _JumpClock:
    """Advance a fake clock by a fixed step on each read."""

    def __init__(self, step: float) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        self._now += self._step
        return self._now


def test_time_budget_stops_between_batches(tiny_model: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=100, eval_batch_size=4)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0))
    out = mcts.run(
        GoState.new_game(board_size=5, komi=2.5),
        temperature=1.0,
        add_root_noise=False,
        time_budget_seconds=5.0,
        clock=_JumpClock(10.0),  # The first post-batch time check exceeds the budget.
    )
    # The first batch always runs. Check the budget between batches.
    assert int(out.visit_counts.sum()) == 4


def test_time_budget_zero_still_runs_one_batch(tiny_model: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=100, eval_batch_size=4)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0))
    out = mcts.run(
        GoState.new_game(board_size=5, komi=2.5),
        temperature=1.0,
        add_root_noise=False,
        time_budget_seconds=0.0,
        clock=_JumpClock(1.0),
    )
    assert int(out.visit_counts.sum()) == 4


def test_without_budget_all_simulations_run(tiny_model: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=20, eval_batch_size=4)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0))
    out = mcts.run(
        GoState.new_game(board_size=5, komi=2.5),
        temperature=1.0,
        add_root_noise=False,
        clock=_JumpClock(1000.0),  # A clock alone has no effect without a deadline.
    )
    assert int(out.visit_counts.sum()) == 20


def test_early_exit_when_leader_unreachable(tiny_model: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=64, eval_batch_size=4)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0))
    state = GoState.new_game(board_size=5, komi=2.5)
    first = mcts.run(state, temperature=1.0, add_root_noise=False, num_simulations=8)
    # Give the leading edge an insurmountable lead to test early exit.
    leader = max(first.root.edges, key=lambda a: first.root.edges[a].visit_count)
    first.root.edges[leader].visit_count += 10_000
    out = mcts.run(
        state,
        temperature=1.0,
        add_root_noise=False,
        root=first.root,
        num_simulations=64,
        early_exit_unreachable=True,
    )
    assert int(out.visit_counts.sum()) == 8 + 10_000 + 4


def test_without_early_exit_full_budget_runs(tiny_model: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=64, eval_batch_size=4)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0))
    state = GoState.new_game(board_size=5, komi=2.5)
    first = mcts.run(state, temperature=1.0, add_root_noise=False, num_simulations=8)
    leader = max(first.root.edges, key=lambda a: first.root.edges[a].visit_count)
    first.root.edges[leader].visit_count += 10_000
    out = mcts.run(
        state,
        temperature=1.0,
        add_root_noise=False,
        root=first.root,
        num_simulations=64,
    )
    assert int(out.visit_counts.sum()) == 8 + 10_000 + 64


# ----------------------------------------------------------------------
# Policy target pruning and forced root playouts
# ----------------------------------------------------------------------


def test_prune_policy_target_zeroes_tail_and_renormalizes() -> None:
    target = np.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)
    pruned = prune_policy_target(target, 0.5)  # A cutoff of 0.25 retains entries 0.5 and 0.3.
    assert pruned.tolist() == pytest.approx([0.625, 0.375, 0.0, 0.0, 0.0])
    assert float(pruned.sum()) == pytest.approx(1.0)


def test_prune_policy_target_keeps_peak_and_its_ties() -> None:
    target = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    pruned = prune_policy_target(target, 0.999)  # Both tied peaks survive a cutoff just below the maximum.
    assert pruned[0] > 0.0 and pruned[1] > 0.0 and pruned[2] == 0.0


def test_prune_policy_target_zero_input_returned_unchanged() -> None:
    target = np.zeros(4, dtype=np.float32)
    assert prune_policy_target(target, 0.5) is target


def _output(visit_counts: list[float], **kwargs: object) -> SearchOutput:
    return SearchOutput(
        policy=np.zeros(26, dtype=np.float32),
        visit_counts=np.array(visit_counts, dtype=np.float32),
        root=MCTSNode(GoState.new_game(5)),
        root_value=0.0,
        best_child_q=0.0,
        root_q=0.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_policy_target_is_pruned_only_when_fraction_is_set() -> None:
    counts = [50.0, 30.0, 10.0, 5.0, 5.0] + [0.0] * 21
    plain = _output(counts)
    assert np.array_equal(plain.policy_target, plain.visit_distribution)

    pruned = _output(counts, target_prune_fraction=0.5)
    target = pruned.policy_target
    assert target[2] == 0.0 and target[3] == 0.0  # Entries 0.10 and 0.05 are below 0.25.
    assert float(target.sum()) == pytest.approx(1.0)
    # Pruning must preserve visits and the policy used for move selection.
    assert np.array_equal(pruned.visit_counts, plain.visit_counts)


def test_policy_target_pruning_applies_to_improved_policy() -> None:
    improved = np.array([0.6, 0.3, 0.1] + [0.0] * 23, dtype=np.float32)
    out = _output([1.0] + [0.0] * 25, improved_policy=improved, target_prune_fraction=0.5)
    assert out.policy_target[0] == pytest.approx(0.6 / 0.9)
    assert out.policy_target[2] == 0.0  # 0.1 < 0.5 * 0.6


def _forced_search(tiny_model: AlphaGoZeroNet, fraction: float) -> AlphaGoZeroMCTS:
    config = SearchConfig(num_simulations=48, eval_batch_size=1, forced_playout_fraction=fraction)
    return AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0))


def test_forced_root_action_picks_the_owing_child_with_the_largest_prior(
    tiny_model: AlphaGoZeroNet,
) -> None:
    search = _forced_search(tiny_model, 0.5)
    root = MCTSNode(
        GoState.new_game(5),
        edges={
            0: EdgeStats(prior=0.5, visit_count=10),
            1: EdgeStats(prior=0.3, visit_count=0),  # The quota is 0.5 * 0.3 * 16 = 2.4 visits.
            2: EdgeStats(prior=0.2, visit_count=6),  # The 1.6-visit quota has already been met.
        },
    )
    assert search._forced_root_action(root) == 1


def test_forced_root_action_none_at_search_start_and_when_paid_off(
    tiny_model: AlphaGoZeroNet,
) -> None:
    search = _forced_search(tiny_model, 0.5)
    fresh = MCTSNode(
        GoState.new_game(5),
        edges={0: EdgeStats(prior=0.6), 1: EdgeStats(prior=0.4)},
    )
    assert search._forced_root_action(fresh) is None  # At zero visits the first descent uses ordinary PUCT.

    paid = MCTSNode(
        GoState.new_game(5),
        edges={
            0: EdgeStats(prior=0.6, visit_count=60),
            1: EdgeStats(prior=0.4, visit_count=40),
        },
    )
    assert search._forced_root_action(paid) is None


def test_forced_playouts_spread_root_visits_against_greedy_selection(
    tiny_model: AlphaGoZeroNet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Greedy selection without forced playouts concentrates visits on one move."""
    logits = np.zeros(26, dtype=np.float32)
    logits[0] = 2.0  # The favorite has prior about 0.23 and each other move about 0.031.
    monkeypatch.setattr(tiny_model, "predict_single", lambda x: (logits.copy(), 0.0))
    state = GoState.new_game(board_size=5, komi=2.5)

    greedy = _forced_search(tiny_model, 0.0)
    monkeypatch.setattr(greedy, "_select_action", lambda node: next(iter(node.edges)))
    control = greedy.run(state, temperature=1.0, add_root_noise=False)
    assert control.visit_counts[0] == 48.0
    assert float(control.visit_counts[1:].sum()) == 0.0

    forced = _forced_search(tiny_model, 0.5)
    monkeypatch.setattr(forced, "_select_action", lambda node: next(iter(node.edges)))
    out = forced.run(state, temperature=1.0, add_root_noise=False)
    legal = state.legal_actions()
    assert out.visit_counts[0] < 48.0  # Some visits went to actions below quota.
    assert all(out.visit_counts[a] >= 1.0 for a in legal)  # Every legal action was explored.


def test_pruning_reaches_policy_target_through_search(
    tiny_model: AlphaGoZeroNet, monkeypatch: pytest.MonkeyPatch
) -> None:
    logits = np.zeros(26, dtype=np.float32)
    logits[0] = 2.0
    monkeypatch.setattr(tiny_model, "predict_single", lambda x: (logits.copy(), 0.0))
    state = GoState.new_game(board_size=5, komi=2.5)

    plain_config = SearchConfig(num_simulations=32, eval_batch_size=1)
    plain = AlphaGoZeroMCTS(tiny_model, plain_config, np.random.default_rng(0)).run(
        state, temperature=1.0, add_root_noise=False
    )
    assert np.array_equal(plain.policy_target, plain.visit_distribution)

    prune_config = SearchConfig(
        num_simulations=32, eval_batch_size=1, policy_target_prune_fraction=0.5
    )
    pruned = AlphaGoZeroMCTS(tiny_model, prune_config, np.random.default_rng(0)).run(
        state, temperature=1.0, add_root_noise=False
    )
    expected = prune_policy_target(pruned.visit_distribution, 0.5)
    assert np.array_equal(pruned.policy_target, expected)
    assert float(pruned.policy_target.sum()) == pytest.approx(1.0)

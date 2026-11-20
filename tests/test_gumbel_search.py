"""Tree-level Gumbel search compared with an independent sequential reference."""

from __future__ import annotations

import numpy as np
import pytest
from gumbel_support import (
    ACTION_SIZE,
    FakeNet,
    RefNode,
    improved_policy,
    make_state,
    reference_search,
    reference_winner,
    two_eyes_state_white_to_move,
)

from alphago_zero.config import SearchConfig
from alphago_zero.game import BLACK, GoState
from alphago_zero.gumbel import GumbelMCTS, considered_visit_sequence, make_self_play_search
from alphago_zero.mcts import AlphaGoZeroMCTS, MCTSNode, SearchOutput, prune_policy_target


def _config(
    simulations: int,
    *,
    max_considered: int = 16,
    batch: int = 1,
    virtual_loss: float = 0.0,
    scale: float = 1.0,
) -> SearchConfig:
    return SearchConfig(
        num_simulations=simulations,
        eval_batch_size=batch,
        virtual_loss=virtual_loss,
        search_algo="gumbel",
        gumbel_max_considered=max_considered,
        gumbel_scale=scale,
    )


def _search(
    simulations: int,
    *,
    seed: int = 7,
    net_seed: int = 3,
    moves: tuple[int, ...] = (),
    noise: bool = True,
    model: FakeNet | None = None,
    max_considered: int = 16,
    batch: int = 1,
    virtual_loss: float = 0.0,
    scale: float = 1.0,
) -> tuple[GumbelMCTS, SearchOutput]:
    config = _config(
        simulations, max_considered=max_considered, batch=batch, virtual_loss=virtual_loss, scale=scale
    )
    search = GumbelMCTS(model or FakeNet(seed=net_seed), config, np.random.default_rng(seed))
    return search, search.run(make_state(moves), temperature=1.0, add_root_noise=noise)


def _assert_same_tree(node: MCTSNode, ref: RefNode, *, path: tuple[int, ...] = ()) -> None:
    """Compare actions, priors, visits, accumulated values and child structure recursively."""
    assert list(node.edges) == ref.actions, f"node action set {path}"
    for action, edge in node.edges.items():
        where = path + (action,)
        assert edge.prior == ref.prior[action], f"edge prior {where}"
        assert edge.visit_count == ref.visits[action], f"edge visits {where}"
        assert edge.value_sum == pytest.approx(ref.total[action], abs=1e-9), f"edge value sum {where}"
        ref_child = ref.children.get(action)
        assert (edge.child is None) == (ref_child is None), f"child presence {where}"
        if edge.child is not None and ref_child is not None:
            _assert_same_tree(edge.child, ref_child, path=where)


# ----------------------------------------------------------------------
# Reference equivalence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("virtual_loss", [0.0, 1.0])
@pytest.mark.parametrize(
    ("simulations", "max_considered"),
    [(1, 16), (5, 16), (16, 16), (33, 16), (32, 4), (64, 8), (64, 2)],
)
@pytest.mark.parametrize("moves", [(), (12, 6, 18, 8, 16)], ids=["empty", "midgame"])
def test_search_matches_independent_reference_tree(
    identity_symmetry: None,
    moves: tuple[int, ...],
    simulations: int,
    max_considered: int,
    virtual_loss: float,
) -> None:
    """With batch one, compare the tree, selected move and improved policy against the independent reference."""
    seed = 7
    _search_obj, output = _search(
        simulations, seed=seed, moves=moves, max_considered=max_considered, virtual_loss=virtual_loss
    )
    state = make_state(moves)
    # Root noise uses identical RNG draws immediately after expansion.
    noise = [float(x) for x in np.random.default_rng(seed).gumbel(size=len(state.legal_actions()))]
    reference = reference_search(
        FakeNet(seed=3), state, simulations=simulations, max_considered=max_considered, gumbel=noise
    )

    _assert_same_tree(output.root, reference)

    for action in range(ACTION_SIZE):
        assert output.visit_counts[action] == reference.visits.get(action, 0)
    assert output.improved_policy is not None
    expected_policy = improved_policy(reference, 50.0, 0.1)
    for action in range(ACTION_SIZE):
        assert output.improved_policy[action] == pytest.approx(expected_policy.get(action, 0.0), abs=1e-6)
    assert int(np.argmax(output.policy)) == reference_winner(reference, noise, 50.0, 0.1)

    total_visits = sum(reference.visits.values())
    total_value = sum(reference.total.values())
    assert output.root_q == pytest.approx(total_value / total_visits, abs=1e-9)


@pytest.mark.parametrize("scale", [0.25, 2.5])
def test_gumbel_noise_is_scaled_by_gumbel_scale(identity_symmetry: None, scale: float) -> None:
    """Scale a common Gumbel noise vector and compare against the reference tree."""
    seed, simulations = 11, 24
    _search_obj, output = _search(simulations, seed=seed, scale=scale)
    state = make_state()
    noise = [scale * float(x) for x in np.random.default_rng(seed).gumbel(size=len(state.legal_actions()))]
    reference = reference_search(
        FakeNet(seed=3), state, simulations=simulations, max_considered=16, gumbel=noise
    )

    _assert_same_tree(output.root, reference)
    assert int(np.argmax(output.policy)) == reference_winner(reference, noise, 50.0, 0.1)


@pytest.mark.parametrize(
    ("moves", "simulations", "max_considered", "batch"),
    [((), 32, 16, 1), ((12, 6, 18, 8, 16), 64, 8, 4), ((), 1, 16, 1), ((), 100, 26, 16)],
)
def test_numerics_raise_no_floating_point_errors(
    moves: tuple[int, ...], simulations: int, max_considered: int, batch: int
) -> None:
    """Check for division by zero, overflow and NaN while allowing softmax rounding."""
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        _search_obj, output = _search(simulations, moves=moves, max_considered=max_considered, batch=batch)
    assert output.improved_policy is not None
    assert np.isfinite(output.improved_policy).all() and np.isfinite(output.root_q)


def test_search_reaches_and_handles_terminal_positions(identity_symmetry: None) -> None:
    """Terminal leaves in a short game must back up exact outcomes as in the reference."""
    moves = (12, 6)
    state = make_state(moves, max_moves=5)  # A five-move limit gives terminal leaves at depth three here.
    noise = [float(x) for x in np.random.default_rng(11).gumbel(size=len(state.legal_actions()))]
    reference = reference_search(FakeNet(seed=4), state, simulations=48, max_considered=6, gumbel=noise)

    search = GumbelMCTS(FakeNet(seed=4), _config(48, max_considered=6), np.random.default_rng(11))
    output = search.run(state, temperature=1.0, add_root_noise=True)

    _assert_same_tree(output.root, reference)
    # The scenario must reach an expanded terminal node without edges.
    terminals = _count_terminals(output.root)
    assert terminals > 0


def _count_terminals(node: MCTSNode) -> int:
    count = 1 if node.state.is_terminal() and node.is_expanded else 0
    return count + sum(_count_terminals(e.child) for e in node.edges.values() if e.child is not None)


# ----------------------------------------------------------------------
# Sequential Halving structure under leaf batching
# ----------------------------------------------------------------------


def _expected_sorted_visits(max_considered: int, simulations: int) -> list[int]:
    """Derive the multiset of visit counts imposed by the schedule, independent of action identity."""
    visits = [0] * max_considered
    for required in considered_visit_sequence(max_considered, simulations):
        index = next(i for i, count in enumerate(visits) if count == required)
        visits[index] += 1
    return sorted(visits, reverse=True)


@pytest.mark.parametrize("batch", [1, 2, 5, 16, 32])
@pytest.mark.parametrize(("simulations", "max_considered"), [(32, 16), (16, 16), (24, 8), (9, 16), (50, 4)])
def test_visit_multiset_is_dictated_by_the_schedule_for_any_batch_size(
    simulations: int, max_considered: int, batch: int
) -> None:
    """In-flight visits count toward the schedule, so batch size preserves the visit multiset."""
    for seed in (1, 2, 3):
        _search_obj, output = _search(
            simulations, seed=seed, net_seed=seed, max_considered=max_considered, batch=batch
        )
        counts = sorted((int(c) for c in output.visit_counts if c > 0), reverse=True)
        expected = [c for c in _expected_sorted_visits(max_considered, simulations) if c > 0]
        assert counts == expected
        assert sum(counts) == simulations


@pytest.mark.parametrize(("config_sims", "call_sims"), [(32, 16), (8, 24)])
def test_budget_override_per_call_drives_the_schedule(config_sims: int, call_sims: int) -> None:
    """Use the per-call simulation budget, including explicit overrides."""
    search = GumbelMCTS(FakeNet(seed=2), _config(config_sims, max_considered=8), np.random.default_rng(3))
    output = search.run(make_state(), temperature=1.0, add_root_noise=True, num_simulations=call_sims)
    counts = sorted((int(c) for c in output.visit_counts if c > 0), reverse=True)
    assert counts == [c for c in _expected_sorted_visits(8, call_sims) if c > 0]
    assert sum(counts) == call_sims


@pytest.mark.parametrize("batch", [1, 3, 8, 64])
def test_no_virtual_loss_residue_and_consistent_visit_counts(batch: int) -> None:
    """A large virtual loss exposes any residual penalty after backup."""
    _search_obj, output = _search(
        40, batch=batch, virtual_loss=1000.0, moves=(12, 6, 18), net_seed=5
    )
    root = output.root
    assert sum(edge.visit_count for edge in root.edges.values()) == 40

    def walk(node: MCTSNode, incoming_visits: int | None) -> None:
        visits_below = sum(edge.visit_count for edge in node.edges.values())
        if incoming_visits is not None and node.edges:
            # The first visit expands the node. At most N-1 simulations can descend further.
            assert visits_below <= incoming_visits - 1
            if batch == 1:
                assert visits_below == incoming_visits - 1
        for edge in node.edges.values():
            assert abs(edge.value_sum) <= edge.visit_count + 1e-6, "residual virtual loss in value sum"
            if edge.child is not None:
                walk(edge.child, edge.visit_count)

    walk(root, None)


def test_winner_is_among_the_most_visited_actions() -> None:
    for seed in range(8):
        _search_obj, output = _search(32, seed=seed, net_seed=seed)
        chosen = int(np.argmax(output.policy))
        assert output.visit_counts[chosen] == output.visit_counts.max()
        assert output.policy.sum() == 1.0


# ----------------------------------------------------------------------
# Gumbel-Top-k and Gumbel-max
# ----------------------------------------------------------------------


def test_first_round_visits_exactly_the_top_k_of_gumbel_plus_logit(identity_symmetry: None) -> None:
    """With one visit per candidate the visited set must equal the top m noisy logits."""
    seed, k = 5, 8
    _search_obj, output = _search(k, seed=seed, net_seed=1, max_considered=k)
    legal = make_state().legal_actions()
    noise = np.random.default_rng(seed).gumbel(size=len(legal))
    log_priors = np.log([output.root.edges[a].prior for a in legal])
    expected = {legal[i] for i in np.argsort(-(noise + log_priors))[:k]}
    assert {int(a) for a in np.flatnonzero(output.visit_counts)} == expected
    assert output.visit_counts.max() == 1.0


def test_single_simulation_samples_the_prior(identity_symmetry: None) -> None:
    """With a single simulation, Gumbel-max samples the first action from the prior."""
    model = FakeNet(seed=2, bias_scale=1.2)
    search = GumbelMCTS(model, _config(1, max_considered=26), np.random.default_rng(123))
    runs = 2000
    counts = np.zeros(ACTION_SIZE)
    priors = np.zeros(ACTION_SIZE)
    for _ in range(runs):
        output = search.run(make_state(), temperature=1.0, add_root_noise=True)
        counts[int(np.argmax(output.visit_counts))] += 1
        for action, edge in output.root.edges.items():
            priors[action] = edge.prior
    frequencies = counts / runs
    tolerance = 5.0 * np.sqrt(priors * (1.0 - priors) / runs) + 1e-3
    assert np.all(np.abs(frequencies - priors) <= tolerance), (frequencies - priors).round(3)
    # A nonuniform predictor makes the distribution test discriminating.
    assert priors.max() > 3.0 / ACTION_SIZE


@pytest.mark.parametrize(("scale", "noise"), [(0.0, True), (1.0, False)], ids=["scale0", "noise-off"])
def test_noise_scale_zero_and_disabled_noise_make_search_deterministic(
    identity_symmetry: None, scale: float, noise: bool
) -> None:
    """With noise and symmetries disabled, changing the seed has no effect."""
    _s1, first = _search(32, seed=1, net_seed=6, scale=scale, noise=noise)
    _s2, second = _search(32, seed=2, net_seed=6, scale=scale, noise=noise)
    np.testing.assert_array_equal(first.visit_counts, second.visit_counts)
    assert first.improved_policy is not None and second.improved_policy is not None
    np.testing.assert_array_equal(first.improved_policy, second.improved_policy)


def test_same_seed_reproduces_and_noise_changes_the_search(identity_symmetry: None) -> None:
    _s, first = _search(32, seed=4)
    _s, again = _search(32, seed=4)
    np.testing.assert_array_equal(first.visit_counts, again.visit_counts)
    outcomes = {tuple(_search(32, seed=s)[1].visit_counts.astype(int)) for s in range(10, 16)}
    assert len(outcomes) > 1, "Gumbel noise must vary the candidate set"


# ----------------------------------------------------------------------
# Improved policy and Q signal
# ----------------------------------------------------------------------


def test_improved_policy_is_a_valid_distribution_over_legal_actions() -> None:
    moves = (12, 6, 18, 8, 16)
    _search_obj, output = _search(32, moves=moves, net_seed=8)
    assert output.improved_policy is not None
    legal = set(make_state(moves).legal_actions())
    assert output.improved_policy.dtype == np.float32
    assert float(output.improved_policy.sum()) == pytest.approx(1.0, abs=1e-5)
    assert (output.improved_policy >= 0.0).all()
    assert {int(a) for a in np.flatnonzero(output.improved_policy)} <= legal
    assert output.policy_target is output.improved_policy


def test_search_sees_value_and_shifts_policy_towards_the_better_move(identity_symmetry: None) -> None:
    """Construct a position where ownership of one point determines the value and test search improvement."""
    # Choose a good move ranked ninth of 26 so it enters the top 16 without leading on prior alone.
    plain_logits, _ = FakeNet(seed=9).predict_single(make_state().encode())
    legal = make_state().legal_actions()
    good = sorted(legal, key=lambda a: -plain_logits[a])[8]
    model = FakeNet(seed=9, good_point=good)
    logits, _ = model.predict_single(make_state().encode())
    prior = np.exp(logits - logits.max())
    prior = prior[legal] / prior[legal].sum()
    prior_of_good = float(prior[legal.index(good)])

    _search_obj, output = _search(32, model=model, scale=0.0, noise=False)
    assert output.improved_policy is not None
    assert int(np.argmax(output.policy)) == good
    assert output.visit_counts[good] == output.visit_counts.max()
    assert output.improved_policy[good] > prior_of_good + 0.2
    assert output.improved_policy[good] == output.improved_policy.max()


def test_best_child_q_requires_enough_total_simulations_for_resign() -> None:
    """Use the conservative +1 sentinel below the visit threshold and the best visited Q thereafter."""
    _s, shallow = _search(4, max_considered=4, noise=False)
    assert shallow.best_child_q == 1.0

    _s, deep = _search(16, max_considered=16, noise=False)
    qs = [e.q_value for e in deep.root.edges.values() if e.visit_count > 0]
    assert deep.best_child_q == pytest.approx(max(qs))
    assert deep.best_child_q < 1.0

    # Exactly eight visits satisfy resign_min_visits=8.
    config = SearchConfig()
    assert config.resign_min_visits == 8
    _s, boundary = _search(8, max_considered=8, noise=False)
    assert boundary.best_child_q < 1.0
    _s, below = _search(7, max_considered=7, noise=False)
    assert below.best_child_q == 1.0


# ----------------------------------------------------------------------
# Edge cases and search contract
# ----------------------------------------------------------------------


def test_single_legal_action_gets_all_simulations_and_full_probability() -> None:
    state = two_eyes_state_white_to_move()
    assert state.legal_actions() == [state.pass_action]
    search = GumbelMCTS(FakeNet(seed=1), _config(8), np.random.default_rng(0))
    output = search.run(state, temperature=1.0, add_root_noise=True)
    assert output.visit_counts[state.pass_action] == 8.0
    assert output.policy[state.pass_action] == 1.0
    assert output.improved_policy is not None
    assert output.improved_policy[state.pass_action] == pytest.approx(1.0)
    assert output.improved_policy.sum() == pytest.approx(1.0)


def test_fewer_legal_actions_than_candidates_still_follows_the_schedule() -> None:
    """With only two legal actions and m=16, cap the candidate count at two."""
    board = np.ones((5, 5), dtype=np.int8)
    board[0, 0] = 0
    board[2, 2] = 0
    # Black can play inside its own eyes or pass.
    state = GoState.from_board(board, to_play=BLACK, komi=2.5)
    legal = state.legal_actions()
    assert 1 < len(legal) < 16
    search = GumbelMCTS(FakeNet(seed=2), _config(12, max_considered=16), np.random.default_rng(1))
    output = search.run(state, temperature=1.0, add_root_noise=True)
    counts = sorted((int(c) for c in output.visit_counts if c > 0), reverse=True)
    assert counts == [c for c in _expected_sorted_visits(len(legal), 12) if c > 0]


def test_terminal_root_returns_empty_result_without_error() -> None:
    state = make_state()
    state = state.play(state.pass_action).play(state.pass_action)
    assert state.is_terminal()
    search = GumbelMCTS(FakeNet(seed=1), _config(8), np.random.default_rng(0))
    output = search.run(state, temperature=1.0, add_root_noise=True)
    assert output.visit_counts.sum() == 0.0
    assert output.policy.sum() == 0.0
    assert output.policy_target.sum() == 0.0


def test_tree_is_never_reused_between_moves() -> None:
    search, output = _search(8)
    action = int(np.argmax(output.policy))
    assert search.advance_root(output.root, action) is None
    assert search.advance_root(None, action) is None


def test_root_with_visits_is_rejected() -> None:
    """Reject a previously searched root because Sequential Halving starts visit counts at zero."""
    search, output = _search(8)
    with pytest.raises(ValueError, match="unvisited root"):
        search.run(make_state(), temperature=1.0, add_root_noise=True, root=output.root)


def test_select_leaf_without_prepared_root_is_noise_free_and_does_not_touch_rng() -> None:
    """Direct _select_leaf calls initialize a noiseless root without consuming randomness."""
    rng = np.random.default_rng(99)
    search = GumbelMCTS(FakeNet(seed=1), _config(8), rng)
    root = MCTSNode(state=make_state())
    search._expand(root)
    state_before = rng.bit_generator.state
    leaf, path = search._select_leaf(root)
    assert rng.bit_generator.state == state_before
    assert path and path[0][0] is root
    assert leaf is root.edges[path[0][1]].child


def test_unprepared_root_still_follows_sequential_halving_at_the_root() -> None:
    """Fallback initialization must apply the root schedule using the configured budget."""
    simulations, candidates = 8, 4
    search = GumbelMCTS(FakeNet(seed=2), _config(simulations, max_considered=candidates), np.random.default_rng(0))
    root = MCTSNode(state=make_state())
    search._expand(root)
    for _ in range(simulations):
        search._select_leaf(root)  # Skip explicit preparation and backup so only visit counts change.
    counts = sorted((e.visit_count for e in root.edges.values() if e.visit_count), reverse=True)
    assert counts == _expected_sorted_visits(candidates, simulations) == [3, 3, 1, 1]


def test_selection_beyond_the_budget_is_clamped_not_crashing() -> None:
    """Selections beyond the supplied budget reuse the final schedule entry."""
    search = GumbelMCTS(FakeNet(seed=1), _config(4, max_considered=4), np.random.default_rng(0))
    root = MCTSNode(state=make_state())
    search._expand(root)
    search._prepare_root(root, 4, add_noise=False)
    for _ in range(10):
        search._select_leaf(root)
    assert sum(e.visit_count for e in root.edges.values()) == 10


def test_broken_schedule_invariant_raises_instead_of_silently_picking_action_zero() -> None:
    """Report an impossible schedule instead of silently choosing the first edge."""
    search = GumbelMCTS(FakeNet(seed=1), _config(8, max_considered=4), np.random.default_rng(0))
    root = MCTSNode(state=make_state())
    search._expand(root)
    search._prepare_root(root, 8, add_noise=False)
    for edge in root.edges.values():
        edge.visit_count = 3  # Corrupt the counters so no candidate meets the schedule.
    with pytest.raises(RuntimeError, match="no candidate"):
        search._select_leaf(root)


def test_policy_target_pruning_applies_to_improved_policy() -> None:
    """Apply policy-target pruning to Gumbel's improved policy."""
    config = SearchConfig(
        num_simulations=32,
        eval_batch_size=1,
        virtual_loss=0.0,
        search_algo="gumbel",
        gumbel_max_considered=8,
        policy_target_prune_fraction=0.5,
    )
    search = GumbelMCTS(FakeNet(seed=2), config, np.random.default_rng(3))
    output = search.run(make_state(), temperature=1.0, add_root_noise=True)

    assert output.improved_policy is not None
    expected = prune_policy_target(output.improved_policy, 0.5)
    assert np.array_equal(output.policy_target, expected)
    assert float(output.policy_target.sum()) == pytest.approx(1.0)
    # The original softmax is positive on every legal action. Pruning must remove some entries.
    assert int((output.policy_target == 0).sum()) > int((output.improved_policy == 0).sum())
    # Pruning leaves the Sequential Halving visit counts intact.
    assert float(output.visit_counts.sum()) == 32.0


def test_factory_returns_search_for_the_configured_algorithm() -> None:
    model = FakeNet(seed=1)
    rng = np.random.default_rng(0)
    gumbel = make_self_play_search(model, SearchConfig(search_algo="gumbel"), rng)
    puct = make_self_play_search(model, SearchConfig(), rng)
    assert type(gumbel) is GumbelMCTS
    assert type(puct) is AlphaGoZeroMCTS

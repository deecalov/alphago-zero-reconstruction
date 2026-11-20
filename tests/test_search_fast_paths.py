"""Differential tests for optimized search and symmetry operations against frozen reference implementations."""

from __future__ import annotations

import itertools
import math
import warnings
import zlib
from types import SimpleNamespace
from typing import Any

import frozen_gumbel_v1 as old_gumbel
import frozen_mcts_v1 as old_mcts
import numpy as np
import pytest
from frozen_augmentations_v1 import INVERSE_DIHDERAL as OLD_INVERSE
from frozen_augmentations_v1 import apply_dihedral_to_board as old_to_board
from frozen_augmentations_v1 import apply_dihedral_to_planes as old_to_planes
from frozen_augmentations_v1 import apply_dihedral_to_policy as old_to_policy
from frozen_augmentations_v1 import invert_dihedral_on_policy as old_invert
from frozen_game_v1 import GoState as FrozenGoState

from alphago_zero import augmentations as new
from alphago_zero import gumbel as new_gumbel
from alphago_zero import mcts as new_mcts
from alphago_zero.config import SearchConfig
from alphago_zero.game import GoState

SIZES = [1, 2, 3, 4, 5, 6, 7, 9, 13, 19]


def _same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare shape, dtype, bytes and C/Fortran contiguity."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    assert actual.flags.writeable and actual.flags.aligned


def _noisy(rng: np.random.Generator, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Include signed zero, infinities and NaNs to ensure permutations preserve the bit pattern."""
    if dtype == "bool":
        return rng.random(shape) < 0.5
    if dtype in ("int8", "uint8", "int64"):
        return rng.integers(0, 100, size=shape).astype(dtype)
    array = rng.standard_normal(shape).astype(dtype)
    flat = array.reshape(-1)
    for index, special in zip(rng.permutation(flat.size)[:4], (-0.0, np.inf, -np.inf, np.nan)):
        flat[index] = special
    return array


# ---------------------------------------------------------------------------
# Dihedral symmetry
# ---------------------------------------------------------------------------
def test_inverse_table_is_unchanged() -> None:
    assert new.INVERSE_DIHDERAL == OLD_INVERSE


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("transform_id", range(8))
@pytest.mark.parametrize("dtype", ["float32", "float64", "uint8", "bool"])
def test_planes_match_frozen(size: int, transform_id: int, dtype: str) -> None:
    rng = np.random.default_rng(size * 100 + transform_id)
    for channels in (1, 17):
        planes = _noisy(rng, (channels, size, size), dtype)
        actual = new.apply_dihedral_to_planes(planes, transform_id)
        _same_bits(actual, old_to_planes(planes, transform_id))
        assert not np.shares_memory(actual, planes)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("transform_id", range(8))
def test_board_matches_frozen(size: int, transform_id: int) -> None:
    rng = np.random.default_rng(size * 100 + transform_id + 7)
    board = _noisy(rng, (size, size), "float32")
    actual = new.apply_dihedral_to_board(board, transform_id)
    _same_bits(actual, old_to_board(board, transform_id))
    assert not np.shares_memory(actual, board)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("transform_id", range(8))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_policy_functions_match_frozen(size: int, transform_id: int, dtype: str) -> None:
    rng = np.random.default_rng(size * 100 + transform_id + 13)
    policy = _noisy(rng, (size * size + 1,), dtype)
    forward = new.apply_dihedral_to_policy(policy, size, transform_id)
    _same_bits(forward, old_to_policy(policy, size, transform_id))
    assert not np.shares_memory(forward, policy)
    restored = new.invert_dihedral_on_policy(policy, size, transform_id)
    _same_bits(restored, old_invert(policy, size, transform_id))
    assert restored[-1].tobytes() == policy[-1].tobytes()  # Pass retains its index.


@pytest.mark.parametrize("transform_id", range(8))
def test_invert_undoes_apply_exactly(transform_id: int) -> None:
    rng = np.random.default_rng(transform_id)
    policy = _noisy(rng, (82,), "float32")
    forward = new.apply_dihedral_to_policy(policy, 9, transform_id)
    assert new.invert_dihedral_on_policy(forward, 9, transform_id).tobytes() == policy.tobytes()


def _layout(name: str, rng: np.random.Generator) -> np.ndarray:
    """Represent identical [17, 9, 9] values with different memory layouts."""
    base = _noisy(rng, (17, 9, 9), "float32")
    if name == "fortran":
        return np.asfortranarray(base)
    if name == "reversed-view":
        return base[:, ::-1, :]
    if name == "transposed-view":
        return base.transpose(0, 2, 1)
    if name == "strided-channels":
        return _noisy(rng, (34, 9, 9), "float32")[::2]
    raise AssertionError(name)


@pytest.mark.parametrize("transform_id", range(8))
@pytest.mark.parametrize("layout", ["fortran", "reversed-view", "transposed-view", "strided-channels"])
def test_planes_accept_any_memory_layout(layout: str, transform_id: int) -> None:
    planes = _layout(layout, np.random.default_rng(transform_id + 50))
    actual = new.apply_dihedral_to_planes(planes, transform_id)
    _same_bits(actual, old_to_planes(planes, transform_id))
    assert not np.shares_memory(actual, planes)


@pytest.mark.parametrize("transform_id", range(8))
def test_planes_with_other_leading_dimensions(transform_id: int) -> None:
    rng = np.random.default_rng(transform_id + 70)
    # Check the cache boundary at 64 planes and the reference fallback above it.
    shapes = ((9, 9), (3, 9, 9), (2, 3, 9, 9), (64, 3, 3), (65, 3, 3), (2, 40, 5, 5), (0, 9, 9), (4, 0, 5, 5))
    for shape in shapes:
        planes = _noisy(rng, shape, "float32")
        _same_bits(new.apply_dihedral_to_planes(planes, transform_id), old_to_planes(planes, transform_id))


@pytest.mark.parametrize("transform_id", range(8))
def test_non_square_input_keeps_old_behaviour(transform_id: int) -> None:
    rng = np.random.default_rng(transform_id + 90)
    board = _noisy(rng, (3, 5), "float32")
    _same_bits(new.apply_dihedral_to_board(board, transform_id), old_to_board(board, transform_id))
    planes = _noisy(rng, (2, 4, 6), "float32")
    _same_bits(new.apply_dihedral_to_planes(planes, transform_id), old_to_planes(planes, transform_id))


@pytest.mark.parametrize("transform_id", [-5, -1, 8, 9, 12, 100])
def test_transform_outside_zero_to_seven_keeps_old_behaviour(transform_id: int) -> None:
    rng = np.random.default_rng(abs(transform_id))
    planes = _noisy(rng, (3, 5, 5), "float32")
    _same_bits(new.apply_dihedral_to_planes(planes, transform_id), old_to_planes(planes, transform_id))
    board = _noisy(rng, (5, 5), "float32")
    _same_bits(new.apply_dihedral_to_board(board, transform_id), old_to_board(board, transform_id))
    policy = _noisy(rng, (26,), "float32")
    _same_bits(new.apply_dihedral_to_policy(policy, 5, transform_id), old_to_policy(policy, 5, transform_id))


@pytest.mark.parametrize("length", [0, 1, 10, 25, 27, 50])
def test_policy_of_wrong_length_fails_like_before(length: int) -> None:
    policy = np.zeros(length, dtype=np.float32)
    with pytest.raises(ValueError):
        old_to_policy(policy, 5, 3)
    with pytest.raises(ValueError):
        new.apply_dihedral_to_policy(policy, 5, 3)
    with pytest.raises(ValueError):
        new.invert_dihedral_on_policy(policy, 5, 3)


def test_policy_matrix_keeps_old_behaviour() -> None:
    rng = np.random.default_rng(4)
    for shape in ((3, 26), (26, 3)):
        matrix = _noisy(rng, shape, "float32")
        outcomes = []
        for function in (old_to_policy, new.apply_dihedral_to_policy):
            try:
                outcomes.append(function(matrix, 5, 2))
            except ValueError as error:
                outcomes.append(type(error))
        old_outcome, new_outcome = outcomes
        if isinstance(old_outcome, type):
            assert new_outcome is old_outcome
        else:
            _same_bits(new_outcome, old_outcome)


def test_numpy_integer_transform_ids_are_accepted() -> None:
    rng = np.random.default_rng(8)
    planes = _noisy(rng, (17, 9, 9), "float32")
    policy = _noisy(rng, (82,), "float32")
    for transform_id in range(8):
        for kind in (np.int64, np.int32, np.uint8):
            tid = kind(transform_id)
            _same_bits(new.apply_dihedral_to_planes(planes, tid), old_to_planes(planes, transform_id))
            _same_bits(new.invert_dihedral_on_policy(policy, 9, tid), old_invert(policy, 9, transform_id))


def test_unusual_argument_types_take_the_reference_path() -> None:
    """Preserve behavior for boolean, float and NumPy integer inputs and ndarray subclasses."""
    rng = np.random.default_rng(9)
    planes = _noisy(rng, (3, 9, 9), "float32")
    policy = _noisy(rng, (82,), "float32")
    # Exercise floating-point symmetry indices accepted by the reference rotations.
    for transform_id in (True, False, 2.0, 5.0, np.float64(3.0), np.float32(6.0), np.float16(1.0)):
        _same_bits(new.apply_dihedral_to_planes(planes, transform_id), old_to_planes(planes, transform_id))
        _same_bits(new.apply_dihedral_to_board(planes[0], transform_id), old_to_board(planes[0], transform_id))
        _same_bits(new.apply_dihedral_to_policy(policy, 9, transform_id), old_to_policy(policy, 9, transform_id))
    for transform_id in range(8):
        _same_bits(
            new.apply_dihedral_to_policy(policy, np.int64(9), transform_id), old_to_policy(policy, 9, transform_id)
        )
    masked = np.ma.masked_invalid(planes)
    for transform_id in range(8):
        actual = new.apply_dihedral_to_planes(masked, transform_id)
        expected = old_to_planes(masked, transform_id)
        assert type(actual) is type(expected)
        assert np.ma.getdata(actual).tobytes() == np.ma.getdata(expected).tobytes()
        assert np.ma.getmaskarray(actual).tobytes() == np.ma.getmaskarray(expected).tobytes()
    as_list = planes.tolist()
    for transform_id in range(8):
        _same_bits(new.apply_dihedral_to_planes(as_list, transform_id), old_to_planes(as_list, transform_id))
    # A flattened np.matrix remains two-dimensional and must use the reference path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)  # NumPy discourages np.matrix.
        matrix = np.matrix(planes[0])
        for transform_id in range(8):
            outcomes: list[Any] = []
            for function in (old_to_board, new.apply_dihedral_to_board):
                try:
                    outcomes.append(function(matrix, transform_id))
                except ValueError as error:
                    outcomes.append(type(error))
            old_outcome, new_outcome = outcomes
            if isinstance(old_outcome, type):
                assert new_outcome is old_outcome
            else:
                assert type(new_outcome) is type(old_outcome)
                assert new_outcome.shape == old_outcome.shape and new_outcome.tobytes() == old_outcome.tobytes()


@pytest.mark.parametrize("shape", [(), (5,), (0,)])
def test_arrays_with_fewer_than_two_dimensions_fail_like_before(shape: tuple[int, ...]) -> None:
    array = np.zeros(shape, dtype=np.float32)
    for transform_id in (0, 3, 5):
        for old_function, new_function in (
            (old_to_planes, new.apply_dihedral_to_planes),
            (old_to_board, new.apply_dihedral_to_board),
        ):
            with pytest.raises(ValueError) as old_error:
                old_function(array, transform_id)
            with pytest.raises(ValueError) as new_error:
                new_function(array, transform_id)
            assert str(new_error.value) == str(old_error.value)


def test_gather_tables_cannot_be_modified() -> None:
    """Shared permutation tables must reject writes."""
    for table in (new._cell_permutations(9)[3], new._policy_permutations(9)[5], new._gather_indices(9, 17)[1]):
        assert not table.flags.writeable
        with pytest.raises(ValueError):
            table[0] = 0


# ---------------------------------------------------------------------------
# Shared search test helpers
# ---------------------------------------------------------------------------
def _searcher(module: Any, config: SearchConfig | None = None, seed: int = 0) -> Any:
    """Construct an old or new search without a predictor for methods that never evaluate a leaf."""
    return module.AlphaGoZeroMCTS(SimpleNamespace(), config or SearchConfig(), np.random.default_rng(seed))


def _playout_states(board_size: int, games: int, seed: int, max_moves: int = 60) -> list[GoState]:
    """Sample nonterminal positions at different depths, including captures, ko and passes."""
    rng = np.random.default_rng(seed)
    states: list[GoState] = []
    for _ in range(games):
        state = GoState.new_game(board_size=board_size, komi=7.5 if board_size > 5 else 0.5)
        for _ in range(max_moves):
            if state.is_terminal():
                break
            states.append(state)
            legal = state.legal_actions()
            # Pass infrequently to keep random games long enough to exercise captures.
            action = legal[-1] if (len(legal) == 1 or rng.random() < 0.02) else int(rng.choice(legal[:-1]))
            state = state.play(action)
    return states


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("board_size", [5, 7, 9, 13])
def test_encode_matches_frozen_game_on_whole_playouts(board_size: int) -> None:
    """Compare the optimized history stack against the reference planes and memory layout."""
    rng = np.random.default_rng(board_size + 40)
    checked = 0
    for _game in range(3):
        new_state = GoState.new_game(board_size=board_size, komi=0.5)
        old_state = FrozenGoState.new_game(board_size=board_size, komi=0.5)
        for _step in range(70):
            if new_state.is_terminal():
                break
            _same_bits(new_state.encode(), old_state.encode())
            for history_length in (1, 2, 3, 4, 7, 9, 12):  # Exercise requested depths below, above and equal to available history.
                _same_bits(new_state.encode(history_length), old_state.encode(history_length))
            checked += 1
            legal = new_state.legal_actions()
            action = legal[-1] if (len(legal) == 1 or rng.random() < 0.03) else int(rng.choice(legal[:-1]))
            new_state, old_state = new_state.play(action), old_state.play(action)
    assert checked > 100


def test_encode_planes_do_not_alias_the_board_history() -> None:
    state = GoState.new_game(board_size=5, komi=0.5).play(12).play(13)
    snapshot = [board.copy() for board in state.board_history]
    planes = state.encode()
    planes[:] = 7.0
    assert all(np.array_equal(board, saved) for board, saved in zip(state.board_history, snapshot))


# ---------------------------------------------------------------------------
# Legal-move softmax and edge construction
# ---------------------------------------------------------------------------
def _random_logits(rng: np.random.Generator, kind: int, size: int) -> np.ndarray:
    """Generate six logit families including ties, extremes and nonfinite values."""
    if kind == 0:
        return rng.normal(0.0, 1.5, size=size)
    if kind == 1:
        return rng.normal(0.0, 30.0, size=size)
    if kind == 2:
        return rng.normal(0.0, 1000.0, size=size)
    if kind == 3:
        return np.round(rng.normal(0.0, 3.0, size=size))  # Many identical entries.
    if kind == 4:
        logits = rng.normal(0.0, 1.0, size=size)
        logits[rng.integers(0, size, size=3)] = rng.choice([np.nan, np.inf, -np.inf], size=3)
        return logits
    return rng.normal(-800.0, 5.0, size=size)


def _random_legal(rng: np.random.Generator, action_size: int) -> list[int]:
    """Select a sorted subset of actions as legal_actions would return."""
    size = int(rng.integers(1, action_size + 1))
    return sorted(rng.choice(action_size, size=size, replace=False).tolist())


@pytest.mark.parametrize("kind", range(6))
def test_legal_priors_match_frozen_masked_softmax(kind: int) -> None:
    rng = np.random.default_rng(100 + kind)
    old_search, new_search = _searcher(old_mcts), _searcher(new_mcts)
    for case in range(500):
        action_size = (26, 82, 82, 362)[case % 4]
        logits = _random_logits(rng, kind, action_size).astype(np.float32 if case % 3 else np.float64)
        legal = _random_legal(rng, action_size)
        with np.errstate(all="ignore"):  # Both implementations can warn on inf - inf and NaN.
            expected_vector = old_search._masked_softmax(logits, legal)
            actual = new_search._legal_priors(logits, legal)
            actual_vector = new_search._masked_softmax(logits, legal)
        expected = [float(expected_vector[action]) for action in legal]

        assert actual == expected
        assert all(type(prior) is float for prior in actual)
        _same_bits(actual_vector, expected_vector)


@pytest.mark.parametrize("bad", ["nan", "inf", "all-minus-inf"])
def test_legal_priors_fall_back_to_uniform_on_non_finite_logits(bad: str) -> None:
    logits = np.linspace(-1.0, 1.0, 82).astype(np.float32)
    legal = [1, 3, 81]
    if bad == "all-minus-inf":
        logits[legal] = -np.inf  # A maximum of -inf gives NaN when subtracted from itself.
    else:
        logits[3] = float(bad)
    expected = float(np.float32(1.0 / 3.0))  # Round each uniform probability to float32 exactly once.
    assert expected != 1.0 / 3.0
    with np.errstate(all="ignore"):
        assert new_mcts.AlphaGoZeroMCTS._legal_priors(logits, legal) == [expected] * 3
        assert _searcher(old_mcts)._masked_softmax(logits, legal)[legal].tolist() == [expected] * 3


def test_minus_infinity_on_one_logit_gives_zero_prior_not_fallback() -> None:
    logits = np.linspace(-1.0, 1.0, 82).astype(np.float32)
    logits[3] = -np.inf
    legal = [1, 3, 81]
    with np.errstate(all="ignore"):
        priors = new_mcts.AlphaGoZeroMCTS._legal_priors(logits, legal)
        reference = _searcher(old_mcts)._masked_softmax(logits, legal)
    assert priors[1] == 0.0 and priors[0] > 0.0 and priors[2] > 0.0
    assert priors == [float(reference[action]) for action in legal]


def test_legal_priors_match_frozen_on_a_float32_rounding_boundary_case() -> None:
    """These logits distinguish direct division from multiplication by a rounded reciprocal."""
    logits = np.array([-1.537663221359253, -0.9204521775245667, -1.2199881076812744], dtype=np.float32)
    legal = [0, 1, 2]
    expected = _searcher(old_mcts)._masked_softmax(logits, legal)[legal]
    exponents = np.exp(logits.astype(np.float64) - logits.astype(np.float64).max())
    reciprocal_variant = (exponents * (1.0 / float(exponents.sum()))).astype(np.float32)
    if reciprocal_variant.tobytes() == expected.tobytes():
        pytest.skip("this NumPy build does not distinguish division from reciprocal multiplication for this case")
    assert new_mcts.AlphaGoZeroMCTS._legal_priors(logits, legal) == expected.tolist()


def test_legal_priors_leave_inputs_untouched_and_handle_tiny_sets() -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(size=82).astype(np.float32)
    snapshot = logits.copy()
    legal = [4, 17, 40, 81]
    new_mcts.AlphaGoZeroMCTS._legal_priors(logits, legal)
    assert logits.tobytes() == snapshot.tobytes()
    assert legal == [4, 17, 40, 81]
    assert new_mcts.AlphaGoZeroMCTS._legal_priors(logits, [81]) == [1.0]  # Only pass is available.
    assert math.isclose(sum(new_mcts.AlphaGoZeroMCTS._legal_priors(logits, legal)), 1.0, rel_tol=1e-6)


def test_legal_priors_with_no_legal_actions_fail_like_before() -> None:
    logits = np.zeros(82, dtype=np.float32)
    with pytest.raises(ValueError):
        _searcher(old_mcts)._masked_softmax(logits, [])
    with pytest.raises(ValueError):
        new_mcts.AlphaGoZeroMCTS._legal_priors(logits, [])


@pytest.mark.parametrize("board_size", [5, 9])
def test_apply_network_output_matches_frozen_on_real_positions(board_size: int) -> None:
    rng = np.random.default_rng(board_size)
    old_search, new_search = _searcher(old_mcts), _searcher(new_mcts)
    action_size = board_size * board_size + 1
    states = _playout_states(board_size, games=6, seed=board_size)
    assert len(states) > 100
    for index, state in enumerate(states):
        logits = _random_logits(rng, index % 4, action_size).astype(np.float32)
        old_node = old_mcts.MCTSNode(state=state)
        new_node = new_mcts.MCTSNode(state=state)
        old_value = old_search._apply_network_output(old_node, logits, 0.3 - 0.01 * (index % 7))
        new_value = new_search._apply_network_output(new_node, logits, 0.3 - 0.01 * (index % 7))

        assert new_value == old_value
        assert new_node.node_value == old_node.node_value and new_node.is_expanded and old_node.is_expanded
        assert list(new_node.edges) == list(old_node.edges) == state.legal_actions()
        for action, edge in new_node.edges.items():
            reference = old_node.edges[action]
            assert type(edge.prior) is float
            assert edge.prior == reference.prior
            assert (edge.visit_count, edge.value_sum, edge.child) == (0, 0.0, None)


def test_new_edges_are_independent_objects() -> None:
    state = GoState.new_game(board_size=5, komi=0.5)
    node = new_mcts.MCTSNode(state=state)
    _searcher(new_mcts)._apply_network_output(node, np.zeros(26, dtype=np.float32), 0.0)
    edges = list(node.edges.values())
    assert len({id(edge) for edge in edges}) == len(edges)
    edges[0].visit_count += 5
    edges[0].value_sum -= 2.0
    assert all(edge.visit_count == 0 and edge.value_sum == 0.0 for edge in edges[1:])


# ---------------------------------------------------------------------------
# PUCT action selection
# ---------------------------------------------------------------------------
def _outcome(search: Any, node: Any) -> tuple[str, Any]:
    """Capture the chosen action or exception type to compare error behavior as well."""
    try:
        return "ok", search._select_action(node)
    except (AssertionError, ZeroDivisionError, ValueError, OverflowError) as error:
        # Negative total visits can cause a square-root error and a visit count of -1 a zero denominator.
        return "error", type(error)


def _node_pair(
    action_priors_visits: list[tuple[int, Any, Any, float]], node_value: float
) -> tuple[Any, Any]:
    """Construct equivalent nodes in the reference and optimized search classes."""
    old_edges = {
        action: old_mcts.EdgeStats(prior=prior, visit_count=visits, value_sum=value_sum)
        for action, prior, visits, value_sum in action_priors_visits
    }
    new_edges = {
        action: new_mcts.EdgeStats(prior=prior, visit_count=visits, value_sum=value_sum)
        for action, prior, visits, value_sum in action_priors_visits
    }
    old_node = old_mcts.MCTSNode(state=None, edges=old_edges, is_expanded=True, node_value=node_value)  # type: ignore[arg-type]
    new_node = new_mcts.MCTSNode(state=None, edges=new_edges, is_expanded=True, node_value=node_value)  # type: ignore[arg-type]
    return old_node, new_node


def _random_node_spec(rng: np.random.Generator, case: int) -> tuple[list[tuple[int, Any, Any, float]], float]:
    """Generate seven types of edge input, including ties and mixed numeric types."""
    count = int(rng.integers(1, 83))
    actions = sorted(rng.choice(82, size=count, replace=False).tolist())
    if case % 2:  # Dictionary insertion order determines ties even when actions are unsorted.
        rng.shuffle(actions)
    style = case % 7
    if style == 0:
        priors: list[Any] = rng.dirichlet(np.full(count, 0.3)).tolist()
        visits: list[Any] = rng.choice([0, 0, 0, 0, 1, 2, 5, 30, 200], size=count).tolist()
    elif style == 1:
        priors = rng.choice([0.0, 0.1, 0.2, 0.2, 0.3], size=count).tolist()
        visits = rng.choice([0, 0, 1, 2], size=count).tolist()
    elif style == 2:
        priors, visits = [0.125] * count, [0] * count
    elif style == 3:
        priors = [np.float32(p) for p in rng.dirichlet(np.full(count, 0.5))]
        visits = rng.choice([0, 3, 7], size=count).astype(float).tolist()
    elif style == 4:
        priors = rng.choice([0.0, 1e-9, 0.5, 1.0, 1e9, np.inf, np.nan], size=count).tolist()
        visits = rng.choice([0, 0, 1, 3], size=count).tolist()
    elif style == 5:
        priors = rng.dirichlet(np.full(count, 1.0)).tolist()
        visits = rng.choice([-2, -1, -0.5, 0, 0, 0.5, 1.5, 4], size=count).tolist()
    else:
        priors = rng.dirichlet(np.full(count, 0.2)).tolist()
        visits = rng.integers(0, 3, size=count).tolist()
    sums = [float(rng.uniform(-1.0, 1.0)) * (v if v > 0 else 1.0) for v in visits]
    return list(zip(actions, priors, visits, sums)), float(rng.uniform(-1.0, 1.0))


SELECT_CONFIGS = [
    SearchConfig(),
    SearchConfig(c_puct=0.3),
    SearchConfig(c_puct=4.0, fpu_reduction=0.0),
    SearchConfig(fpu_reduction=0.25),
    SearchConfig(c_puct=2.2, fpu_reduction=0.5),
    SearchConfig(c_puct=1.0, fpu_reduction=3.0),  # Exercise the -1 lower bound on parent value minus FPU.
]


@pytest.mark.parametrize("config_index", range(len(SELECT_CONFIGS)))
def test_select_action_matches_frozen_on_random_nodes(config_index: int) -> None:
    config = SELECT_CONFIGS[config_index]
    rng = np.random.default_rng(300 + config_index)
    old_search, new_search = _searcher(old_mcts, config), _searcher(new_mcts, config)
    for case in range(700):
        spec, node_value = _random_node_spec(rng, case)
        old_node, new_node = _node_pair(spec, node_value)
        with np.errstate(all="ignore"):
            expected = _outcome(old_search, old_node)
            actual = _outcome(new_search, new_node)
        assert actual == expected, (case, expected, actual)


def test_select_action_ties_go_to_the_first_edge_in_dict_order() -> None:
    for order in ([7, 3, 11, 0], [0, 3, 7, 11], [11, 7, 3, 0]):
        spec = [(action, 0.25, 0, 0.0) for action in order]
        old_node, new_node = _node_pair(spec, 0.0)
        assert _searcher(new_mcts)._select_action(new_node) == order[0]
        assert _searcher(old_mcts)._select_action(old_node) == order[0]


def test_select_action_without_edges_fails_like_before() -> None:
    old_node, new_node = _node_pair([], 0.0)
    assert _outcome(_searcher(old_mcts), old_node) == _outcome(_searcher(new_mcts), new_node) == ("error", AssertionError)


def test_select_action_does_not_modify_the_node() -> None:
    rng = np.random.default_rng(11)
    spec, node_value = _random_node_spec(rng, 0)
    _, node = _node_pair(spec, node_value)
    before = [(a, e.prior, e.visit_count, e.value_sum, e.child) for a, e in node.edges.items()]
    _searcher(new_mcts)._select_action(node)
    assert before == [(a, e.prior, e.visit_count, e.value_sum, e.child) for a, e in node.edges.items()]


def _first_maximum(scores: list[float]) -> int:
    """Return the first maximum under strict comparison, matching _select_action."""
    best, best_score = -1, -math.inf
    for index, score in enumerate(scores):
        if score > best_score:
            best, best_score = index, score
    return best


def _winners_by_grouping(priors: list[float], visits: list[int], c_puct: float) -> tuple[int, int, int]:
    """Compute the winning action using the original operation order and two alternative groupings."""
    normalizer = math.sqrt(sum(visits) + 1.0)
    exact: list[float] = []
    regrouped_p: list[float] = []
    regrouped_n: list[float] = []
    for prior, count in zip(priors, visits):
        if count == 0:
            score = 0.0 + c_puct * prior * normalizer
            exact.append(score)
            regrouped_p.append(score)
            regrouped_n.append(score)
        else:
            exact.append(0.0 / count + c_puct * prior * normalizer / (1.0 + count))
            regrouped_p.append(0.0 / count + c_puct * (prior * normalizer) / (1.0 + count))
            regrouped_n.append(0.0 / count + (c_puct * prior) * (normalizer / (1.0 + count)))
    return _first_maximum(exact), _first_maximum(regrouped_p), _first_maximum(regrouped_n)


def test_select_action_keeps_the_float_operation_order_of_the_formula() -> None:
    """Use rounding-sensitive positions where reassociating the PUCT bonus changes the winning action."""
    nice_priors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]
    discriminating = 0
    for c_puct in (1.0, 1.5, 2.5):
        config = SearchConfig(c_puct=c_puct)
        old_search, new_search = _searcher(old_mcts, config), _searcher(new_mcts, config)
        for prior_a, prior_b in itertools.product(nice_priors, repeat=2):
            for visits_a, visits_b, visits_c in itertools.product(range(8), range(8), (0, 5)):
                priors, visits = [prior_a, prior_b, 0.05], [visits_a, visits_b, visits_c]
                if len(set(_winners_by_grouping(priors, visits, c_puct))) == 1:
                    continue
                discriminating += 1
                spec = [(index, p, v, 0.0) for index, (p, v) in enumerate(zip(priors, visits))]
                old_node, new_node = _node_pair(spec, 0.0)
                assert _outcome(new_search, new_node) == _outcome(old_search, old_node), (priors, visits, c_puct)
    assert discriminating >= 20  # Require the case to distinguish the implementations.


# ---------------------------------------------------------------------------
# Complete search output and tree equivalence
# ---------------------------------------------------------------------------
class _HashNet:
    """Build logits and values solely from the bytes of the input planes."""

    def __init__(self, action_size: int) -> None:
        self._action_size = action_size

    def predict_single(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        seed = zlib.crc32(np.ascontiguousarray(planes, dtype=np.float32).tobytes())
        rng = np.random.default_rng(seed)
        logits = rng.normal(0.0, 1.2, size=self._action_size).astype(np.float32)
        return logits, float(rng.uniform(-0.9, 0.9))

    def predict_batch(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = [self.predict_single(sample) for sample in planes]
        logits = np.stack([logits for logits, _value in outputs]).astype(np.float32)
        values = np.asarray([value for _logits, value in outputs], dtype=np.float32)
        return logits, values


def _assert_same_tree(old_root: Any, new_root: Any) -> int:
    """Traverse both trees and compare every node bitwise. Return the number of nodes checked."""
    stack = [(old_root, new_root)]
    nodes = 0
    while stack:
        old_node, new_node = stack.pop()
        nodes += 1
        assert new_node.is_expanded == old_node.is_expanded
        assert new_node.node_value == old_node.node_value
        assert new_node.state.to_play == old_node.state.to_play
        assert new_node.state.board.tobytes() == old_node.state.board.tobytes()
        assert list(new_node.edges) == list(old_node.edges)
        for action, new_edge in new_node.edges.items():
            old_edge = old_node.edges[action]
            assert type(new_edge.prior) is type(old_edge.prior)
            assert (new_edge.prior, new_edge.visit_count, new_edge.value_sum) == (
                old_edge.prior,
                old_edge.visit_count,
                old_edge.value_sum,
            )
            assert (new_edge.child is None) == (old_edge.child is None)
            if new_edge.child is not None:
                stack.append((old_edge.child, new_edge.child))
    return nodes


def _assert_same_output(old_out: Any, new_out: Any) -> None:
    for name in ("policy", "visit_counts"):
        actual, expected = getattr(new_out, name), getattr(old_out, name)
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes(), name
    assert (new_out.root_value, new_out.best_child_q, new_out.root_q) == (
        old_out.root_value,
        old_out.best_child_q,
        old_out.root_q,
    )
    if old_out.improved_policy is None:
        assert new_out.improved_policy is None
    else:
        assert new_out.improved_policy.dtype == old_out.improved_policy.dtype
        assert new_out.improved_policy.tobytes() == old_out.improved_policy.tobytes()


def _compare_searches(old_search: Any, new_search: Any, state: GoState, noise: bool, temperature: float) -> int:
    old_out = old_search.run(state, temperature=temperature, add_root_noise=noise)
    new_out = new_search.run(state, temperature=temperature, add_root_noise=noise)
    _assert_same_output(old_out, new_out)
    nodes = _assert_same_tree(old_out.root, new_out.root)
    # Verify identical RNG consumption for symmetry selection and root noise.
    assert new_search.rng.bit_generator.state == old_search.rng.bit_generator.state
    return nodes


PUCT_CASES = [
    (5, {"num_simulations": 40, "eval_batch_size": 1}),
    (5, {"num_simulations": 48, "eval_batch_size": 4, "fpu_reduction": 0.25}),
    (9, {"num_simulations": 64, "eval_batch_size": 16, "c_puct": 2.0, "virtual_loss": 2.0}),
    (9, {"num_simulations": 33, "eval_batch_size": 8, "root_dirichlet_epsilon": 0.4, "fpu_reduction": 0.0}),
]


@pytest.mark.parametrize(("board_size", "overrides"), PUCT_CASES)
def test_puct_search_matches_frozen_whole_tree(board_size: int, overrides: dict[str, Any]) -> None:
    config = SearchConfig(**overrides)
    network = _HashNet(board_size * board_size + 1)
    states = _playout_states(board_size, games=2, seed=board_size)[::9][:5]
    assert len(states) >= 3
    compared = 0
    for index, state in enumerate(states):
        for noise, temperature in ((True, 1.0), (False, 1e-6)):
            old_search = old_mcts.AlphaGoZeroMCTS(network, config, np.random.default_rng(index))
            new_search = new_mcts.AlphaGoZeroMCTS(network, config, np.random.default_rng(index))
            compared += _compare_searches(old_search, new_search, state, noise, temperature)
    assert compared > 100


def test_puct_game_with_tree_reuse_and_budget_override_matches_frozen() -> None:
    config = SearchConfig(num_simulations=32, eval_batch_size=4, fpu_reduction=0.1)
    network = _HashNet(26)
    old_search = old_mcts.AlphaGoZeroMCTS(network, config, np.random.default_rng(3))
    new_search = new_mcts.AlphaGoZeroMCTS(network, config, np.random.default_rng(3))
    state = GoState.new_game(board_size=5, komi=0.5)
    old_root = new_root = None
    for move in range(14):
        budget = 12 if move % 3 == 0 else None  # Cheap move under randomized playout caps.
        old_out = old_search.run(state, 1.0, True, root=old_root, num_simulations=budget)
        new_out = new_search.run(state, 1.0, True, root=new_root, num_simulations=budget)
        _assert_same_output(old_out, new_out)
        _assert_same_tree(old_out.root, new_out.root)
        action = int(np.argmax(old_out.policy))
        old_root = old_search.advance_root(old_out.root, action)
        new_root = new_search.advance_root(new_out.root, action)
        assert (old_root is None) == (new_root is None)
        state = state.play(action)
        if state.is_terminal():
            break
    assert new_search.rng.bit_generator.state == old_search.rng.bit_generator.state


# ---------------------------------------------------------------------------
# Gumbel completed Q, edge selection, root scheduling and full search
# ---------------------------------------------------------------------------
def _gumbel_searcher(module: Any, config: SearchConfig, seed: int = 0) -> Any:
    """Construct either Gumbel selector without a predictor."""
    return module.GumbelMCTS(SimpleNamespace(), config, np.random.default_rng(seed))


def _action_outcome(select: Any, node: Any) -> tuple[Any, ...]:
    """Return an action or the exception type and text for comparison."""
    try:
        return ("ok", select(node))
    except (AssertionError, ZeroDivisionError, ValueError, OverflowError, RuntimeError, IndexError) as error:
        return ("error", type(error), str(error))


GUMBEL_SELECT_CONFIGS = [
    SearchConfig(search_algo="gumbel"),
    SearchConfig(search_algo="gumbel", gumbel_c_visit=0.0, gumbel_c_scale=1.0),
    SearchConfig(search_algo="gumbel", gumbel_c_visit=5.0, gumbel_c_scale=2.5),
    SearchConfig(search_algo="gumbel", gumbel_c_visit=-3.0, gumbel_c_scale=0.1),
    # An overflowing scale on an unvisited node must fall back to the full computation.
    SearchConfig(search_algo="gumbel", gumbel_c_visit=1e308, gumbel_c_scale=100.0),
]


def _gumbel_visits(rng: np.random.Generator, style: int, count: int) -> np.ndarray:
    """Generate six kinds of float64 visit arrays with integer values, including huge counts."""
    if style == 0:
        return np.zeros(count)
    if style == 1:
        visits = np.zeros(count)
        visits[int(rng.integers(count))] = float(rng.integers(1, 40))
        return visits
    if style == 2:
        return rng.choice([0.0, 0.0, 1.0, 2.0, 5.0, 30.0], size=count)
    if style == 3:
        return rng.integers(1, 5000, size=count).astype(np.float64)
    if style == 4:
        return rng.choice([0.0, 3.0], size=count)
    return rng.choice([0.0, 1.0, 2.0**40], size=count)  # Even 82 * 2**40 fits exactly below 2**53.


@pytest.mark.parametrize("style", range(6))
def test_sigma_from_stats_matches_frozen_completed_q_scores(style: int) -> None:
    """Compare precomputed-statistics completed Q against the original formulas bit for bit."""
    rng = np.random.default_rng(900 + style)
    for case in range(400):
        count = int(rng.integers(1, 83))
        probs = old_gumbel._softmax(rng.normal(0.0, 1.5, size=count))
        visits = _gumbel_visits(rng, style, count)
        q_values = np.full(count, 0.25) if style == 4 else rng.uniform(-1.0, 1.0, size=count)  # Style four ties every edge.
        if case % 3 == 0:
            q_values = np.where(visits > 0, q_values, 0.0)  # Initialize unvisited Q exactly as the production implementation does.
        if case % 4:
            raw_value = float(rng.uniform(-1.0, 1.0))
        else:
            raw_value = [0.0, -0.0, 1.0, -1.0, math.nan, math.inf][(case // 4) % 6]
        c_visit, c_scale = [(50.0, 0.1), (0.0, 1.0), (1.0, 0.0), (100.0, 2.5), (-5.0, 0.1)][case % 5]
        visit_list = visits.astype(np.int64).tolist()
        with np.errstate(all="ignore"):
            expected = old_gumbel.completed_q_scores(raw_value, q_values, visits, probs, c_visit, c_scale)
            actual = new_gumbel._sigma_from_stats(
                raw_value, q_values, visits > 0, float(sum(visit_list)), float(max(visit_list)), probs, c_visit, c_scale
            )
        _same_bits(actual, expected)


def test_edge_arrays_match_frozen_edge_view() -> None:
    rng = np.random.default_rng(1700)
    for case in range(700):
        spec, node_value = _random_node_spec(rng, case)
        old_node, new_node = _node_pair(spec, node_value)
        with np.errstate(all="ignore"):
            _actions, _priors, old_visits, old_q = old_gumbel._edge_view(old_node)
            visit_list = [edge.visit_count for edge in new_node.edges.values()]
            visits, q_values, visited = new_gumbel._edge_arrays(new_node.edges, visit_list)
        _same_bits(visits, old_visits)
        _same_bits(q_values, old_q)
        assert visited.dtype == np.bool_ and visited.tobytes() == (old_visits > 0).tobytes()


@pytest.mark.parametrize("config_index", range(len(GUMBEL_SELECT_CONFIGS)))
def test_gumbel_interior_action_matches_frozen_on_random_nodes(config_index: int) -> None:
    config = GUMBEL_SELECT_CONFIGS[config_index]
    rng = np.random.default_rng(1300 + config_index)
    old_search, new_search = _gumbel_searcher(old_gumbel, config), _gumbel_searcher(new_gumbel, config)
    for case in range(600):
        spec, node_value = _random_node_spec(rng, case)
        old_node, new_node = _node_pair(spec, node_value)
        with np.errstate(all="ignore"):
            expected = _action_outcome(old_search._select_interior_action, old_node)
            actual = _action_outcome(new_search._select_interior_action, new_node)
            again = _action_outcome(new_search._select_interior_action, new_node)  # The second call exercises the node cache.
        assert actual == expected, (case, expected, actual)
        assert again == expected, (case, expected, again)


@pytest.mark.parametrize("config_index", range(len(GUMBEL_SELECT_CONFIGS)))
def test_gumbel_zero_visit_shortcut_matches_frozen(config_index: int) -> None:
    """Select by prior alone at an unvisited node only when the original arithmetic permits it."""
    config = GUMBEL_SELECT_CONFIGS[config_index]
    rng = np.random.default_rng(1350 + config_index)
    old_search, new_search = _gumbel_searcher(old_gumbel, config), _gumbel_searcher(new_gumbel, config)
    node_values = [0.0, -0.0, 0.3, -1.0, 1.0, math.nan, math.inf, -math.inf]
    for case in range(500):
        count = int(rng.integers(1, 83))
        actions = rng.permutation(82)[:count].tolist()  # Shuffle insertion order to exercise tie-breaking.
        kind = case % 6
        if kind == 0:
            priors = old_gumbel._softmax(rng.normal(0.0, 1.5, size=count)).astype(np.float32).tolist()
        elif kind == 1:
            priors = [0.25] * count  # All priors equal.
        elif kind == 2:
            priors = rng.choice([0.0, 0.0, 0.1, 0.1, 0.8], size=count).tolist()  # Zeros and a tied maximum.
        elif kind == 3:
            priors = [1.0] + [0.0] * (count - 1)
            rng.shuffle(priors)
        elif kind == 4:
            priors = rng.dirichlet(np.full(count, 0.05)).astype(np.float32).tolist()  # Nearly degenerate distribution.
        else:
            # Sub-floor priors tie after clamping, while larger priors remain distinguishable.
            priors = rng.choice([0.0, 1e-45, 1e-40, 1e-38, 1e-35, 1e-33, 1e-31], size=count).tolist()
        node_value = node_values[(case // 6) % len(node_values)]
        old_node, new_node = _node_pair([(a, p, 0, 0.0) for a, p in zip(actions, priors)], node_value)
        with np.errstate(all="ignore"):
            expected = _action_outcome(old_search._select_interior_action, old_node)
            actual = _action_outcome(new_search._select_interior_action, new_node)
        assert actual == expected, (case, node_value, expected, actual)


@pytest.mark.parametrize("config_index", range(len(GUMBEL_SELECT_CONFIGS)))
def test_gumbel_interior_choices_match_frozen_through_a_visit_sequence(config_index: int) -> None:
    """Advance both trees one simulation at a time while retaining the prior cache."""
    config = GUMBEL_SELECT_CONFIGS[config_index]
    rng = np.random.default_rng(1400 + config_index)
    old_search, new_search = _gumbel_searcher(old_gumbel, config), _gumbel_searcher(new_gumbel, config)
    checked = 0
    for case in range(30):
        count = int(rng.integers(2, 40))
        priors = old_gumbel._softmax(rng.normal(0.0, 1.5, size=count)).astype(np.float32).tolist()
        actions = rng.permutation(82)[:count].tolist()
        old_node, new_node = _node_pair(
            [(a, p, 0, 0.0) for a, p in zip(actions, priors)], float(rng.uniform(-1.0, 1.0))
        )
        for step in range(90):
            with np.errstate(all="ignore"):
                expected = _action_outcome(old_search._select_interior_action, old_node)
                actual = _action_outcome(new_search._select_interior_action, new_node)
            assert actual == expected, (case, step, expected, actual)
            checked += 1
            if expected[0] != "ok":
                break
            value = float(rng.uniform(-1.0, 1.0))
            for node in (old_node, new_node):  # Back up the chosen edge's visit and value in both copies.
                edge = node.edges[expected[1]]
                edge.visit_count += 1
                edge.value_sum += value
    assert checked > 1000


ROOT_CASES = [
    # Edge count, simulation budget, candidate count and noise scale.
    (2, 8, 16, 1.0),
    (7, 16, 4, 1.0),
    (30, 32, 16, 1.0),
    (82, 64, 16, 0.0),
    (50, 33, 8, 2.0),
    (12, 10, 100, 1.0),  # Cap the number of candidates at the number of actions.
]


@pytest.mark.parametrize(("count", "simulations", "candidates", "scale"), ROOT_CASES)
def test_gumbel_root_schedule_matches_frozen_step_by_step(
    count: int, simulations: int, candidates: int, scale: float
) -> None:
    """Compare each Sequential Halving root choice, including selections beyond the planned budget."""
    config = SearchConfig(
        search_algo="gumbel", num_simulations=simulations, gumbel_max_considered=candidates, gumbel_scale=scale
    )
    rng = np.random.default_rng(1500 + count)
    old_search, new_search = _gumbel_searcher(old_gumbel, config), _gumbel_searcher(new_gumbel, config)
    checked = 0
    for _case in range(8):
        priors = old_gumbel._softmax(rng.normal(0.0, 1.5, size=count)).astype(np.float32).tolist()
        actions = rng.permutation(82)[:count].tolist()
        old_node, new_node = _node_pair(
            [(a, p, 0, 0.0) for a, p in zip(actions, priors)], float(rng.uniform(-1.0, 1.0))
        )
        noise = scale * rng.gumbel(size=count)
        for search, node in ((old_search, old_node), (new_search, new_node)):
            search._prepare_root(node, simulations, add_noise=False)
            search._gumbel = noise.copy()  # Use identical candidate noise in both implementations.
        for step in range(simulations + 4):  # Past the budget, reuse the last schedule entry.
            with np.errstate(all="ignore"):
                expected = _action_outcome(old_search._select_action, old_node)
                actual = _action_outcome(new_search._select_action, new_node)
            assert actual == expected, (count, step, expected, actual)
            checked += 1
            if expected[0] != "ok":
                break
            value = float(rng.uniform(-1.0, 1.0))
            for node in (old_node, new_node):
                edge = node.edges[expected[1]]
                edge.visit_count += 1
                edge.value_sum += value
    assert checked >= 8 * simulations


def test_gumbel_root_with_broken_counters_fails_like_before() -> None:
    """Raise the same RuntimeError for a schedule with no eligible candidate."""
    config = SearchConfig(search_algo="gumbel", num_simulations=8, gumbel_max_considered=4)
    spec = [(action, 0.25, 3, 0.0) for action in (4, 9, 20, 41)]
    outcomes = []
    for module, mcts_module in ((old_gumbel, old_mcts), (new_gumbel, new_mcts)):
        search = _gumbel_searcher(module, config)
        edges = {a: mcts_module.EdgeStats(prior=p, visit_count=v, value_sum=s) for a, p, v, s in spec}
        node = mcts_module.MCTSNode(state=None, edges=edges, is_expanded=True, node_value=0.0)  # type: ignore[arg-type]
        search._root, search._budget, search._gumbel = node, 8, np.zeros(4)
        outcomes.append(_action_outcome(search._select_action, node))
    assert outcomes[0][0] == "error" and outcomes[0][1] is RuntimeError
    assert outcomes[1] == outcomes[0]


def test_gumbel_selection_without_edges_fails_like_before() -> None:
    config = SearchConfig(search_algo="gumbel")
    old_node, new_node = _node_pair([], 0.0)
    old_search, new_search = _gumbel_searcher(old_gumbel, config), _gumbel_searcher(new_gumbel, config)
    expected = _action_outcome(old_search._select_interior_action, old_node)
    assert expected[:2] == ("error", ValueError)
    assert _action_outcome(new_search._select_interior_action, new_node) == expected
    for search, node in ((old_search, old_node), (new_search, new_node)):
        search._root, search._budget, search._gumbel = node, 8, np.zeros(0)
    assert _action_outcome(new_search._select_action, new_node) == _action_outcome(old_search._select_action, old_node)


def test_gumbel_invariants_are_rebuilt_when_the_edges_are_replaced() -> None:
    """Invalidate stale cache data after re-expansion or a change in the number of edges."""
    config = SearchConfig(search_algo="gumbel")
    old_search, new_search = _gumbel_searcher(old_gumbel, config), _gumbel_searcher(new_gumbel, config)
    old_node, new_node = _node_pair([(5, 0.7, 0, 0.0), (6, 0.2, 0, 0.0), (7, 0.1, 0, 0.0)], 0.1)
    assert _action_outcome(new_search._select_interior_action, new_node) == ("ok", 5)
    replaced_old, replaced_new = _node_pair([(5, 0.1, 0, 0.0), (6, 0.2, 0, 0.0), (7, 0.7, 0, 0.0)], 0.1)
    old_node.edges, new_node.edges = replaced_old.edges, replaced_new.edges
    expected = _action_outcome(old_search._select_interior_action, old_node)
    assert expected == ("ok", 7)
    assert _action_outcome(new_search._select_interior_action, new_node) == expected
    # Append an edge to the same dictionary object.
    old_node.edges[9] = old_mcts.EdgeStats(prior=0.95)
    new_node.edges[9] = new_mcts.EdgeStats(prior=0.95)
    expected = _action_outcome(old_search._select_interior_action, old_node)
    assert expected == ("ok", 9)
    assert _action_outcome(new_search._select_interior_action, new_node) == expected


def test_gumbel_prepare_root_clears_the_invariants_cache() -> None:
    search = _gumbel_searcher(new_gumbel, SearchConfig(search_algo="gumbel"))
    _, node = _node_pair([(2, 0.6, 0, 0.0), (3, 0.4, 0, 0.0)], 0.0)
    search._select_interior_action(node)
    assert id(node) in search._invariants
    search._prepare_root(node, 8, add_noise=False)
    assert search._invariants == {}


def test_gumbel_invariants_keep_their_node_alive_so_ids_are_never_reused() -> None:
    """Repeatedly drop nodes and check that identity reuse cannot retrieve an unrelated cache entry."""
    config = SearchConfig(search_algo="gumbel")
    rng = np.random.default_rng(1800)
    old_search, new_search = _gumbel_searcher(old_gumbel, config), _gumbel_searcher(new_gumbel, config)
    for case in range(300):
        count = int(rng.integers(2, 20))
        priors = old_gumbel._softmax(rng.normal(0.0, 2.0, size=count)).astype(np.float32).tolist()
        old_node, new_node = _node_pair([(a, p, 0, 0.0) for a, p in enumerate(priors)], 0.0)
        expected = _action_outcome(old_search._select_interior_action, old_node)
        assert _action_outcome(new_search._select_interior_action, new_node) == expected, case
        assert new_search._invariants[id(new_node)].node is new_node
        del old_node, new_node


def test_gumbel_cached_arrays_are_read_only() -> None:
    search = _gumbel_searcher(new_gumbel, SearchConfig(search_algo="gumbel"))
    _, node = _node_pair([(2, 0.6, 0, 0.0), (3, 0.4, 0, 0.0)], 0.0)
    search._select_interior_action(node)
    entry = search._invariants[id(node)]
    assert not entry.logits.flags.writeable and not entry.probs.flags.writeable
    with pytest.raises(ValueError):
        entry.logits[0] = 0.0


def test_gumbel_selection_does_not_modify_the_node() -> None:
    rng = np.random.default_rng(1900)
    search = _gumbel_searcher(new_gumbel, SearchConfig(search_algo="gumbel"))
    for case in range(40):
        spec, node_value = _random_node_spec(rng, case)
        _, node = _node_pair(spec, node_value)
        before = [(a, e.prior, e.visit_count, e.value_sum, e.child) for a, e in node.edges.items()]
        with np.errstate(all="ignore"):
            search._select_interior_action(node)
        assert before == [(a, e.prior, e.visit_count, e.value_sum, e.child) for a, e in node.edges.items()]
    _, root = _node_pair([(a, 0.1 * (a + 1), 0, 0.0) for a in range(4)], 0.2)
    search._prepare_root(root, 8, add_noise=True)
    before = [(a, e.prior, e.visit_count, e.value_sum, e.child) for a, e in root.edges.items()]
    search._select_action(root)
    assert before == [(a, e.prior, e.visit_count, e.value_sum, e.child) for a, e in root.edges.items()]


GUMBEL_CASES = [
    (5, {"num_simulations": 16, "eval_batch_size": 1, "gumbel_max_considered": 8}),
    (5, {"num_simulations": 24, "eval_batch_size": 4, "gumbel_max_considered": 16}),
    (9, {"num_simulations": 32, "eval_batch_size": 4}),
    (9, {"num_simulations": 48, "eval_batch_size": 8, "gumbel_max_considered": 4, "gumbel_scale": 0.0}),
    (9, {"num_simulations": 33, "eval_batch_size": 2, "gumbel_c_visit": 10.0, "gumbel_c_scale": 0.5, "gumbel_scale": 2.0}),
]


@pytest.mark.parametrize(("board_size", "overrides"), GUMBEL_CASES)
def test_gumbel_search_matches_frozen_whole_tree(board_size: int, overrides: dict[str, Any]) -> None:
    config = SearchConfig(search_algo="gumbel", **overrides)
    network = _HashNet(board_size * board_size + 1)
    states = _playout_states(board_size, games=2, seed=board_size + 5)[::9][:5]
    assert len(states) >= 3
    compared = 0
    for index, state in enumerate(states):
        for noise, temperature in ((True, 1.0), (False, 1e-6)):
            old_search = old_gumbel.GumbelMCTS(network, config, np.random.default_rng(index))
            new_search = new_gumbel.GumbelMCTS(network, config, np.random.default_rng(index))
            compared += _compare_searches(old_search, new_search, state, noise, temperature)
    assert compared > 60


def test_gumbel_game_with_budget_override_matches_frozen() -> None:
    """Start a fresh Gumbel tree each move and exercise varying search budgets."""
    config = SearchConfig(search_algo="gumbel", num_simulations=24, eval_batch_size=4)
    network = _HashNet(26)
    old_search = old_gumbel.GumbelMCTS(network, config, np.random.default_rng(3))
    new_search = new_gumbel.GumbelMCTS(network, config, np.random.default_rng(3))
    state = GoState.new_game(board_size=5, komi=0.5)
    for move in range(14):
        budget = 8 if move % 3 == 0 else None
        old_out = old_search.run(state, 1.0, True, root=None, num_simulations=budget)
        new_out = new_search.run(state, 1.0, True, root=None, num_simulations=budget)
        _assert_same_output(old_out, new_out)
        _assert_same_tree(old_out.root, new_out.root)
        state = state.play(int(np.argmax(old_out.policy)))
        if state.is_terminal():
            break
    assert new_search.rng.bit_generator.state == old_search.rng.bit_generator.state


def test_gumbel_search_never_changes_the_priors_that_the_cache_relies_on() -> None:
    """Check that expansion priors stay fixed while root noise is stored separately."""
    config = SearchConfig(search_algo="gumbel", num_simulations=32, eval_batch_size=4)
    state = _playout_states(9, games=1, seed=2)[5]
    search = new_gumbel.GumbelMCTS(_HashNet(82), config, np.random.default_rng(7))
    expanded: list[tuple[Any, dict[int, float]]] = []
    original = search._apply_network_output

    def spy(node: Any, policy_logits: np.ndarray, value: float) -> float:
        result = original(node, policy_logits, value)
        expanded.append((node, {a: e.prior for a, e in node.edges.items()}))  # Priors immediately after expansion.
        return result

    search._apply_network_output = spy  # type: ignore[method-assign]
    search.run(state, temperature=1.0, add_root_noise=True)
    assert len(expanded) > 20  # Check roots and internal nodes, both expanded by _apply_network_output.
    for node, priors in expanded:
        assert {a: e.prior for a, e in node.edges.items()} == priors

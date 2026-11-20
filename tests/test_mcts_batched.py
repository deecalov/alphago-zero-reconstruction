from __future__ import annotations

"""Batched MCTS, virtual loss compensation and network evaluation tests."""

import numpy as np
import pytest

from alphago_zero.config import SearchConfig
from alphago_zero.game import BLACK, GoState
from alphago_zero.mcts import AlphaGoZeroMCTS
from alphago_zero.network import AlphaGoZeroNet


def _make_config(eval_batch_size: int, virtual_loss: float = 1.0, sims: int = 48) -> SearchConfig:
    return SearchConfig(
        num_simulations=sims,
        c_puct=1.5,
        eval_batch_size=eval_batch_size,
        virtual_loss=virtual_loss,
        root_dirichlet_epsilon=0.0,  # Disable noise for repeatability.
    )


@pytest.mark.parametrize("eval_batch_size", [1, 4, 16])
def test_visit_counts_conserved(tiny_model: AlphaGoZeroNet, eval_batch_size: int) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    config = _make_config(eval_batch_size)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0))

    out = mcts.run(state, temperature=1.0, add_root_noise=False)
    assert int(out.visit_counts.sum()) == config.num_simulations
    # Only legal actions may receive visits.
    legal = set(state.legal_actions())
    nonzero = {int(a) for a in np.nonzero(out.visit_counts)[0]}
    assert nonzero.issubset(legal)


def test_batch_size_one_independent_of_virtual_loss(tiny_model: AlphaGoZeroNet) -> None:
    # With batch one the temporary penalty is gone before the next selection.
    state = GoState.new_game(board_size=5, komi=2.5)

    results = []
    for virtual_loss in (0.0, 1.0, 3.0):
        config = _make_config(eval_batch_size=1, virtual_loss=virtual_loss)
        mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(7))
        out = mcts.run(state, temperature=1.0, add_root_noise=False)
        results.append(out.visit_counts.copy())

    assert np.array_equal(results[0], results[1])
    assert np.array_equal(results[1], results[2])


def test_q_values_unaffected_by_virtual_loss_after_search(tiny_model: AlphaGoZeroNet) -> None:
    # All virtual losses must be removed after search.
    state = GoState.new_game(board_size=5, komi=2.5)
    config = _make_config(eval_batch_size=8, virtual_loss=5.0)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(3))

    out = mcts.run(state, temperature=1.0, add_root_noise=False)
    for edge in out.root.edges.values():
        assert -1.0 - 1e-6 <= edge.q_value <= 1.0 + 1e-6


def test_batched_search_is_deterministic(tiny_model: AlphaGoZeroNet) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    config = _make_config(eval_batch_size=4)

    out1 = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(42)).run(
        state, temperature=1.0, add_root_noise=False
    )
    out2 = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(42)).run(
        state, temperature=1.0, add_root_noise=False
    )
    assert np.array_equal(out1.visit_counts, out2.visit_counts)


def test_backup_sign_in_batched_path(
    tiny_model: AlphaGoZeroNet, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Check backup sign in the predict_batch path using a fixed leaf value.
    action_size = 5 * 5 + 1

    def fake_predict_single(_planes: np.ndarray) -> tuple[np.ndarray, float]:
        return np.zeros(action_size, dtype=np.float32), 0.8

    def fake_predict_batch(planes_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = planes_batch.shape[0]
        return (
            np.zeros((batch, action_size), dtype=np.float32),
            np.full(batch, 0.8, dtype=np.float32),
        )

    monkeypatch.setattr(tiny_model, "predict_single", fake_predict_single)
    monkeypatch.setattr(tiny_model, "predict_batch", fake_predict_batch)

    state = GoState.new_game(board_size=5, komi=2.5)
    config = _make_config(eval_batch_size=8, sims=8)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(1))
    out = mcts.run(state, temperature=1.0, add_root_noise=False)

    visited = [edge for edge in out.root.edges.values() if edge.visit_count > 0]
    assert visited, "at least one root edge must be visited"
    for edge in visited:
        # Every depth-one leaf is worth +0.8 to the opponent of the root player.
        assert edge.q_value == pytest.approx(-0.8, abs=1e-6)


def test_leaf_collisions_evaluated_once_and_statistics_exact(
    tiny_model: AlphaGoZeroNet, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One batch of 64 paths over 26 root moves forces leaf collisions.
    action_size = 5 * 5 + 1
    leaf_batches: list[int] = []

    def fake_predict_single(_planes: np.ndarray) -> tuple[np.ndarray, float]:
        return np.zeros(action_size, dtype=np.float32), 0.5

    def fake_predict_batch(planes_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = planes_batch.shape[0]
        leaf_batches.append(batch)
        return (
            np.zeros((batch, action_size), dtype=np.float32),
            np.full(batch, 0.5, dtype=np.float32),
        )

    monkeypatch.setattr(tiny_model, "predict_single", fake_predict_single)
    monkeypatch.setattr(tiny_model, "predict_batch", fake_predict_batch)

    state = GoState.new_game(board_size=5, komi=2.5)
    # Use virtual loss 3.0 so a leftover penalty dominates the valid Q range.
    config = _make_config(eval_batch_size=64, virtual_loss=3.0, sims=64)
    out = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(0)).run(
        state, temperature=1.0, add_root_noise=False
    )

    # Evaluate each distinct leaf once and reuse the result for all paths.
    assert leaf_batches == [len(state.legal_actions())]

    edges = list(out.root.edges.values())
    assert sum(edge.visit_count for edge in edges) == config.num_simulations
    assert max(edge.visit_count for edge in edges) > 1, "this scenario must produce leaf collisions"
    for edge in edges:
        # Each first-ply leaf gives +0.5 to the opponent, fixing the expected edge Q.
        assert edge.q_value == pytest.approx(-0.5, abs=1e-9)


def test_batched_search_prefers_immediately_winning_pass(tiny_model: AlphaGoZeroNet) -> None:
    board = np.zeros((5, 5), dtype=np.int8)
    board[:, 0:3] = BLACK
    state = GoState.from_board(board, to_play=BLACK, komi=2.5, consecutive_passes=1)
    assert state.winner_color() == BLACK

    config = _make_config(eval_batch_size=8, sims=100)
    mcts = AlphaGoZeroMCTS(tiny_model, config, np.random.default_rng(5))
    out = mcts.run(state, temperature=1.0, add_root_noise=False)

    assert int(np.argmax(out.visit_counts)) == state.pass_action


def test_predict_batch_matches_predict_single(tiny_model: AlphaGoZeroNet) -> None:
    # Batched predictions must agree with single-position inference.
    rng = np.random.default_rng(11)
    states = rng.random((4, 17, 5, 5)).astype(np.float32)

    logits_batch, values_batch = tiny_model.predict_batch(states)
    assert logits_batch.shape == (4, 26)
    assert values_batch.shape == (4,)

    for i in range(states.shape[0]):
        logits_one, value_one = tiny_model.predict_single(states[i])
        assert np.allclose(logits_batch[i], logits_one, atol=1e-5)
        assert values_batch[i] == pytest.approx(value_one, abs=1e-5)


def test_predict_batch_rejects_wrong_ndim(tiny_model: AlphaGoZeroNet) -> None:
    with pytest.raises(ValueError):
        tiny_model.predict_batch(np.zeros((17, 5, 5), dtype=np.float32))


def test_model_mode_restored_after_batched_search(tiny_model: AlphaGoZeroNet) -> None:
    # Restore training mode after predict_batch when needed.
    tiny_model.train()
    states = np.zeros((2, 17, 5, 5), dtype=np.float32)
    tiny_model.predict_batch(states)
    assert tiny_model.training

from __future__ import annotations

"""Exercise batched inference and game completion in the parallel self-play driver."""

import numpy as np
import pytest

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.game import BLACK, WHITE
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import (
    _ParallelSelfPlayDriver,
    play_parallel_self_play_games,
)
from alphago_zero.pipeline import run_training_pipeline
from alphago_zero.selfplay import ResignationController


def _make_controller() -> ResignationController:
    return ResignationController(
        threshold=-0.95,
        target_false_positive_rate=0.05,
        adjustment=0.02,
        window=100,
    )


def _run_games(
    tiny_model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    num_games: int,
    parallel_games: int,
    seed: int,
):
    config.training.parallel_games = parallel_games
    rng = np.random.default_rng(seed)
    return play_parallel_self_play_games(
        model=tiny_model,
        config=config,
        rng=rng,
        resignation_controller=_make_controller(),
        num_games=num_games,
    )


def test_parallel_selfplay_produces_requested_games(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    results = _run_games(tiny_model, tiny_config, num_games=5, parallel_games=3, seed=0)
    assert len(results) == 5
    for result in results:
        assert result.winner in (BLACK, WHITE)
        assert result.num_moves > 0
        assert len(result.examples) == result.num_moves


def test_parallel_selfplay_reports_end_reasons(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    results = _run_games(tiny_model, tiny_config, num_games=3, parallel_games=2, seed=3)
    for result in results:
        assert result.ended_by in ("double_pass", "resign", "max_moves")
        assert (result.ended_by == "resign") == result.resigned


def test_parallel_selfplay_examples_are_consistent(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    results = _run_games(tiny_model, tiny_config, num_games=3, parallel_games=2, seed=1)
    action_size = tiny_config.game.board_size ** 2 + 1

    for result in results:
        for example in result.examples:
            assert example.state.shape == (17, 5, 5)
            assert example.policy.shape == (action_size,)
            # The PUCT learning target is a normalized visit distribution.
            assert example.policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert np.all(example.policy >= 0.0)
            assert example.value in (1.0, -1.0)

        # Outcome labels follow the player in the position. The first example belongs to Black.
        first = result.examples[0]
        expected = 1.0 if result.winner == BLACK else -1.0
        assert first.value == expected


def test_parallel_selfplay_is_deterministic_by_seed(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    results_a = _run_games(tiny_model, tiny_config, num_games=4, parallel_games=2, seed=99)
    results_b = _run_games(tiny_model, tiny_config, num_games=4, parallel_games=2, seed=99)

    assert [r.winner for r in results_a] == [r.winner for r in results_b]
    assert [r.num_moves for r in results_a] == [r.num_moves for r in results_b]
    for ra, rb in zip(results_a, results_b):
        for ea, eb in zip(ra.examples, rb.examples):
            assert np.array_equal(ea.state, eb.state)
            assert np.array_equal(ea.policy, eb.policy)
            assert ea.value == eb.value


def test_parallel_selfplay_zero_games(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    assert _run_games(tiny_model, tiny_config, num_games=0, parallel_games=2, seed=0) == []


def test_parallel_slots_capped_by_num_games(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    # Play exactly num_games even when more slots are configured.
    results = _run_games(tiny_model, tiny_config, num_games=2, parallel_games=8, seed=5)
    assert len(results) == 2


def test_parallel_leaf_collisions_evaluated_once_and_statistics_exact(
    tiny_model: AlphaGoZeroNet,
    tiny_config: AlphaGoZeroConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise the parallel driver's own deduplication and backup when leaf requests collide.
    action_size = 5 * 5 + 1
    batches: list[int] = []

    def fake_predict_batch(planes_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = planes_batch.shape[0]
        batches.append(batch)
        return (
            np.zeros((batch, action_size), dtype=np.float32),
            np.full(batch, 0.5, dtype=np.float32),
        )

    monkeypatch.setattr(tiny_model, "predict_batch", fake_predict_batch)
    tiny_model.eval()

    config = tiny_config
    config.training.parallel_games = 1
    config.search.num_simulations = 64
    config.search.eval_batch_size = 64
    config.search.virtual_loss = 3.0  # A leftover penalty would be larger than the expected Q.
    config.search.root_dirichlet_epsilon = 0.0  # Uniform priors without noise.
    config.validate()

    root_edge_stats: list[tuple[int, float]] = []

    class RecordingDriver(_ParallelSelfPlayDriver):
        def _finalize_ready_moves(self) -> None:
            # Read root statistics before the slot advances.
            for slot in self.slots:
                if slot is not None and slot.root.is_expanded and slot.sims_done >= slot.move_sims:
                    root_edge_stats.extend(
                        (edge.visit_count, edge.value_sum) for edge in slot.root.edges.values()
                    )
            super()._finalize_ready_moves()

    driver = RecordingDriver(
        model=tiny_model,
        config=config,
        rng=np.random.default_rng(0),
        resignation_controller=_make_controller(),
        num_games=1,
    )
    first_slot = driver.slots[0]
    assert first_slot is not None
    legal_actions = len(first_slot.state.legal_actions())
    driver._run_round()  # Round one expands one root.
    driver._run_round()  # Round two searches 64 paths in one batch and plays a move.

    assert batches == [1, legal_actions]  # The second round evaluates one row per unique leaf.
    assert sum(visits for visits, _ in root_edge_stats) == config.search.num_simulations
    assert max(visits for visits, _ in root_edge_stats) > 1, "this scenario must produce leaf collisions"
    for visits, value_sum in root_edge_stats:
        assert value_sum / visits == pytest.approx(-0.5, abs=1e-9)


def test_pipeline_with_parallel_selfplay(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.parallel_games = 2
    tiny_config.training.self_play_games_per_iteration = 3
    tiny_config.validate()

    summary = run_training_pipeline(tiny_config)
    assert summary["replay_buffer_size"] > 0
    assert summary["iterations"] == 1

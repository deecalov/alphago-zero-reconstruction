from __future__ import annotations

"""Test league self-play against frozen opponents."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig, NetworkConfig
from alphago_zero.game import BLACK, WHITE
from alphago_zero.mcts import EdgeStats
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import _ParallelSelfPlayDriver, play_parallel_self_play_games
from alphago_zero.selfplay import ResignationController, SelfPlayResult


def _make_controller(config: AlphaGoZeroConfig) -> ResignationController:
    return ResignationController(
        threshold=config.search.resign_threshold,
        target_false_positive_rate=config.search.target_false_positive_rate,
        adjustment=config.search.resignation_adjustment,
        window=config.search.resignation_window,
    )


class _CountingNet(AlphaGoZeroNet):
    """Count calls to each network's batched predictor to check league routing."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__(config)
        self.batch_calls = 0

    def predict_batch(self, states_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.batch_calls += 1
        return super().predict_batch(states_planes)


def _league_setup(
    tiny_config: AlphaGoZeroConfig,
) -> tuple[_CountingNet, _CountingNet, AlphaGoZeroConfig]:
    torch.manual_seed(0)
    best = _CountingNet(tiny_config.network)
    torch.manual_seed(7)  # Use a distinct seed for the frozen champion.
    frozen = _CountingNet(tiny_config.network)
    tiny_config.training.parallel_games = 2
    tiny_config.search.num_simulations = 4
    tiny_config.search.disable_resign_prob = 1.0
    return best, frozen, tiny_config


def test_league_games_use_both_networks(
    tiny_config: AlphaGoZeroConfig, rng: np.random.Generator
) -> None:
    best, frozen, cfg = _league_setup(tiny_config)
    cfg.training.league_prob = 1.0
    cfg.training.league_checkpoints = ["frozen_champion.pt"]  # Placeholder for configuration validation.
    cfg.validate()

    results = play_parallel_self_play_games(
        model=best,
        config=cfg,
        rng=rng,
        resignation_controller=_make_controller(cfg),
        num_games=3,
        league=[("frozen_champion", frozen, cfg.search)],
    )

    assert len(results) == 3
    assert best.batch_calls > 0 and frozen.batch_calls > 0
    for r in results:
        assert r.league_opponent == "frozen_champion"
        assert r.league_color in (BLACK, WHITE)
        assert r.winner in (BLACK, WHITE)
        assert len(r.examples) == r.num_moves > 0
        for ex in r.examples:
            assert ex.state.shape == (17, 5, 5)
            assert float(ex.policy.sum()) == pytest.approx(1.0, abs=1e-4)
            assert ex.value in (1.0, -1.0)


def test_league_prob_zero_never_uses_frozen(
    tiny_config: AlphaGoZeroConfig, rng: np.random.Generator
) -> None:
    best, frozen, cfg = _league_setup(tiny_config)
    cfg.training.league_prob = 0.0

    results = play_parallel_self_play_games(
        model=best,
        config=cfg,
        rng=rng,
        resignation_controller=_make_controller(cfg),
        num_games=2,
        league=[("frozen_champion", frozen, cfg.search)],
    )

    assert frozen.batch_calls == 0
    assert all(r.league_opponent is None for r in results)
    assert all(r.league_color is None for r in results)


def test_league_config_validation(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.league_prob = 0.5
    tiny_config.training.league_checkpoints = []
    tiny_config.training.parallel_games = 4
    with pytest.raises(ValueError):
        tiny_config.validate()

    tiny_config.training.league_checkpoints = ["x.pt"]
    tiny_config.training.parallel_games = 1
    with pytest.raises(ValueError):
        tiny_config.validate()

    tiny_config.training.parallel_games = 4
    tiny_config.validate()  # A valid league configuration passes.


@pytest.mark.parametrize("seed", [0, 1])
def test_frozen_league_search_keeps_full_budget_with_playout_cap(
    tiny_config: AlphaGoZeroConfig, seed: int,
) -> None:
    best, frozen, cfg = _league_setup(tiny_config)
    cfg.training.league_prob = 1.0
    cfg.search.playout_cap_prob = 0.01
    cfg.search.playout_cap_cheap_sims = 1
    cfg.search.eval_batch_size = 1
    frozen_search = replace(cfg.search, num_simulations=11, eval_batch_size=3)
    observed: list[tuple[bool, int, bool]] = []

    class RecordingDriver(_ParallelSelfPlayDriver):
        def _finalize_ready_moves(self) -> None:
            for slot in self.slots:
                if slot is not None and slot.root.is_expanded and slot.sims_done >= slot.move_sims:
                    observed.append((
                        slot.state.to_play == slot.league_color,
                        sum(edge.visit_count for edge in slot.root.edges.values()),
                        slot.move_is_full,
                    ))
            super()._finalize_ready_moves()

    driver = RecordingDriver(
        best, cfg, np.random.default_rng(seed), _make_controller(cfg), 2,
        [("champion", frozen, frozen_search)],
    )
    results = driver.run()
    champion_moves = [entry for entry in observed if entry[0]]
    assert champion_moves
    assert all(visits == 11 and full for _, visits, full in champion_moves)
    assert any(not champion and visits == 1 and not full for champion, visits, full in observed)
    for result in results:
        for index, example in enumerate(result.examples):
            if (BLACK if index % 2 == 0 else WHITE) == result.league_color:
                assert example.policy_weight == 1.0


# ---------------------------------------------------------------------------
# Resignation and calibration in league games
# ---------------------------------------------------------------------------
class _ConstantValueNet(AlphaGoZeroNet):
    """Return uniform policy and a fixed value in every position to control resignation evidence."""

    def __init__(self, config: NetworkConfig, value: float) -> None:
        super().__init__(config)
        self._constant_value = value

    def predict_single(self, state_planes: np.ndarray) -> tuple[np.ndarray, float]:
        action_size = state_planes.shape[-1] * state_planes.shape[-2] + 1
        return np.zeros(action_size, dtype=np.float32), self._constant_value

    def predict_batch(self, states_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = states_planes.shape[0]
        action_size = states_planes.shape[-1] * states_planes.shape[-2] + 1
        return (
            np.zeros((batch, action_size), dtype=np.float32),
            np.full(batch, self._constant_value, dtype=np.float32),
        )


def _run_resign_league(
    tiny_config: AlphaGoZeroConfig,
    *,
    learner_value: float,
    champion_value: float,
    disable_resign_prob: float,
    num_games: int = 8,
    seed: int = 0,
    league_prob: float = 1.0,
    exclude_from_calibration: bool = False,
) -> tuple[list[SelfPlayResult], ResignationController]:
    """Play stub networks with different values, normally using only league games."""
    cfg = tiny_config
    cfg.training.parallel_games = 2
    cfg.training.league_prob = league_prob
    cfg.training.exclude_league_from_resign_calibration = exclude_from_calibration
    cfg.training.league_checkpoints = ["champion.pt"]  # Placeholder for configuration validation.
    cfg.search.num_simulations = 4
    cfg.search.eval_batch_size = 1
    cfg.search.resign_min_visits = 1
    cfg.search.disable_resign_prob = disable_resign_prob
    cfg.validate()

    learner = _ConstantValueNet(cfg.network, learner_value)
    champion = _ConstantValueNet(cfg.network, champion_value)
    controller = _make_controller(cfg)
    results = play_parallel_self_play_games(
        model=learner,
        config=cfg,
        rng=np.random.default_rng(seed),
        resignation_controller=controller,
        num_games=num_games,
        league=[("champion", champion, cfg.search)],
    )
    assert len(results) == num_games
    if league_prob == 1.0:
        assert all(r.league_opponent == "champion" for r in results)
    return results, controller


def test_frozen_champion_never_resigns(tiny_config: AlphaGoZeroConfig) -> None:
    # The champion predicts a loss throughout while the learner remains neutral.
    #
    # If the champion resigned, the learner would be recorded as the winner.
    results, _ = _run_resign_league(
        tiny_config, learner_value=0.0, champion_value=1.0, disable_resign_prob=0.0
    )
    champion_resigns = [r for r in results if r.resigned and r.winner != r.league_color]
    assert champion_resigns == []
    assert all(r.winner in (BLACK, WHITE) for r in results)


def test_champion_hypothetical_resign_does_not_feed_calibration(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    # A frozen champion's hypothetical resignations must not enter calibration.
    results, controller = _run_resign_league(
        tiny_config, learner_value=0.0, champion_value=1.0, disable_resign_prob=1.0
    )
    assert all(r.resignation_disabled for r in results)
    assert any(r.winner == r.league_color for r in results), "the scenario must include champion wins"
    assert not any(r.false_positive_resignation for r in results)
    assert not any(controller.history)


def test_learner_still_resigns_in_league_games(tiny_config: AlphaGoZeroConfig) -> None:
    # The learner can still resign when its value predicts a loss.
    results, _ = _run_resign_league(
        tiny_config, learner_value=1.0, champion_value=0.0, disable_resign_prob=0.0
    )
    assert all(r.resigned for r in results)
    # The learner resigned, so the champion won.
    assert all(r.winner == r.league_color for r in results)


def test_false_positive_is_attributed_to_learner_only(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    # If both would resign, only a learner win counts as a false learner resignation.
    results, controller = _run_resign_league(
        tiny_config, learner_value=1.0, champion_value=1.0, disable_resign_prob=1.0
    )
    assert len(controller.history) == len(results)
    for result in results:
        learner_won = result.winner != result.league_color
        assert result.false_positive_resignation == learner_won


@pytest.mark.parametrize(("exclude", "expected_observations"), [(False, 8), (True, 0)])
def test_exclude_league_from_resign_calibration_flag(
    tiny_config: AlphaGoZeroConfig, exclude: bool, expected_observations: int
) -> None:
    # All eight games reach scoring with both predictors signaling resignation.
    results, controller = _run_resign_league(
        tiny_config,
        learner_value=1.0,
        champion_value=1.0,
        disable_resign_prob=1.0,
        exclude_from_calibration=exclude,
    )
    assert len(controller.history) == expected_observations
    # Excluding a game from calibration must preserve its descriptive false-positive label.
    for result in results:
        assert result.false_positive_resignation == (result.winner != result.league_color)


def test_excluding_league_keeps_regular_games_in_calibration(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    # Exclude only league games from a mixed sample.
    results, controller = _run_resign_league(
        tiny_config,
        learner_value=1.0,
        champion_value=1.0,
        disable_resign_prob=1.0,
        num_games=12,
        seed=3,
        league_prob=0.5,
        exclude_from_calibration=True,
    )
    league_games = [r for r in results if r.league_opponent is not None]
    regular_games = [r for r in results if r.league_opponent is None]
    assert league_games and regular_games, "the scenario must include league games and ordinary self-play"
    assert len(controller.history) == len(regular_games)


def test_exclude_league_flag_requires_league(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.exclude_league_from_resign_calibration = True
    with pytest.raises(ValueError, match="exclude_league_from_resign_calibration"):
        tiny_config.validate()  # Without a league the flag would have no effect.

    tiny_config.training.league_prob = 0.5
    tiny_config.training.league_checkpoints = ["x.pt"]
    tiny_config.training.parallel_games = 4
    tiny_config.validate()  # The flag is valid when league games are enabled.


@pytest.mark.parametrize(
    ("champion_turn", "resignation_disabled", "expect_resigned", "expect_hypothetical"),
    [
        (True, False, False, False),  # The champion does not resign
        (True, True, False, False),  # or record hypothetical resignations.
        (False, False, True, False),  # The learner can resign
        (False, True, False, True),  # or record a hypothetical resignation in a protected game.
    ],
)
def test_resign_decision_belongs_to_side_to_move(
    tiny_config: AlphaGoZeroConfig,
    champion_turn: bool,
    resignation_disabled: bool,
    expect_resigned: bool,
    expect_hypothetical: bool,
) -> None:
    # Use a synthetic losing root to test the current player's decision directly.
    cfg = tiny_config
    cfg.training.parallel_games = 2
    cfg.training.league_prob = 1.0
    cfg.training.league_checkpoints = ["champion.pt"]  # Placeholder for configuration validation.
    cfg.search.resign_min_visits = 8
    cfg.validate()

    controller = _make_controller(cfg)
    driver = _ParallelSelfPlayDriver(
        _ConstantValueNet(cfg.network, 0.0),
        cfg,
        np.random.default_rng(0),
        controller,
        1,
        [("champion", _ConstantValueNet(cfg.network, 0.0), cfg.search)],
    )
    slot = driver.slots[0]
    assert slot is not None and slot.league_search is not None

    # Black plays first. Assign the champion either Black or White.
    slot.league_color = BLACK if champion_turn else WHITE
    slot.resignation_disabled = resignation_disabled
    slot.move_is_full = True
    slot.move_sims = slot.sims_done = 8

    action = 12
    slot.root.is_expanded = True
    slot.root.node_value = -1.0
    slot.root.edges = {action: EdgeStats(prior=1.0, visit_count=8, value_sum=-8.0)}

    driver._finalize_ready_moves()

    if expect_resigned:
        assert driver.slots[0] is None
        assert len(driver.results) == 1
        result = driver.results[0]
        # The learner resigned, so the champion won.
        assert result.resigned and result.winner == result.league_color
        return

    assert driver.results == []
    survivor = driver.slots[0]
    assert survivor is not None and not survivor.resigned
    assert survivor.moves_played == [action]
    # Only learner-side hypothetical resignations belong in the record.
    assert survivor.hypothetical_resign_players == ({BLACK} if expect_hypothetical else set())
    assert len(controller.history) == 0  # Update calibration at game end.

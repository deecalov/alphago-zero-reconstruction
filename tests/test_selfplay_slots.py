from __future__ import annotations

"""Compare requested slot counts, actual slot counts and measured utilization in parallel self-play."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import (
    ParallelSelfPlayStats,
    _ParallelSelfPlayDriver,
    play_parallel_self_play_games,
)
from alphago_zero.pipeline import run_training_pipeline
from alphago_zero.selfplay import ResignationController, SelfPlayResult
from examples import gpu_train


def _controller() -> ResignationController:
    return ResignationController(
        threshold=-0.95, target_false_positive_rate=0.05, adjustment=0.02, window=100
    )


def _play(
    model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    *,
    num_games: int,
    parallel_games: int,
    seed: int = 11,
    stats: ParallelSelfPlayStats | None = None,
) -> list[SelfPlayResult]:
    config.training.parallel_games = parallel_games
    return play_parallel_self_play_games(
        model=model,
        config=config,
        rng=np.random.default_rng(seed),
        resignation_controller=_controller(),
        num_games=num_games,
        stats=stats,
    )


def _assert_same_games(a: list[SelfPlayResult], b: list[SelfPlayResult]) -> None:
    assert [r.moves for r in a] == [r.moves for r in b]
    assert [r.winner for r in a] == [r.winner for r in b]
    for x, y in zip(a, b):
        for ex, ey in zip(x.examples, y.examples):
            assert np.array_equal(ex.state, ey.state)
            assert np.array_equal(ex.policy, ey.policy)
            assert ex.value == ey.value


def _slot_warnings(config: AlphaGoZeroConfig) -> list[str]:
    """Collect only slot-capacity warnings emitted by validation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.validate()
    return [
        str(item.message)
        for item in caught
        if issubclass(item.category, UserWarning) and "parallel_games=" in str(item.message)
    ]


# ----------------------------------------------------------------------
# Effective slots from configuration
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("parallel_games", "games", "workers", "per_process", "effective"),
    [
        (32, 24, 1, 24, 24),  # The recipe requests 32 slots but can fill only 24.
        (16, 16, 1, 16, 16),  # Experiment branch requesting 16 slots.
        (32, 16, 1, 16, 16),  # The branch requesting 32 has the same effective capacity here.
        (16, 24, 1, 24, 16),  # Fewer slots than games needs no clamp.
        (16, 72, 3, 24, 16),
        (32, 72, 3, 24, 24),
        (8, 25, 3, 9, 8),  # The first worker gets the extra game, producing 9 + 8 + 8.
        (16, 25, 3, 9, 9),
        (1, 8, 1, 8, 1),
        (4, 0, 1, 0, 1),  # No games run, while the configured minimum remains one.
        (4, 2, 4, 1, 1),
    ],
)
def test_effective_parallel_games_is_capped_by_the_games_of_one_process(
    tiny_config: AlphaGoZeroConfig,
    parallel_games: int,
    games: int,
    workers: int,
    per_process: int,
    effective: int,
) -> None:
    training = tiny_config.training
    training.parallel_games = parallel_games
    training.self_play_games_per_iteration = games
    training.selfplay_workers = workers
    assert training.self_play_games_per_process() == per_process
    assert training.effective_parallel_games() == effective


def test_effective_parallel_games_matches_what_the_driver_opens(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    """Check that reported capacity agrees with the actual driver."""
    for parallel_games, games in [(8, 2), (2, 5), (3, 3), (1, 4)]:
        tiny_config.training.parallel_games = parallel_games
        tiny_config.training.self_play_games_per_iteration = games
        driver = _ParallelSelfPlayDriver(
            model=tiny_model,
            config=tiny_config,
            rng=np.random.default_rng(0),
            resignation_controller=_controller(),
            num_games=games,
        )
        assert len(driver.slots) == tiny_config.training.effective_parallel_games()
        assert driver.stats.slots == len(driver.slots)


# ----------------------------------------------------------------------
# Validation warnings
# ----------------------------------------------------------------------
def test_validate_warns_when_parallel_games_exceeds_the_games(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.parallel_games = 32
    tiny_config.training.self_play_games_per_iteration = 24
    messages = _slot_warnings(tiny_config)
    assert len(messages) == 1
    assert "parallel_games=32" in messages[0]
    assert "Active slots: 24 instead of 32" in messages[0]


def test_validate_warning_accounts_for_workers(tiny_config: AlphaGoZeroConfig) -> None:
    """With 24 games on three workers, a request for 16 slots is capped at eight per worker."""
    tiny_config.training.parallel_games = 16
    tiny_config.training.self_play_games_per_iteration = 24
    tiny_config.training.selfplay_workers = 3
    messages = _slot_warnings(tiny_config)
    assert len(messages) == 1
    assert "Active slots: 8 instead of 16" in messages[0]
    assert "24 games across 3 worker(s)" in messages[0]


@pytest.mark.parametrize(
    ("parallel_games", "games", "workers"),
    [
        (24, 24, 1),  # One game for each slot.
        (16, 24, 1),
        (16, 72, 3),
        (24, 72, 3),
        (1, 8, 1),  # Sequential self-play.
        (4, 0, 1),  # No slot warning when self-play is disabled.
    ],
)
def test_validate_is_silent_when_nothing_is_clamped(
    tiny_config: AlphaGoZeroConfig, parallel_games: int, games: int, workers: int
) -> None:
    tiny_config.training.parallel_games = parallel_games
    tiny_config.training.self_play_games_per_iteration = games
    tiny_config.training.selfplay_workers = workers
    assert _slot_warnings(tiny_config) == []


def test_clamped_config_stays_valid(tiny_config: AlphaGoZeroConfig) -> None:
    """Clamping reports a warning while accepting historical recipes and snapshots."""
    tiny_config.training.parallel_games = 32
    tiny_config.training.self_play_games_per_iteration = 24
    with pytest.warns(UserWarning, match="parallel_games=32"):
        tiny_config.validate()


def test_parallel_games_above_the_games_is_the_same_configuration(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    """Three and eight requested slots produce the same trajectory when only three games are available."""
    exact = _play(tiny_model, tiny_config, num_games=3, parallel_games=3)
    more = _play(tiny_model, tiny_config, num_games=3, parallel_games=8)
    _assert_same_games(exact, more)
    # Fewer slots than games changes the schedule.
    fewer = _play(tiny_model, tiny_config, num_games=3, parallel_games=2)
    assert [r.moves for r in fewer] != [r.moves for r in exact]


# ----------------------------------------------------------------------
# Slot utilization counters
# ----------------------------------------------------------------------
def test_stats_do_not_change_the_games(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    plain = _play(tiny_model, tiny_config, num_games=4, parallel_games=2)
    counted = _play(
        tiny_model, tiny_config, num_games=4, parallel_games=2, stats=ParallelSelfPlayStats()
    )
    _assert_same_games(plain, counted)


def test_stats_match_an_independent_count_of_rounds_and_batches(
    tiny_model: AlphaGoZeroNet,
    tiny_config: AlphaGoZeroConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare against an independent per-round count of occupied slots and inference positions."""
    batches: list[int] = []
    real_predict_batch = tiny_model.predict_batch

    def spying_predict_batch(planes_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batches.append(int(planes_batch.shape[0]))
        return real_predict_batch(planes_batch)

    monkeypatch.setattr(tiny_model, "predict_batch", spying_predict_batch)

    active_per_round: list[int] = []

    class CountingDriver(_ParallelSelfPlayDriver):
        def _run_round(self) -> None:
            active_per_round.append(sum(slot is not None for slot in self.slots))
            super()._run_round()

    tiny_config.training.parallel_games = 3
    driver = CountingDriver(
        model=tiny_model,
        config=tiny_config,
        rng=np.random.default_rng(4),
        resignation_controller=_controller(),
        num_games=5,
    )
    results = driver.run()
    stats = driver.stats

    assert len(results) == 5
    assert stats.slots == 3
    assert stats.rounds == len(active_per_round)
    assert stats.slot_rounds == 3 * len(active_per_round)
    assert stats.active_slot_rounds == sum(active_per_round)
    assert stats.network_calls == len(batches)
    assert stats.positions == sum(batches)
    assert stats.fill == pytest.approx(sum(active_per_round) / (3 * len(active_per_round)))
    assert stats.mean_batch == pytest.approx(sum(batches) / len(batches))
    # Five games and three slots should exercise both full occupancy and a partially empty tail.
    assert max(active_per_round) == 3
    assert min(active_per_round) < 3
    assert stats.fill is not None and 0.0 < stats.fill < 1.0


def test_a_single_slot_is_always_full(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    """The next game immediately fills a single slot when its predecessor finishes."""
    stats = ParallelSelfPlayStats()
    _play(tiny_model, tiny_config, num_games=3, parallel_games=1, stats=stats)
    assert stats.slots == 1
    assert stats.fill == 1.0
    assert stats.active_slot_rounds == stats.slot_rounds == stats.rounds


def test_stats_slots_report_the_clamped_number(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    stats = ParallelSelfPlayStats()
    _play(tiny_model, tiny_config, num_games=2, parallel_games=8, stats=stats)
    assert stats.slots == 2


def test_stats_accumulate_over_calls(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    first, second, both = ParallelSelfPlayStats(), ParallelSelfPlayStats(), ParallelSelfPlayStats()
    _play(tiny_model, tiny_config, num_games=3, parallel_games=2, seed=1, stats=first)
    _play(tiny_model, tiny_config, num_games=2, parallel_games=2, seed=2, stats=second)
    _play(tiny_model, tiny_config, num_games=3, parallel_games=2, seed=1, stats=both)
    _play(tiny_model, tiny_config, num_games=2, parallel_games=2, seed=2, stats=both)
    for name in ("slots", "rounds", "slot_rounds", "active_slot_rounds", "network_calls", "positions"):
        assert getattr(both, name) == getattr(first, name) + getattr(second, name)


def test_zero_games_leave_the_stats_untouched(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    stats = ParallelSelfPlayStats()
    assert _play(tiny_model, tiny_config, num_games=0, parallel_games=4, stats=stats) == []
    assert stats == ParallelSelfPlayStats()
    assert stats.fill is None and stats.mean_batch is None
    assert stats.to_dict() == {
        "slots": 0,
        "rounds": 0,
        "fill": None,
        "network_calls": 0,
        "positions": 0,
        "mean_batch": None,
    }


def test_to_dict_rounds_the_ratios() -> None:
    stats = ParallelSelfPlayStats(
        slots=3, rounds=7, slot_rounds=21, active_slot_rounds=16, network_calls=7, positions=50
    )
    assert stats.to_dict() == {
        "slots": 3,
        "rounds": 7,
        "fill": round(16 / 21, 4),
        "network_calls": 7,
        "positions": 50,
        "mean_batch": round(50 / 7, 2),
    }


# ----------------------------------------------------------------------
# Pipeline and CLI
# ----------------------------------------------------------------------
def _first_iteration(summary: dict[str, object]) -> dict[str, object]:
    metrics = json.loads(Path(str(summary["metrics_log_path"])).read_text(encoding="utf-8"))
    row: dict[str, object] = metrics["iterations"][0]
    return row


def test_pipeline_logs_slot_usage_for_parallel_selfplay(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.parallel_games = 4
    tiny_config.training.self_play_games_per_iteration = 3
    with pytest.warns(UserWarning, match="Active slots: 3 instead of 4"):
        summary = run_training_pipeline(tiny_config)
    slots = _first_iteration(summary)["selfplay_slots"]
    assert isinstance(slots, dict)
    assert set(slots) == {
        "parallel_games", "slots", "rounds", "fill", "network_calls", "positions", "mean_batch",
    }
    assert slots["parallel_games"] == 4  # Requested.
    assert slots["slots"] == 3  # Actually opened.
    assert 0.0 < slots["fill"] <= 1.0
    assert slots["rounds"] > 0 and slots["positions"] >= slots["network_calls"] > 0


def test_pipeline_logs_no_slot_usage_for_sequential_selfplay(tiny_config: AlphaGoZeroConfig) -> None:
    assert tiny_config.training.parallel_games == 1
    summary = run_training_pipeline(tiny_config)
    assert _first_iteration(summary)["selfplay_slots"] is None


class _PipelineReached(Exception):
    pass


def _header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *cli_args: str
) -> str:
    def fake_pipeline(config: AlphaGoZeroConfig) -> dict[str, object]:
        raise _PipelineReached

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_train, "run_training_pipeline", fake_pipeline)
    monkeypatch.setattr(
        sys, "argv", ["gpu_train.py", "--board", "5", "--channels", "8", "--res-blocks", "1", *cli_args]
    )
    with pytest.raises(_PipelineReached):
        gpu_train.main()
    return capsys.readouterr().out


def test_header_prints_the_effective_slots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _header(monkeypatch, capsys, "--parallel-games", "16", "--self-play-games", "24")
    assert "PARALLEL_GAMES 16 SELFPLAY_GAMES 24 SELFPLAY_SLOTS 16" in output


def test_header_shows_the_clamp_of_the_g1_recipe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requesting 32 slots for 24 games warns and reports 24 effective slots."""
    with pytest.warns(UserWarning, match="Active slots: 24 instead of 32"):
        output = _header(monkeypatch, capsys, "--parallel-games", "32", "--self-play-games", "24")
    assert "PARALLEL_GAMES 32 SELFPLAY_GAMES 24 SELFPLAY_SLOTS 24" in output


def test_header_counts_slots_per_worker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.warns(UserWarning, match="Active slots: 8 instead of 16"):
        output = _header(
            monkeypatch, capsys,
            "--selfplay-workers", "3", "--parallel-games", "16", "--self-play-games", "24",
        )
    assert "SELFPLAY_WORKERS 3 PARALLEL_GAMES 16 SELFPLAY_GAMES 24 SELFPLAY_SLOTS 8" in output

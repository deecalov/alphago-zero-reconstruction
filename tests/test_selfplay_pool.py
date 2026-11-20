from __future__ import annotations

"""Persistent multiprocessing self-play, calibration, error propagation and determinism tests."""

import copy
import json
import multiprocessing as mp
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from alphago_zero import selfplay_pool
from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import ParallelSelfPlayStats, play_parallel_self_play_games
from alphago_zero.pipeline import run_training_pipeline
from alphago_zero.selfplay import ResignationController, SelfPlayResult
from alphago_zero.selfplay_pool import (
    SelfPlayPool,
    play_worker_share,
    recording_controller,
    split_games,
    worker_seed,
)
from examples import gpu_train


def _assert_same_results(first: list[SelfPlayResult], second: list[SelfPlayResult]) -> None:
    """Compare moves, outcomes and all training arrays bit for bit across two game lists."""
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.moves == b.moves
        assert (a.winner, a.num_moves, a.resigned, a.ended_by) == (b.winner, b.num_moves, b.resigned, b.ended_by)
        assert a.resignation_disabled == b.resignation_disabled
        assert a.false_positive_resignation == b.false_positive_resignation
        assert len(a.examples) == len(b.examples)
        for x, y in zip(a.examples, b.examples):
            assert np.array_equal(x.state, y.state)
            assert np.array_equal(x.policy, y.policy)
            assert (x.value, x.score, x.policy_weight) == (y.value, y.score, y.policy_weight)
            assert (x.ownership is None) == (y.ownership is None)
            if x.ownership is not None and y.ownership is not None:
                assert np.array_equal(x.ownership, y.ownership)


def _weights(model: AlphaGoZeroNet) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}


@pytest.fixture
def pool_config(tiny_config: AlphaGoZeroConfig) -> AlphaGoZeroConfig:
    """Create a tiny parallel configuration that plays deterministic games through scoring."""
    tiny_config.training.parallel_games = 2
    tiny_config.training.self_play_games_per_iteration = 3
    tiny_config.training.selfplay_workers = 2
    tiny_config.validate()
    return tiny_config


# ----------------------------------------------------------------------
# Game partitioning and worker seeds
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("games", "workers", "expected"),
    [(24, 3, [8, 8, 8]), (25, 3, [9, 8, 8]), (2, 4, [1, 1, 0, 0]), (0, 2, [0, 0]), (5, 1, [5])],
)
def test_split_games_is_even_and_complete(games: int, workers: int, expected: list[int]) -> None:
    assert split_games(games, workers) == expected


def test_split_games_always_sums_to_the_total() -> None:
    for games in range(0, 40):
        for workers in range(1, 9):
            shares = split_games(games, workers)
            assert sum(shares) == games
            assert max(shares) - min(shares) <= 1
            assert shares == sorted(shares, reverse=True)  # Assign remainder games to the first workers.


@pytest.mark.parametrize(("games", "workers"), [(4, 0), (4, -1), (-1, 2)])
def test_split_games_rejects_invalid_arguments(games: int, workers: int) -> None:
    with pytest.raises(ValueError):
        split_games(games, workers)


def test_worker_seed_streams_differ_by_every_component() -> None:
    def draws(seed: list[int]) -> list[int]:
        return np.random.default_rng(seed).integers(0, 2**31, size=4).tolist()

    base = draws(worker_seed(7, 3, 0))
    assert base == draws(worker_seed(7, 3, 0))  # The same seed triple gives the same stream.
    assert base != draws(worker_seed(7, 3, 1))  # Change the worker index.
    assert base != draws(worker_seed(7, 4, 0))  # Change the iteration.
    assert base != draws(worker_seed(8, 3, 0))  # Change the run seed.


def test_recording_controller_keeps_the_threshold_and_all_observations() -> None:
    """The ordinary controller adjusts after enough observations. The recording controller must stay fixed."""
    controller = recording_controller(threshold=-0.9, target_false_positive_rate=0.05)
    flags = [True] * 300 + [False] * 300
    for flag in flags:
        controller.observe_false_positive(flag)
    assert controller.threshold == -0.9
    assert list(controller.history) == flags


def test_recording_controller_does_not_clip_an_out_of_range_threshold() -> None:
    """Play with the exact supplied threshold even when it falls outside the ordinary adjustment clamp."""
    controller = recording_controller(threshold=-1.0, target_false_positive_rate=0.05)
    for _ in range(50):
        controller.observe_false_positive(True)
    assert controller.threshold == -1.0


# ----------------------------------------------------------------------
# One worker's share executed in the test process
# ----------------------------------------------------------------------
def test_play_worker_share_is_reproducible_and_streams_are_independent(
    pool_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    def share(seed: list[int]) -> list[SelfPlayResult]:
        results, _observations = play_worker_share(
            tiny_model, pool_config, threshold=-0.95, seed=seed, num_games=2
        )
        return results

    first = share(worker_seed(7, 1, 0))
    _assert_same_results(first, share(worker_seed(7, 1, 0)))
    assert len(first) == 2
    other_worker = share(worker_seed(7, 1, 1))
    other_iteration = share(worker_seed(7, 2, 0))
    assert [r.moves for r in first] != [r.moves for r in other_worker]
    assert [r.moves for r in first] != [r.moves for r in other_iteration]


def test_play_worker_share_returns_the_observations_an_ordinary_controller_would_get(
    monkeypatch: pytest.MonkeyPatch, pool_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    """Collect exactly the false-positive observations the ordinary controller would receive."""
    pool_config.search.disable_resign_prob = 1.0
    pool_config.search.resign_min_visits = 1
    seed = worker_seed(7, 1, 0)

    calls: list[bool] = []
    original = ResignationController.observe_false_positive

    def spy(self: ResignationController, flag: bool) -> None:
        calls.append(bool(flag))
        original(self, flag)

    monkeypatch.setattr(ResignationController, "observe_false_positive", spy)
    ordinary = ResignationController(
        threshold=-0.05, target_false_positive_rate=0.05, adjustment=0.0, window=100
    )
    reference_results = play_parallel_self_play_games(
        tiny_model, pool_config, np.random.default_rng(seed), ordinary, num_games=6
    )
    reference_calls = list(calls)
    calls.clear()

    results, observations = play_worker_share(
        tiny_model, pool_config, threshold=-0.05, seed=seed, num_games=6
    )
    _assert_same_results(results, reference_results)
    assert observations == reference_calls
    assert len(observations) >= 2
    assert observations.count(True) == sum(r.false_positive_resignation for r in results)


# ----------------------------------------------------------------------
# Worker protocol without starting a subprocess
# ----------------------------------------------------------------------
def test_worker_loop_plays_a_share_and_stops(
    pool_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    parent, child = mp.Pipe()
    thread = threading.Thread(target=selfplay_pool._worker_main, args=(child, pool_config, 1))
    thread.start()
    try:
        assert parent.recv() == ("ready",)
        parent.send(("play", _weights(tiny_model), -0.95, 5, 2))
        kind, results, observations, seconds, stats = parent.recv()
        assert kind == "ok" and seconds >= 0.0
        # Two games fill both slots of this worker.
        assert stats.slots == 2 and stats.rounds > 0 and stats.positions > 0
        reference, reference_observations = play_worker_share(
            tiny_model, pool_config, threshold=-0.95, seed=worker_seed(pool_config.training.seed, 5, 1), num_games=2
        )
        _assert_same_results(results, reference)
        assert observations == reference_observations
        parent.send(("stop",))
    finally:
        thread.join(timeout=30.0)
        parent.close()
    assert not thread.is_alive()


def test_worker_loop_reports_an_error_and_exits(
    pool_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    """An incompatible weight shape must send a traceback to the parent process."""
    parent, child = mp.Pipe()
    thread = threading.Thread(target=selfplay_pool._worker_main, args=(child, pool_config, 0))
    thread.start()
    try:
        assert parent.recv() == ("ready",)
        broken = _weights(tiny_model)
        broken["input_block.conv.weight"] = np.zeros((1, 1, 1, 1), dtype=np.float32)
        parent.send(("play", broken, -0.95, 1, 1))
        kind, text = parent.recv()
        assert kind == "error"
        assert "size mismatch" in text and "Traceback" in text
    finally:
        thread.join(timeout=30.0)
        parent.close()
    assert not thread.is_alive()


# ----------------------------------------------------------------------
# Real process pool
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_pool_matches_the_in_process_reference_and_is_reproducible(
    pool_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    with SelfPlayPool(pool_config) as pool:
        assert pool.num_workers == 2
        outcome = pool.play(tiny_model, threshold=-0.95, iteration=4)
        again = pool.play(tiny_model, threshold=-0.95, iteration=4)
        other = pool.play(tiny_model, threshold=-0.95, iteration=5)
        single = pool.play(tiny_model, threshold=-0.95, iteration=4, num_games=1)
        nothing = pool.play(tiny_model, threshold=-0.95, iteration=4, num_games=0)

    # Split three games 2 + 1 and collect worker zero's games first.
    seed = pool_config.training.seed
    reference: list[SelfPlayResult] = []
    reference_observations: list[bool] = []
    reference_stats = ParallelSelfPlayStats()
    for index, share in enumerate([2, 1]):
        results, observations = play_worker_share(
            tiny_model,
            pool_config,
            threshold=-0.95,
            seed=worker_seed(seed, 4, index),
            num_games=share,
            stats=reference_stats,
        )
        reference.extend(results)
        reference_observations.extend(observations)
    _assert_same_results(outcome.results, reference)
    assert outcome.observations == reference_observations
    # Aggregate slot counters from the worker with two games and the worker with one.
    assert outcome.stats == reference_stats
    assert outcome.stats.slots == 3
    assert nothing.stats == ParallelSelfPlayStats()
    assert len(outcome.worker_seconds) == 2 and all(s > 0.0 for s in outcome.worker_seconds)

    _assert_same_results(again.results, outcome.results)  # Repeating the same iteration reproduces its games.
    assert [r.moves for r in other.results] != [r.moves for r in outcome.results]

    # With one game the other worker remains idle and records zero game time.
    lone, _ = play_worker_share(
        tiny_model, pool_config, threshold=-0.95, seed=worker_seed(seed, 4, 0), num_games=1
    )
    _assert_same_results(single.results, lone)
    assert single.worker_seconds[1] == 0.0
    assert nothing.results == [] and nothing.observations == []


@pytest.mark.slow
def test_pool_uses_the_weights_it_was_given(
    pool_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    """Workers must play with the weights sent by the parent."""
    torch.manual_seed(123)
    other_model = AlphaGoZeroNet(tiny_model.config)
    with SelfPlayPool(pool_config, num_workers=1) as pool:
        with_first = pool.play(tiny_model, threshold=-0.95, iteration=1, num_games=2)
        with_second = pool.play(other_model, threshold=-0.95, iteration=1, num_games=2)
    reference, _ = play_worker_share(
        other_model, pool_config, threshold=-0.95, seed=worker_seed(pool_config.training.seed, 1, 0), num_games=2
    )
    _assert_same_results(with_second.results, reference)
    assert [r.moves for r in with_first.results] != [r.moves for r in with_second.results]


@pytest.mark.slow
def test_pool_turns_a_worker_failure_into_a_runtime_error_and_closes_twice(
    pool_config: AlphaGoZeroConfig,
) -> None:
    wide = copy.deepcopy(pool_config)
    wide.network.channels = 16  # Use parent weights different from worker initialization.
    torch.manual_seed(0)
    wrong_model = AlphaGoZeroNet(wide.network)
    pool = SelfPlayPool(pool_config, num_workers=1)
    try:
        with pytest.raises(RuntimeError, match="self-play worker 0 failed"):
            pool.play(wrong_model, threshold=-0.95, iteration=1, num_games=1)
    finally:
        pool.close()
        pool.close()  # Closing twice is safe.


def test_pool_rejects_a_non_positive_worker_count(pool_config: AlphaGoZeroConfig) -> None:
    with pytest.raises(ValueError, match="num_workers"):
        SelfPlayPool(pool_config, num_workers=0)


# ----------------------------------------------------------------------
# Pipeline, configuration and CLI integration
# ----------------------------------------------------------------------
def _stable(iteration_log: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(iteration_log)
    cleaned.pop("timings", None)
    return cleaned


@pytest.mark.slow
def test_pipeline_with_two_workers_is_reproducible(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    logs: list[list[dict[str, Any]]] = []
    for name in ("first", "second"):
        config = copy.deepcopy(tiny_config)
        config.runtime.work_dir = tmp_path / name
        config.training.parallel_games = 2
        config.training.self_play_games_per_iteration = 3
        config.training.selfplay_workers = 2
        config.training.num_iterations = 2
        summary = run_training_pipeline(config)
        metrics = json.loads(Path(summary["metrics_log_path"]).read_text(encoding="utf-8"))
        logs.append(metrics["iterations"])

    first, second = logs
    assert [_stable(row) for row in first] == [_stable(row) for row in second]
    for row in first:
        assert sum(row["game_endings"].values()) == 3
        assert row["examples_generated"] > 0
        assert len(row["timings"]["selfplay_worker_sec"]) == 2
        # Three games and two workers open two slots and one slot respectively.
        assert row["selfplay_slots"]["parallel_games"] == 2
        assert row["selfplay_slots"]["slots"] == 3


def test_selfplay_workers_must_be_positive(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.selfplay_workers = 0
    with pytest.raises(ValueError, match="selfplay_workers"):
        tiny_config.validate()


class _PipelineReached(Exception):
    pass


def _cli_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *cli_args: str
) -> tuple[AlphaGoZeroConfig, str]:
    captured: dict[str, AlphaGoZeroConfig] = {}

    def fake_pipeline(config: AlphaGoZeroConfig) -> dict[str, object]:
        captured["config"] = config
        raise _PipelineReached

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_train, "run_training_pipeline", fake_pipeline)
    monkeypatch.setattr(
        sys, "argv", ["gpu_train.py", "--board", "5", "--channels", "8", "--res-blocks", "1", *cli_args]
    )
    with pytest.raises(_PipelineReached):
        gpu_train.main()
    return captured["config"], capsys.readouterr().out


def test_selfplay_workers_flag_defaults_to_one_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _cli_config(monkeypatch, capsys)
    assert config.training.selfplay_workers == 1
    assert "SELFPLAY_WORKERS 1 PARALLEL_GAMES 4 SELFPLAY_GAMES 4" in output


def test_selfplay_workers_flag_reaches_the_config_and_header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _cli_config(
        monkeypatch, capsys, "--selfplay-workers", "3", "--parallel-games", "16", "--self-play-games", "72"
    )
    assert config.training.selfplay_workers == 3
    assert "SELFPLAY_WORKERS 3 PARALLEL_GAMES 16 SELFPLAY_GAMES 72" in output


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "abc"])
def test_selfplay_workers_flag_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", f"--selfplay-workers={value}"])
    with pytest.raises(SystemExit) as raised:
        gpu_train.main()
    assert raised.value.code == 2
    message = capsys.readouterr().err.strip().splitlines()[-1]
    assert "--selfplay-workers" in message and "expected" in message

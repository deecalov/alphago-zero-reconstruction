from __future__ import annotations

"""Held-out games, generalization metrics and snapshot compatibility tests."""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.pipeline import holdout_game_flags, run_training_pipeline
from alphago_zero.replay_buffer import ReplayBuffer, TrainingExample
from alphago_zero.train import evaluate_on_buffer
from examples import gpu_train


def _filled_buffer(model: AlphaGoZeroNet, count: int, seed: int = 3) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=count, board_size=model.board_size, rng=rng)
    for index in range(count):
        raw = rng.random(model.action_size).astype(np.float32)
        buffer.add(
            TrainingExample(
                state=rng.integers(0, 2, size=(17, model.board_size, model.board_size)).astype(np.float32),
                policy=raw / raw.sum(),
                value=float(rng.choice([-1.0, 1.0])),
                policy_weight=0.0 if index % 3 == 0 else 1.0,
            )
        )
    return buffer


# ----------------------------------------------------------------------
# Whole-game holdout sampling
# ----------------------------------------------------------------------
def test_zero_fraction_holds_nothing_out_and_creates_no_generator(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a zero holdout fraction must not create a generator")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    assert holdout_game_flags(tiny_config, iteration=5, num_games=4) == [False] * 4
    assert holdout_game_flags(tiny_config, iteration=5, num_games=0) == []


def test_holdout_lottery_is_reproducible_and_depends_on_the_iteration(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    tiny_config.training.holdout_game_fraction = 0.5
    first = holdout_game_flags(tiny_config, iteration=3, num_games=64)
    assert first == holdout_game_flags(tiny_config, iteration=3, num_games=64)
    assert first != holdout_game_flags(tiny_config, iteration=4, num_games=64)
    other_seed = copy.deepcopy(tiny_config)
    other_seed.training.seed += 1
    assert first != holdout_game_flags(other_seed, iteration=3, num_games=64)
    assert all(isinstance(flag, bool) for flag in first)


@pytest.mark.parametrize("fraction", [0.05, 0.25, 0.5])
def test_holdout_lottery_matches_the_requested_fraction(
    tiny_config: AlphaGoZeroConfig, fraction: float
) -> None:
    tiny_config.training.holdout_game_fraction = fraction
    flags = [
        flag
        for iteration in range(1, 201)
        for flag in holdout_game_flags(tiny_config, iteration, num_games=24)
    ]
    # Across 4,800 trials, three standard deviations of the fraction are at most 0.022.
    assert abs(sum(flags) / len(flags) - fraction) < 0.025


# ----------------------------------------------------------------------
# Evaluate a buffer without training
# ----------------------------------------------------------------------
def test_evaluate_on_buffer_matches_a_manual_computation(tiny_model: AlphaGoZeroNet) -> None:
    buffer = _filled_buffer(tiny_model, count=20)
    measured = evaluate_on_buffer(
        tiny_model, buffer, positions=20, rng=np.random.default_rng(1), chunk_size=7
    )
    assert measured is not None and measured["positions"] == 20.0

    batch = buffer.sample_batch_full(batch_size=20, augment=False, rng=np.random.default_rng(1))
    tiny_model.eval()
    with torch.no_grad():
        logits, value = tiny_model(torch.from_numpy(batch.states))
        per_sample = -(torch.from_numpy(batch.policies) * torch.log_softmax(logits, dim=1)).sum(dim=1)
        weights = torch.from_numpy(batch.policy_weights)
        expected_ce = float((per_sample * weights).sum() / weights.sum())
        expected_mse = float(((value.squeeze(1) - torch.from_numpy(batch.values)) ** 2).mean())
    assert measured["policy_ce"] == pytest.approx(expected_ce, rel=1e-5)
    assert measured["value_mse"] == pytest.approx(expected_mse, rel=1e-5)


def test_evaluate_on_buffer_ignores_policy_targets_of_cheap_search_moves(
    tiny_model: AlphaGoZeroNet,
) -> None:
    """Changing zero-weight policy targets must leave policy cross-entropy unchanged."""
    buffer = _filled_buffer(tiny_model, count=12)
    base = evaluate_on_buffer(tiny_model, buffer, positions=12, rng=np.random.default_rng(1))
    assert base is not None

    spike = np.zeros(tiny_model.action_size, dtype=np.float32)
    spike[0] = 1.0
    cheap = _filled_buffer(tiny_model, count=12)
    full = _filled_buffer(tiny_model, count=12)
    for index in range(12):
        target = cheap if index % 3 == 0 else full  # Every third example has policy weight zero.
        target._buffer[index].policy = spike.copy()
    same = evaluate_on_buffer(tiny_model, cheap, positions=12, rng=np.random.default_rng(1))
    changed = evaluate_on_buffer(tiny_model, full, positions=12, rng=np.random.default_rng(1))
    assert same is not None and changed is not None
    assert same["policy_ce"] == base["policy_ce"]
    assert changed["policy_ce"] != pytest.approx(base["policy_ce"], rel=1e-3)
    assert same["value_mse"] == base["value_mse"] == changed["value_mse"]


@pytest.mark.parametrize("training", [True, False])
def test_evaluate_on_buffer_keeps_the_model_mode_and_weights(
    tiny_model: AlphaGoZeroNet, training: bool
) -> None:
    buffer = _filled_buffer(tiny_model, count=12)
    tiny_model.train(training)
    before = {name: tensor.clone() for name, tensor in tiny_model.state_dict().items()}
    evaluate_on_buffer(tiny_model, buffer, positions=12, rng=np.random.default_rng(0))
    assert tiny_model.training is training
    # Evaluation must preserve both parameters and BatchNorm running statistics.
    for name, tensor in tiny_model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_evaluate_on_buffer_handles_empty_and_small_buffers(tiny_model: AlphaGoZeroNet) -> None:
    empty = ReplayBuffer(capacity=8, board_size=tiny_model.board_size)
    assert evaluate_on_buffer(tiny_model, empty, positions=16, rng=np.random.default_rng(0)) is None
    small = _filled_buffer(tiny_model, count=5)
    assert evaluate_on_buffer(tiny_model, small, positions=0, rng=np.random.default_rng(0)) is None
    measured = evaluate_on_buffer(tiny_model, small, positions=100, rng=np.random.default_rng(0))
    assert measured is not None and measured["positions"] == 5.0


# ----------------------------------------------------------------------
# Pipeline integration
# ----------------------------------------------------------------------
def _run(config: AlphaGoZeroConfig, work_dir: Path, **training: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = copy.deepcopy(config)
    cfg.runtime.work_dir = work_dir
    for name, value in training.items():
        setattr(cfg.training, name, value)
    summary = run_training_pipeline(cfg)
    metrics = json.loads(Path(summary["metrics_log_path"]).read_text(encoding="utf-8"))
    return summary, metrics["iterations"]


def test_pipeline_without_holdout_logs_none(tiny_config: AlphaGoZeroConfig, tmp_path: Path) -> None:
    _summary, rows = _run(tiny_config, tmp_path / "plain")
    assert rows[0]["generalization"] is None


def test_pipeline_holds_games_out_of_training(tiny_config: AlphaGoZeroConfig, tmp_path: Path) -> None:
    options = {"self_play_games_per_iteration": 6, "num_iterations": 3, "replay_buffer_size": 4096}
    summary, rows = _run(tiny_config, tmp_path / "holdout", holdout_game_fraction=0.5, **options)
    _plain_summary, plain_rows = _run(tiny_config, tmp_path / "plain", **options)

    held_out_examples = 0
    for iteration, row in enumerate(rows, start=1):
        general = row["generalization"]
        expected_games = sum(holdout_game_flags_for(tiny_config, iteration, 6))
        assert general["holdout_games"] == expected_games
        held_out_examples += general["holdout_examples"]
        assert general["holdout_buffer_size"] == held_out_examples
        # All positions are accounted for across disjoint training and holdout buffers.
        generated = sum(r["examples_generated"] for r in rows[:iteration])
        assert row["replay_buffer_size"] == generated - held_out_examples
        for side in ("train", "holdout"):
            if general[side] is not None:
                assert general[side]["value_mse"] >= 0.0 and general[side]["policy_ce"] > 0.0
                assert general[side]["positions"] <= general["holdout_buffer_size"]
    assert 0 < held_out_examples < sum(r["examples_generated"] for r in rows)
    assert rows[-1]["generalization"]["holdout"] is not None
    assert summary["replay_buffer_size"] == rows[-1]["replay_buffer_size"]

    # Dedicated sampling streams keep the first self-play iteration independent of monitoring.
    assert rows[0]["examples_generated"] == plain_rows[0]["examples_generated"]
    assert rows[0]["avg_game_length"] == plain_rows[0]["avg_game_length"]
    assert rows[0]["game_endings"] == plain_rows[0]["game_endings"]


def holdout_game_flags_for(config: AlphaGoZeroConfig, iteration: int, games: int) -> list[bool]:
    probe = copy.deepcopy(config)
    probe.training.holdout_game_fraction = 0.5
    return holdout_game_flags(probe, iteration, games)


def test_holdout_buffer_survives_resume_and_old_snapshots_still_load(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    options = {"self_play_games_per_iteration": 6, "holdout_game_fraction": 0.5}
    first_summary, first_rows = _run(tiny_config, tmp_path / "first", num_iterations=2, **options)
    snapshot_path = Path(first_summary["work_dir"]) / "resume_state.pt"
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    saved = snapshot["holdout_buffer"]
    assert saved is not None and saved["states"].shape[0] == first_rows[-1]["generalization"]["holdout_buffer_size"]

    resumed = copy.deepcopy(tiny_config)
    resumed.runtime.resume_from = snapshot_path
    _summary, resumed_rows = _run(resumed, tmp_path / "resumed", num_iterations=3, **options)
    third = resumed_rows[-1]["generalization"]
    assert third["holdout_buffer_size"] == saved["states"].shape[0] + third["holdout_examples"]

    # Resume an older snapshot without holdout_buffer using an empty holdout.
    legacy_path = tmp_path / "legacy_resume.pt"
    snapshot.pop("holdout_buffer")
    torch.save(snapshot, legacy_path)
    legacy = copy.deepcopy(tiny_config)
    legacy.runtime.resume_from = legacy_path
    _summary, legacy_rows = _run(legacy, tmp_path / "legacy", num_iterations=3, **options)
    legacy_third = legacy_rows[-1]["generalization"]
    assert legacy_third["holdout_buffer_size"] == legacy_third["holdout_examples"]


def test_holdout_metric_can_be_switched_on_at_resume(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    """Enable monitoring when resuming a run that stored holdout_buffer=None."""
    first_summary, _rows = _run(tiny_config, tmp_path / "first", self_play_games_per_iteration=4)
    snapshot_path = Path(first_summary["work_dir"]) / "resume_state.pt"
    assert torch.load(snapshot_path, map_location="cpu", weights_only=False)["holdout_buffer"] is None
    resumed = copy.deepcopy(tiny_config)
    resumed.runtime.resume_from = snapshot_path
    _summary, rows = _run(
        resumed, tmp_path / "resumed", num_iterations=2,
        self_play_games_per_iteration=4, holdout_game_fraction=0.5,
    )
    assert rows[0]["generalization"] is None  # Iteration from the earlier run.
    assert rows[1]["generalization"] is not None


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
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


def test_holdout_flags_are_off_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _cli_config(monkeypatch, capsys)
    assert config.training.holdout_game_fraction == 0.0
    assert "HOLDOUT_FRACTION 0.0 HOLDOUT_BUFFER 8192" in output


def test_holdout_flags_reach_the_config_and_header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _cli_config(
        monkeypatch, capsys, "--holdout-fraction", "0.05", "--holdout-buffer", "4096"
    )
    assert config.training.holdout_game_fraction == 0.05
    assert config.training.holdout_buffer_size == 4096
    assert "HOLDOUT_FRACTION 0.05 HOLDOUT_BUFFER 4096" in output


def test_holdout_fraction_of_one_is_rejected_by_validate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="holdout_game_fraction"):
        _cli_config(monkeypatch, capsys, "--holdout-fraction", "1.0")


@pytest.mark.parametrize(("flag", "value"), [
    ("--holdout-fraction", "-0.1"), ("--holdout-fraction", "nan"), ("--holdout-fraction", "abc"),
    ("--holdout-buffer", "0"), ("--holdout-buffer", "1.5"),
])
def test_holdout_flags_reject_invalid_values_in_the_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flag: str, value: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", f"{flag}={value}"])
    with pytest.raises(SystemExit) as raised:
        gpu_train.main()
    assert raised.value.code == 2
    message = capsys.readouterr().err.strip().splitlines()[-1]
    assert flag in message and "expected" in message

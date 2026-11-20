from __future__ import annotations

"""Exercise self-play, training, evaluation and artifact writing end to end."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.pipeline import run_training_pipeline
from alphago_zero.evaluate import EvaluationSummary


@pytest.mark.slow
def test_pipeline_checkpoints_carry_calibrated_resign_threshold(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    """Persist the calibrated resignation threshold in checkpoint metadata."""
    run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)
    for path in (
        work / "final_best.pt",
        work / "final_candidate.pt",
        work / "checkpoints" / "best.pt",
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        threshold = payload["resign_threshold"]
        assert isinstance(threshold, float)
        assert -1.0 <= threshold <= 0.0


@pytest.mark.slow
def test_pipeline_end_to_end_creates_artifacts(tiny_config: AlphaGoZeroConfig) -> None:
    summary = run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)

    # Check the primary artifacts.
    assert (work / "summary.json").exists()
    assert (work / "config.json").exists()
    assert (work / "final_best.pt").exists()
    assert (work / "final_candidate.pt").exists()
    assert (work / "logs" / "metrics.json").exists()
    assert (work / "checkpoints" / "best.pt").exists()

    # Check environment metadata and iteration count in metrics.json.
    metrics = json.loads((work / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert "environment" in metrics
    assert metrics["seed"] == tiny_config.training.seed
    assert len(metrics["iterations"]) == tiny_config.training.num_iterations

    # Keep the GPU statistics key on CPU with an explicit None.
    iteration_log = metrics["iterations"][0]
    assert "gpu" in iteration_log
    assert iteration_log["gpu"] is None

    # Ending-reason counts must account for every game in the iteration.
    endings = iteration_log["game_endings"]
    assert set(endings) == {"double_pass", "resign", "max_moves"}
    assert sum(endings.values()) == tiny_config.training.self_play_games_per_iteration
    assert endings["resign"] == iteration_log["resignations"]
    assert iteration_log["game_draws"] == 0

    # The summary includes reproducibility metadata.
    assert summary["device"] == "cpu"
    assert "environment" in summary
    assert summary["seed"] == tiny_config.training.seed
    assert summary["iterations"] == tiny_config.training.num_iterations


def test_pipeline_respects_gate_decision_over_observed_win_rate(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig,
) -> None:
    monkeypatch.setattr("alphago_zero.pipeline.evaluate_candidate", lambda **kwargs: EvaluationSummary(
        candidate_wins=5, best_wins=3, draws=0, games=8, candidate_win_rate=0.625,
        promoted=False, required_wins=7, games_planned=8, gate="binomial",
    ))
    summary = run_training_pipeline(tiny_config)
    assert summary["best_iteration"] == 0
    metrics = json.loads(Path(summary["metrics_log_path"]).read_text(encoding="utf-8"))
    assert not metrics["iterations"][0]["promoted"]


def test_pipeline_without_gate_promotes_every_iteration(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig,
) -> None:
    """With the gate disabled, the newest learner supplies the next iteration's self-play."""

    def forbidden(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("no promotion games should run with gate 'none'")

    monkeypatch.setattr("alphago_zero.evaluate.play_evaluation_game", forbidden)
    tiny_config.training = replace(
        tiny_config.training, evaluation_gate="none", num_iterations=3
    )
    summary = run_training_pipeline(tiny_config)
    assert summary["best_iteration"] == 3

    metrics = json.loads(Path(summary["metrics_log_path"]).read_text(encoding="utf-8"))
    for row in metrics["iterations"]:
        assert row["promoted"]
        assert row["evaluation"]["gate"] == "none"
        assert row["evaluation"]["games"] == 0

    best = torch.load(summary["final_best_path"], map_location="cpu", weights_only=False)
    candidate = torch.load(
        summary["final_candidate_path"], map_location="cpu", weights_only=False
    )
    for name, tensor in candidate["model_state_dict"].items():
        assert torch.equal(tensor, best["model_state_dict"][name]), name


@pytest.mark.slow
def test_pipeline_time_budget_stops_on_iteration_boundary(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    """Always complete the first iteration and stop at a boundary after the time budget expires."""
    tiny_config.training = replace(tiny_config.training, num_iterations=50)
    tiny_config.runtime = replace(tiny_config.runtime, time_budget_sec=1e-6)
    tiny_config.validate()

    summary = run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)

    assert summary["stopped_by_time_budget"] is True
    assert summary["completed_iterations"] == 1
    metrics = json.loads((work / "logs" / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["iterations"]) == 1
    # Write final artifacts for either termination condition.
    assert (work / "final_best.pt").exists()
    assert (work / "summary.json").exists()


@pytest.mark.slow
def test_pipeline_checkpoint_every_saves_periodic_snapshots(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    """Save periodic checkpoints even with per-iteration saves disabled."""
    tiny_config.training = replace(
        tiny_config.training,
        num_iterations=4,
        save_every_iteration=False,
        checkpoint_every=2,
    )
    tiny_config.validate()

    summary = run_training_pipeline(tiny_config)
    checkpoints = Path(tiny_config.runtime.work_dir) / "checkpoints"

    assert summary["stopped_by_time_budget"] is False
    assert summary["completed_iterations"] == 4
    saved = sorted(p.name for p in checkpoints.glob("candidate_iter_*.pt"))
    assert saved == ["candidate_iter_002.pt", "candidate_iter_004.pt"]
    assert (checkpoints / "best.pt").exists()


@pytest.mark.slow
def test_pipeline_frozen_checkpoint_anchor(tiny_config: AlphaGoZeroConfig) -> None:
    """Use a previous run's frozen checkpoint as a third benchmark opponent."""
    # The first run supplies an anchor checkpoint.
    run_training_pipeline(tiny_config)
    anchor_path = Path(tiny_config.runtime.work_dir) / "final_best.pt"
    assert anchor_path.exists()

    # The second run uses it as the third anchor.
    second_dir = Path(tiny_config.runtime.work_dir).parent / "second_run"
    tiny_config.runtime = replace(tiny_config.runtime, work_dir=second_dir)
    tiny_config.training = replace(
        tiny_config.training,
        benchmark_every=1,
        benchmark_games=2,
        benchmark_anchor_checkpoint=anchor_path,
        benchmark_anchor_rating=150.0,
    )
    tiny_config.validate()
    run_training_pipeline(tiny_config)

    metrics = json.loads(
        (second_dir / "logs" / "metrics.json").read_text(encoding="utf-8")
    )
    benchmark = metrics["iterations"][0]["benchmark"]
    assert set(benchmark["anchors"]) == {"random", "raw_policy_init", "frozen_checkpoint"}
    frozen = benchmark["anchors"]["frozen_checkpoint"]
    assert frozen["games"] == 2
    assert frozen["anchor_rating"] == 150.0


@pytest.mark.slow
def test_pipeline_benchmark_and_sgf(tiny_config: AlphaGoZeroConfig) -> None:
    """Enable benchmarking and SGF export in the complete training loop."""
    tiny_config.training = replace(
        tiny_config.training, benchmark_every=1, benchmark_games=2
    )
    tiny_config.runtime = replace(tiny_config.runtime, save_selfplay_sgf=True)
    tiny_config.validate()

    run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)

    metrics = json.loads((work / "logs" / "metrics.json").read_text(encoding="utf-8"))
    iteration_log = metrics["iterations"][0]

    benchmark = iteration_log["benchmark"]
    assert benchmark is not None
    assert "elo_estimate" in benchmark
    assert set(benchmark["anchors"]) == {"random", "raw_policy_init"}
    for anchor_stats in benchmark["anchors"].values():
        assert anchor_stats["games"] == 2
    assert iteration_log["timings"]["benchmark_sec"] >= 0.0
    assert "stopped_early" in iteration_log["evaluation"]

    # Write one SGF for each self-play game in the iteration.
    sgf_dir = work / "sgf" / "iter_001"
    sgf_files = sorted(sgf_dir.glob("*.sgf"))
    assert len(sgf_files) == tiny_config.training.self_play_games_per_iteration
    content = sgf_files[0].read_text(encoding="utf-8")
    assert content.startswith("(;GM[1]FF[4]")
    assert f"SZ[{tiny_config.game.board_size}]" in content

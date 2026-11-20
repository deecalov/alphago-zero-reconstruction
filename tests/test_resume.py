from __future__ import annotations

"""Full resume equivalence and mixed-precision training tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.pipeline import run_training_pipeline
from alphago_zero.replay_buffer import ReplayBuffer, TrainingExample
from alphago_zero.train import train_for_steps, create_optimizer
from alphago_zero.network import AlphaGoZeroNet


def _strip_timings(iteration_log: dict) -> dict:
    """Remove wall-clock timings before comparing runs."""
    cleaned = copy.deepcopy(iteration_log)
    cleaned.pop("timings", None)
    return cleaned


@pytest.mark.parametrize("benchmark_every", [0, 1])
def test_resume_matches_continuous_run(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path, benchmark_every: int,
) -> None:
    tiny_config.training.benchmark_every = benchmark_every
    tiny_config.training.benchmark_games = 2
    # Run A completes two uninterrupted iterations.
    config_a = copy.deepcopy(tiny_config)
    config_a.training.num_iterations = 2
    config_a.training.self_play_games_per_iteration = 2
    config_a.runtime.work_dir = tmp_path / "run_a"
    run_training_pipeline(config_a)
    metrics_a = json.loads(
        (Path(config_a.runtime.work_dir) / "logs" / "metrics.json").read_text(encoding="utf-8")
    )

    # Run B saves after one iteration and resumes through iteration two.
    config_b = copy.deepcopy(tiny_config)
    config_b.training.num_iterations = 1
    config_b.training.self_play_games_per_iteration = 2
    config_b.runtime.work_dir = tmp_path / "run_b"
    run_training_pipeline(config_b)

    resume_path = Path(config_b.runtime.work_dir) / "resume_state.pt"
    assert resume_path.exists()

    config_b2 = copy.deepcopy(config_b)
    config_b2.training.num_iterations = 2
    config_b2.runtime.resume_from = resume_path
    summary_b2 = run_training_pipeline(config_b2)
    assert summary_b2["start_iteration"] == 2

    metrics_b = json.loads(
        (Path(config_b2.runtime.work_dir) / "logs" / "metrics.json").read_text(encoding="utf-8")
    )

    # The resumed log retains both iterations.
    assert [it["iteration"] for it in metrics_a["iterations"]] == [1, 2]
    assert [it["iteration"] for it in metrics_b["iterations"]] == [1, 2]

    # Compare second-iteration results bit for bit, excluding elapsed times.
    assert _strip_timings(metrics_b["iterations"][1]) == _strip_timings(metrics_a["iterations"][1])


@pytest.mark.parametrize("section,key,value", [
    ("game", "komi", 7.5),
    ("game", "superko_rule", "situational"),
    ("game", "max_moves_multiplier", 3),
    ("training", "momentum", 0.5),
    ("training", "weight_decay_exclude_norm_bias", True),
    ("network", "channels", 16),
])
def test_resume_rejects_changed_semantics_before_overwriting_config(
    tiny_config: AlphaGoZeroConfig, section: str, key: str, value: object,
) -> None:
    run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)
    original = (work / "config.json").read_bytes()
    tiny_config.runtime.resume_from = work / "resume_state.pt"
    setattr(getattr(tiny_config, section), key, value)
    with pytest.raises(ValueError, match=key):
        run_training_pipeline(tiny_config)
    assert (work / "config.json").read_bytes() == original


def test_snapshot_without_weight_decay_flag_means_flag_off(tiny_config: AlphaGoZeroConfig) -> None:
    # Emulate a snapshot predating weight_decay_exclude_norm_bias.
    run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)
    path = work / "resume_state.pt"
    payload = torch.load(path, weights_only=False)
    payload["config"]["training"].pop("weight_decay_exclude_norm_bias")
    torch.save(payload, path)
    tiny_config.runtime.resume_from = path
    tiny_config.training.num_iterations = 2

    # Reject a change that would alter the restored optimizer's parameter groups.
    original = (work / "config.json").read_bytes()
    tiny_config.training.weight_decay_exclude_norm_bias = True
    with pytest.raises(ValueError, match="weight_decay_exclude_norm_bias"):
        run_training_pipeline(tiny_config)
    assert (work / "config.json").read_bytes() == original

    # The historical default allows the resume to proceed.
    tiny_config.training.weight_decay_exclude_norm_bias = False
    summary = run_training_pipeline(tiny_config)
    assert summary["start_iteration"] == 2


def test_resume_roundtrip_keeps_weight_decay_param_groups(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.weight_decay_exclude_norm_bias = True
    run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)
    tiny_config.runtime.resume_from = work / "resume_state.pt"
    tiny_config.training.num_iterations = 2
    summary = run_training_pipeline(tiny_config)

    assert summary["start_iteration"] == 2
    payload = torch.load(work / "resume_state.pt", weights_only=False)
    groups = payload["optimizer_state_dict"]["param_groups"]
    assert [group["weight_decay"] for group in groups] == [tiny_config.training.weight_decay, 0.0]


def test_legacy_resume_uses_original_sidecar(tiny_config: AlphaGoZeroConfig) -> None:
    run_training_pipeline(tiny_config)
    path = Path(tiny_config.runtime.work_dir) / "resume_state.pt"
    payload = torch.load(path, weights_only=False)
    payload.pop("config")
    payload.pop("schema_version")
    torch.save(payload, path)
    tiny_config.runtime.resume_from = path
    tiny_config.training.num_iterations = 2
    with pytest.warns(UserWarning, match="Legacy resume"):
        summary = run_training_pipeline(tiny_config)
    assert summary["start_iteration"] == 2


def test_resume_restores_buffer_and_counters(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    config = copy.deepcopy(tiny_config)
    config.training.num_iterations = 1
    config.training.self_play_games_per_iteration = 2
    config.runtime.work_dir = tmp_path / "run"
    summary1 = run_training_pipeline(config)

    resumed = copy.deepcopy(config)
    resumed.training.num_iterations = 3
    resumed.runtime.resume_from = Path(config.runtime.work_dir) / "resume_state.pt"
    summary2 = run_training_pipeline(resumed)

    # Continue global_step and retain the populated buffer.
    steps_per_iter = config.training.gradient_steps_per_iteration
    assert summary1["global_step"] == steps_per_iter
    assert summary2["global_step"] == steps_per_iter * 3
    assert summary2["replay_buffer_size"] >= summary1["replay_buffer_size"]
    assert summary2["resumed_from"] == str(resumed.runtime.resume_from)


def test_resume_from_invalid_file_raises(tiny_config: AlphaGoZeroConfig, tmp_path: Path) -> None:
    bad_path = tmp_path / "not_resume.pt"
    torch.save({"kind": "best"}, bad_path)

    config = copy.deepcopy(tiny_config)
    config.runtime.resume_from = bad_path
    with pytest.raises(ValueError, match="resume"):
        run_training_pipeline(config)


def test_save_resume_every_zero_disables_snapshots(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    config = copy.deepcopy(tiny_config)
    config.runtime.work_dir = tmp_path / "run"
    config.runtime.save_resume_every = 0
    run_training_pipeline(config)
    assert not (Path(config.runtime.work_dir) / "resume_state.pt").exists()


def test_replay_buffer_state_dict_roundtrip(rng: np.random.Generator) -> None:
    buffer = ReplayBuffer(capacity=4, board_size=5, rng=rng)
    for i in range(6):  # Overflow capacity to exercise the write pointer.
        state = (rng.random((17, 5, 5)) > 0.5).astype(np.float32)  # Binary state planes.
        policy = rng.random(26).astype(np.float32)
        policy /= policy.sum()
        buffer.add(TrainingExample(state=state, policy=policy, value=1.0 if i % 2 else -1.0))

    payload = buffer.state_dict()
    restored = ReplayBuffer(capacity=1, board_size=3, rng=rng)
    restored.load_state_dict(payload)

    assert len(restored) == len(buffer)
    assert restored.capacity == buffer.capacity
    assert restored.board_size == buffer.board_size
    for original, loaded in zip(buffer._buffer, restored._buffer):
        assert np.array_equal(original.state, loaded.state)
        assert np.allclose(original.policy, loaded.policy)
        assert original.value == loaded.value


def test_amp_training_produces_finite_losses(
    tiny_config: AlphaGoZeroConfig, rng: np.random.Generator
) -> None:
    # Use CPU bfloat16 autocast and verify finite training losses.
    config = copy.deepcopy(tiny_config)
    config.runtime.use_amp = True
    config.training.gradient_steps_per_iteration = 2

    torch.manual_seed(0)
    model = AlphaGoZeroNet(config.network)
    optimizer = create_optimizer(model, config)

    buffer = ReplayBuffer(capacity=64, board_size=5, rng=rng)
    for _ in range(16):
        state = (rng.random((17, 5, 5)) > 0.5).astype(np.float32)
        policy = rng.random(26).astype(np.float32)
        policy /= policy.sum()
        buffer.add(TrainingExample(state=state, policy=policy, value=1.0))

    metrics, global_step = train_for_steps(
        model=model,
        optimizer=optimizer,
        replay_buffer=buffer,
        config=config,
        rng=rng,
        global_step=0,
    )
    assert global_step == 2
    assert np.isfinite(metrics.total_loss)
    assert np.isfinite(metrics.policy_loss)
    assert np.isfinite(metrics.value_loss)


def test_replay_buffer_keep_capacity_trims_to_new_capacity(rng: np.random.Generator) -> None:
    """Keep the configured capacity and the newest examples."""
    src = ReplayBuffer(capacity=6, board_size=5, rng=rng)
    for i in range(6):
        state = (rng.random((17, 5, 5)) > 0.5).astype(np.float32)
        policy = rng.random(26).astype(np.float32)
        policy /= policy.sum()
        src.add(TrainingExample(state=state, policy=policy, value=1.0 if i % 2 else -1.0))

    dst = ReplayBuffer(capacity=4, board_size=5, rng=rng)
    dst.load_state_dict(src.state_dict(), keep_capacity=True)

    assert dst.capacity == 4  # The snapshot's capacity of six must not replace the target capacity.
    assert len(dst) == 4  # Keep the newest four examples.
    for original, loaded in zip(src._buffer[-4:], dst._buffer):
        assert np.array_equal(original.state, loaded.state)
        assert original.value == loaded.value


@pytest.mark.parametrize("count", [3, 5, 8, 13])
@pytest.mark.parametrize("capacity", [2, 5, 8])
def test_warmstart_preserves_recency_after_ring_wrap(count: int, capacity: int) -> None:
    def example(index: int) -> TrainingExample:
        policy = np.zeros(26, dtype=np.float32)
        policy[index % 26] = 1.0
        return TrainingExample(
            state=np.zeros((17, 5, 5), dtype=np.float32),
            policy=policy, value=1.0, score=index / 100.0,
        )

    source = ReplayBuffer(capacity=5, board_size=5)
    source.add_many(example(i) for i in range(count))
    target = ReplayBuffer(capacity=capacity, board_size=5)
    target.load_state_dict(source.state_dict(), keep_capacity=True)
    expected = list(range(max(0, count - 5, count - capacity), count))
    assert [round(e.score * 100) for e in target._buffer] == expected
    assert [int(e.policy.argmax()) for e in target._buffer] == expected

    # Continue past the next wrap: eviction must still remove the oldest data.
    end = count + capacity + 2
    target.add_many(example(i) for i in range(count, end))
    chronological = target._buffer[target._position:] + target._buffer[:target._position]
    assert [round(e.score * 100) for e in chronological] == list(range(end - capacity, end))


def test_warmstart_buffer_only_uses_fresh_scaled_network(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    """Warm-start a fresh, larger network with replay from a previous run."""
    producer = copy.deepcopy(tiny_config)
    producer.training.num_iterations = 1
    producer.training.self_play_games_per_iteration = 2
    producer.runtime.work_dir = tmp_path / "producer"
    summary_p = run_training_pipeline(producer)
    resume_path = Path(producer.runtime.work_dir) / "resume_state.pt"
    assert resume_path.exists()
    assert summary_p["replay_buffer_size"] > 0

    scaled = copy.deepcopy(tiny_config)
    scaled.network.channels = 16  # Use more channels than the source network's eight.
    scaled.network.num_res_blocks = 2
    scaled.network.value_head_hidden = 16
    scaled.training.num_iterations = 1
    scaled.runtime.work_dir = tmp_path / "scaled"
    scaled.runtime.warmstart_buffer_from = resume_path
    summary_s = run_training_pipeline(scaled)

    # Start at iteration one with new weights.
    assert summary_s["start_iteration"] == 1
    assert summary_s["warmstart_buffer_from"] == str(resume_path)
    assert summary_s["warmstart_buffer_examples"] > 0
    # Use the larger target architecture.
    assert summary_s["current_model"]["channels"] == 16
    assert summary_s["current_model"]["num_res_blocks"] == 2


def test_warmstart_buffer_respects_new_capacity(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    producer = copy.deepcopy(tiny_config)
    producer.training.num_iterations = 1
    producer.training.self_play_games_per_iteration = 2
    producer.runtime.work_dir = tmp_path / "producer"
    summary_p = run_training_pipeline(producer)
    resume_path = Path(producer.runtime.work_dir) / "resume_state.pt"
    assert summary_p["replay_buffer_size"] > 4  # The source contains more examples than the target can hold.

    scaled = copy.deepcopy(tiny_config)
    scaled.training.replay_buffer_size = 4
    scaled.training.batch_size = 4
    scaled.training.num_iterations = 1
    scaled.runtime.work_dir = tmp_path / "scaled"
    scaled.runtime.warmstart_buffer_from = resume_path
    summary_s = run_training_pipeline(scaled)

    assert summary_s["warmstart_buffer_examples"] == 4  # Limit copied examples to the target capacity.


def test_warmstart_and_resume_are_mutually_exclusive(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    config = copy.deepcopy(tiny_config)
    config.runtime.resume_from = tmp_path / "a.pt"
    config.runtime.warmstart_buffer_from = tmp_path / "b.pt"
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()


def test_warmstart_buffer_board_size_mismatch_raises(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    producer = copy.deepcopy(tiny_config)
    producer.training.num_iterations = 1
    producer.runtime.work_dir = tmp_path / "producer"
    run_training_pipeline(producer)
    resume_path = Path(producer.runtime.work_dir) / "resume_state.pt"

    scaled = copy.deepcopy(tiny_config)
    scaled.game.board_size = 7
    scaled.network.board_size = 7
    scaled.runtime.work_dir = tmp_path / "scaled"
    scaled.runtime.warmstart_buffer_from = resume_path
    with pytest.raises(ValueError, match="board_size"):
        run_training_pipeline(scaled)


def test_metrics_contain_stage_timings(tiny_config: AlphaGoZeroConfig) -> None:
    summary = run_training_pipeline(tiny_config)
    metrics = json.loads(
        (Path(summary["metrics_log_path"])).read_text(encoding="utf-8")
    )
    timings = metrics["iterations"][0]["timings"]
    assert set(timings) == {
        "selfplay_sec",
        "train_sec",
        "evaluation_sec",
        "benchmark_sec",
        "selfplay_moves_per_sec",
        "selfplay_worker_sec",
    }
    assert timings["selfplay_sec"] >= 0.0
    assert timings["selfplay_moves_per_sec"] > 0.0
    # Keep a stable schema with no worker timings in single-process mode.
    assert timings["selfplay_worker_sec"] == []

from __future__ import annotations

"""Checkpoint averaging, BatchNorm recalibration and command-line integration tests."""

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from alphago_zero.config import AlphaGoZeroConfig, NetworkConfig
from alphago_zero.network import AlphaGoZeroNet
from examples import average_checkpoints, eval_match


def _checkpoint(path: Path, config: AlphaGoZeroConfig, net: AlphaGoZeroNet, **extra: Any) -> Path:
    """Create a pipeline-style candidate checkpoint with optimizer state."""
    torch.save(
        {
            "kind": "candidate",
            "config": asdict(config),
            "model_state_dict": copy.deepcopy(net.state_dict()),
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            **extra,
        },
        path,
    )
    return path


def _trajectory(
    tmp_path: Path, config: AlphaGoZeroConfig, network: NetworkConfig, count: int = 3
) -> list[Path]:
    """Generate nearby checkpoints on one synthetic trajectory with distinct BatchNorm statistics."""
    torch.manual_seed(3)
    base = AlphaGoZeroNet(network)
    paths: list[Path] = []
    for index in range(count):
        net = AlphaGoZeroNet(network)
        net.load_state_dict(base.state_dict())
        generator = torch.Generator().manual_seed(100 + index)
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(0.01 * torch.randn(parameter.shape, generator=generator))
            for module in net.modules():
                if isinstance(module, _BatchNorm):
                    module.running_mean.add_(0.1 * (index + 1))
                    module.running_var.mul_(1.0 + 0.2 * index)
                    module.num_batches_tracked.fill_(10 * (index + 1))
        paths.append(
            _checkpoint(tmp_path / f"candidate_iter_{index:03d}.pt", config, net, iteration=index)
        )
    return paths


def _buffer(path: Path, network: NetworkConfig, count: int = 40, seed: int = 9) -> Path:
    rng = np.random.default_rng(seed)
    board = network.board_size
    states = rng.integers(0, 2, size=(count, 17, board, board), dtype=np.uint8)
    raw = rng.random((count, network.action_size), dtype=np.float32)
    weights = np.ones(count, dtype=np.float32)
    weights[::4] = 0.0  # Every fourth move uses cheap search.
    torch.save(
        {
            "kind": "resume_state",
            "replay_buffer": {
                "states": states,
                "policies": raw / raw.sum(axis=1, keepdims=True),
                "values": rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=count),
                "policy_weights": weights,
            },
        },
        path,
    )
    return path


def _error_message(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err.strip().splitlines()[-1]


def _argv(paths: list[Path], out: Path, *extra: str) -> list[str]:
    argv: list[str] = []
    for path in paths:
        argv += ["--checkpoint", str(path)]
    return [*argv, "--out", str(out), "--bn-batches", "3", "--batch-size", "8", *extra]


# ----------------------------------------------------------------------
# Weight averaging
# ----------------------------------------------------------------------
def test_average_state_dicts_is_the_exact_mean_and_keeps_the_last_counters() -> None:
    states = [
        {"w": torch.tensor([1.0, 2.0]), "bn.num_batches_tracked": torch.tensor(5)},
        {"w": torch.tensor([3.0, 6.0]), "bn.num_batches_tracked": torch.tensor(7)},
        {"w": torch.tensor([5.0, 1.0]), "bn.num_batches_tracked": torch.tensor(9)},
    ]
    averaged = average_checkpoints.average_state_dicts(states)
    assert torch.equal(averaged["w"], torch.tensor([3.0, 3.0]))
    assert averaged["w"].dtype == torch.float32
    assert int(averaged["bn.num_batches_tracked"]) == 9
    # Averaging must leave the input tensors intact.
    assert torch.equal(states[0]["w"], torch.tensor([1.0, 2.0]))


def test_average_state_dicts_accumulates_in_double_precision() -> None:
    """Float64 accumulation retains small terms next to large float32 values."""
    big, tiny = 16777216.0, 1.0  # At 2**24, float32 rounds big + tiny back to big.
    states = [{"w": torch.tensor([big])}, {"w": torch.tensor([tiny])}, {"w": torch.tensor([tiny])}]
    averaged = average_checkpoints.average_state_dicts(states)
    assert float(averaged["w"]) == np.float32((big + 2.0 * tiny) / 3.0)
    # Both small terms would disappear with float32 accumulation.
    assert float(averaged["w"]) != float(np.float32(big) / np.float32(3.0))


def test_average_state_dicts_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        average_checkpoints.average_state_dicts([])


def test_relative_distances_are_zero_for_identical_checkpoints_and_grow_with_noise(
    tmp_path: Path, tiny_config: AlphaGoZeroConfig, tiny_network_config: NetworkConfig
) -> None:
    paths = _trajectory(tmp_path, tiny_config, tiny_network_config)
    states = [average_checkpoints.load_payload(path)["model_state_dict"] for path in paths]
    same = [states[0], states[0]]
    assert average_checkpoints.relative_distances(
        same, average_checkpoints.average_state_dicts(same)
    ) == [0.0, 0.0]
    distances = average_checkpoints.relative_distances(
        states, average_checkpoints.average_state_dicts(states)
    )
    assert len(distances) == 3 and all(0.0 < d < 0.2 for d in distances)


# ----------------------------------------------------------------------
# Checkpoint compatibility
# ----------------------------------------------------------------------
def test_require_same_model_rejects_other_architecture_rules_and_tensors(
    tmp_path: Path, tiny_config: AlphaGoZeroConfig, tiny_network_config: NetworkConfig
) -> None:
    paths = _trajectory(tmp_path, tiny_config, tiny_network_config, count=2)
    payloads = [average_checkpoints.load_payload(path) for path in paths]
    names = [str(path) for path in paths]
    average_checkpoints.require_same_model(payloads, names)  # Accept checkpoints from the same trajectory.

    wider = copy.deepcopy(payloads[1])
    wider["config"]["network"]["channels"] = 16
    with pytest.raises(ValueError, match="architecture"):
        average_checkpoints.require_same_model([payloads[0], wider], names)

    other_komi = copy.deepcopy(payloads[1])
    other_komi["config"]["game"]["komi"] = 7.5
    with pytest.raises(ValueError, match="rules"):
        average_checkpoints.require_same_model([payloads[0], other_komi], names)

    missing = copy.deepcopy(payloads[1])
    missing["model_state_dict"].pop(next(iter(missing["model_state_dict"])))
    with pytest.raises(ValueError, match="tensor"):
        average_checkpoints.require_same_model([payloads[0], missing], names)


def test_load_payload_rejects_a_file_that_is_not_a_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"replay_buffer": {}}, path)
    with pytest.raises(ValueError, match="expected a pipeline checkpoint"):
        average_checkpoints.load_payload(path)


# ----------------------------------------------------------------------
# BatchNorm recalibration
# ----------------------------------------------------------------------
def test_recompute_batchnorm_matches_a_manual_average_over_batches(
    monkeypatch: pytest.MonkeyPatch, tiny_model: AlphaGoZeroNet
) -> None:
    """Compare the first layer's running mean with a manual batch calculation using fixed orientation."""
    monkeypatch.setattr(average_checkpoints, "random_dihedral_index", lambda rng: 0)
    states = np.random.default_rng(1).integers(0, 2, size=(24, 17, 5, 5), dtype=np.uint8)
    first_bn = tiny_model.input_block.bn
    with torch.no_grad():
        first_bn.running_mean.fill_(123.0)  # Insert stale running statistics that recalibration must replace.
        first_bn.running_var.fill_(456.0)
    weights_before = {
        name: tensor.detach().clone() for name, tensor in tiny_model.named_parameters()
    }

    seen = average_checkpoints.recompute_batchnorm(
        tiny_model, states, batches=3, batch_size=8,
        rng=np.random.default_rng(77), device=torch.device("cpu"),
    )
    assert seen == 24

    replay = np.random.default_rng(77)
    means, variances = [], []
    with torch.no_grad():
        for _ in range(3):
            indices = replay.choice(24, size=8, replace=False)
            activations = tiny_model.input_block.conv(torch.from_numpy(states[indices].astype(np.float32)))
            means.append(activations.mean(dim=(0, 2, 3)))
            variances.append(activations.var(dim=(0, 2, 3), unbiased=True))
    assert torch.allclose(first_bn.running_mean, torch.stack(means).mean(dim=0), atol=1e-5)
    assert torch.allclose(first_bn.running_var, torch.stack(variances).mean(dim=0), atol=1e-5)

    # Keep trainable weights fixed and restore evaluation mode and BatchNorm momentum.
    for name, tensor in tiny_model.named_parameters():
        assert torch.equal(tensor, weights_before[name]), name
    assert not tiny_model.training
    assert all(
        module.momentum == 0.1 for module in tiny_model.modules() if isinstance(module, _BatchNorm)
    )


def test_recompute_batchnorm_covers_aux_heads_and_small_buffers(
    tiny_network_config: NetworkConfig,
) -> None:
    """Recalibrate the auxiliary heads too, sampling with replacement from a small buffer."""
    network = NetworkConfig(**{**asdict(tiny_network_config), "aux_heads": True})
    torch.manual_seed(4)
    model = AlphaGoZeroNet(network)
    assert model.ownership_head is not None
    with torch.no_grad():
        model.ownership_head.bn.running_mean.fill_(99.0)
    states = np.random.default_rng(2).integers(0, 2, size=(3, 17, 5, 5), dtype=np.uint8)
    seen = average_checkpoints.recompute_batchnorm(
        model, states, batches=2, batch_size=8,
        rng=np.random.default_rng(5), device=torch.device("cpu"),
    )
    assert seen == 16
    assert float(model.ownership_head.bn.running_mean.abs().max()) < 50.0
    assert int(model.ownership_head.bn.num_batches_tracked) == 2


def test_recompute_batchnorm_restores_the_mode_when_the_forward_pass_fails(
    monkeypatch: pytest.MonkeyPatch, tiny_model: AlphaGoZeroNet
) -> None:
    def broken(_x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RuntimeError("forward pass failed")

    monkeypatch.setattr(tiny_model, "forward_full", broken)
    states = np.zeros((4, 17, 5, 5), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="failed"):
        average_checkpoints.recompute_batchnorm(
            tiny_model, states, batches=1, batch_size=4,
            rng=np.random.default_rng(0), device=torch.device("cpu"),
        )
    assert not tiny_model.training
    assert all(
        module.momentum == 0.1 for module in tiny_model.modules() if isinstance(module, _BatchNorm)
    )


# ----------------------------------------------------------------------
# Replay loss measurement
# ----------------------------------------------------------------------
def test_buffer_losses_normalise_policy_ce_by_the_weight_sum(
    tmp_path: Path, tiny_model: AlphaGoZeroNet, tiny_network_config: NetworkConfig
) -> None:
    buffer = average_checkpoints.load_buffer(_buffer(tmp_path / "buf.pt", tiny_network_config), tiny_network_config)
    indices = np.arange(buffer["states"].shape[0])
    losses = average_checkpoints.buffer_losses(
        tiny_model, buffer, indices, batch_size=7, device=torch.device("cpu")
    )

    with torch.no_grad():
        logits, value = tiny_model(torch.from_numpy(buffer["states"].astype(np.float32)))
        per_sample = -(torch.from_numpy(buffer["policies"]) * torch.log_softmax(logits, dim=1)).sum(dim=1)
        weights = torch.from_numpy(buffer["policy_weights"])
        expected_ce = float((per_sample * weights).sum() / weights.sum())
        expected_mse = float(((value.squeeze(1) - torch.from_numpy(buffer["values"])) ** 2).mean())
    assert losses["policy_ce"] == pytest.approx(expected_ce, abs=1e-4)
    assert losses["value_mse"] == pytest.approx(expected_mse, abs=1e-4)
    # A plain batch mean would incorrectly include the cheap moves in the denominator.
    assert losses["policy_ce"] != pytest.approx(float(per_sample.mean()), abs=1e-3)


def test_load_buffer_defaults_policy_weights_and_rejects_bad_shapes(
    tmp_path: Path, tiny_network_config: NetworkConfig
) -> None:
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "replay_buffer": {
                "states": np.zeros((4, 17, 5, 5), dtype=np.uint8),
                "policies": np.full((4, 26), 1.0 / 26.0, dtype=np.float32),
                "values": np.ones(4, dtype=np.float32),
            }
        },
        legacy,
    )
    buffer = average_checkpoints.load_buffer(legacy, tiny_network_config)
    assert np.array_equal(buffer["policy_weights"], np.ones(4, dtype=np.float32))

    wrong_board = tmp_path / "wrong_board.pt"
    torch.save({"replay_buffer": {"states": np.zeros((4, 17, 9, 9), dtype=np.uint8)}}, wrong_board)
    with pytest.raises(ValueError, match="expected states"):
        average_checkpoints.load_buffer(wrong_board, tiny_network_config)

    empty = tmp_path / "empty.pt"
    torch.save({"replay_buffer": {"states": np.zeros((0,), dtype=np.uint8)}}, empty)
    with pytest.raises(ValueError, match="expected states"):
        average_checkpoints.load_buffer(empty, tiny_network_config)

    not_a_snapshot = tmp_path / "weights_only.pt"
    torch.save({"model_state_dict": {}}, not_a_snapshot)
    with pytest.raises(ValueError, match="replay_buffer"):
        average_checkpoints.load_buffer(not_a_snapshot, tiny_network_config)


# ----------------------------------------------------------------------
# End-to-end CLI
# ----------------------------------------------------------------------
def test_main_writes_an_averaged_checkpoint_that_eval_match_can_load(
    tmp_path: Path,
    tiny_config: AlphaGoZeroConfig,
    tiny_network_config: NetworkConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _trajectory(tmp_path, tiny_config, tiny_network_config)
    buffer_path = _buffer(tmp_path / "resume_state.pt", tiny_network_config)
    out = tmp_path / "nested" / "swa.pt"
    average_checkpoints.main(_argv(paths, out, "--buffer", str(buffer_path), "--seed", "5"))

    saved = torch.load(out, map_location="cpu", weights_only=False)
    assert saved["kind"] == "swa_checkpoint"
    assert "optimizer_state_dict" not in saved
    assert saved["iteration"] == 2  # Use metadata from the newest checkpoint.
    assert saved["config"] == asdict(tiny_config)
    meta = saved["swa"]
    assert meta["sources"] == [str(path) for path in paths]
    assert (meta["count"], meta["batchnorm"], meta["bn_positions"]) == (3, "recomputed", 24)
    assert (meta["buffer"], meta["seed"]) == (str(buffer_path), 5)
    report = meta["report"]
    assert set(report) == {
        "relative_distance_to_average", "last_checkpoint", "average_bn_averaged",
        "average_bn_recomputed", "check_positions",
    }
    assert report["check_positions"] == 40
    assert "average_bn_recomputed" in capsys.readouterr().out

    # Trainable weights equal the three-way average and BatchNorm statistics come from recalibration.
    sources = [torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"] for path in paths]
    name = "input_block.conv.weight"
    expected = (sources[0][name].double() + sources[1][name].double() + sources[2][name].double()) / 3.0
    assert torch.allclose(saved["model_state_dict"][name].double(), expected, atol=1e-7)
    bn_mean = "input_block.bn.running_mean"
    averaged_stats = (sources[0][bn_mean] + sources[1][bn_mean] + sources[2][bn_mean]) / 3.0
    assert not torch.allclose(saved["model_state_dict"][bn_mean], averaged_stats, atol=1e-3)
    assert int(saved["model_state_dict"]["input_block.bn.num_batches_tracked"]) == 3

    network, search, game = eval_match.load_checkpoint(out, torch.device("cpu"))
    assert not network.training
    assert search.num_simulations == tiny_config.search.num_simulations
    assert game.komi == tiny_config.game.komi
    logits, value = network.predict_single(np.zeros((17, 5, 5), dtype=np.float32))
    assert logits.shape == (26,) and np.isfinite(logits).all() and -1.0 <= value <= 1.0


def test_main_without_a_buffer_averages_the_batchnorm_statistics(
    tmp_path: Path,
    tiny_config: AlphaGoZeroConfig,
    tiny_network_config: NetworkConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _trajectory(tmp_path, tiny_config, tiny_network_config, count=2)
    out = tmp_path / "swa.pt"
    average_checkpoints.main(_argv(paths, out))
    assert "Warning" in capsys.readouterr().out

    saved = torch.load(out, map_location="cpu", weights_only=False)
    meta = saved["swa"]
    assert (meta["batchnorm"], meta["bn_positions"], meta["buffer"]) == ("averaged", 0, None)
    assert set(meta["report"]) == {"relative_distance_to_average"}
    sources = [torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"] for path in paths]
    for name in ("input_block.bn.running_mean", "input_block.bn.running_var"):
        assert torch.allclose(saved["model_state_dict"][name], (sources[0][name] + sources[1][name]) / 2.0)
    # Take the batch counter from the last checkpoint.
    assert int(saved["model_state_dict"]["input_block.bn.num_batches_tracked"]) == 20


# ----------------------------------------------------------------------
# CLI validation
# ----------------------------------------------------------------------
def _forbid_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("checkpoints must not load after a usage error")

    monkeypatch.setattr(average_checkpoints, "load_payload", forbidden)


def test_a_single_checkpoint_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_loading(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        average_checkpoints.main(_argv([tmp_path / "only.pt"], tmp_path / "swa.pt"))
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert "--checkpoint" in message and "expected" in message


def test_a_repeated_checkpoint_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject duplicate paths because they would overweight one checkpoint."""
    _forbid_loading(monkeypatch)
    same = tmp_path / "a.pt"
    with pytest.raises(SystemExit) as exit_info:
        average_checkpoints.main(_argv([same, same], tmp_path / "swa.pt"))
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert "--checkpoint" in message and "expected" in message


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--bn-batches", "0"), ("--bn-batches", "-3"), ("--bn-batches", "abc"),
        ("--batch-size", "0"), ("--batch-size", "1.5"),
        ("--check-positions", "0"), ("--check-positions", "abc"),
    ],
)
def test_invalid_numeric_flags_are_rejected_before_loading_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    _forbid_loading(monkeypatch)
    argv = _argv([tmp_path / "a.pt", tmp_path / "b.pt"], tmp_path / "swa.pt", f"{flag}={value}")
    with pytest.raises(SystemExit) as exit_info:
        average_checkpoints.main(argv)
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert flag in message and "expected" in message


def test_unusable_out_directory_is_rejected_before_loading_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_loading(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("file in place of a directory", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        average_checkpoints.main(_argv([tmp_path / "a.pt", tmp_path / "b.pt"], blocker / "swa.pt"))
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert "--out" in message and "blocker" in message


def test_incompatible_checkpoints_fail_without_writing_a_result(
    tmp_path: Path, tiny_config: AlphaGoZeroConfig, tiny_network_config: NetworkConfig
) -> None:
    first = _trajectory(tmp_path, tiny_config, tiny_network_config, count=1)[0]
    wide_network = NetworkConfig(**{**asdict(tiny_network_config), "channels": 16})
    wide_config = copy.deepcopy(tiny_config)
    wide_config.network = wide_network
    torch.manual_seed(8)
    second = _checkpoint(tmp_path / "wide.pt", wide_config, AlphaGoZeroNet(wide_network))
    out = tmp_path / "swa.pt"
    with pytest.raises(ValueError, match="architecture"):
        average_checkpoints.main(_argv([first, second], out))
    assert not out.exists()

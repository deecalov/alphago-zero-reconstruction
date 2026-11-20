from __future__ import annotations

"""Average checkpoints from one SGD trajectory, then optionally recompute BatchNorm statistics."""

import argparse
import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from alphago_zero.augmentations import apply_dihedral_to_planes, random_dihedral_index
from alphago_zero.checkpoint_io import atomic_output
from alphago_zero.cli_types import ensure_directory, positive_int
from alphago_zero.config import GameConfig, NetworkConfig
from alphago_zero.network import AlphaGoZeroNet


def _filter_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep recognized dataclass fields when reading older configurations."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load the entire pipeline checkpoint on CPU for weight averaging."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "config" not in payload:
        raise ValueError(f"{path}: expected a pipeline checkpoint containing config and model_state_dict")
    return payload


def require_same_model(payloads: Sequence[dict[str, Any]], paths: Sequence[str]) -> None:
    """Require identical network architecture, rule settings, tensor names and shapes before averaging."""
    reference = payloads[0]
    network = NetworkConfig(**_filter_fields(NetworkConfig, reference["config"]["network"]))
    game = GameConfig(**_filter_fields(GameConfig, reference["config"]["game"]))
    shapes = {name: tuple(tensor.shape) for name, tensor in reference["model_state_dict"].items()}
    for payload, path in zip(payloads[1:], paths[1:]):
        other_network = NetworkConfig(**_filter_fields(NetworkConfig, payload["config"]["network"]))
        other_game = GameConfig(**_filter_fields(GameConfig, payload["config"]["game"]))
        if other_network != network:
            raise ValueError(f"{path}: network architecture differs from {paths[0]}")
        if other_game != game:
            raise ValueError(f"{path}: game rules (board size, komi or superko) differ from {paths[0]}")
        other_shapes = {name: tuple(tensor.shape) for name, tensor in payload["model_state_dict"].items()}
        if other_shapes != shapes:
            raise ValueError(f"{path}: tensor names or shapes differ from {paths[0]}")


def average_state_dicts(states: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Average floating-point tensors with equal weights and float64 accumulation."""
    if not states:
        raise ValueError("at least one set of weights is required")
    averaged: dict[str, Tensor] = {}
    for name, last in states[-1].items():
        if not last.is_floating_point():
            averaged[name] = last.clone()
            continue
        total = torch.zeros_like(last, dtype=torch.float64)
        for state in states:
            total += state[name].to(torch.float64)
        averaged[name] = (total / len(states)).to(last.dtype)
    return averaged


def relative_distances(states: Sequence[dict[str, Tensor]], averaged: dict[str, Tensor]) -> list[float]:
    """Report each checkpoint's relative distance from the mean weight vector."""
    names = [
        name
        for name, tensor in averaged.items()
        if tensor.is_floating_point() and not name.endswith(("running_mean", "running_var"))
    ]
    norm = math.sqrt(sum(float((averaged[name].double() ** 2).sum()) for name in names))
    distances: list[float] = []
    for state in states:
        delta = math.sqrt(
            sum(float(((state[name].double() - averaged[name].double()) ** 2).sum()) for name in names)
        )
        distances.append(delta / norm if norm > 0.0 else 0.0)
    return distances


def load_buffer(path: str | Path, network: NetworkConfig) -> dict[str, np.ndarray]:
    """Read encoded positions and training targets from a run's replay snapshot."""
    snapshot = torch.load(Path(path), map_location="cpu", weights_only=False)
    raw = snapshot.get("replay_buffer") if isinstance(snapshot, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: missing replay_buffer (supply the run\'s resume_state.pt)")
    states = np.asarray(raw["states"])
    expected = (network.num_input_planes, network.board_size, network.board_size)
    if states.ndim != 4 or states.shape[0] == 0 or tuple(states.shape[1:]) != expected:
        raise ValueError(
            f"{path}: expected states [N, {expected[0]}, {expected[1]}, {expected[2]}] "
            f"with N > 0, got {states.shape}"
        )
    count = states.shape[0]
    weights = raw.get("policy_weights")
    return {
        "states": states,
        "policies": np.asarray(raw["policies"], dtype=np.float32),
        "values": np.asarray(raw["values"], dtype=np.float32),
        # Older snapshots assigned full policy weight to every position.
        "policy_weights": (
            np.asarray(weights, dtype=np.float32)
            if weights is not None and np.asarray(weights).shape[0] == count
            else np.ones(count, dtype=np.float32)
        ),
    }


@torch.no_grad()
def recompute_batchnorm(
    model: AlphaGoZeroNet,
    states: np.ndarray,
    *,
    batches: int,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    """Reset BatchNorm statistics and recompute them on replay using the averaged weights."""
    layers = [module for module in model.modules() if isinstance(module, _BatchNorm)]
    saved_momentum = [layer.momentum for layer in layers]
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
    model.train()
    count = states.shape[0]
    seen = 0
    try:
        for _ in range(batches):
            indices = rng.choice(count, size=batch_size, replace=count < batch_size)
            batch = np.stack(
                [
                    apply_dihedral_to_planes(
                        states[int(index)].astype(np.float32), random_dihedral_index(rng)
                    )
                    for index in indices
                ],
                axis=0,
            ).astype(np.float32)
            model.forward_full(torch.from_numpy(batch).to(device))
            seen += len(indices)
    finally:
        for layer, momentum in zip(layers, saved_momentum):
            layer.momentum = momentum
        model.eval()
    return seen


@torch.no_grad()
def buffer_losses(
    model: AlphaGoZeroNet,
    buffer: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure cross-entropy on full-search policies and value MSE on replay positions."""
    model.eval()
    ce_sum, weight_sum, mse_sum = 0.0, 0.0, 0.0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        x = torch.from_numpy(buffer["states"][chunk].astype(np.float32)).to(device)
        target_policy = torch.from_numpy(buffer["policies"][chunk]).to(device)
        target_value = torch.from_numpy(buffer["values"][chunk]).to(device)
        weights = torch.from_numpy(buffer["policy_weights"][chunk]).to(device)
        logits, value = model(x)
        per_sample = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1)
        ce_sum += float((per_sample * weights).sum().item())
        weight_sum += float(weights.sum().item())
        mse_sum += float(((value.squeeze(1) - target_value) ** 2).sum().item())
    return {
        "policy_ce": round(ce_sum / max(weight_sum, 1.0), 4),
        "value_mse": round(mse_sum / max(len(indices), 1), 4),
    }


def _build_model(network: NetworkConfig, state: dict[str, Tensor], device: torch.device) -> AlphaGoZeroNet:
    model = AlphaGoZeroNet(network).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Average weights from one run and optionally recalibrate BatchNorm",
    )
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="checkpoint (.pt), repeat at least twice from oldest to newest. "
             "The newest checkpoint supplies the output configuration and metadata",
    )
    parser.add_argument(
        "--buffer", default=None,
        help="resume_state.pt from the same run for BatchNorm recalibration and "
             "loss measurement. Without it, BatchNorm statistics are averaged",
    )
    parser.add_argument(
        "--bn-batches", type=positive_int, default=200,
        help="number of positive-sized passes over batches for BatchNorm recalibration",
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=256,
        help="batch size for BatchNorm recalibration and loss measurement",
    )
    parser.add_argument(
        "--check-positions", type=positive_int, default=8192,
        help="number of replay positions to use for measuring loss",
    )
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--out", required=True, help="output path for the averaged checkpoint (.pt)")
    args = parser.parse_args(argv)

    paths = [str(path) for path in args.checkpoint]
    if len(paths) < 2:
        parser.error("--checkpoint: expected at least two checkpoints")
    if len(set(paths)) != len(paths):
        parser.error("--checkpoint: expected distinct paths")
    out_path = Path(args.out)
    ensure_directory(parser, out_path.parent, "--out")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    rng = np.random.default_rng(args.seed)

    print("== checkpoints ==", flush=True)
    payloads = [load_payload(path) for path in paths]
    require_same_model(payloads, paths)
    states = [payload["model_state_dict"] for payload in payloads]
    last = payloads[-1]
    network = NetworkConfig(**_filter_fields(NetworkConfig, last["config"]["network"]))

    averaged = average_state_dicts(states)
    distances = relative_distances(states, averaged)
    for path, distance in zip(paths, distances):
        print(f"  {path}: |w - mean| / |mean| = {distance:.4f}", flush=True)

    model = _build_model(network, averaged, device)
    report: dict[str, Any] = {"relative_distance_to_average": [round(d, 5) for d in distances]}
    batchnorm = "averaged"
    bn_positions = 0
    if args.buffer is None:
        print(
            "Warning: without --buffer, BatchNorm statistics are averaged. "
            "Recalibrate them on replay before comparing match strength",
            flush=True,
        )
    else:
        print("== replay and BatchNorm ==", flush=True)
        buffer = load_buffer(args.buffer, network)
        total = buffer["states"].shape[0]
        check = rng.permutation(total)[: min(args.check_positions, total)]
        report["last_checkpoint"] = buffer_losses(
            _build_model(network, states[-1], device), buffer, check,
            batch_size=args.batch_size, device=device,
        )
        report["average_bn_averaged"] = buffer_losses(
            model, buffer, check, batch_size=args.batch_size, device=device
        )
        bn_positions = recompute_batchnorm(
            model, buffer["states"],
            batches=args.bn_batches, batch_size=args.batch_size, rng=rng, device=device,
        )
        batchnorm = "recomputed"
        report["average_bn_recomputed"] = buffer_losses(
            model, buffer, check, batch_size=args.batch_size, device=device
        )
        report["check_positions"] = int(len(check))
        for key in ("last_checkpoint", "average_bn_averaged", "average_bn_recomputed"):
            print(f"  {key}: {report[key]}", flush=True)

    checkpoint = {key: value for key, value in last.items() if key != "optimizer_state_dict"}
    checkpoint.update(
        {
            "kind": "swa_checkpoint",
            "model_state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
            "model_summary": model.summary_dict(),
            "swa": {
                "sources": paths,
                "count": len(paths),
                "batchnorm": batchnorm,
                "bn_positions": bn_positions,
                "buffer": args.buffer,
                "seed": args.seed,
                "report": report,
            },
        }
    )
    with atomic_output(out_path) as stream:
        torch.save(checkpoint, stream)
    print(f"saved {out_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

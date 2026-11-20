from __future__ import annotations

"""Distill the averaged policy and value predictions of two teachers into a single network."""

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from alphago_zero.augmentations import (
    apply_dihedral_to_planes,
    apply_dihedral_to_policy,
    random_dihedral_index,
)
from alphago_zero.cli_types import (
    ensure_directory,
    non_negative_float,
    non_negative_int,
    positive_float,
    positive_int,
)
from alphago_zero.config import NetworkConfig
from alphago_zero.network import AlphaGoZeroNet


def _filter_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields recognized by the configuration dataclass."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def load_teacher(path: str | Path, device: torch.device) -> tuple[AlphaGoZeroNet, dict[str, Any]]:
    """Load a teacher network and return it and the full checkpoint payload."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = NetworkConfig(**_filter_fields(NetworkConfig, payload["config"]["network"]))
    net = AlphaGoZeroNet(cfg).to(device)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    return net, payload


def load_positions(buffer_paths: list[str], max_positions: int, seed: int) -> np.ndarray:
    """Collect encoded positions from the replay buffers in resume snapshots."""
    parts: list[np.ndarray] = []
    for p in buffer_paths:
        snap = torch.load(Path(p), map_location="cpu", weights_only=False)
        states = np.asarray(snap["replay_buffer"]["states"])
        if states.ndim != 4:
            raise ValueError(f"{p}: expected states [N,C,H,W], got {states.shape}")
        parts.append(states.astype(np.uint8))
        print(f"  buffer {p}: {states.shape[0]} positions", flush=True)
    corpus = np.concatenate(parts, axis=0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(corpus.shape[0])
    corpus = corpus[order]
    if 0 < max_positions < corpus.shape[0]:
        corpus = corpus[:max_positions]
    return corpus


@torch.no_grad()
def label_corpus(
    teachers: list[AlphaGoZeroNet],
    corpus: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Label positions with the mean teacher policy and value."""
    n = corpus.shape[0]
    action_size = teachers[0].action_size
    policies = np.zeros((n, action_size), dtype=np.float32)
    values = np.zeros((n,), dtype=np.float32)
    for start in range(0, n, batch_size):
        chunk = corpus[start : start + batch_size]
        x = torch.from_numpy(chunk).float().to(device)
        probs_sum: torch.Tensor | None = None
        values_sum: torch.Tensor | None = None
        for net in teachers:
            logits, value = net(x)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            v = value.squeeze(1)
            values_sum = v if values_sum is None else values_sum + v
        assert probs_sum is not None and values_sum is not None
        policies[start : start + batch_size] = (probs_sum / len(teachers)).cpu().numpy()
        values[start : start + batch_size] = (values_sum / len(teachers)).cpu().numpy()
    return policies, values


def _augment_batch(
    states: np.ndarray,
    policies: np.ndarray,
    board_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one random symmetry per example to both input and policy target."""
    out_s = np.empty_like(states)
    out_p = np.empty_like(policies)
    for i in range(states.shape[0]):
        t = random_dihedral_index(rng)
        out_s[i] = apply_dihedral_to_planes(states[i], t)
        out_p[i] = apply_dihedral_to_policy(policies[i], board_size, t)
    return out_s, out_p


def train_student(
    student: AlphaGoZeroNet,
    corpus: np.ndarray,
    target_policy: np.ndarray,
    target_value: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    value_weight: float,
    device: torch.device,
    seed: int,
    holdout_fraction: float = 0.02,
) -> dict[str, Any]:
    """Train the student against ensemble predictions and report metrics per epoch."""
    rng = np.random.default_rng(seed)
    n = corpus.shape[0]
    n_hold = max(256, int(n * holdout_fraction))
    train_idx, hold_idx = np.arange(n - n_hold), np.arange(n - n_hold, n)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    history: list[dict[str, float]] = []
    student.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train_idx)
        policy_sum, value_sum, steps = 0.0, 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            s, p = _augment_batch(
                corpus[idx].astype(np.float32),
                target_policy[idx],
                student.board_size,
                rng,
            )
            x = torch.from_numpy(s).to(device)
            tp = torch.from_numpy(p).to(device)
            tv = torch.from_numpy(target_value[idx]).to(device)

            logits, value = student(x)
            policy_loss = -(tp * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value.squeeze(1), tv)
            loss = policy_loss + value_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            policy_sum += float(policy_loss.item())
            value_sum += float(value_loss.item())
            steps += 1

        hold = evaluate_student(
            student, corpus[hold_idx], target_policy[hold_idx], target_value[hold_idx],
            batch_size=batch_size, device=device,
        )
        row = {
            "epoch": epoch,
            "train_policy_ce": round(policy_sum / max(1, steps), 4),
            "train_value_mse": round(value_sum / max(1, steps), 4),
            "holdout_policy_ce": hold[0],
            "holdout_value_mse": hold[1],
        }
        history.append(row)
        print(f"  epoch {epoch}: {row}", flush=True)
        student.train()
    return {"epochs": history, "holdout_size": int(n_hold)}


@torch.no_grad()
def evaluate_student(
    student: AlphaGoZeroNet,
    states: np.ndarray,
    target_policy: np.ndarray,
    target_value: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Measure student cross-entropy and MSE on held-out teacher targets without augmentation."""
    student.eval()
    policy_sum, value_sum, batches = 0.0, 0.0, 0
    for start in range(0, states.shape[0], batch_size):
        x = torch.from_numpy(states[start : start + batch_size].astype(np.float32)).to(device)
        tp = torch.from_numpy(target_policy[start : start + batch_size]).to(device)
        tv = torch.from_numpy(target_value[start : start + batch_size]).to(device)
        logits, value = student(x)
        policy_sum += float((-(tp * F.log_softmax(logits, dim=1)).sum(dim=1)).mean().item())
        value_sum += float(F.mse_loss(value.squeeze(1), tv).item())
        batches += 1
    return round(policy_sum / max(1, batches), 4), round(value_sum / max(1, batches), 4)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Distill two AlphaGo Zero teachers into one network",
    )
    parser.add_argument("--teacher-a", required=True, help="teacher A checkpoint (.pt), which supplies the student's configuration")
    parser.add_argument("--teacher-b", required=True, help="teacher B checkpoint (.pt)")
    parser.add_argument(
        "--buffer", action="append", required=True,
        help="resume_state.pt containing replay data (repeatable)",
    )
    parser.add_argument(
        "--student-init", default=None,
        help="checkpoint for initializing the student (defaults to teacher A)",
    )
    parser.add_argument(
        "--max-positions",
        type=non_negative_int,
        default=0,
        help="maximum corpus size, or zero to use every position",
    )
    parser.add_argument("--epochs", type=positive_int, default=6, help="number of training epochs (positive)")
    parser.add_argument("--batch-size", type=positive_int, default=256, help="training batch size (positive)")
    parser.add_argument(
        "--label-batch",
        type=positive_int,
        default=512,
        help="teacher labeling batch size (positive)",
    )
    parser.add_argument(
        "--lr",
        type=positive_float,
        default=5e-4,
        help="positive finite SGD learning rate for the student",
    )
    parser.add_argument(
        "--value-weight",
        type=non_negative_float,
        default=1.0,
        help="nonnegative weight for value loss (zero for policy only)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--out", required=True, help="output student checkpoint (.pt)")
    args = parser.parse_args(argv)
    out_path = Path(args.out)
    ensure_directory(parser, out_path.parent, "--out")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    print("== loading teachers ==", flush=True)
    teacher_a, payload_a = load_teacher(args.teacher_a, device)
    teacher_b, _payload_b = load_teacher(args.teacher_b, device)
    print(f"  A: {teacher_a.summary_dict()}", flush=True)
    print(f"  B: {teacher_b.summary_dict()}", flush=True)

    print("== position corpus ==", flush=True)
    corpus = load_positions(args.buffer, args.max_positions, args.seed)
    print(f"  total positions: {corpus.shape}", flush=True)

    print("== labeling with the ensemble ==", flush=True)
    t0 = time.time()
    target_policy, target_value = label_corpus(
        [teacher_a, teacher_b], corpus, args.label_batch, device
    )
    print(f"  labeling time: {time.time() - t0:.1f}s", flush=True)

    print("== student ==", flush=True)
    init_path = args.student_init or args.teacher_a
    student, _ = load_teacher(init_path, device)
    print(f"  init from {init_path}", flush=True)

    t0 = time.time()
    report = train_student(
        student, corpus, target_policy, target_value,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        value_weight=args.value_weight, device=device, seed=args.seed,
    )
    print(f"  training time: {time.time() - t0:.1f}s", flush=True)

    checkpoint = {
        "kind": "distilled_checkpoint",
        "config": payload_a["config"],
        "model_state_dict": {k: v.cpu() for k, v in student.state_dict().items()},
        "distill": {
            "teacher_a": str(args.teacher_a),
            "teacher_b": str(args.teacher_b),
            "buffers": list(args.buffer),
            "positions": int(corpus.shape[0]),
            "epochs": args.epochs,
            "lr": args.lr,
            "value_weight": args.value_weight,
            "seed": args.seed,
            "report": report,
        },
    }
    torch.save(checkpoint, out_path)
    print(f"saved {out_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

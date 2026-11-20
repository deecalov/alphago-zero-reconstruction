from __future__ import annotations

"""Train on replay using policy cross-entropy, squared value error and SGD weight decay."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .config import AlphaGoZeroConfig, paper_learning_rate
from .network import AlphaGoZeroNet
from .replay_buffer import ReplayBuffer


@dataclass(slots=True)
class TrainMetrics:
    """Mean metrics across a sequence of optimizer steps."""

    total_loss: float
    policy_loss: float
    value_loss: float
    learning_rate: float
    steps: int

    # Auxiliary terms are zero when the corresponding heads are disabled.
    ownership_loss: float = 0.0
    score_loss: float = 0.0


def create_optimizer(model: AlphaGoZeroNet, config: AlphaGoZeroConfig) -> torch.optim.Optimizer:
    """Build the momentum SGD optimizer with the configured weight-decay groups."""
    training = config.training
    if not training.weight_decay_exclude_norm_bias:
        return torch.optim.SGD(
            model.parameters(),
            lr=training.learning_rate,
            momentum=training.momentum,
            weight_decay=training.weight_decay,
        )
    parameters = list(model.parameters())
    return torch.optim.SGD(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": training.weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=training.learning_rate,
        momentum=training.momentum,
    )


def _get_learning_rate(global_step: int, config: AlphaGoZeroConfig) -> float:
    """Return the learning rate for the current optimizer step."""
    if config.training.lr_schedule == "paper":
        return float(paper_learning_rate(global_step))
    return float(config.training.learning_rate)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _compute_losses(
    policy_logits: Tensor,
    value_pred: Tensor,
    target_policy: Tensor,
    target_value: Tensor,
    policy_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute policy cross-entropy and value MSE, with optional per-position policy weights."""
    log_probs = F.log_softmax(policy_logits, dim=1)

    # The search target is a distribution and may assign probability to several moves.
    per_sample_ce = -(target_policy * log_probs).sum(dim=1)
    if policy_weights is not None:
        policy_loss = (per_sample_ce * policy_weights).sum() / policy_weights.sum().clamp_min(
            1.0
        )
    else:
        policy_loss = per_sample_ce.mean()

    # Flatten [B, 1] predictions to match [B] outcomes.
    value_loss = F.mse_loss(value_pred.squeeze(1), target_value)

    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss


def _aux_weight_scale(global_step: int, warmup_steps: int) -> float:
    """Return the auxiliary warmup multiplier in [0, 1]. Zero warmup steps apply full weight immediately."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, global_step / float(warmup_steps)))


def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compute MSE only where auxiliary targets are present, for vector or scalar predictions."""
    if pred.dim() == 1:
        per_sample = (pred - target) ** 2
    else:
        per_sample = ((pred - target) ** 2).mean(dim=1)
    denom = mask.sum().clamp_min(1.0)
    return (per_sample * mask).sum() / denom


def evaluate_on_buffer(
    model: AlphaGoZeroNet,
    replay_buffer: ReplayBuffer,
    positions: int,
    rng: np.random.Generator,
    chunk_size: int = 1024,
) -> dict[str, float] | None:
    """Measure policy cross-entropy and value MSE on sampled replay positions without updating the model."""
    count = min(int(positions), len(replay_buffer))
    if count <= 0:
        return None
    batch = replay_buffer.sample_batch_full(batch_size=count, augment=False, rng=rng)

    was_training = model.training
    model.eval()
    ce_sum, weight_sum, squared_error_sum = 0.0, 0.0, 0.0
    try:
        with torch.no_grad():
            for start in range(0, count, chunk_size):
                stop = start + chunk_size
                states = torch.from_numpy(batch.states[start:stop]).float().to(model.device)
                target_policy = torch.from_numpy(batch.policies[start:stop]).float().to(model.device)
                target_value = torch.from_numpy(batch.values[start:stop]).float().to(model.device)
                weights = torch.from_numpy(batch.policy_weights[start:stop]).float().to(model.device)
                policy_logits, value = model(states)
                per_sample_ce = -(target_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
                ce_sum += float((per_sample_ce * weights).sum().item())
                weight_sum += float(weights.sum().item())
                squared_error_sum += float(((value.squeeze(1) - target_value) ** 2).sum().item())
    finally:
        if was_training:
            model.train(True)
    return {
        "policy_ce": ce_sum / max(weight_sum, 1.0),
        "value_mse": squared_error_sum / count,
        "positions": float(count),
    }


def train_for_steps(
    model: AlphaGoZeroNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    config: AlphaGoZeroConfig,
    rng: np.random.Generator,
    global_step: int,
) -> tuple[TrainMetrics, int]:
    """Run a sequence of optimizer steps on samples from replay."""
    if len(replay_buffer) == 0 or config.training.gradient_steps_per_iteration == 0:
        metrics = TrainMetrics(
            total_loss=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            learning_rate=_get_learning_rate(global_step, config),
            steps=0,
        )
        return metrics, global_step

    model.train()

    # Use bfloat16 autocast for CUDA training without gradient scaling.
    amp_enabled = bool(config.runtime.use_amp)
    device_type = model.device.type

    aux_enabled = bool(config.network.aux_heads)
    ownership_weight = float(config.training.ownership_loss_weight)
    score_weight = float(config.training.score_loss_weight)

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    ownership_loss_sum = 0.0
    score_loss_sum = 0.0
    last_lr = 0.0

    for _ in range(config.training.gradient_steps_per_iteration):
        lr = _get_learning_rate(global_step, config)
        _set_optimizer_lr(optimizer, lr)
        last_lr = lr

        batch = replay_buffer.sample_batch_full(
            batch_size=config.training.batch_size,
            augment=True,
            rng=rng,
        )

        states = torch.from_numpy(batch.states).float().to(model.device)
        target_policy = torch.from_numpy(batch.policies).float().to(model.device)
        target_value = torch.from_numpy(batch.values).float().to(model.device)
        policy_weights = torch.from_numpy(batch.policy_weights).float().to(model.device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
            outputs = model.forward_full(states)
            total_loss, policy_loss, value_loss = _compute_losses(
                policy_logits=outputs["policy_logits"],
                value_pred=outputs["value"],
                target_policy=target_policy,
                target_value=target_value,
                policy_weights=policy_weights,
            )

            ownership_loss: Tensor | None = None
            score_loss: Tensor | None = None
            if aux_enabled and "ownership" in outputs:
                aux_mask = torch.from_numpy(batch.aux_mask).float().to(model.device)
                target_ownership = (
                    torch.from_numpy(batch.ownerships).float().to(model.device)
                )
                target_score = torch.from_numpy(batch.scores).float().to(model.device)
                ownership_loss = _masked_mse(outputs["ownership"], target_ownership, aux_mask)
                score_loss = _masked_mse(outputs["score"].squeeze(1), target_score, aux_mask)
                warmup_scale = _aux_weight_scale(
                    global_step, config.training.aux_warmup_steps
                )
                total_loss = (
                    total_loss
                    + warmup_scale * ownership_weight * ownership_loss
                    + warmup_scale * score_weight * score_loss
                )

        total_loss.backward()
        optimizer.step()

        total_loss_sum += float(total_loss.item())
        policy_loss_sum += float(policy_loss.item())
        value_loss_sum += float(value_loss.item())
        if ownership_loss is not None:
            ownership_loss_sum += float(ownership_loss.item())
        if score_loss is not None:
            score_loss_sum += float(score_loss.item())
        global_step += 1

    steps = config.training.gradient_steps_per_iteration
    metrics = TrainMetrics(
        total_loss=total_loss_sum / steps,
        policy_loss=policy_loss_sum / steps,
        value_loss=value_loss_sum / steps,
        learning_rate=last_lr,
        steps=steps,
        ownership_loss=ownership_loss_sum / steps,
        score_loss=score_loss_sum / steps,
    )
    return metrics, global_step

from __future__ import annotations

"""A shared residual trunk with policy and value heads, plus optional auxiliary training heads."""

from typing import Any, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import NetworkConfig


@runtime_checkable
class PolicyValueModel(Protocol):
    """Inference interface shared by search, match play and the GTP engine."""

    def predict_single(self, state_planes: np.ndarray, /) -> tuple[np.ndarray, float]:
        """Evaluate one [C, H, W] position and return action logits and a scalar value in [-1, 1]."""
        ...

    def predict_batch(self, states_planes: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate [B, C, H, W] positions and return [B, A] logits and [B] values."""
        ...

    def eval(self) -> Any:
        """Enable inference mode so BatchNorm uses its running statistics."""
        ...


class ConvBlock(nn.Module):
    """Convolution, BatchNorm and ReLU block used by the input stem and output heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # Apply the operations in the order used in the paper.
        # conv -> batch norm -> rectifier non-linearity.
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=False)
        return x


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection. Apply the final ReLU after adding the input."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        # First convolutional branch with activation.
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=False)

        # Leave the second convolution unactivated until after the skip addition.
        out = self.conv2(out)
        out = self.bn2(out)

        # Learn a residual correction to the identity.
        out = out + residual
        out = F.relu(out, inplace=False)
        return out


class PolicyHead(nn.Module):
    """Project to two channels, flatten and emit one logit per board point plus pass."""

    def __init__(self, channels: int, board_size: int, action_size: int) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_size * board_size, action_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=False)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits


class ValueHead(nn.Module):
    """Project to one channel, flatten and predict a scalar outcome through a hidden layer and tanh."""

    def __init__(self, channels: int, board_size: int, hidden_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=False)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x


class OwnershipHead(nn.Module):
    """Predict terminal ownership at each point from the player-to-move perspective."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        # Flatten [B, 1, H, W] to [B, H*W] with values in [-1, 1].
        return torch.tanh(torch.flatten(x, start_dim=1))


class ScoreHead(nn.Module):
    """Predict the normalized terminal score margin from the player-to-move perspective."""

    def __init__(self, channels: int, board_size: int, hidden_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=False)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.fc2(x)
        return torch.tanh(x)


class AlphaGoZeroNet(nn.Module):
    """Residual policy/value network taking [B, 17, H, W] and returning action logits and bounded values."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.board_size = config.board_size
        self.action_size = config.action_size

        # Input stem of the residual tower.
        self.input_block = ConvBlock(
            in_channels=config.num_input_planes,
            out_channels=config.channels,
            kernel_size=3,
        )

        # Residual tower.
        self.residual_tower = nn.ModuleList(
            [ResidualBlock(config.channels) for _ in range(config.num_res_blocks)]
        )

        # Two heads share the same board representation.
        self.policy_head = PolicyHead(
            channels=config.channels,
            board_size=config.board_size,
            action_size=config.action_size,
        )
        self.value_head = ValueHead(
            channels=config.channels,
            board_size=config.board_size,
            hidden_size=config.value_head_hidden,
        )

        # Instantiate optional heads only when requested to preserve compatibility with classic checkpoints.
        self.ownership_head: OwnershipHead | None = None
        self.score_head: ScoreHead | None = None
        if config.aux_heads:
            self.ownership_head = OwnershipHead(channels=config.channels)
            self.score_head = ScoreHead(
                channels=config.channels,
                board_size=config.board_size,
                hidden_size=config.value_head_hidden,
            )

        # Set the initialization explicitly for reproducibility.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use Kaiming initialization for layers with ReLU activations."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Map [batch, planes, height, width] to [B, action_size] logits and [B, 1] values."""
        x = self.input_block(x)
        for block in self.residual_tower:
            x = block(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def forward_full(self, x: Tensor) -> dict[str, Tensor]:
        """Evaluate all configured heads. Return policy logits, values and any auxiliary ownership and score
        predictions by name.
        """
        x = self.input_block(x)
        for block in self.residual_tower:
            x = block(x)

        outputs: dict[str, Tensor] = {
            "policy_logits": self.policy_head(x),
            "value": self.value_head(x),
        }
        if self.ownership_head is not None:
            outputs["ownership"] = self.ownership_head(x)
        if self.score_head is not None:
            outputs["score"] = self.score_head(x)
        return outputs

    @property
    def device(self) -> torch.device:
        """Return the device holding the model parameters."""
        return next(self.parameters()).device

    def predict_single(self, state_planes: np.ndarray) -> tuple[np.ndarray, float]:
        """Evaluate one encoded position and return NumPy logits and a scalar value for search."""
        if state_planes.ndim != 3:
            raise ValueError(
                "expected a state tensor of shape [C, H, W], "
                f"got {state_planes.shape}"
            )

        x = torch.from_numpy(state_planes).float().unsqueeze(0).to(self.device)
        policy_logits, value = self._forward_inference(x)

        policy_logits_np = policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
        value_float = float(value.squeeze(0).item())
        return policy_logits_np, value_float

    def predict_batch(self, states_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of MCTS leaves in one pass. Return [B, A] logits and [B] values."""
        if states_planes.ndim != 4:
            raise ValueError(
                "expected a state batch of shape [B, C, H, W], "
                f"got {states_planes.shape}"
            )

        x = torch.from_numpy(states_planes).float().to(self.device)
        policy_logits, value = self._forward_inference(x)

        logits_np = policy_logits.detach().cpu().numpy().astype(np.float32)
        values_np = value.squeeze(1).detach().cpu().numpy().astype(np.float32)
        return logits_np, values_np

    def _forward_inference(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Run inference without gradients. Avoid recursively switching modes when already in eval mode and
        restore the previous training state.
        """
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                policy_logits, value = self(x)
        finally:
            if was_training:
                self.train(True)
        return policy_logits, value

    def summary_dict(self) -> dict[str, int]:
        """Summarize network dimensions and parameter counts for run logs."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "board_size": self.board_size,
            "action_size": self.action_size,
            "channels": self.config.channels,
            "num_res_blocks": self.config.num_res_blocks,
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
        }

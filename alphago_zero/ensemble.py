"""Combine several networks in one search. Average policy probabilities and values so networks of different
widths or depths can contribute to the same leaf evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import GameConfig
from .network import AlphaGoZeroNet, PolicyValueModel

# Floor probabilities before taking their logarithms to handle actions that underflow for every
# member.
_PROBABILITY_FLOOR = float(np.finfo(np.float64).tiny)


class EnsembleNet(PolicyValueModel):
    """Average member policies after softmax and average scalar values."""

    def __init__(self, members: Sequence[AlphaGoZeroNet]) -> None:
        networks = tuple(members)
        if not networks:
            raise ValueError("an ensemble requires at least one network")
        reference = networks[0].config
        for index, member in enumerate(networks[1:], start=1):
            for field in ("board_size", "num_input_planes"):
                expected = getattr(reference, field)
                actual = getattr(member.config, field)
                if actual != expected:
                    raise ValueError(
                        f"ensemble member #{index} is incompatible with member #0: {field} = {actual}, expected "
                        f"{expected} (all networks must receive the same input planes and choose "
                        "from the same action space)"
                    )
        self.members: tuple[AlphaGoZeroNet, ...] = networks
        self.board_size: int = reference.board_size
        self.action_size: int = reference.action_size

    def predict_single(self, state_planes: np.ndarray) -> tuple[np.ndarray, float]:
        """Evaluate one [C, H, W] position and return [A] log probabilities and a mean value."""
        outputs = [member.predict_single(state_planes) for member in self.members]
        logits, values = self._combine(
            np.stack([member_logits for member_logits, _value in outputs]),
            np.asarray([member_value for _logits, member_value in outputs], dtype=np.float64),
        )
        return logits, float(values)

    def predict_batch(self, states_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a [B, C, H, W] batch and return [B, A] log probabilities and [B] values."""
        outputs = [member.predict_batch(states_planes) for member in self.members]
        return self._combine(
            np.stack([member_logits for member_logits, _values in outputs]),
            np.stack([member_values for _logits, member_values in outputs]),
        )

    def eval(self) -> EnsembleNet:
        """Put all members in inference mode and return the ensemble."""
        for member in self.members:
            member.eval()
        return self

    @staticmethod
    def _combine(logits: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reduce the member dimension of logits and values. Shift logits by their maxima and compute
        softmax in float64 for numerical stability.
        """
        shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        mean_probabilities = probabilities.mean(axis=0)
        log_policy = np.log(np.maximum(mean_probabilities, _PROBABILITY_FLOOR))
        mean_values = values.astype(np.float64).mean(axis=0)
        return log_policy.astype(np.float32), mean_values.astype(np.float32)


def require_same_rules(games: Sequence[GameConfig], labels: Sequence[str]) -> None:
    """Check board size and komi across checkpoint configurations. Use the supplied member labels in diagnostics."""
    if not games:
        return
    reference = games[0]
    for label, game in zip(labels[1:], games[1:], strict=True):
        if game.board_size != reference.board_size:
            raise ValueError(
                f"{label}: board size {game.board_size} differs from {labels[0]} "
                f"({reference.board_size}). Ensemble members must share the same rules"
            )
        if game.komi != reference.komi:
            raise ValueError(
                f"{label}: komi {game.komi:g} differs from {labels[0]} ({reference.komi:g}); "
                "ensemble members must share the same rules because value depends on komi"
            )

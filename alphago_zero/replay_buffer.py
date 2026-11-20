from __future__ import annotations

"""Store self-play training positions and sample uniformly from recent experience."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .augmentations import (
    apply_dihedral_to_board,
    apply_dihedral_to_planes,
    apply_dihedral_to_policy,
    augment_state_and_policy,
    random_dihedral_index,
)


@dataclass(slots=True)
class TrainingExample:
    """One training position with encoded board, search target and game outcome."""

    state: np.ndarray
    policy: np.ndarray
    value: float
    ownership: np.ndarray | None = None
    score: float = 0.0
    policy_weight: float = 1.0


@dataclass(slots=True)
class SampleBatch:
    """Training batch with auxiliary targets and masks. Missing ownership targets are padded with zeros."""

    states: np.ndarray  # [B, C, H, W] float32
    policies: np.ndarray  # [B, A] float32
    values: np.ndarray  # [B] float32
    ownerships: np.ndarray  # [B, H*W] float32
    scores: np.ndarray  # [B] float32
    aux_mask: np.ndarray  # [B] float32, one where ownership and score targets are available.
    policy_weights: np.ndarray  # [B] float32


class ReplayBuffer:
    """Fixed-capacity ring of self-play examples."""

    def __init__(
        self,
        capacity: int,
        board_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.board_size = int(board_size)
        # List storage provides constant-time random indexing for minibatches.
        self._buffer: list[TrainingExample] = []
        self._position = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, example: TrainingExample) -> None:
        """Append an example, overwriting the oldest when the ring is full."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(example)
        else:
            self._buffer[self._position] = example
            self._position = (self._position + 1) % self.capacity

    def add_many(self, examples: Iterable[TrainingExample]) -> None:
        for example in examples:
            self.add(example)

    def add_game(self, examples: Iterable[TrainingExample]) -> None:
        """Add all examples from a completed game."""
        self.add_many(examples)

    def sample_batch(
        self,
        batch_size: int,
        augment: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample state, policy and value arrays, applying the same board symmetry to state and policy."""
        if len(self._buffer) == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        rng = rng if rng is not None else self.rng
        replace = len(self._buffer) < batch_size
        indices = rng.choice(len(self._buffer), size=batch_size, replace=replace)

        states: list[np.ndarray] = []
        policies: list[np.ndarray] = []
        values: list[float] = []

        # Direct list indexing has constant cost.
        for idx in indices:
            example = self._buffer[int(idx)]

            state = example.state.astype(np.float32, copy=True)
            policy = example.policy.astype(np.float32, copy=True)
            value = float(example.value)

            if augment:
                state, policy = augment_state_and_policy(
                    state_planes=state,
                    policy=policy,
                    board_size=self.board_size,
                    rng=rng,
                )

            states.append(state)
            policies.append(policy)
            values.append(value)

        return (
            np.stack(states, axis=0).astype(np.float32),
            np.stack(policies, axis=0).astype(np.float32),
            np.asarray(values, dtype=np.float32),
        )

    def sample_batch_full(
        self,
        batch_size: int,
        augment: bool = True,
        rng: np.random.Generator | None = None,
    ) -> SampleBatch:
        """Sample a full training batch. Apply a common symmetry to the board, policy and ownership map for
        each example.
        """
        if len(self._buffer) == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        rng = rng if rng is not None else self.rng
        replace = len(self._buffer) < batch_size
        indices = rng.choice(len(self._buffer), size=batch_size, replace=replace)

        area = self.board_size * self.board_size
        states: list[np.ndarray] = []
        policies: list[np.ndarray] = []
        values = np.empty(batch_size, dtype=np.float32)
        ownerships = np.zeros((batch_size, area), dtype=np.float32)
        scores = np.zeros(batch_size, dtype=np.float32)
        aux_mask = np.zeros(batch_size, dtype=np.float32)
        policy_weights = np.empty(batch_size, dtype=np.float32)

        for i, idx in enumerate(indices):
            example = self._buffer[int(idx)]

            state = example.state.astype(np.float32, copy=True)
            policy = example.policy.astype(np.float32, copy=True)
            ownership = (
                example.ownership.astype(np.float32, copy=True)
                if example.ownership is not None
                else None
            )

            if augment:
                t = random_dihedral_index(rng)
                state = apply_dihedral_to_planes(state, t)
                policy = apply_dihedral_to_policy(policy, self.board_size, t)
                if ownership is not None:
                    ownership = apply_dihedral_to_board(ownership, t)

            states.append(state)
            policies.append(policy)
            values[i] = float(example.value)
            if ownership is not None:
                ownerships[i] = ownership.reshape(-1)
                scores[i] = float(example.score)
                aux_mask[i] = 1.0
            policy_weights[i] = float(example.policy_weight)

        return SampleBatch(
            states=np.stack(states, axis=0).astype(np.float32),
            policies=np.stack(policies, axis=0).astype(np.float32),
            values=values,
            ownerships=ownerships,
            scores=scores,
            aux_mask=aux_mask,
            policy_weights=policy_weights,
        )

    def summary_dict(self) -> dict[str, int]:
        """Return a short diagnostic summary."""
        return {
            "capacity": self.capacity,
            "size": len(self),
            "board_size": self.board_size,
        }

    # ------------------------------------------------------------------
    # Serialization for full-run resume
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Serialize the ring and its write index. Binary state planes fit losslessly in uint8."""
        if self._buffer:
            states = np.stack([e.state for e in self._buffer], axis=0).astype(np.uint8)
            policies = np.stack([e.policy for e in self._buffer], axis=0).astype(np.float32)
            values = np.asarray([e.value for e in self._buffer], dtype=np.float32)

            zero_ownership = np.zeros(
                (self.board_size, self.board_size), dtype=np.int8
            )
            ownerships = np.stack(
                [
                    e.ownership.astype(np.int8) if e.ownership is not None else zero_ownership
                    for e in self._buffer
                ],
                axis=0,
            )
            aux_mask = np.asarray(
                [e.ownership is not None for e in self._buffer], dtype=np.uint8
            )
            scores = np.asarray([e.score for e in self._buffer], dtype=np.float32)
            policy_weights = np.asarray(
                [e.policy_weight for e in self._buffer], dtype=np.float32
            )
        else:
            states = np.zeros((0,), dtype=np.uint8)
            policies = np.zeros((0,), dtype=np.float32)
            values = np.zeros((0,), dtype=np.float32)
            ownerships = np.zeros((0,), dtype=np.int8)
            aux_mask = np.zeros((0,), dtype=np.uint8)
            scores = np.zeros((0,), dtype=np.float32)
            policy_weights = np.zeros((0,), dtype=np.float32)
        return {
            "capacity": self.capacity,
            "board_size": self.board_size,
            "position": self._position,
            "states": states,
            "policies": policies,
            "values": values,
            "ownerships": ownerships,
            "aux_mask": aux_mask,
            "scores": scores,
            "policy_weights": policy_weights,
        }

    def load_state_dict(self, payload: dict[str, Any], *, keep_capacity: bool = False) -> None:
        """Restore the ring and write index. When keeping a different capacity, retain the newest examples
        in chronological order.
        """
        states = np.asarray(payload["states"])
        policies = np.asarray(payload["policies"])
        values = np.asarray(payload["values"])

        # Supply defaults for snapshots created before auxiliary targets and policy weights were introduced.
        n = states.shape[0]
        raw_ownerships = payload.get("ownerships")
        raw_aux_mask = payload.get("aux_mask")
        raw_scores = payload.get("scores")
        raw_policy_weights = payload.get("policy_weights")

        has_aux = (
            raw_ownerships is not None
            and raw_aux_mask is not None
            and raw_scores is not None
            and np.asarray(raw_ownerships).shape[0] == n
        )
        ownerships = np.asarray(raw_ownerships) if has_aux else np.zeros((n, 1, 1), np.int8)
        aux_mask = np.asarray(raw_aux_mask) if has_aux else np.zeros(n, np.uint8)
        scores = np.asarray(raw_scores) if has_aux else np.zeros(n, np.float32)
        if raw_policy_weights is not None and np.asarray(raw_policy_weights).shape[0] == n:
            policy_weights = np.asarray(raw_policy_weights)
        else:
            policy_weights = np.ones(n, dtype=np.float32)

        if not keep_capacity:
            self.capacity = int(payload["capacity"])
        self.board_size = int(payload["board_size"])
        examples = [
            TrainingExample(
                state=states[i].astype(np.float32),
                policy=policies[i].astype(np.float32),
                value=float(values[i]),
                ownership=(
                    ownerships[i].astype(np.float32)
                    if has_aux and bool(aux_mask[i])
                    else None
                ),
                score=float(scores[i]) if has_aux else 0.0,
                policy_weight=float(policy_weights[i]),
            )
            for i in range(n)
        ]
        if keep_capacity:
            # A full ring is stored in physical order. Rotate to oldest-first
            # before trimming, also when the destination grows or stays equal.
            if n == int(payload["capacity"]):
                position = int(payload["position"])
                examples = examples[position:] + examples[:position]
            examples = examples[-self.capacity :]
        self._buffer = examples
        # Warmstart now stores oldest-first; full resume retains physical order.
        self._position = 0 if keep_capacity else int(payload["position"])

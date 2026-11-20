"""Deterministic fake network and independent reference implementation for Gumbel tests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.gumbel import considered_visit_sequence

BOARD_SIZE = 5
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1
_TINY = float(np.finfo(np.float32).tiny)


class FakeNet:
    """Generate logits and values as functions of input planes and a designated board point."""

    def __init__(
        self,
        seed: int = 0,
        *,
        good_point: int | None = None,
        bias_scale: float = 0.7,
    ) -> None:
        rng = np.random.default_rng(seed)
        area = BOARD_SIZE * BOARD_SIZE
        flat = 17 * area
        self._w_policy = rng.normal(0.0, 0.05, size=(ACTION_SIZE, flat))
        self._w_value = rng.normal(0.0, 0.15, size=flat)
        self._w_value[16 * area :] = 0.0  # Ignore the side-to-play plane in this estimate.
        self._bias = rng.normal(0.0, bias_scale, size=ACTION_SIZE)
        self.good_point = good_point
        self.calls = 0

    def predict_single(self, planes: np.ndarray, /) -> tuple[np.ndarray, float]:
        self.calls += 1
        flat = np.asarray(planes, dtype=np.float64).ravel()
        logits = (self._w_policy @ flat + self._bias).astype(np.float32)
        value = float(np.tanh(self._w_value @ flat))
        if self.good_point is not None:
            # Planes zero and one hold the current player's and opponent's stones.
            own = float(planes[0].ravel()[self.good_point])
            opponent = float(planes[1].ravel()[self.good_point])
            value = max(-1.0, min(1.0, value + 1.5 * (own - opponent)))
        # Choose an exactly representable float32 value for scalar and batched inference.
        return logits, float(np.float32(value))

    def predict_batch(self, planes: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        outputs = [self.predict_single(item) for item in planes]
        logits = np.stack([o[0] for o in outputs]).astype(np.float32)
        values = np.asarray([o[1] for o in outputs], dtype=np.float32)
        return logits, values

    def eval(self) -> FakeNet:
        return self


def make_state(moves: tuple[int, ...] = (), max_moves: int | None = None) -> GoState:
    """Construct a 5x5 position with komi 2.5 after the specified moves."""
    state = GoState.new_game(board_size=BOARD_SIZE, komi=2.5, max_moves=max_moves)
    for action in moves:
        state = state.play(action)
    return state


def two_eyes_state_white_to_move() -> GoState:
    """Fill the board with Black except for two eyes. White has only the pass action."""
    board = np.full((BOARD_SIZE, BOARD_SIZE), BLACK, dtype=np.int8)
    board[0, 0] = 0
    board[2, 2] = 0
    return GoState.from_board(board, to_play=WHITE, komi=2.5)


# ----------------------------------------------------------------------
# Independent sequential Gumbel reference
# ----------------------------------------------------------------------


@dataclass
class RefNode:
    """Reference node represented with plain dictionaries."""

    state: GoState
    value: float = 0.0
    expanded: bool = False
    actions: list[int] = field(default_factory=list)
    prior: dict[int, float] = field(default_factory=dict)
    visits: dict[int, int] = field(default_factory=dict)
    total: dict[int, float] = field(default_factory=dict)
    children: dict[int, RefNode] = field(default_factory=dict)


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    norm = sum(exps)
    return [e / norm for e in exps]


def _expand(model: FakeNet, node: RefNode) -> None:
    """Expand from legal-action softmax priors rounded to float32."""
    node.expanded = True
    if node.state.is_terminal():
        node.value = node.state.outcome_for_player(node.state.to_play)
        return
    logits, value = model.predict_single(node.state.encode())
    node.value = value
    node.actions = node.state.legal_actions()
    probs = _softmax([float(logits[a]) for a in node.actions])
    for action, p in zip(node.actions, probs, strict=True):
        # Mirror the precision of the production prior representation.
        node.prior[action] = float(np.float32(p))
        node.visits[action] = 0
        node.total[action] = 0.0


def _stats(node: RefNode) -> tuple[list[int], list[float], list[float]]:
    visits = [node.visits[a] for a in node.actions]
    q = [node.total[a] / node.visits[a] if node.visits[a] > 0 else 0.0 for a in node.actions]
    logs = [math.log(max(node.prior[a], _TINY)) for a in node.actions]
    return visits, q, logs


def _sigma(node: RefNode, c_visit: float, c_scale: float) -> list[float]:
    """Compute completed Q with explicit loops for mixing, filling, normalizing and scaling."""
    visits, q, logs = _stats(node)
    count = len(visits)
    probs = _softmax(logs)
    total_visits = sum(visits)
    visited = [i for i in range(count) if visits[i] > 0]
    if visited:
        mass = sum(max(probs[i], _TINY) for i in visited)
        weighted_q = sum(max(probs[i], _TINY) * q[i] for i in visited) / mass
        mixed = (node.value + total_visits * weighted_q) / (total_visits + 1)
    else:
        mixed = node.value
    completed = [q[i] if visits[i] > 0 else mixed for i in range(count)]
    low, high = min(completed), max(completed)
    rescaled = [(c - low) / max(high - low, 1e-8) for c in completed]
    scale = (c_visit + max(visits)) * c_scale
    return [scale * r for r in rescaled]


def improved_policy(node: RefNode, c_visit: float, c_scale: float) -> dict[int, float]:
    """Form the improved policy as softmax(logits + transformed Q)."""
    _visits, _q, logs = _stats(node)
    sigma = _sigma(node, c_visit, c_scale)
    probs = _softmax([logs[i] + sigma[i] for i in range(len(logs))])
    return dict(zip(node.actions, probs, strict=True))


def _first_max(scores: list[float]) -> int:
    best = 0
    for index, score in enumerate(scores):
        if score > scores[best]:
            best = index
    return best


def _root_action(
    root: RefNode, gumbel: list[float], budget: int, max_considered: int, c_visit: float, c_scale: float
) -> int:
    visits, _q, logs = _stats(root)
    sigma = _sigma(root, c_visit, c_scale)
    sequence = considered_visit_sequence(min(max_considered, len(visits)), budget)
    considered = sequence[min(sum(visits), len(sequence) - 1)]
    top = max(logs)
    scores = []
    for i in range(len(visits)):
        score = max(-1e9, gumbel[i] + (logs[i] - top) + sigma[i])
        scores.append(score if visits[i] == considered else -math.inf)
    return root.actions[_first_max(scores)]


def _interior_action(node: RefNode, c_visit: float, c_scale: float) -> int:
    visits, _q, logs = _stats(node)
    sigma = _sigma(node, c_visit, c_scale)
    improved = _softmax([logs[i] + sigma[i] for i in range(len(visits))])
    total = sum(visits)
    scores = [improved[i] - visits[i] / (1 + total) for i in range(len(visits))]
    return node.actions[_first_max(scores)]


def reference_search(
    model: FakeNet,
    state: GoState,
    *,
    simulations: int,
    max_considered: int,
    gumbel: list[float],
    c_visit: float = 50.0,
    c_scale: float = 0.1,
) -> RefNode:
    """Implement sequential Gumbel search directly with supplied noise in legal-action order."""
    root = RefNode(state=state)
    _expand(model, root)
    for _ in range(simulations):
        node, path = root, []
        while node.expanded and not node.state.is_terminal():
            if node is root:
                action = _root_action(root, gumbel, simulations, max_considered, c_visit, c_scale)
            else:
                action = _interior_action(node, c_visit, c_scale)
            path.append((node, action))
            child = node.children.get(action)
            if child is None:
                child = RefNode(state=node.state.play(action))
                node.children[action] = child
            node = child
        if not node.expanded:
            _expand(model, node)
        value = node.value
        # Each edge belongs to the parent player, so flip the sign at every ply.
        for parent, action in reversed(path):
            value = -value
            parent.visits[action] += 1
            parent.total[action] += value
    return root


def reference_winner(root: RefNode, gumbel: list[float], c_visit: float, c_scale: float) -> int:
    """Choose the best noise + logit + transformed Q among the most visited root actions."""
    visits, _q, logs = _stats(root)
    sigma = _sigma(root, c_visit, c_scale)
    top = max(logs)
    most = max(visits)
    scores = []
    for i in range(len(visits)):
        score = max(-1e9, gumbel[i] + (logs[i] - top) + sigma[i])
        scores.append(score if visits[i] == most else -math.inf)
    return root.actions[_first_max(scores)]

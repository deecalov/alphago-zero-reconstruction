"""Gumbel AlphaZero search based on Danihelka et al. Root candidates compete under Sequential Halving and
interior nodes track the improved policy.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import SearchConfig
from .mcts import (
    AlphaGoZeroMCTS,
    EdgeStats,
    MCTSNode,
    SearchOutput,
    root_visit_statistics,
)
from .network import PolicyValueModel

# Floor priors before taking logs. A float32 underflow must not introduce infinities or NaNs.
_TINY = float(np.finfo(np.float32).tiny)
# Floor eligible scores to keep negative infinity reserved for ineligible visit counts.
_LOW_LOGIT = -1e9
# Avoid division by zero when all completed Q values are equal.
_EPSILON = 1e-8


# ----------------------------------------------------------------------
# Pure search mathematics, checked against mctx reference formulas.
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def considered_visit_sequence(max_considered: int, num_simulations: int) -> tuple[int, ...]:
    """Return the eligible visit count for each simulation in a Sequential Halving schedule."""
    if max_considered <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_considered)))
    sequence: list[int] = []
    visits = [0] * max_considered
    num_considered = max_considered
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for index in range(num_considered):
                visits[index] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


def mixed_value(
    raw_value: float,
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    prior_probs: np.ndarray,
) -> float:
    """Mix the network value with the prior-weighted mean Q of visited actions. The network value has weight
    one and the visited estimate has weight equal to the total visits.
    """
    visited = visit_counts > 0
    if not visited.any():
        return float(raw_value)
    total_visits = float(visit_counts.sum())
    probs = np.maximum(prior_probs[visited], _TINY)
    weighted_q = float((probs * q_values[visited]).sum() / probs.sum())
    return float((raw_value + total_visits * weighted_q) / (total_visits + 1.0))


def completed_q_scores(
    raw_value: float,
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    prior_probs: np.ndarray,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """Fill unvisited actions with the mixed value, normalize the completed Q values and apply the
    visit-dependent scale.
    """
    completed = np.where(
        visit_counts > 0,
        q_values,
        mixed_value(raw_value, q_values, visit_counts, prior_probs),
    )
    lowest = float(completed.min())
    spread = max(float(completed.max()) - lowest, _EPSILON)
    rescaled = (completed - lowest) / spread
    return (c_visit + float(visit_counts.max())) * c_scale * rescaled


def score_considered(
    considered_visit: float,
    gumbel: np.ndarray,
    logits: np.ndarray,
    sigma_q: np.ndarray,
    visit_counts: np.ndarray,
) -> np.ndarray:
    """Score eligible root candidates by Gumbel noise plus prior logit plus transformed Q."""
    shifted = logits - logits.max()
    penalty = np.where(visit_counts == considered_visit, 0.0, -np.inf)
    return np.maximum(_LOW_LOGIT, gumbel + shifted + sigma_q) + penalty


def _softmax(values: np.ndarray) -> np.ndarray:
    exp = np.exp(values - values.max())
    return exp / exp.sum()


def _edge_view(node: MCTSNode) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Collect actions, priors, visits and mean Q. Include in-flight visits and their temporary virtual losses."""
    edges = node.edges
    count = len(edges)
    actions = list(edges)
    priors = np.fromiter((edge.prior for edge in edges.values()), dtype=np.float64, count=count)
    visits = np.fromiter((edge.visit_count for edge in edges.values()), dtype=np.float64, count=count)
    sums = np.fromiter((edge.value_sum for edge in edges.values()), dtype=np.float64, count=count)
    q_values = np.divide(sums, visits, out=np.zeros(count, dtype=np.float64), where=visits > 0)
    return actions, priors, visits, q_values


# ----------------------------------------------------------------------
# Fast edge selection
# ----------------------------------------------------------------------
# Cache quantities determined by immutable priors. Preserve the reference order of floating-point
# operations.
_PRIOR = operator.attrgetter("prior")
_VISIT_COUNT = operator.attrgetter("visit_count")
_VALUE_SUM = operator.attrgetter("value_sum")


@dataclass(slots=True)
class _NodeInvariants:
    """Cache data derived from a node's priors after expansion."""

    node: MCTSNode  # A strong reference prevents reuse of this node's identity.
    edges: dict[int, EdgeStats]  # The original edge dictionary identifies the expansion.
    actions: list[int]
    logits: np.ndarray
    probs: np.ndarray  # Normalized prior probabilities used for completed Q.
    first_action: int  # On the first visit, the first maximum prior wins.


def _sigma_from_stats(
    raw_value: float,
    q_values: np.ndarray,
    visited: np.ndarray,
    total_visits: float,
    max_visits: float,
    prior_probs: np.ndarray,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """Compute completed Q using precomputed visit statistics."""
    if max_visits > 0:
        probs = np.maximum(prior_probs[visited], _TINY)
        weighted_q = float((probs * q_values[visited]).sum() / probs.sum())
        mixed = float((raw_value + total_visits * weighted_q) / (total_visits + 1.0))
    else:
        mixed = float(raw_value)
    completed = np.where(visited, q_values, mixed)
    lowest = float(completed.min())
    spread = max(float(completed.max()) - lowest, _EPSILON)
    rescaled = (completed - lowest) / spread
    return (c_visit + max_visits) * c_scale * rescaled


def _edge_arrays(
    edges: dict[int, EdgeStats], visit_list: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build visits, mean Q and a visited mask from the collected counters."""
    count = len(visit_list)
    visits = np.array(visit_list, dtype=np.float64)
    sums = np.fromiter(map(_VALUE_SUM, edges.values()), dtype=np.float64, count=count)
    visited = visits > 0
    q_values = np.divide(sums, visits, out=np.zeros(count, dtype=np.float64), where=visited)
    return visits, q_values, visited


class GumbelMCTS(AlphaGoZeroMCTS):
    """Use Gumbel-Top-k and Sequential Halving at the root and improved-policy selection inside the tree.
    Reuse the base search's batched leaf evaluation and backup.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        config: SearchConfig,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(model, config, rng)
        # Per-move root, simulation budget and noise in root edge order.
        self._root: MCTSNode | None = None
        self._budget: int = config.num_simulations
        self._gumbel: np.ndarray = np.zeros(0, dtype=np.float64)
        # Discard cached prior statistics when starting a new Gumbel root.
        self._invariants: dict[int, _NodeInvariants] = {}

    # ------------------------------------------------------------------
    # AlphaGoZeroMCTS extension points
    # ------------------------------------------------------------------
    def _prepare_root(self, root: MCTSNode, num_simulations: int, add_noise: bool) -> None:
        """Initialize the root budget and candidate noise. Disabling noise makes this step deterministic."""
        if any(edge.visit_count for edge in root.edges.values()):
            raise ValueError(
                "Gumbel search requires an unvisited root. Sequential Halving schedules "
                "visits from zero and cannot reuse a previous tree"
            )
        count = len(root.edges)
        scale = float(self.config.gumbel_scale)
        if add_noise and scale > 0.0 and count:
            noise = scale * self.rng.gumbel(size=count)
        else:
            noise = np.zeros(count, dtype=np.float64)
        self._root = root
        self._budget = int(num_simulations)
        self._gumbel = noise
        self._invariants.clear()

    def _select_leaf(self, root: MCTSNode) -> tuple[MCTSNode, list[tuple[MCTSNode, int]]]:
        """Select a leaf and initialize an unprepared root without noise for direct callers."""
        if root is not self._root:
            self._prepare_root(root, self.config.num_simulations, add_noise=False)
        return super()._select_leaf(root)

    def _select_action(self, node: MCTSNode) -> int:
        """Use Sequential Halving at the root and improved-policy selection elsewhere."""
        if node is self._root:
            return self._select_root_action(node)
        return self._select_interior_action(node)

    def advance_root(self, root: MCTSNode | None, action: int) -> MCTSNode | None:
        """Start a fresh tree for each Gumbel move."""
        return None

    def summarize(self, root: MCTSNode, temperature: float) -> SearchOutput:
        """Return the Sequential Halving winner as the move and the improved root policy as the learning target."""
        visit_counts, root_q = root_visit_statistics(root)
        policy = np.zeros_like(visit_counts)
        improved = np.zeros_like(visit_counts)
        if not root.edges:
            return SearchOutput(
                policy=policy,
                visit_counts=visit_counts,
                root=root,
                root_value=float(root.node_value),
                best_child_q=float(root.node_value),
                root_q=root_q,
                improved_policy=improved,
            )

        actions, priors, visits, q_values = _edge_view(root)
        logits = np.log(np.maximum(priors, _TINY))
        sigma = completed_q_scores(
            root.node_value, q_values, visits, _softmax(logits),
            self.config.gumbel_c_visit, self.config.gumbel_c_scale,
        )
        # Among the most visited actions, maximize noise + logit + transformed Q.
        scores = score_considered(
            float(visits.max()), self._gumbel_for(root, len(actions)), logits, sigma, visits
        )
        policy[actions[int(np.argmax(scores))]] = 1.0
        improved[actions] = _softmax(logits + sigma).astype(np.float32)

        visited = visits > 0
        total_visits = float(visits.sum())
        if not visited.any() or total_visits < self.config.resign_min_visits:
            best_child_q = 1.0  # Conservative sentinel when the search has too little evidence.
        else:
            best_child_q = float(q_values[visited].max())
        return SearchOutput(
            policy=policy,
            visit_counts=visit_counts,
            root=root,
            root_value=float(root.node_value),
            best_child_q=best_child_q,
            root_q=root_q,
            improved_policy=improved,
            target_prune_fraction=self.config.policy_target_prune_fraction,
        )

    # ------------------------------------------------------------------
    # Edge selection
    # ------------------------------------------------------------------
    def _gumbel_for(self, root: MCTSNode, count: int) -> np.ndarray:
        """Return root noise, or zeros for a different root or incompatible action count."""
        if root is self._root and len(self._gumbel) == count:
            return self._gumbel
        return np.zeros(count, dtype=np.float64)

    def _node_invariants(self, node: MCTSNode) -> _NodeInvariants:
        """Cache prior-derived quantities by node and edge dictionary identity for one move."""
        entry = self._invariants.get(id(node))
        edges = node.edges
        if entry is not None and entry.node is node and entry.edges is edges and len(entry.actions) == len(edges):
            return entry
        actions = list(edges)
        priors = np.fromiter(map(_PRIOR, edges.values()), dtype=np.float64, count=len(actions))
        logits = np.log(np.maximum(priors, _TINY))
        probs = _softmax(logits)
        logits.flags.writeable = False  # Keep shared cached arrays read-only.
        probs.flags.writeable = False
        entry = _NodeInvariants(node, edges, actions, logits, probs, actions[int(np.argmax(probs))])
        self._invariants[id(node)] = entry
        return entry

    def _select_root_action(self, root: MCTSNode) -> int:
        """Choose the next root action under Sequential Halving."""
        invariants = self._node_invariants(root)
        actions, logits = invariants.actions, invariants.logits
        visit_list = [edge.visit_count for edge in root.edges.values()]
        visits, q_values, visited = _edge_arrays(root.edges, visit_list)
        total_visits = float(sum(visit_list))
        sigma = _sigma_from_stats(
            root.node_value, q_values, visited, total_visits, float(max(visit_list)), invariants.probs,
            self.config.gumbel_c_visit, self.config.gumbel_c_scale,
        )
        considered = min(self.config.gumbel_max_considered, len(actions))
        sequence = considered_visit_sequence(considered, max(1, self._budget))
        # The simulation index includes pending root visits.
        index = min(int(total_visits), len(sequence) - 1)
        scores = score_considered(
            sequence[index], self._gumbel_for(root, len(actions)), logits, sigma, visits
        )
        best = int(np.argmax(scores))
        if scores[best] == -np.inf:
            # The schedule must admit a candidate. An unrestricted argmax could silently select an ineligible
            # action.
            raise RuntimeError(
                "Sequential Halving: no candidate with visit count "
                f"{sequence[index]} (root counts: {visits.astype(int).tolist()})"
            )
        return actions[best]

    def _select_interior_action(self, node: MCTSNode) -> int:
        """Choose the action maximizing improved probability minus N(a)/(1 + total visits)."""
        invariants = self._node_invariants(node)
        actions = invariants.actions
        visit_list = [edge.visit_count for edge in node.edges.values()]
        c_visit, c_scale = self.config.gumbel_c_visit, self.config.gumbel_c_scale
        if (
            not any(visit_list)
            and math.isfinite(node.node_value)
            and math.isfinite((c_visit + 0.0) * c_scale)
        ):
            # Without visits the transformed Q bonus vanishes and selection reduces to the first maximum prior.
            return invariants.first_action
        visits, q_values, visited = _edge_arrays(node.edges, visit_list)
        total_visits = float(sum(visit_list))
        sigma = _sigma_from_stats(
            node.node_value, q_values, visited, total_visits, float(max(visit_list)), invariants.probs,
            c_visit, c_scale,
        )
        improved = _softmax(invariants.logits + sigma)
        return actions[int(np.argmax(improved - visits / (1.0 + total_visits)))]


def make_self_play_search(
    model: PolicyValueModel,
    config: SearchConfig,
    rng: np.random.Generator,
) -> AlphaGoZeroMCTS:
    """Construct the trainable side's self-play search from config.search_algo for both game drivers."""
    if config.search_algo == "gumbel":
        return GumbelMCTS(model, config, rng)
    return AlphaGoZeroMCTS(model, config, rng)

# Frozen PUCT implementation from before the search fast paths.
#
# Reference PUCT uses the original edge loops, scalar softmax and symmetry functions.
# `.config/.game/.network` -> `alphago_zero.*`, `.augmentations` -> `frozen_augmentations_v1`.
# Keep reference behavior stable for bitwise differential tests.
# (tests/test_search_fast_paths.py).
from __future__ import annotations

"""AlphaGo Zero Monte Carlo tree search. Use a network to supply legal-move priors and leaf values, then
improve the move distribution through simulation.
"""

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from alphago_zero.config import SearchConfig
from alphago_zero.game import GoState
from alphago_zero.network import PolicyValueModel
from frozen_augmentations_v1 import (
    apply_dihedral_to_planes,
    invert_dihedral_on_policy,
    random_dihedral_index,
)


@dataclass(slots=True)
class EdgeStats:
    """Prior, visit count, accumulated value and optional child for one state-action edge."""

    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: Optional["MCTSNode"] = None

    @property
    def q_value(self) -> float:
        """Mean backed-up value from the parent player's perspective. An unvisited edge starts at zero."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(slots=True)
class MCTSNode:
    """A node in the search tree."""

    state: GoState
    edges: dict[int, EdgeStats] = field(default_factory=dict)
    is_expanded: bool = False
    node_value: float = 0.0


@dataclass(slots=True)
class SearchOutput:
    """Search results distinguish the move-selection policy, raw visit distribution and training target."""

    policy: np.ndarray
    visit_counts: np.ndarray
    root: MCTSNode
    root_value: float
    best_child_q: float
    root_q: float
    # Gumbel supplies its improved policy here. PUCT uses its visit distribution.
    improved_policy: np.ndarray | None = None

    @property
    def policy_target(self) -> np.ndarray:
        """Return the policy learning target for the self-play driver."""
        if self.improved_policy is not None:
            return self.improved_policy
        return self.visit_distribution

    @property
    def visit_distribution(self) -> np.ndarray:
        """Normalize root visit counts into the AlphaGo Zero policy target."""
        total = float(self.visit_counts.sum())
        if total <= 0.0:
            # A terminal or empty root with no visits has a zero distribution.
            return np.zeros_like(self.visit_counts, dtype=np.float32)
        return (self.visit_counts / total).astype(np.float32)


def resignation_child_q(root: MCTSNode, min_visits: int) -> float:
    """Best observed Q, guarded by evidence on the most visited action.

    Unvisited Q=0 is an exploration default, not a value estimate. Retain a
    conservative +1 sentinel until the preferred action has enough visits.
    """
    visited = [edge for edge in root.edges.values() if edge.visit_count > 0]
    if not visited or max(edge.visit_count for edge in visited) < min_visits:
        return 1.0
    return max(edge.q_value for edge in visited)


def root_visit_statistics(root: MCTSNode) -> tuple[np.ndarray, float]:
    """Return root counts in action order and the visit-weighted mean root Q."""
    visit_counts = np.zeros(root.state.action_size, dtype=np.float32)
    for action, edge in root.edges.items():
        visit_counts[action] = float(edge.visit_count)

    total_visits = sum(edge.visit_count for edge in root.edges.values())
    if total_visits > 0:
        total_value = sum(edge.value_sum for edge in root.edges.values())
        return visit_counts, float(total_value / total_visits)
    return visit_counts, float(root.node_value)


class AlphaGoZeroMCTS:
    """Run PUCT using a policy/value predictor. A leaf value belongs to the player to move at that leaf.
    Change its sign at every step up the path.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        config: SearchConfig,
        rng: np.random.Generator,
    ) -> None:
        # Both a single network and an ensemble satisfy the PolicyValueModel interface.
        self.model = model
        self.config = config
        self.rng = rng

    # ------------------------------------------------------------------
    # Public protocol interface
    # ------------------------------------------------------------------
    def run(
        self,
        root_state: GoState,
        temperature: float,
        add_root_noise: bool,
        root: MCTSNode | None = None,
        num_simulations: int | None = None,
    ) -> SearchOutput:
        """Search from the supplied root using the configured simulation budget."""
        target_simulations = (
            self.config.num_simulations if num_simulations is None else num_simulations
        )
        if root is None:
            root = MCTSNode(state=root_state)

        if root.state.is_terminal():
            # Return a well-defined empty result for a terminal root.
            visit_counts = np.zeros(root_state.action_size, dtype=np.float32)
            policy = np.zeros_like(visit_counts)
            root_value = float(root_state.outcome_for_player(root_state.to_play))
            return SearchOutput(
                policy=policy,
                visit_counts=visit_counts,
                root=root,
                root_value=root_value,
                best_child_q=root_value,
                root_q=root_value,
            )

        # Expand the root before the first simulation.
        if not root.is_expanded:
            self._expand(root)

        self._prepare_root(root, target_simulations, add_root_noise)

        simulations_done = 0
        while simulations_done < target_simulations:
            batch_target = min(
                self.config.eval_batch_size,
                target_simulations - simulations_done,
            )
            simulations_done += self._run_simulation_batch(root, batch_target)

        return self.summarize(root, temperature)

    def summarize(self, root: MCTSNode, temperature: float) -> SearchOutput:
        """Build a search result from root statistics. The sequential and parallel drivers share this
        implementation.
        """
        visit_counts, root_q = root_visit_statistics(root)
        policy = self._policy_from_visit_counts(visit_counts, temperature)
        best_child_q = resignation_child_q(root, self.config.resign_min_visits)

        return SearchOutput(
            policy=policy,
            visit_counts=visit_counts,
            root=root,
            root_value=float(root.node_value),
            best_child_q=float(best_child_q),
            root_q=root_q,
        )

    def advance_root(self, root: MCTSNode | None, action: int) -> MCTSNode | None:
        """Advance to the chosen child to preserve subtree statistics when a child exists."""
        if root is None:
            return None
        edge = root.edges.get(action)
        if edge is None:
            return None
        return edge.child

    # ------------------------------------------------------------------
    # Batched simulations collect leaves with virtual loss, evaluate them and back up the results.
    # ------------------------------------------------------------------
    def _run_simulation_batch(self, root: MCTSNode, batch_target: int) -> int:
        """Run a batch of simulations and return the actual number completed."""
        # Group paths by leaf identity. Insertion order preserves deterministic symmetry sampling.
        pending: dict[int, tuple[MCTSNode, list[list[tuple[MCTSNode, int]]]]] = {}
        simulations = 0

        while simulations < batch_target:
            leaf, path = self._select_leaf(root)
            simulations += 1

            if leaf.state.is_terminal():
                # Use the exact outcome at terminal leaves.
                value = float(leaf.state.outcome_for_player(leaf.state.to_play))
                leaf.node_value = value
                leaf.is_expanded = True
                self._backup(path, value)
                continue

            key = id(leaf)
            entry = pending.get(key)
            if entry is None:
                pending[key] = (leaf, [path])
            else:
                entry[1].append(path)

        if pending:
            self._evaluate_and_backup(pending)

        return simulations

    def _select_leaf(self, root: MCTSNode) -> tuple[MCTSNode, list[tuple[MCTSNode, int]]]:
        """Descend to a leaf, counting visits immediately and applying a temporary virtual loss to each
        selected edge.
        """
        virtual_loss = self.config.virtual_loss
        node = root
        path: list[tuple[MCTSNode, int]] = []

        while node.is_expanded and not node.state.is_terminal():
            action = self._select_action(node)
            edge = node.edges[action]
            path.append((node, action))

            edge.visit_count += 1
            edge.value_sum -= virtual_loss

            # Allocate a child only when selection first traverses its edge.
            if edge.child is None:
                edge.child = MCTSNode(state=node.state.play(action))
            node = edge.child

        return node, path

    def _evaluate_and_backup(
        self,
        pending: dict[int, tuple[MCTSNode, list[list[tuple[MCTSNode, int]]]]],
    ) -> None:
        """Evaluate distinct leaves in one network batch, undo their input symmetries and back up each
        selected path.
        """
        nodes = [node for node, _paths in pending.values()]
        transform_ids = [random_dihedral_index(self.rng) for _ in nodes]

        if len(nodes) == 1:
            # Use predict_single for a singleton to preserve the sequential search result.
            node = nodes[0]
            encoded = apply_dihedral_to_planes(node.state.encode(), transform_ids[0])
            policy_logits, value = self.model.predict_single(encoded)
            logits_batch = policy_logits[np.newaxis, :]
            values_batch = np.asarray([value], dtype=np.float32)
        else:
            encoded_batch = np.stack(
                [
                    apply_dihedral_to_planes(node.state.encode(), transform_id)
                    for node, transform_id in zip(nodes, transform_ids)
                ],
                axis=0,
            ).astype(np.float32)
            logits_batch, values_batch = self.model.predict_batch(encoded_batch)

        for (node, paths), transform_id, logits, value in zip(
            pending.values(), transform_ids, logits_batch, values_batch
        ):
            restored_logits = invert_dihedral_on_policy(
                policy=logits,
                board_size=node.state.board_size,
                transform_id=transform_id,
            )
            node_value = self._apply_network_output(node, restored_logits, float(value))
            for path in paths:
                self._backup(path, node_value)

    def _backup(self, path: list[tuple[MCTSNode, int]], leaf_value: float) -> None:
        """Undo virtual loss and back up the value with a sign flip at each ply."""
        virtual_loss = self.config.virtual_loss
        value = leaf_value
        for parent, action in reversed(path):
            value = -value
            edge = parent.edges[action]
            edge.value_sum += value + virtual_loss

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------
    def _select_action(self, node: MCTSNode) -> int:
        """Select the edge maximizing Q plus the prior-weighted exploration term."""
        best_action: int | None = None
        best_score = -float("inf")

        fpu = self.config.fpu_reduction
        unvisited_q = 0.0 if fpu is None else max(-1.0, node.node_value - fpu)

        total_visits = sum(edge.visit_count for edge in node.edges.values())
        normalizer = math.sqrt(total_visits + 1.0)

        for action, edge in node.edges.items():
            q = edge.q_value if edge.visit_count > 0 else unvisited_q
            u = (
                self.config.c_puct
                * edge.prior
                * normalizer
                / (1.0 + edge.visit_count)
            )
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None, "an expanded node must have at least one legal move"
        return best_action

    # ------------------------------------------------------------------
    # Expand & evaluate
    # ------------------------------------------------------------------
    def _expand(self, node: MCTSNode) -> float:
        """Encode and evaluate the root, then initialize its legal edges from the resulting priors."""
        if node.state.is_terminal():
            node.node_value = float(node.state.outcome_for_player(node.state.to_play))
            node.is_expanded = True
            node.edges = {}
            return node.node_value

        encoded = node.state.encode()

        # Apply a random square-board symmetry before network inference.
        transform_id = random_dihedral_index(self.rng)
        encoded_aug = apply_dihedral_to_planes(encoded, transform_id)

        policy_logits, value = self.model.predict_single(encoded_aug)

        # Restore logits to the board's original orientation.
        policy_logits = invert_dihedral_on_policy(
            policy=policy_logits,
            board_size=node.state.board_size,
            transform_id=transform_id,
        )
        return self._apply_network_output(node, policy_logits, float(value))

    def _apply_network_output(
        self,
        node: MCTSNode,
        policy_logits: np.ndarray,
        value: float,
    ) -> float:
        """Expand legal edges from the network output and store its value."""
        legal_actions = node.state.legal_actions()
        priors = self._masked_softmax(policy_logits, legal_actions)

        node.edges = {
            action: EdgeStats(prior=float(priors[action]))
            for action in legal_actions
        }
        node.node_value = float(value)
        node.is_expanded = True
        return node.node_value

    # ------------------------------------------------------------------
    # Root noise
    # ------------------------------------------------------------------
    def _prepare_root(self, root: MCTSNode, num_simulations: int, add_noise: bool) -> None:
        """Prepare an expanded root for the given simulation budget in either driver."""
        if add_noise:
            self._add_dirichlet_noise(root)

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Mix the prior with a Dirichlet sample at the root during exploratory search."""
        if not root.edges:
            return

        actions = list(root.edges.keys())
        alpha = self.config.root_dirichlet_alpha
        epsilon = self.config.root_dirichlet_epsilon

        noise = self.rng.dirichlet(np.full(len(actions), alpha, dtype=np.float64))
        for action, eta in zip(actions, noise):
            edge = root.edges[action]
            edge.prior = float((1.0 - epsilon) * edge.prior + epsilon * eta)

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _masked_softmax(self, logits: np.ndarray, legal_actions: list[int]) -> np.ndarray:
        """Normalize network logits over legal actions only."""
        probs = np.zeros_like(logits, dtype=np.float32)
        legal_logits = logits[legal_actions].astype(np.float64)

        # Subtract the maximum before exponentiation.
        legal_logits = legal_logits - np.max(legal_logits)
        exp_logits = np.exp(legal_logits)
        total = float(np.sum(exp_logits))

        if total <= 0.0 or not np.isfinite(total):
            # Fall back to a uniform legal policy for nonfinite network output.
            uniform = 1.0 / len(legal_actions)
            for action in legal_actions:
                probs[action] = uniform
            return probs

        probs_subset = exp_logits / total
        for action, p in zip(legal_actions, probs_subset):
            probs[action] = float(p)
        return probs

    def _policy_from_visit_counts(self, visit_counts: np.ndarray, temperature: float) -> np.ndarray:
        """Build the move policy with weights proportional to N^(1/temperature)."""
        policy = np.zeros_like(visit_counts, dtype=np.float32)

        if np.sum(visit_counts) <= 0:
            return policy

        # At near-zero temperature select the most visited edge.
        if temperature <= 1e-6:
            best_action = int(np.argmax(visit_counts))
            policy[best_action] = 1.0
            return policy

        # Apply the temperature exponent to visit counts.
        counts = visit_counts.astype(np.float64)
        counts_power = np.power(counts, 1.0 / temperature)
        total = float(np.sum(counts_power))

        if total <= 0.0 or not np.isfinite(total):
            # Fall back to the most visited action if normalization fails.
            best_action = int(np.argmax(visit_counts))
            policy[best_action] = 1.0
            return policy

        policy = (counts_power / total).astype(np.float32)
        return policy

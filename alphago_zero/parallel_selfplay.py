from __future__ import annotations

"""Batch inference across simultaneous self-play games to keep the device busy when individual trees have
few pending leaves.
"""

from dataclasses import dataclass, field

import numpy as np

from .augmentations import (
    apply_dihedral_to_planes,
    invert_dihedral_on_policy,
    random_dihedral_index,
)
from .config import AlphaGoZeroConfig, SearchConfig
from .game import BLACK, WHITE, GoState
from .gumbel import make_self_play_search
from .mcts import AlphaGoZeroMCTS, MCTSNode
from .network import AlphaGoZeroNet, PolicyValueModel
from .replay_buffer import TrainingExample
from .selfplay import (
    PendingExample,
    ResignationController,
    SelfPlayResult,
    _argmax_action,
    _aux_targets_from_final_state,
    _draw_move_budget,
    _sample_action,
    _search_temperature,
    game_end_reason,
)


@dataclass(slots=True)
class ParallelSelfPlayStats:
    """Measure slot occupancy and batch width as games finish and fewer positions remain active."""

    # Slots opened across drivers.
    slots: int = 0
    # Rounds of leaf collection, inference and move application.
    rounds: int = 0
    # Sum available and occupied slots over all rounds.
    slot_rounds: int = 0
    active_slot_rounds: int = 0
    # Network calls and positions evaluated. League members may need separate calls.
    network_calls: int = 0
    positions: int = 0

    def add(self, other: ParallelSelfPlayStats) -> None:
        self.slots += other.slots
        self.rounds += other.rounds
        self.slot_rounds += other.slot_rounds
        self.active_slot_rounds += other.active_slot_rounds
        self.network_calls += other.network_calls
        self.positions += other.positions

    @property
    def fill(self) -> float | None:
        """Return the mean occupied fraction of available slots."""
        if self.slot_rounds == 0:
            return None
        return self.active_slot_rounds / self.slot_rounds

    @property
    def mean_batch(self) -> float | None:
        """Return mean positions per inference call."""
        if self.network_calls == 0:
            return None
        return self.positions / self.network_calls

    def to_dict(self) -> dict[str, float | int | None]:
        fill, mean_batch = self.fill, self.mean_batch
        return {
            "slots": self.slots,
            "rounds": self.rounds,
            "fill": None if fill is None else round(fill, 4),
            "network_calls": self.network_calls,
            "positions": self.positions,
            "mean_batch": None if mean_batch is None else round(mean_batch, 2),
        }


@dataclass(slots=True)
class _GameSlot:
    """State for one active game in the parallel driver."""

    state: GoState
    search: AlphaGoZeroMCTS
    root: MCTSNode
    pending_examples: list[PendingExample] = field(default_factory=list)
    moves_played: list[int] = field(default_factory=list)
    resignation_disabled: bool = False
    resigned: bool = False
    hypothetical_resign_players: set[int] = field(default_factory=set)
    sims_done: int = 0
    noise_added: bool = False

    # Draw the search budget when starting a game and after each move.
    move_sims: int = 0
    move_is_full: bool = True

    # Optional frozen league opponent, with its own color, model and search settings.
    league_search: AlphaGoZeroMCTS | None = None
    league_color: int = 0
    league_name: str | None = None


class _ParallelSelfPlayDriver:
    """Schedule concurrent self-play games in rounds."""

    def __init__(
        self,
        model: AlphaGoZeroNet,
        config: AlphaGoZeroConfig,
        rng: np.random.Generator,
        resignation_controller: ResignationController,
        num_games: int,
        league: list[tuple[str, AlphaGoZeroNet, SearchConfig]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.rng = rng
        self.resignation_controller = resignation_controller
        self.num_games = num_games
        # Pool entries contain a name, frozen network and search configuration.
        self.league = league or []

        self.games_started = 0
        self.results: list[SelfPlayResult] = []
        self.slots: list[_GameSlot | None] = []

        # Cap slots at the number of games available to avoid opening permanently idle slots.
        num_slots = max(1, min(config.training.parallel_games, num_games))
        self.stats = ParallelSelfPlayStats(slots=num_slots)
        for _ in range(num_slots):
            self.slots.append(self._start_new_game())

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------
    def _start_new_game(self) -> _GameSlot | None:
        if self.games_started >= self.num_games:
            return None
        self.games_started += 1

        state = GoState.new_game(
            board_size=self.config.game.board_size,
            komi=self.config.game.komi,
            max_moves=self.config.game.max_moves,
            superko_rule=self.config.game.superko_rule,
        )
        # Use the configured search algorithm for the learner and PUCT for frozen league opponents.
        search = make_self_play_search(self.model, self.config.search, self.rng)
        move_sims, move_is_full = _draw_move_budget(self.config, self.rng)
        slot = _GameSlot(
            state=state,
            search=search,
            root=MCTSNode(state=state),
            resignation_disabled=bool(self.rng.random() < self.config.search.disable_resign_prob),
            move_sims=move_sims,
            move_is_full=move_is_full,
        )

        # Sample a frozen opponent and its color using league_prob. Its search uses its own configuration.
        if self.league and self.rng.random() < self.config.training.league_prob:
            name, frozen_model, frozen_search = self.league[
                int(self.rng.integers(len(self.league)))
            ]
            slot.league_search = AlphaGoZeroMCTS(
                model=frozen_model, config=frozen_search, rng=self.rng
            )
            slot.league_color = BLACK if self.rng.random() < 0.5 else WHITE
            slot.league_name = name
            if slot.state.to_play == slot.league_color:
                slot.move_sims = frozen_search.num_simulations
                slot.move_is_full = True
        return slot

    @staticmethod
    def _champion_search(slot: _GameSlot) -> AlphaGoZeroMCTS | None:
        """Return the frozen opponent's search when it is that opponent's turn."""
        if slot.league_search is not None and slot.state.to_play == slot.league_color:
            return slot.league_search
        return None

    def _active_search(self, slot: _GameSlot) -> AlphaGoZeroMCTS:
        """Select the search belonging to the player to move in this slot."""
        champion_search = self._champion_search(slot)
        return slot.search if champion_search is None else champion_search

    def _move_budget(self, slot: _GameSlot) -> tuple[int, bool]:
        # Frozen champions supply full-search expert targets. Applying the
        # learner's cap would silently weaken the anchor and discard targets.
        champion_search = self._champion_search(slot)
        if champion_search is not None:
            return champion_search.config.num_simulations, True
        return _draw_move_budget(self.config, self.rng)

    def _finalize_game(
        self,
        slot: _GameSlot,
        winner: int | None,
        final_state: GoState | None = None,
    ) -> None:
        """Complete each pending training example with the game outcome."""
        false_positive = False
        if slot.resignation_disabled and slot.hypothetical_resign_players:
            false_positive = winner in slot.hypothetical_resign_players
            # Keep the descriptive false-resignation metric even if league games are excluded from calibration.
            league_excluded = (
                slot.league_name is not None
                and self.config.training.exclude_league_from_resign_calibration
            )
            if not league_excluded:
                self.resignation_controller.observe_false_positive(false_positive)

        # Only scored games supply ownership and margin targets. Flip Black's perspective for White examples.
        ownership_black: np.ndarray | None = None
        normalized_diff_black = 0.0
        if final_state is not None:
            ownership_black, normalized_diff_black = _aux_targets_from_final_state(final_state)

        examples: list[TrainingExample] = []
        for item in slot.pending_examples:
            # A draw gives both players z=0.
            z = 0.0 if winner is None else (1.0 if item.player_to_play == winner else -1.0)
            sign = 1.0 if item.player_to_play == BLACK else -1.0
            examples.append(
                TrainingExample(
                    state=item.state,
                    policy=item.policy,
                    value=z,
                    ownership=(
                        (ownership_black * sign).astype(np.float32)
                        if ownership_black is not None
                        else None
                    ),
                    score=normalized_diff_black * sign,
                    policy_weight=item.policy_weight,
                )
            )

        score_margin: float | None = None
        if final_state is not None:
            black_score, white_score = final_state.area_score()
            score_margin = abs(black_score - white_score)

        self.results.append(
            SelfPlayResult(
                examples=examples,
                winner=winner,
                num_moves=len(slot.pending_examples),
                resigned=slot.resigned,
                resignation_disabled=slot.resignation_disabled,
                false_positive_resignation=false_positive,
                moves=slot.moves_played,
                score_margin=score_margin,
                ended_by=game_end_reason(final_state, slot.resigned),
                league_opponent=slot.league_name,
                league_color=slot.league_color if slot.league_name else None,
            )
        )

    # ------------------------------------------------------------------
    # Main driver loop
    # ------------------------------------------------------------------
    def run(self) -> list[SelfPlayResult]:
        # Use evaluation mode for all self-play inference.
        self.model.eval()

        while any(slot is not None for slot in self.slots):
            self._run_round()
        return self.results

    def _run_round(self) -> None:
        """Collect requests, evaluate shared batches and advance ready games."""
        # Each request records the slot, owning search, leaf and paths for backup. Roots have no pending path.
        eval_requests: list[
            tuple[_GameSlot, AlphaGoZeroMCTS, MCTSNode, list[list[tuple[MCTSNode, int]]]]
        ] = []

        self.stats.rounds += 1
        self.stats.slot_rounds += len(self.slots)
        self.stats.active_slot_rounds += sum(slot is not None for slot in self.slots)

        for slot in self.slots:
            if slot is None:
                continue

            search = self._active_search(slot)

            if not slot.root.is_expanded:
                eval_requests.append((slot, search, slot.root, []))
                continue

            # Add root noise only on full-search moves that supply policy targets.
            if not slot.noise_added:
                search._prepare_root(slot.root, slot.move_sims, add_noise=slot.move_is_full)
                slot.noise_added = True

            remaining = slot.move_sims - slot.sims_done
            if remaining <= 0:
                continue

            chunk = min(search.config.eval_batch_size, remaining)
            pending: dict[int, tuple[MCTSNode, list[list[tuple[MCTSNode, int]]]]] = {}
            collected = 0
            while collected < chunk:
                leaf, path = search._select_leaf(slot.root)
                collected += 1
                if leaf.state.is_terminal():
                    value = float(leaf.state.outcome_for_player(leaf.state.to_play))
                    leaf.node_value = value
                    leaf.is_expanded = True
                    search._backup(path, value)
                    continue
                key = id(leaf)
                entry = pending.get(key)
                if entry is None:
                    pending[key] = (leaf, [path])
                else:
                    entry[1].append(path)
            slot.sims_done += collected

            for node, paths in pending.values():
                eval_requests.append((slot, search, node, paths))

        if eval_requests:
            self._evaluate_requests(eval_requests)

        self._finalize_ready_moves()

    def _evaluate_requests(
        self,
        eval_requests: list[
            tuple[_GameSlot, AlphaGoZeroMCTS, MCTSNode, list[list[tuple[MCTSNode, int]]]]
        ],
    ) -> None:
        """Batch requests by network so each frozen opponent evaluates only its own positions."""
        transform_ids = [random_dihedral_index(self.rng) for _ in eval_requests]
        encoded = np.stack(
            [
                apply_dihedral_to_planes(node.state.encode(), transform_id)
                for (_slot, _search, node, _paths), transform_id in zip(
                    eval_requests, transform_ids
                )
            ],
            axis=0,
        ).astype(np.float32)

        # Group request indices by network identity.
        groups: dict[int, list[int]] = {}
        models: dict[int, PolicyValueModel] = {}
        for i, (_slot, search, _node, _paths) in enumerate(eval_requests):
            key = id(search.model)
            groups.setdefault(key, []).append(i)
            models[key] = search.model

        logits_batch: list[np.ndarray | None] = [None] * len(eval_requests)
        values_batch: list[float] = [0.0] * len(eval_requests)
        for key, indices in groups.items():
            group_logits, group_values = models[key].predict_batch(encoded[indices])
            self.stats.network_calls += 1
            self.stats.positions += len(indices)
            for j, idx in enumerate(indices):
                logits_batch[idx] = group_logits[j]
                values_batch[idx] = float(group_values[j])

        for (slot, search, node, paths), transform_id, maybe_logits, value in zip(
            eval_requests, transform_ids, logits_batch, values_batch
        ):
            assert maybe_logits is not None
            restored = invert_dihedral_on_policy(
                policy=maybe_logits,
                board_size=node.state.board_size,
                transform_id=transform_id,
            )
            node_value = search._apply_network_output(node, restored, float(value))

            if not paths:
                # Prepare the expanded root with this move's noise and search budget.
                search._prepare_root(node, slot.move_sims, add_noise=slot.move_is_full)
                slot.noise_added = True
            else:
                for path in paths:
                    search._backup(path, node_value)

    def _finalize_ready_moves(self) -> None:
        """Play moves in slots that have spent their search budget and close any completed games."""
        for index, slot in enumerate(self.slots):
            if slot is None or not slot.root.is_expanded:
                continue
            if slot.sims_done < slot.move_sims:
                continue

            state = slot.state
            root = slot.root
            search = self._active_search(slot)

            # Use the same result builder as sequential search for move probabilities and policy targets.
            temperature = _search_temperature(state, self.config)
            output = search.summarize(root, temperature)
            policy = output.policy

            # Record the example before applying the chosen action.
            slot.pending_examples.append(
                PendingExample(
                    state=state.encode().astype(np.float32),
                    policy=output.policy_target,
                    player_to_play=state.to_play,
                    policy_weight=1.0 if slot.move_is_full else 0.0,
                )
            )

            # Use visit-weighted root Q for resignation decisions, as in sequential self-play.
            root_q = output.root_q
            best_child_q = output.best_child_q
            threshold = self.resignation_controller.threshold
            # Only learner moves contribute resignation evidence. A frozen opponent has a different value
            # calibration.
            champion_turn = self._champion_search(slot) is not None
            if (
                slot.move_is_full
                and not champion_turn
                and root_q < threshold
                and best_child_q < threshold
            ):
                if slot.resignation_disabled:
                    slot.hypothetical_resign_players.add(state.to_play)
                else:
                    slot.resigned = True
                    self._finalize_game(slot, winner=-state.to_play)
                    self.slots[index] = self._start_new_game()
                    continue

            if state.move_number < self.config.search.temperature_moves:
                action = _sample_action(policy, self.rng)
            else:
                action = _argmax_action(policy)

            # Discard trees between players in league games to avoid mixing networks' statistics.
            reuse = self.config.search.reuse_tree and slot.league_search is None
            next_root = slot.search.advance_root(root, action) if reuse else None
            slot.moves_played.append(action)
            next_state = state.play(action)

            if next_state.is_terminal():
                self._finalize_game(slot, winner=next_state.winner_color(), final_state=next_state)
                self.slots[index] = self._start_new_game()
                continue

            slot.state = next_state
            slot.root = next_root if next_root is not None else MCTSNode(state=next_state)
            slot.sims_done = 0
            slot.noise_added = False
            slot.move_sims, slot.move_is_full = self._move_budget(slot)


def play_parallel_self_play_games(
    model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    rng: np.random.Generator,
    resignation_controller: ResignationController,
    num_games: int,
    league: list[tuple[str, AlphaGoZeroNet, SearchConfig]] | None = None,
    stats: ParallelSelfPlayStats | None = None,
) -> list[SelfPlayResult]:
    """Play the requested games with up to parallel_games active and return results in completion order."""
    if num_games <= 0:
        return []
    driver = _ParallelSelfPlayDriver(
        model=model,
        config=config,
        rng=rng,
        resignation_controller=resignation_controller,
        num_games=num_games,
        league=league,
    )
    results = driver.run()
    if stats is not None:
        stats.add(driver.stats)
    return results

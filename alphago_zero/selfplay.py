from __future__ import annotations

"""Generate training data by searching and playing both sides of Go."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import AlphaGoZeroConfig
from .game import BLACK, GoState
from .gumbel import make_self_play_search
from .network import AlphaGoZeroNet
from .replay_buffer import TrainingExample


@dataclass(slots=True)
class PendingExample:
    """Hold the state, policy and player until the game supplies an outcome."""

    state: np.ndarray
    policy: np.ndarray
    player_to_play: int
    policy_weight: float = 1.0


def game_end_reason(final_state: GoState | None, resigned: bool) -> str:
    """Classify the ending as resignation, two passes or the move limit in both self-play drivers."""
    if resigned:
        return "resign"
    if final_state is not None and final_state.consecutive_passes >= 2:
        return "double_pass"
    return "max_moves"


@dataclass(slots=True)
class SelfPlayResult:
    """Result and training data from one self-play game."""

    examples: list[TrainingExample]
    # Winning color, or None for a draw.
    winner: int | None
    num_moves: int
    resigned: bool
    resignation_disabled: bool
    false_positive_resignation: bool

    # Move history for SGF export and the absolute terminal score difference when available.
    moves: list[int] = field(default_factory=list)
    score_margin: float | None = None

    # Track move-limit endings so truncated games can be counted separately.
    ended_by: str = "unknown"

    # Frozen league opponent and its color, if any.
    league_opponent: str | None = None
    league_color: int | None = None


@dataclass(slots=True)
class ResignationController:
    """Calibrate the resignation threshold with games that continue after a player would resign."""

    threshold: float
    target_false_positive_rate: float
    adjustment: float
    window: int
    # Keep a sliding window sized during initialization.
    history: deque[bool] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        self.history = deque(self.history, maxlen=self.window)

    def observe_false_positive(self, flag: bool) -> None:
        self.history.append(bool(flag))
        self._maybe_adjust()

    @property
    def false_positive_rate(self) -> float:
        if not self.history:
            return 0.0
        return float(sum(self.history)) / float(len(self.history))

    def _maybe_adjust(self) -> None:
        """Adjust the negative threshold using the observed false-resignation rate."""
        if len(self.history) < max(10, self.window // 5):
            return

        rate = self.false_positive_rate

        if rate > self.target_false_positive_rate:
            # Too many winning players would have resigned. Require a worse position.
            self.threshold -= self.adjustment
        elif rate < self.target_false_positive_rate / 2.0:
            # Few false positives allow a more aggressive threshold.
            self.threshold += self.adjustment

        # Clamp the threshold to the supported range.
        self.threshold = float(np.clip(self.threshold, -0.999, -0.05))

    def summary_dict(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold),
            "false_positive_rate": float(self.false_positive_rate),
            "observations": float(len(self.history)),
        }


def _sample_action(policy: np.ndarray, rng: np.random.Generator) -> int:
    """Draw a move from the policy probabilities."""
    action = int(rng.choice(len(policy), p=policy))
    return action


def _argmax_action(policy: np.ndarray) -> int:
    """Select a maximum-probability move."""
    return int(np.argmax(policy))


def _search_temperature(state: GoState, config: AlphaGoZeroConfig) -> float:
    """Use temperature one in the opening and the configured low temperature afterward."""
    if state.move_number < config.search.temperature_moves:
        return 1.0
    return config.search.deterministic_temperature


def _draw_move_budget(
    config: AlphaGoZeroConfig, rng: np.random.Generator
) -> tuple[int, bool]:
    """Draw the simulation budget for this move and indicate whether it supplies a full policy target."""
    prob = config.search.playout_cap_prob
    cheap = config.search.playout_cap_cheap_sims
    if prob <= 0.0 or cheap <= 0:
        return config.search.num_simulations, True
    if rng.random() < prob:
        return config.search.num_simulations, True
    return cheap, False


def _aux_targets_from_final_state(
    final_state: GoState,
) -> tuple[np.ndarray, float]:
    """Compute terminal ownership and normalized score margin from Black's perspective."""
    ownership_black = final_state.ownership_map().astype(np.float32)
    black_score, white_score = final_state.area_score()
    area = float(final_state.board_size * final_state.board_size)
    normalized_diff_black = float(np.clip((black_score - white_score) / area, -1.0, 1.0))
    return ownership_black, normalized_diff_black


def play_self_play_game(
    model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    rng: np.random.Generator,
    resignation_controller: ResignationController,
) -> SelfPlayResult:
    """Play one self-play game, then attach outcomes to the saved pre-move positions."""
    state = GoState.new_game(
        board_size=config.game.board_size,
        komi=config.game.komi,
        max_moves=config.game.max_moves,
        superko_rule=config.game.superko_rule,
    )

    # Use evaluation mode throughout search to avoid switching before every leaf prediction.
    model.eval()

    # Select PUCT or Gumbel once. The game loop uses their common search interface.
    search = make_self_play_search(model, config.search, rng)
    root = None

    pending_examples: list[PendingExample] = []
    moves_played: list[int] = []

    resignation_disabled = bool(rng.random() < config.search.disable_resign_prob)
    resigned = False
    winner: int | None = None  # Winning color or None for a draw.
    hypothetical_resign_players: set[int] = set()
    score_margin: float | None = None

    final_state: GoState | None = None

    while not state.is_terminal():
        temperature = _search_temperature(state, config)
        # Cheap moves use fewer simulations and no root noise. Their policy targets carry zero weight.
        move_sims, is_full_search = _draw_move_budget(config, rng)
        search_output = search.run(
            root_state=state,
            temperature=temperature,
            add_root_noise=is_full_search,
            root=root if config.search.reuse_tree else None,
            num_simulations=move_sims,
        )

        # Save the position before playing the move whose target it supplies.
        #
        # Train on the visit distribution or Gumbel improved policy independently of move-sampling
        # temperature.
        pending_examples.append(
            PendingExample(
                state=state.encode().astype(np.float32),
                policy=search_output.policy_target,
                player_to_play=state.to_play,
                policy_weight=1.0 if is_full_search else 0.0,
            )
        )

        # Check search Q and the best child's Q for resignation, and track false positives in games played to
        # completion.
        if is_full_search and (
            search_output.root_q < resignation_controller.threshold
            and search_output.best_child_q < resignation_controller.threshold
        ):
            if resignation_disabled:
                # Remember a would-be resignation and allow the game to continue.
                hypothetical_resign_players.add(state.to_play)
            else:
                resigned = True
                winner = -state.to_play
                break

        # Sample early moves and choose late moves nearly deterministically.
        if state.move_number < config.search.temperature_moves:
            action = _sample_action(search_output.policy, rng)
        else:
            action = _argmax_action(search_output.policy)

        # Advance the root along the played action when reusing the tree.
        next_root = search.advance_root(search_output.root, action) if config.search.reuse_tree else None

        moves_played.append(action)
        state = state.play(action)
        root = next_root
    else:
        # Normal loop exhaustion means the board reached a terminal state.
        winner = state.winner_color()
        black_score, white_score = state.area_score()
        score_margin = abs(black_score - white_score)
        final_state = state

    false_positive_resignation = False
    if resignation_disabled and hypothetical_resign_players:
        # A would-be resigning player who wins produces a false-positive observation.
        false_positive_resignation = winner in hypothetical_resign_players
        resignation_controller.observe_false_positive(false_positive_resignation)

    # Compute auxiliary targets once and convert them to each example's player perspective.
    ownership_black: np.ndarray | None = None
    normalized_diff_black = 0.0
    if final_state is not None:
        ownership_black, normalized_diff_black = _aux_targets_from_final_state(final_state)

    examples: list[TrainingExample] = []
    for item in pending_examples:
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

    return SelfPlayResult(
        examples=examples,
        winner=winner,
        num_moves=len(pending_examples),
        resigned=resigned,
        resignation_disabled=resignation_disabled,
        false_positive_resignation=false_positive_resignation,
        moves=moves_played,
        score_margin=score_margin,
        ended_by=game_end_reason(final_state, resigned),
    )

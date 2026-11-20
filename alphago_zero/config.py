from __future__ import annotations

"""Configuration for the reconstruction, from small local experiments to settings modeled on the AlphaGo
Zero paper.
"""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Warn above this leaf batch size in Gumbel search. At 32 simulations on a small 5x5 network, batches
# of 2, 4, 8 and 16 changed the sequential move about 5%, 8%, 18% and 25% of the time. Recheck the
# threshold on the target network with examples/probe_gumbel_batch.py.
GUMBEL_RECOMMENDED_MAX_EVAL_BATCH = 4


def _require_finite_non_negative(name: str, value: float) -> None:
    """Require a finite, nonnegative value. Check the allowed range explicitly because comparisons against
    NaN can silently accept invalid search and scoring settings.
    """
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and >= 0, got {value}")


def _require_finite_positive(name: str, value: float) -> None:
    """Require a finite, positive value. An infinite exploration constant ties scores at infinity, while NaN
    can leave selection without a move.
    """
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")


@dataclass(slots=True)
class NetworkConfig:
    """Policy/value network dimensions. The paper used 17 input planes on a 19x19 board, 256 channels and a
    deep residual tower. The demo uses the same structure at a smaller scale.
    """

    board_size: int = 5
    num_input_planes: int = 17
    channels: int = 64
    num_res_blocks: int = 4
    value_head_hidden: int = 64

    # Optional auxiliary heads predict final ownership at each point (+1 for the current player, -1 for
    # the opponent, 0 for neutral) and a normalized score margin. The shared trunk receives a denser
    # training signal.
    aux_heads: bool = False

    @property
    def action_size(self) -> int:
        """Return the number of board points plus the pass action."""
        return self.board_size * self.board_size + 1


@dataclass(slots=True)
class SearchConfig:
    """Monte Carlo tree search and PUCT settings."""

    num_simulations: int = 64
    c_puct: float = 1.5

    # Collect up to eval_batch_size leaves with virtual loss before one network call. A batch size of one
    # gives sequential search.
    eval_batch_size: int = 8
    virtual_loss: float = 1.0

    # Root noise from the paper, P' = (1 - eps) * p + eps * Dir(alpha).
    root_dirichlet_alpha: float = 0.30
    root_dirichlet_epsilon: float = 0.25

    # The paper samples with tau = 1 for the first 30 moves and then uses a nearly deterministic
    # selection.
    temperature_moves: int = 10
    deterministic_temperature: float = 1e-6

    reuse_tree: bool = True

    # First-play urgency. None assigns zero to unvisited edges. A nonnegative reduction r uses the
    # parent's value minus r.
    fpu_reduction: float | None = None

    # Playout cap randomization mixes deep, noisy searches with cheaper moves. Deep searches provide
    # policy targets. Both kinds of position can contribute outcome and ownership targets.
    playout_cap_prob: float = 0.0
    playout_cap_cheap_sims: int = 0

    # Gumbel search for the trainable side of self-play. Gumbel-Top-k and Sequential Halving allocate root
    # visits. Interior search follows the improved policy, which also provides the training target.
    search_algo: Literal["puct", "gumbel"] = "puct"
    # Number of root actions considered by Sequential Halving, ranked by prior logit plus Gumbel noise.
    gumbel_max_considered: int = 16
    # Root noise scale. A value of one samples from the prior with Gumbel-Top-k. Zero makes search
    # deterministic.
    gumbel_scale: float = 1.0
    # Scale normalized completed Q in [0, 1] by (c_visit + max N) * c_scale. The paper uses 50 and 0.1.
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 0.1

    # Two independent controls for policy target quality. Both default to zero.
    #
    # Give underexplored PUCT root actions at least fraction * prior * total_root_visits visits. Gumbel
    # allocates its root budget through Sequential Halving.
    forced_playout_fraction: float = 0.0
    # Zero training-target entries below fraction * max(target), then renormalize. Move sampling and raw
    # visit counts still use the original distribution.
    policy_target_prune_fraction: float = 0.0

    # Resignation settings.
    resign_threshold: float = -0.95
    resign_min_visits: int = 8
    # In live GTP play, require a minimum move number and optionally a predicted score worse than
    # -resign_score_margin before resigning. Self-play calibrates its own threshold.
    resign_min_move: int = 0
    resign_score_margin: float | None = None
    disable_resign_prob: float = 0.10
    target_false_positive_rate: float = 0.05
    resignation_adjustment: float = 0.02
    resignation_window: int = 100


@dataclass(slots=True)
class TrainingConfig:
    """Training, evaluation and checkpoint settings."""

    replay_buffer_size: int = 50_000
    batch_size: int = 128
    gradient_steps_per_iteration: int = 32
    self_play_games_per_iteration: int = 8
    num_iterations: int = 5

    # Batch leaf evaluations across concurrent games. Each process can fill at most as many slots as it
    # has games. With one slot, games run sequentially.
    parallel_games: int = 1

    # Distribute games across worker processes to use additional CPU cores. Each worker has a network
    # replica on the selected device and batches its own active games.
    selfplay_workers: int = 1

    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # Optionally exclude BatchNorm parameters and biases from weight decay. Changing parameter groups is
    # incompatible with restoring the optimizer from a full snapshot.
    weight_decay_exclude_norm_bias: bool = False
    lr_schedule: Literal["constant", "paper"] = "constant"

    # Auxiliary loss weights apply when the network has auxiliary heads. Keep them small so the shared
    # trunk still learns primarily from policy and value.
    ownership_loss_weight: float = 0.15
    score_loss_weight: float = 0.05

    # Ramp each auxiliary weight by min(1, global_step / aux_warmup_steps). This protects a pretrained
    # trunk from large gradients from freshly initialized heads.
    aux_warmup_steps: int = 0

    # Play a fraction of games against randomly selected frozen league members, with random colors. Both
    # sides supply examples, including expert policy targets from the frozen opponent.
    league_prob: float = 0.0
    league_checkpoints: list[Path | str] = field(default_factory=list)

    # Optionally exclude league games from resignation calibration if their outcomes bias the
    # false-positive estimate.
    exclude_league_from_resign_calibration: bool = False

    # Reserve whole games for monitoring generalization. Seen positions can yield a much lower value MSE
    # than equally recent unseen positions. Holdout examples stay out of the training buffer.
    holdout_game_fraction: float = 0.0
    holdout_buffer_size: int = 8192

    evaluation_games: int = 128
    replace_threshold: float = 0.55
    # The binomial gate requires the win threshold and an exact test. The fixed gate uses the threshold
    # alone. With no gate, promote every candidate and skip the match.
    evaluation_gate: Literal["fixed", "binomial", "none"] = "binomial"
    evaluation_significance: float = 0.05

    # Early stopping ends a match once the promotion decision is mathematically fixed. Sampling the first
    # K moves varies openings across evaluation games.
    evaluation_early_stop: bool = True
    evaluation_opening_moves: int = 0

    # Benchmark the best network against fixed random and initial raw-policy opponents every N iterations.
    # Zero disables this evaluation.
    benchmark_every: int = 0
    benchmark_games: int = 8

    # Add a frozen MCTS checkpoint with a known rating when the model wins every game against the weaker
    # anchors. Set its rating independently.
    benchmark_anchor_checkpoint: Path | str | None = None
    benchmark_anchor_rating: float = 0.0

    seed: int = 42
    save_every_iteration: bool = True

    # Save candidate_iter_NNN.pt and best.pt at regular intervals even when per-iteration snapshots are
    # disabled. Zero disables periodic checkpoints.
    checkpoint_every: int = 0

    def self_play_games_per_process(self) -> int:
        """Return the largest game allocation to a worker. The first workers receive any remainder from an
        even split.
        """
        workers = max(1, self.selfplay_workers)
        return -(-self.self_play_games_per_iteration // workers)

    def effective_parallel_games(self) -> int:
        """Return the number of slots that can be active in the busiest worker."""
        return max(1, min(self.parallel_games, self.self_play_games_per_process()))


@dataclass(slots=True)
class GameConfig:
    """Go rules and game length."""

    board_size: int = 5
    komi: float = 2.5
    max_moves_multiplier: int = 2

    # Positional superko rejects a repeated board. Situational superko also considers the next player.
    # Both variants allow passing.
    superko_rule: Literal["positional", "situational"] = "positional"

    @property
    def max_moves(self) -> int:
        # The paper used a limit of 19 * 19 * 2 = 722 moves.
        return self.board_size * self.board_size * self.max_moves_multiplier


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime, device and filesystem settings."""

    device: str = "cpu"
    work_dir: Path = field(default_factory=lambda: Path("runs"))

    # Use bfloat16 autocast in training. Its exponent range matches float32, so gradient scaling is
    # unnecessary.
    use_amp: bool = False

    # Resume the training loop from a snapshot containing the optimizer, replay buffer and RNG states. A
    # save interval of zero disables resume snapshots.
    resume_from: Path | str | None = None
    save_resume_every: int = 1

    # Warm-start the replay buffer with fresh weights and optimizer state. This permits changing network
    # width or depth while keeping experience.
    warmstart_buffer_from: Path | str | None = None

    # Warm-start weights by name and shape. Newly introduced parameters keep their initial values. This
    # option can be combined with a replay-buffer warm start.
    init_weights_from: Path | str | None = None

    # Export self-play games under work_dir/sgf for review in an SGF viewer.
    save_selfplay_sgf: bool = False

    # Stop at the first iteration boundary after the time budget expires so the final artifacts are
    # complete.
    time_budget_sec: float | None = None


@dataclass(slots=True)
class AlphaGoZeroConfig:
    """Top-level experiment configuration."""

    game: GameConfig = field(default_factory=GameConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def _validate_gumbel_search(self) -> None:
        """Validate Gumbel settings even when PUCT is selected, so a later change of search algorithm cannot
        activate an invalid configuration.
        """
        search = self.search
        if search.search_algo not in ("puct", "gumbel"):
            raise ValueError(
                f"search_algo must be \'puct\' or \'gumbel\', got {search.search_algo!r}"
            )
        if search.gumbel_max_considered < 2:
            # Sequential Halving needs at least two candidates for a comparison.
            raise ValueError(
                f"gumbel_max_considered must be >= 2, got {search.gumbel_max_considered}"
            )
        _require_finite_non_negative("gumbel_scale", search.gumbel_scale)
        _require_finite_non_negative("gumbel_c_visit", search.gumbel_c_visit)
        _require_finite_positive("gumbel_c_scale", search.gumbel_c_scale)
        for name in ("forced_playout_fraction", "policy_target_prune_fraction"):
            value = getattr(search, name)
            if not (math.isfinite(value) and 0.0 <= value < 1.0):
                raise ValueError(
                    f"{name} must be finite and in [0, 1), got {value}"
                )
        if search.search_algo == "gumbel" and search.forced_playout_fraction > 0.0:
            # Sequential Halving already allocates the root budget. A second allocator would invalidate its
            # schedule.
            raise ValueError(
                "search_algo='gumbel' is incompatible with forced playouts "
                f"(forced_playout_fraction={search.forced_playout_fraction}): "
                "keep forced_playout_fraction=0"
            )
        if search.search_algo == "gumbel" and search.playout_cap_prob > 0.0:
            # Sequential Halving already produces improvement targets at low budgets. Mixing it with cheap
            # PUCT-style searches would change the meaning of those targets.
            raise ValueError(
                "search_algo='gumbel' is incompatible with playout cap randomization "
                f"(playout_cap_prob={search.playout_cap_prob}): keep playout_cap_prob=0"
            )
        if search.search_algo == "gumbel" and search.eval_batch_size > GUMBEL_RECOMMENDED_MAX_EVAL_BATCH:
            # Warn when batched inference makes elimination decisions before earlier leaf evaluations return.
            warnings.warn(
                f"search_algo=\'gumbel\' with eval_batch_size={search.eval_batch_size}: Sequential "
                "Halving can eliminate moves while leaf evaluations are pending. In a "
                "5x5 test with 32 simulations, batches of 2, 4, 8 and 16 changed "
                "the sequential move in about 5%, 8%, 18% and 25% of positions "
                "(docs/rnd_experiments.md and examples/probe_gumbel_batch.py). Use "
                f"eval_batch_size <= {GUMBEL_RECOMMENDED_MAX_EVAL_BATCH} with a larger parallel_games value "
                "to fill inference batches across games.",
                UserWarning,
                stacklevel=3,
            )

    def validate(self) -> None:
        """Validate the configuration before allocating models or producing run artifacts."""
        # Game and network compatibility
        if self.game.board_size != self.network.board_size:
            raise ValueError(
                "game and network board_size must match: "
                f"{self.game.board_size} != {self.network.board_size}"
            )

        # Board dimensions and rules
        if self.game.board_size <= 1:
            raise ValueError(f"board_size must be > 1, got {self.game.board_size}")
        _require_finite_non_negative("komi", self.game.komi)
        if self.game.max_moves_multiplier <= 0:
            raise ValueError(
                f"max_moves_multiplier must be > 0, got {self.game.max_moves_multiplier}"
            )
        if self.game.superko_rule not in ("positional", "situational"):
            raise ValueError(
                "superko_rule must be 'positional' or 'situational', "
                f"got {self.game.superko_rule!r}"
            )

        # Eight historical boards for each color and one side-to-play plane.
        if self.network.num_input_planes != 17:
            raise ValueError(
                "num_input_planes must be 17 for this state representation, "
                f"got {self.network.num_input_planes}"
            )
        if self.network.channels <= 0:
            raise ValueError(f"channels must be > 0, got {self.network.channels}")
        if self.network.num_res_blocks < 0:
            raise ValueError(
                f"num_res_blocks must be nonnegative, got {self.network.num_res_blocks}"
            )
        if self.network.value_head_hidden <= 0:
            raise ValueError(
                f"value_head_hidden must be > 0, got {self.network.value_head_hidden}"
            )

        # Search settings
        if self.search.num_simulations <= 0:
            # Zero simulations would leave self-play without a visit distribution.
            raise ValueError(
                f"num_simulations must be > 0, got {self.search.num_simulations}"
            )
        _require_finite_positive("c_puct", self.search.c_puct)
        if self.search.eval_batch_size < 1:
            raise ValueError(
                f"eval_batch_size must be >= 1, got {self.search.eval_batch_size}"
            )
        if self.search.resignation_window < 1:
            raise ValueError(
                f"resignation_window must be >= 1, got {self.search.resignation_window}"
            )
        # An invalid virtual loss corrupts every edge on its path, even after backup attempts to undo the
        # penalty.
        _require_finite_non_negative("virtual_loss", self.search.virtual_loss)
        if self.search.resign_min_visits < 1:
            raise ValueError("resign_min_visits must be >= 1")
        fpu = self.search.fpu_reduction
        if fpu is not None and not (math.isfinite(fpu) and fpu >= 0.0):
            raise ValueError(f"fpu_reduction must be None or finite and >= 0, got {fpu}")
        if not (0.0 <= self.search.playout_cap_prob <= 1.0):
            raise ValueError(
                "playout_cap_prob must be in [0, 1], "
                f"got {self.search.playout_cap_prob}"
            )
        if self.search.playout_cap_cheap_sims < 0:
            raise ValueError(
                "playout_cap_cheap_sims must be nonnegative, "
                f"got {self.search.playout_cap_cheap_sims}"
            )
        if (
            0.0 < self.search.playout_cap_prob < 1.0
            and self.search.playout_cap_cheap_sims == 0
        ):
            raise ValueError(
                "when playout cap randomization is enabled (0 < playout_cap_prob < 1), "
                "playout_cap_cheap_sims must be > 0"
            )
        if (
            self.search.playout_cap_cheap_sims > 0
            and self.search.playout_cap_cheap_sims >= self.search.num_simulations
        ):
            raise ValueError(
                "playout_cap_cheap_sims must be less than num_simulations "
                f"for a cheap search, got {self.search.playout_cap_cheap_sims} "
                f">= {self.search.num_simulations}"
            )
        if not (0.0 <= self.search.root_dirichlet_epsilon <= 1.0):
            raise ValueError(
                "root_dirichlet_epsilon must be in [0, 1], "
                f"got {self.search.root_dirichlet_epsilon}"
            )
        _require_finite_positive("root_dirichlet_alpha", self.search.root_dirichlet_alpha)
        # Temperatures at or below 1e-6 use the deterministic branch in _policy_from_visit_counts.
        _require_finite_non_negative(
            "deterministic_temperature", self.search.deterministic_temperature
        )
        if not (0.0 <= self.search.disable_resign_prob <= 1.0):
            raise ValueError(
                "disable_resign_prob must be in [0, 1], "
                f"got {self.search.disable_resign_prob}"
            )
        if not (-1.0 <= self.search.resign_threshold <= 0.0):
            # The resignation threshold is a value in [-1, 0]. Values near -1 predict a loss.
            raise ValueError(
                "resign_threshold must be in [-1, 0], "
                f"got {self.search.resign_threshold}"
            )
        if self.search.resign_min_move < 0:
            raise ValueError(
                f"resign_min_move must be >= 0, got {self.search.resign_min_move}"
            )
        resign_score_margin = self.search.resign_score_margin
        if resign_score_margin is not None and not (
            math.isfinite(resign_score_margin) and resign_score_margin >= 0.0
        ):
            # Reject NaN, which blocks all comparisons, and negative margins, which reverse the intended score
            # guard.
            raise ValueError(
                "resign_score_margin must be None or finite and >= 0, "
                f"got {resign_score_margin}"
            )
        if not (0.0 <= self.search.target_false_positive_rate <= 1.0):
            # Calibration compares the observed false-positive rate with this target. Reject values outside the
            # probability range.
            raise ValueError(
                "target_false_positive_rate must be in [0, 1], "
                f"got {self.search.target_false_positive_rate}"
            )
        # A zero step freezes calibration. A negative step would move the threshold in the wrong direction.
        _require_finite_non_negative(
            "resignation_adjustment", self.search.resignation_adjustment
        )
        self._validate_gumbel_search()

        # Training settings
        if self.training.replay_buffer_size <= 0:
            raise ValueError(
                f"replay_buffer_size must be > 0, got {self.training.replay_buffer_size}"
            )
        if self.training.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.training.batch_size}")
        if self.training.gradient_steps_per_iteration < 0:
            raise ValueError(
                "gradient_steps_per_iteration must be nonnegative, "
                f"got {self.training.gradient_steps_per_iteration}"
            )
        if self.training.self_play_games_per_iteration < 0:
            raise ValueError(
                "self_play_games_per_iteration must be nonnegative, "
                f"got {self.training.self_play_games_per_iteration}"
            )
        if self.training.parallel_games < 1:
            raise ValueError(
                f"parallel_games must be >= 1, got {self.training.parallel_games}"
            )
        if self.training.selfplay_workers < 1:
            raise ValueError(
                f"selfplay_workers must be >= 1, got {self.training.selfplay_workers}"
            )
        games_per_process = self.training.self_play_games_per_process()
        if 0 < games_per_process < self.training.parallel_games:
            # Allow idle slots for compatibility with saved configurations. Warn so that throughput estimates use
            # the number of active slots.
            warnings.warn(
                f"parallel_games={self.training.parallel_games} exceeds the number of games per worker "
                f"per self-play iteration ({games_per_process} = "
                f"{self.training.self_play_games_per_iteration} games across "
                f"{self.training.selfplay_workers} worker(s)). Active slots: "
                f"{games_per_process} instead of {self.training.parallel_games}. To achieve the "
                "requested inference batch width, increase self_play_games_per_iteration "
                "(see docs/performance.md).",
                UserWarning,
                stacklevel=2,
            )
        holdout_fraction = self.training.holdout_game_fraction
        if not (math.isfinite(holdout_fraction) and 0.0 <= holdout_fraction < 1.0):
            # Holding out every game would leave SGD without data.
            raise ValueError(
                "holdout_game_fraction must be finite and in [0, 1), "
                f"got {holdout_fraction}"
            )
        if self.training.holdout_buffer_size < 1:
            raise ValueError(
                f"holdout_buffer_size must be >= 1, got {self.training.holdout_buffer_size}"
            )
        # Validate the fixed learning rate even when the paper schedule currently overrides it.
        _require_finite_non_negative("learning_rate", self.training.learning_rate)
        _require_finite_non_negative("momentum", self.training.momentum)
        _require_finite_non_negative("weight_decay", self.training.weight_decay)
        if self.runtime.save_resume_every < 0:
            raise ValueError(
                "save_resume_every must be nonnegative, "
                f"got {self.runtime.save_resume_every}"
            )
        if self.training.checkpoint_every < 0:
            raise ValueError(
                "checkpoint_every must be nonnegative, "
                f"got {self.training.checkpoint_every}"
            )
        budget = self.runtime.time_budget_sec
        if budget is not None and not (math.isfinite(budget) and budget > 0.0):
            # A nonfinite time budget can silently turn a bounded run into an unlimited one.
            raise ValueError(
                f"time_budget_sec must be a positive finite number or None, got {budget}"
            )
        if (
            self.runtime.resume_from is not None
            and self.runtime.warmstart_buffer_from is not None
        ):
            raise ValueError(
                "resume_from and warmstart_buffer_from are mutually exclusive. The former "
                "restores weights, optimizer state and replay. The latter "
                "copies only replay into a new run with fresh weights."
            )
        if (
            self.runtime.resume_from is not None
            and self.runtime.init_weights_from is not None
        ):
            raise ValueError(
                "resume_from and init_weights_from are mutually exclusive. Resume already "
                "restores weights. Use init_weights_from to initialize weights "
                "in a new run, optionally together with warmstart_buffer_from."
            )
        if self.training.evaluation_opening_moves < 0:
            raise ValueError(
                "evaluation_opening_moves must be nonnegative, "
                f"got {self.training.evaluation_opening_moves}"
            )
        if self.training.benchmark_every < 0:
            raise ValueError(
                f"benchmark_every must be nonnegative, got {self.training.benchmark_every}"
            )
        if self.training.benchmark_games < 1:
            raise ValueError(
                f"benchmark_games must be >= 1, got {self.training.benchmark_games}"
            )
        if not math.isfinite(self.training.benchmark_anchor_rating):
            # Ratings are relative and may be negative. A nonfinite anchor invalidates the likelihood calculation.
            raise ValueError(
                "benchmark_anchor_rating must be finite, "
                f"got {self.training.benchmark_anchor_rating}"
            )
        if self.training.num_iterations < 0:
            raise ValueError(
                f"num_iterations must be nonnegative, got {self.training.num_iterations}"
            )
        if self.training.evaluation_games < 0:
            raise ValueError(
                f"evaluation_games must be nonnegative, got {self.training.evaluation_games}"
            )
        if self.training.evaluation_gate not in ("fixed", "binomial", "none"):
            raise ValueError(
                "evaluation_gate must be 'fixed', 'binomial' or 'none', "
                f"got {self.training.evaluation_gate!r}"
            )
        if not 0.0 < self.training.evaluation_significance < 0.5:
            raise ValueError("evaluation_significance must be in (0, 0.5)")
        if (
            self.training.evaluation_gate == "binomial"
            and self.training.evaluation_games > 0
            and 2.0 ** (-self.training.evaluation_games) > self.training.evaluation_significance
        ):
            raise ValueError("evaluation_games is too small for evaluation_significance")
        if not (0.0 <= self.training.replace_threshold <= 1.0):
            # The promotion threshold is a fraction of games won.
            raise ValueError(
                "replace_threshold must be in [0, 1], "
                f"got {self.training.replace_threshold}"
            )
        _require_finite_non_negative("ownership_loss_weight", self.training.ownership_loss_weight)
        _require_finite_non_negative("score_loss_weight", self.training.score_loss_weight)
        if self.training.aux_warmup_steps < 0:
            raise ValueError(
                "aux_warmup_steps must be nonnegative, "
                f"got {self.training.aux_warmup_steps}"
            )
        if not (0.0 <= self.training.league_prob <= 1.0):
            raise ValueError(
                f"league_prob must be in [0, 1], got {self.training.league_prob}"
            )
        if self.training.league_prob > 0.0:
            if not self.training.league_checkpoints:
                raise ValueError(
                    "league_prob > 0 requires at least one checkpoint in league_checkpoints"
                )
            if self.training.parallel_games < 2:
                raise ValueError(
                    "league play uses the parallel self-play driver: "
                    "league_prob > 0 requires parallel_games > 1"
                )
        elif self.training.exclude_league_from_resign_calibration:
            raise ValueError(
                "exclude_league_from_resign_calibration requires "
                "league_prob > 0 to have league games available"
            )

    @property
    def action_size(self) -> int:
        return self.network.action_size


def paper_learning_rate(step: int) -> float:
    """Return the learning rate for an optimizer step using Extended Data Table 3. The table expresses
    milestones in thousands of steps.
    """
    kstep = step / 1000.0
    if kstep < 400:
        return 1e-2
    if kstep < 600:
        return 1e-3
    return 1e-4


def make_demo_config(work_dir: str | Path = "runs/demo") -> AlphaGoZeroConfig:
    """Build a small configuration for running the entire training loop locally."""
    cfg = AlphaGoZeroConfig(
        game=GameConfig(board_size=5, komi=2.5),
        network=NetworkConfig(board_size=5, channels=64, num_res_blocks=4, value_head_hidden=64),
        search=SearchConfig(
            num_simulations=32,
            c_puct=1.5,
            root_dirichlet_alpha=0.30,
            root_dirichlet_epsilon=0.25,
            temperature_moves=8,
            deterministic_temperature=1e-6,
            reuse_tree=True,
            resign_threshold=-0.95,
            disable_resign_prob=0.10,
        ),
        training=TrainingConfig(
            replay_buffer_size=10_000,
            batch_size=64,
            gradient_steps_per_iteration=16,
            self_play_games_per_iteration=4,
            num_iterations=3,
            learning_rate=1e-2,
            momentum=0.9,
            weight_decay=1e-4,
            lr_schedule="constant",
            evaluation_games=4,
            evaluation_gate="fixed",
            replace_threshold=0.55,
        ),
        runtime=RuntimeConfig(device="cpu", work_dir=Path(work_dir)),
    )
    cfg.validate()
    return cfg


def make_paper_like_config(work_dir: str | Path = "runs/paper_like") -> AlphaGoZeroConfig:
    """Build a configuration with the published board, model and search settings. Reproducing the original
    training run requires substantially more compute.
    """
    cfg = AlphaGoZeroConfig(
        game=GameConfig(board_size=19, komi=7.5),
        network=NetworkConfig(
            board_size=19,
            channels=256,
            num_res_blocks=19,
            value_head_hidden=256,
        ),
        search=SearchConfig(
            num_simulations=1600,
            c_puct=1.5,
            root_dirichlet_alpha=0.03,
            root_dirichlet_epsilon=0.25,
            temperature_moves=30,
            deterministic_temperature=1e-6,
            reuse_tree=True,
            resign_threshold=-0.95,
            disable_resign_prob=0.10,
        ),
        training=TrainingConfig(
            replay_buffer_size=500_000,
            batch_size=2048,
            gradient_steps_per_iteration=1000,
            self_play_games_per_iteration=25_000,
            num_iterations=700,
            learning_rate=1e-2,
            momentum=0.9,
            weight_decay=1e-4,
            lr_schedule="paper",
            evaluation_games=400,
            evaluation_gate="fixed",
            replace_threshold=0.55,
        ),
        runtime=RuntimeConfig(device="cpu", work_dir=Path(work_dir)),
    )
    cfg.validate()
    return cfg

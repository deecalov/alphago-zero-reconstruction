from __future__ import annotations

"""Evaluate a trained candidate against the current best network before promotion."""

from dataclasses import dataclass
import math

import numpy as np

from .config import AlphaGoZeroConfig
from .game import BLACK, WHITE, GoState
from .mcts import AlphaGoZeroMCTS
from .network import AlphaGoZeroNet


@dataclass(slots=True)
class EvaluationSummary:
    """Candidate match summary. Integer komi can produce draws. Half-integer komi breaks ties under area scoring."""

    candidate_wins: int
    best_wins: int
    draws: int
    games: int
    candidate_win_rate: float
    stopped_early: bool = False
    promoted: bool = False
    required_wins: int = 0
    games_planned: int = 0
    gate: str = "fixed"


def required_wins_for_promotion(
    games: int, threshold: float, significance: float | None = None,
) -> int:
    """Find the minimum win count that passes the gate. Use exactly the same floating-point comparison as
    the promotion decision.
    """
    if games <= 0:
        return 0
    # Correct any one-game rounding discrepancy in the initial ceiling estimate.
    wins = max(0, min(games, math.ceil(threshold * games)))
    while wins > 0 and (wins - 1) / games >= threshold:
        wins -= 1
    while wins <= games and wins / games < threshold:
        wins += 1
    if significance is not None:
        # Exact one-sided test of p <= 0.5 at the PREDECLARED sample size.
        # Integer binomial coefficients avoid cancellation in the upper tail.
        denominator = 1 << games
        tail = 0
        critical = games + 1
        for k in range(games, -1, -1):
            tail += math.comb(games, k)
            if tail / denominator > significance:
                break
            critical = k
        wins = max(wins, critical)
    return wins


def play_evaluation_game(
    black_model: AlphaGoZeroNet,
    white_model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    rng: np.random.Generator,
) -> int | None:
    """Play one evaluation game. Disable root noise and resignation. Return None for a draw."""
    state = GoState.new_game(
        board_size=config.game.board_size,
        komi=config.game.komi,
        max_moves=config.game.max_moves,
        superko_rule=config.game.superko_rule,
    )

    # Switch both networks to evaluation mode once before the search loop.
    black_model.eval()
    white_model.eval()

    # Keep separate search trees because the networks supply different priors and values.
    black_search = AlphaGoZeroMCTS(black_model, config.search, rng)
    white_search = AlphaGoZeroMCTS(white_model, config.search, rng)

    while not state.is_terminal():
        current_search = black_search if state.to_play == BLACK else white_search
        output = current_search.run(
            root_state=state,
            temperature=config.search.deterministic_temperature,
            add_root_noise=False,
            root=None,  # Discard the tree when switching models.
        )

        # Sample opening moves from visit counts to avoid replaying one deterministic game.
        if state.move_number < config.training.evaluation_opening_moves:
            distribution = output.visit_distribution
            total = float(distribution.sum())
            if total > 0.0:
                action = int(rng.choice(len(distribution), p=distribution / total))
            else:
                action = int(np.argmax(output.policy))
        else:
            action = int(np.argmax(output.policy))
        state = state.play(action)

    return state.winner_color()


def evaluate_candidate(
    candidate_model: AlphaGoZeroNet,
    best_model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    rng: np.random.Generator,
) -> EvaluationSummary:
    """Evaluate the candidate with alternating colors. Early stopping is valid once the remaining results
    cannot affect promotion.
    """
    if config.training.evaluation_gate == "none":
        return EvaluationSummary(
            candidate_wins=0,
            best_wins=0,
            draws=0,
            games=0,
            candidate_win_rate=0.0,
            promoted=True,
            gate="none",
        )

    candidate_wins = 0
    best_wins = 0
    draws = 0
    games_planned = config.training.evaluation_games
    games_played = 0
    stopped_early = False

    required_wins = required_wins_for_promotion(
        games_planned, config.training.replace_threshold,
        config.training.evaluation_significance if config.training.evaluation_gate == "binomial" else None,
    )

    for game_idx in range(games_planned):
        if game_idx % 2 == 0:
            winner = play_evaluation_game(
                black_model=candidate_model,
                white_model=best_model,
                config=config,
                rng=rng,
            )
            candidate_color = BLACK
        else:
            winner = play_evaluation_game(
                black_model=best_model,
                white_model=candidate_model,
                config=config,
                rng=rng,
            )
            candidate_color = WHITE

        games_played += 1
        if winner == candidate_color:
            candidate_wins += 1
        elif winner == -candidate_color:
            best_wins += 1
        else:
            draws += 1

        if config.training.evaluation_early_stop:
            remaining = games_planned - games_played
            promotion_secured = candidate_wins >= required_wins
            promotion_impossible = candidate_wins + remaining < required_wins
            if (promotion_secured or promotion_impossible) and remaining > 0:
                stopped_early = True
                break

    candidate_win_rate = candidate_wins / games_played if games_played > 0 else 0.0
    return EvaluationSummary(
        candidate_wins=candidate_wins,
        best_wins=best_wins,
        draws=draws,
        games=games_played,
        candidate_win_rate=float(candidate_win_rate),
        stopped_early=stopped_early,
        promoted=games_played > 0 and candidate_wins >= required_wins,
        required_wins=required_wins,
        games_planned=games_planned,
        gate=config.training.evaluation_gate,
    )

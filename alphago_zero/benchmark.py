from __future__ import annotations

"""Measure playing strength against fixed opponents. The random player defines the zero point of the Elo
scale. A raw-policy player and a frozen checkpoint can provide stronger anchors as training progresses.
"""

from typing import Any, Mapping

import numpy as np

from .config import AlphaGoZeroConfig
from .elo import MatchRecord, estimate_rating
from .game import BLACK, WHITE
from .network import AlphaGoZeroNet
from .players import MCTSPlayer, Player, play_match


def play_anchor_series(
    candidate: Player,
    anchor: Player,
    config: AlphaGoZeroConfig,
    games: int,
) -> tuple[int, int, int]:
    """Play alternating colors and return wins, losses and draws from the candidate's perspective. Integer
    komi can produce a draw.
    """
    wins = 0
    losses = 0
    draws = 0
    for game_idx in range(games):
        if game_idx % 2 == 0:
            winner = play_match(
                black=candidate,
                white=anchor,
                board_size=config.game.board_size,
                komi=config.game.komi,
                max_moves=config.game.max_moves,
                superko_rule=config.game.superko_rule,
            )
            candidate_color = BLACK
        else:
            winner = play_match(
                black=anchor,
                white=candidate,
                board_size=config.game.board_size,
                komi=config.game.komi,
                max_moves=config.game.max_moves,
                superko_rule=config.game.superko_rule,
            )
            candidate_color = WHITE
        if winner is None:
            draws += 1
        elif winner == candidate_color:
            wins += 1
        else:
            losses += 1
    return wins, losses, draws


def benchmark_model(
    model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    rng: np.random.Generator,
    anchors: Mapping[str, tuple[Player, float]],
    games_per_anchor: int,
) -> dict[str, Any]:
    """Play the MCTS model against each named (agent, rating) anchor. Return per-opponent scores and the
    maximum-likelihood Elo estimate for metrics.json.
    """
    model.eval()
    candidate = MCTSPlayer(model=model, search_config=config.search, rng=rng)

    results: dict[str, Any] = {}
    records: list[MatchRecord] = []
    for name, (anchor, anchor_rating) in anchors.items():
        wins, losses, draws = play_anchor_series(candidate, anchor, config, games_per_anchor)
        games_played = wins + losses + draws
        results[name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games_played,
            # Each draw earns half a point, as in MatchRecord.points.
            "win_rate": (wins + 0.5 * draws) / max(1, games_played),
            "anchor_rating": anchor_rating,
        }
        records.append(
            MatchRecord(opponent_rating=anchor_rating, wins=wins, losses=losses, draws=draws)
        )

    return {
        "anchors": results,
        "elo_estimate": round(estimate_rating(records), 1),
        "games_per_anchor": games_per_anchor,
    }

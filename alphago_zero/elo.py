from __future__ import annotations

"""Estimate Elo ratings from match results with the logistic expected-score model."""

from dataclasses import dataclass
from typing import Sequence

# Bracket the solution within 2,000 Elo of the anchors.
_RATING_SEARCH_HALF_WIDTH = 2000.0
# Add one virtual draw to keep estimates finite after a sweep.
_SMOOTHING_GAMES = 1.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass(slots=True)
class MatchRecord:
    """Results against one rated opponent. A draw contributes half a point and one game."""

    opponent_rating: float
    wins: int
    losses: int
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def points(self) -> float:
        """Total points, counting a win as one and a draw as one half."""
        return self.wins + 0.5 * self.draws


def rating_from_win_rate(win_rate: float, opponent_rating: float = 0.0) -> float:
    """Infer a rating from the score against one opponent. Clamp extreme fractions to keep the estimate finite."""
    epsilon = 1e-3
    clamped = min(max(win_rate, epsilon), 1.0 - epsilon)
    import math

    return opponent_rating + 400.0 * math.log10(clamped / (1.0 - clamped))


def estimate_rating(records: Sequence[MatchRecord]) -> float:
    """Estimate a rating against multiple anchors by bisection of the likelihood equation. Smooth each
    series with one virtual draw.
    """
    played = [r for r in records if r.games > 0]
    if not played:
        raise ValueError("At least one completed game is required to estimate a rating")

    def excess_wins(rating: float) -> float:
        total = 0.0
        for record in played:
            wins = record.points + _SMOOTHING_GAMES / 2.0
            games = record.games + _SMOOTHING_GAMES
            total += wins - games * expected_score(rating, record.opponent_rating)
        return total

    low = min(r.opponent_rating for r in played) - _RATING_SEARCH_HALF_WIDTH
    high = max(r.opponent_rating for r in played) + _RATING_SEARCH_HALF_WIDTH

    # The endpoint signs bracket the root. Sixty bisections suffice at double precision.
    for _ in range(60):
        mid = (low + high) / 2.0
        if excess_wins(mid) > 0.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0
